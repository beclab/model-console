// Package lifecycle is the state machine that gets a model from "process
// just booted" to "ready to serve requests". It is the only component
// allowed to write phase transitions on progress.Manager: every other
// piece of llm-init reports raw events (OnBytes / OnFileDone / OnError),
// and lifecycle decides what those events mean.
//
// The lifecycle state machine is implemented in this package.
// Startup order branches on Adapter.AliveBeforeBoot(); the two orders
// follow this boot order:
//
//   - ollama (AliveBeforeBoot()==true):
//     1. WaitAlive (poll /api/tags until 200; the daemon was brought
//     up by docker-compose before llm-init started).
//     2. Spawn engineHealthLoop (10s tick on adapter.Ready()).
//     3. for { ensure() ; wait retry/ctx } — ensure() iterates
//     cfg.Sources and dispatches per ms.Kind.
//
//   - proxy / vllm / llamacpp / sglang (AliveBeforeBoot()==false):
//     1. for { ensure() ; WaitAlive ; PhaseReady ; wait retry/ctx } —
//     each iteration downloads (idempotent on re-ensure), writes the
//     sentinel the engine wrapper blocks on, then waits for the engine
//     to report alive before flipping PhaseReady.
//     2. Refresh lastReady so /readyz can flip immediately.
//     3. Spawn engineHealthLoop after the first successful WaitAlive.
//
// Lifecycle consumes cfg.Sources directly via ms.Kind dispatch. The first
// source drives the engine-facing path and later sources are extras.
package lifecycle

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/obs"
	"github.com/llm-init/llm-init/internal/progress"
)

// HFRunFunc is the function signature lifecycle calls per HF download.
// Defaults to hfwrap.Run; tests inject a stub so they don't have to
// stand up a real Python interpreter.
type HFRunFunc func(ctx context.Context, cfg hfwrap.Config, sink progress.Sink) (hfwrap.Result, error)

// Options bundles dependencies. New rejects an Options without Adapter,
// Manager, or Downloader because there is no sane stub for any of
// them in production. As of v1.1.0 there is no Source field; lifecycle
// reads cfg.Sources directly.
type Options struct {
	Config     config.Config
	Adapter    adapter.Adapter
	Manager    progress.Manager
	Downloader fetch.RangeDownloader

	// HealthInterval defaults to 10s; tests override to keep CI fast.
	HealthInterval time.Duration

	// EngineDownGrace is how long the engine must keep failing its
	// health probe before a ready model is called degraded. Defaults to
	// defaultEngineDownGrace, which is sized against the engine restart
	// a model-card edit triggers; see its godoc before shortening it.
	EngineDownGrace time.Duration

	// RelaunchInFlight reports whether a relaunch of the engine has been
	// asked for and not yet completed, normally
	// handoff.Tracker.Relaunching. Nil leaves the grace window as the
	// only thing standing between a failing probe and `degraded`.
	//
	// A failing probe is ambiguous on its own: an engine being relaunched
	// in place and an engine that crashed look identical from here, which
	// is why EngineDownGrace has to be long enough to cover a relaunch.
	// The cost of that is a relaunch reporting nothing at all — the phase
	// holds at `ready` through the whole dip and lands back on `ready`,
	// so a console that asked for one has nothing to show for it. This
	// hook is the missing fact, and it is a function rather than a
	// dependency because the tracker that knows it is owned by the
	// control plane, not by the state machine.
	RelaunchInFlight func() bool

	// RelaunchProbeInterval replaces HealthInterval while RelaunchInFlight
	// reports one. Defaults to defaultRelaunchProbeInterval; a value that
	// is not shorter than HealthInterval is ignored, since the point is to
	// look more often rather than less.
	RelaunchProbeInterval time.Duration

	// RelaunchRequested tells the health loop to end its current wait,
	// so the gap after it is chosen knowing a relaunch is outstanding.
	// Without it the switch to RelaunchProbeInterval only takes effect at
	// the end of a wait that was armed before the signal — up to a whole
	// HealthInterval, which is longer than the relaunch it is meant to
	// catch. Wire it to handoff.Config.OnRequest through a buffered
	// channel of one; nil leaves the loop on its own schedule.
	RelaunchRequested <-chan struct{}

	// MaxConcurrentDownloads caps the parallel fan-out for URL fetches
	// and the hf wrapper's --max-workers flag. 0 → 4.
	MaxConcurrentDownloads int

	// HFCacheRoot is HF_HUB_CACHE. Empty means defaultHFCacheRoot, which
	// is what the deploy contract pins it to; only tests set it, so a
	// pass that reads or writes the cache stays in a temp directory.
	HFCacheRoot string

	// MaxDownloadBytes caps total bytes written per URL download
	// (fetch.RangeOptions.MaxBytes). 0 means unlimited.
	MaxDownloadBytes int64

	// RetriableBackoff is the sleep window between successive ensure
	// passes after a *retriable* failure. Defaults to 30s; tests use
	// a smaller value to keep CI fast. Permanent failures sit idle
	// regardless of this value.
	RetriableBackoff time.Duration

	// NowFunc is the time source. Defaults to time.Now.
	NowFunc func() time.Time

	// Metrics is the Prometheus registry used by healthLoop and
	// ensure. May be nil.
	Metrics *obs.Metrics

	// HFRun lets tests substitute a fake hfwrap.Run that never spawns
	// a subprocess. Production wiring leaves this nil so ensureHF
	// uses hfwrap.Run directly.
	HFRun HFRunFunc
}

// RetryOptions are the per-iteration overrides supplied via
// `POST /api/retry?force=…&level=…`.
type RetryOptions struct {
	// Force, when true, makes the next ensure iteration re-fetch from
	// scratch instead of reusing already-complete bytes: the download
	// records are dropped, the URL path deletes the local file
	// (+ .part), and the HF path sets ForceReload.
	Force bool

	// Level overrides VERIFY_LEVEL for this one pass: "size", "sha256"
	// or "remote". Anything else falls back to the configured default
	// rather than failing the pass — see passLevel.
	//
	// "remote" is reachable only from here. As a standing default it
	// would make every boot depend on the upstream being up, which is
	// the thing the download records exist to avoid.
	Level string
}

// Manager is the lifecycle state machine.
type Manager struct {
	opts Options

	// installer is how this engine is told about a model, resolved once
	// from the adapter. Ollama has something to do here; the engines
	// that read a path do not, and get one that does nothing, so no
	// call site asks which kind it has.
	installer adapter.ModelInstaller

	// pass is the current ensure pass, owned by the goroutine running
	// it. See passState.
	pass *passState

	mu          sync.Mutex
	retry       chan struct{}
	pendingOpts RetryOptions
	stopped     atomic.Bool
	lastReady   atomic.Pointer[adapter.ReadyState]

	// unhealthySince is when the current run of failing probes started,
	// nil while the engine is answering. Written only by the health
	// loop; atomic so a test can read it without racing that goroutine.
	unhealthySince atomic.Pointer[time.Time]

	// relaunchLoading records that the health loop is the one who put the
	// current PhaseLoading there, so it may take it away again. Without
	// it the loop could not tell its own loading from the one ensure()
	// writes while a model boots, and would call a model ready off a
	// probe ensure had not finished acting on. Cleared at the top of
	// every ensure pass, which is the moment that stops being true.
	relaunchLoading atomic.Bool
}

// New returns a Manager.
func New(opts Options) (*Manager, error) {
	if opts.Adapter == nil {
		return nil, errors.New("lifecycle: Adapter is required")
	}
	if opts.Manager == nil {
		return nil, errors.New("lifecycle: Manager is required")
	}
	if opts.Downloader == nil {
		return nil, errors.New("lifecycle: Downloader is required")
	}
	if opts.HealthInterval == 0 {
		opts.HealthInterval = 10 * time.Second
	}
	if opts.RelaunchProbeInterval == 0 {
		opts.RelaunchProbeInterval = defaultRelaunchProbeInterval
	}
	if opts.EngineDownGrace == 0 {
		opts.EngineDownGrace = defaultEngineDownGrace
	}
	if opts.MaxConcurrentDownloads == 0 {
		opts.MaxConcurrentDownloads = 4
	}
	if opts.RetriableBackoff == 0 {
		opts.RetriableBackoff = 30 * time.Second
	}
	if opts.NowFunc == nil {
		opts.NowFunc = time.Now
	}
	return &Manager{
		opts:      opts,
		installer: adapter.InstallerFor(opts.Adapter),
		retry:     make(chan struct{}),
	}, nil
}

// Run is the single-goroutine main loop.
func (m *Manager) Run(ctx context.Context) error {
	if m.stopped.Load() {
		return errors.New("lifecycle: Run called after Stop")
	}

	aliveBeforeBoot := m.opts.Adapter.AliveBeforeBoot()

	if aliveBeforeBoot {
		if err := m.opts.Adapter.WaitAlive(ctx); err != nil {
			m.opts.Manager.Update(func(s *progress.State) {
				s.Phase = progress.PhaseFailed
				s.LastError = "engine never came alive: " + err.Error()
			})
			return err
		}
	}

	healthCtx, cancelHealth := context.WithCancel(ctx)
	defer cancelHealth()
	if aliveBeforeBoot {
		go m.healthLoop(healthCtx)
	}

	healthLoopStarted := aliveBeforeBoot

	for {
		err := m.ensure(ctx)
		if err == nil {
			if !aliveBeforeBoot {
				if waitErr := m.opts.Adapter.WaitAlive(ctx); waitErr != nil {
					if errors.Is(waitErr, context.Canceled) ||
						errors.Is(waitErr, context.DeadlineExceeded) {
						return waitErr
					}
					// Engine never came alive during PhaseLoading. Per the
					// state machine this is terminal (PhaseFailed), but keep
					// the loop alive so an operator /api/retry re-runs ensure
					// (which is idempotent) and re-attempts WaitAlive.
					m.opts.Manager.Update(func(s *progress.State) {
						s.Phase = progress.PhaseFailed
						s.LastError = "engine never came alive: " + waitErr.Error()
					})
					if retryErr := m.waitForRetryOrCtx(ctx, 0); retryErr != nil {
						return retryErr
					}
					continue
				}
				if rs, readyErr := m.opts.Adapter.Ready(ctx); readyErr == nil {
					m.lastReady.Store(&rs)
				}
				m.setPhase(progress.PhaseReady)
				if !healthLoopStarted {
					go m.healthLoop(healthCtx)
					healthLoopStarted = true
				}
			}
			// A ready model parks here until something asks for another
			// pass. Nothing re-runs ensure on a timer: a pass that cannot
			// reuse what is already on disk enters PhaseDownload, which
			// bounces every /v1/* request with 503 and reaches the
			// upstream registry, so a timer would let an unreachable
			// huggingface.co move a model that is complete on disk into
			// degraded. The two woken paths are POST /api/retry and the
			// health loop's re-registration.
			if waitErr := m.waitForRetryOrCtx(ctx, 0); waitErr != nil {
				return waitErr
			}
			continue
		}
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}

		var rerr retriableErr
		max := time.Duration(0)
		if errors.As(err, &rerr) {
			max = m.opts.RetriableBackoff
		}
		if waitErr := m.waitForRetryOrCtx(ctx, max); waitErr != nil {
			return waitErr
		}
	}
}

// Retry is shorthand for RetryWith(RetryOptions{}).
func (m *Manager) Retry() { m.RetryWith(RetryOptions{}) }

// RetryWith asks the lifecycle loop to re-run ensure with the supplied
// per-iteration overrides.
func (m *Manager) RetryWith(opts RetryOptions) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pendingOpts = opts
	m.wakeLocked()
}

// requestReensure wakes the loop for a plain ensure pass without touching
// pendingOpts. The health loop calls it, and an operator's /api/retry
// ?force=true may be sitting in pendingOpts unconsumed at that moment;
// writing RetryOptions{} over it would quietly downgrade the re-download
// they asked for into an ordinary pass.
func (m *Manager) requestReensure() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.wakeLocked()
}

// wakeLocked closes the current wake channel and installs a fresh one.
// Caller holds m.mu.
func (m *Manager) wakeLocked() {
	select {
	case <-m.retry:
	default:
		close(m.retry)
	}
	m.retry = make(chan struct{})
}

// consumePendingOpts atomically reads-and-clears the most recent
// RetryWith opts.
func (m *Manager) consumePendingOpts() RetryOptions {
	m.mu.Lock()
	defer m.mu.Unlock()
	opts := m.pendingOpts
	m.pendingOpts = RetryOptions{}
	return opts
}

// Stop marks the manager so a subsequent Run errors out.
func (m *Manager) Stop() { m.stopped.Store(true) }

// waitForRetryOrCtx blocks until ctx fires or Retry() is signaled or
// max elapses (whichever happens first). max==0 means no timer.
func (m *Manager) waitForRetryOrCtx(ctx context.Context, max time.Duration) error {
	m.mu.Lock()
	ch := m.retry
	m.mu.Unlock()

	if max == 0 {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ch:
			return nil
		}
	}
	timer := time.NewTimer(max)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-ch:
		return nil
	case <-timer.C:
		return nil
	}
}
