package handoff

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// LiveProbe answers whether the engine is reachable right now. It must be
// a live call, not a cached health-loop verdict: the whole point is to
// catch a dip that lasts seconds.
type LiveProbe func(ctx context.Context) bool

// State is what we know about the most recent restart request.
type State string

const (
	// StateIdle means no restart has been requested since boot.
	StateIdle State = "idle"

	// StateWatching means a request was signaled and the engine is
	// being polled for the dip that proves a supervisor acted.
	StateWatching State = "watching"

	// StateConfirmed means the engine went away after the request. It
	// may or may not be back yet; ObservedBackAt says which.
	StateConfirmed State = "confirmed"

	// StateNoSupervisor means the window elapsed with the engine
	// answering throughout. Both the wrapper's poll interval and its
	// SIGTERM grace fit inside that window, so nothing acted on the
	// signal — the engine is running under a plain exec with no
	// supervise loop to notice.
	StateNoSupervisor State = "no-supervisor"

	// StateUnverified means the question could not be answered: no probe
	// is wired, or the engine was down when the request went out and
	// never came back inside the window. Down before and down after
	// proves nothing either way.
	StateUnverified State = "unverified"
)

// Supervision is the durable half of what we have learned, kept in
// StateName across llm-init restarts. It is the only evidence available
// at the moment a request is made, which is when a caller wants to know
// whether the signal will be acted on.
type Supervision string

const (
	// SupervisionUnknown means no restart has ever been confirmed here.
	// It is not the same as "unsupervised": it is the honest answer
	// before the first observation.
	SupervisionUnknown Supervision = "unknown"

	// SupervisionConfirmed means a restart request has been observed to
	// stop the engine at least once in this run dir's history.
	SupervisionConfirmed Supervision = "confirmed"
)

// Receipt is what a caller learns synchronously. Confirmation cannot be
// part of it: the wrapper polls every RESTART_POLL_INTERVAL and then
// gives the engine ENGINE_STOP_GRACE_SECONDS to drain, so the dip can
// start up to 17s after the signal with stock settings — far past any
// sane HTTP deadline (Router's control-plane client allows 10s for the
// whole call). Poll Status, or GET /api/engine/restart, for the outcome.
type Receipt struct {
	// Signaled reports that the generation file was written. That is a
	// fact about llm-init's own disk and nothing more.
	Signaled bool `json:"signaled"`

	// Generation identifies this request, so a later Status read can be
	// tied to it rather than to a request that has since been replaced.
	Generation string `json:"generation,omitempty"`

	// Supervision is the prior evidence, not a claim about this
	// request. "confirmed" means a supervisor has acted before and is
	// expected to act again; "unknown" means we have never seen one.
	Supervision Supervision `json:"supervision"`
}

// Status is the full picture, including the part that only becomes known
// after the caller has gone.
type Status struct {
	State       State       `json:"state"`
	Supervision Supervision `json:"supervision"`
	Generation  string      `json:"generation,omitempty"`

	RequestedAt    *time.Time `json:"requested_at,omitempty"`
	ObservedDownAt *time.Time `json:"observed_down_at,omitempty"`
	ObservedBackAt *time.Time `json:"observed_back_at,omitempty"`
}

// Config builds a Tracker. Only RunDir is required, and an empty one
// yields a tracker that reports "cannot signal" instead of panicking, so
// download-only deployments and unit tests need no special case.
type Config struct {
	RunDir string

	// Probe is how the engine is asked whether it is alive. A nil probe
	// leaves every request StateUnverified.
	Probe LiveProbe

	// Window bounds the observation. It must exceed the wrapper's poll
	// interval plus its SIGTERM grace, or a real restart reads as
	// StateNoSupervisor. Zero uses DefaultWindow.
	Window time.Duration

	// Interval is the gap between probes. Zero uses DefaultInterval.
	Interval time.Duration

	// OnRequest is called at the end of Request, once Relaunching has
	// begun answering true. It exists for observers that poll the engine
	// on their own schedule: learning about a relaunch only when they
	// next happen to look means the whole dip can pass unseen, since a
	// healthy one is over in well under ten seconds. It must not block.
	OnRequest func()

	Now func() time.Time
}

const (
	// DefaultWindow leaves ~3x headroom over the stock 2s poll + 15s
	// grace, so a slow drain still reads as a restart rather than as an
	// absent supervisor.
	DefaultWindow = 60 * time.Second

	// DefaultInterval matches the wrapper's own poll cadence. Probes run
	// sequentially, so a slow probe stretches the gap rather than
	// stacking up calls against a dying engine.
	DefaultInterval = 2 * time.Second
)

// Tracker writes the handoff files and remembers what happened next.
// Safe for concurrent use.
type Tracker struct {
	cfg Config

	mu      sync.Mutex
	status  Status
	cancel  context.CancelFunc
	watched sync.WaitGroup
}

// NewTracker seeds Supervision from StateName in the run dir, which is
// why a fresh llm-init process can answer "a supervisor is present"
// about a wrapper it has never signaled itself.
func NewTracker(cfg Config) *Tracker {
	if cfg.Window <= 0 {
		cfg.Window = DefaultWindow
	}
	if cfg.Interval <= 0 {
		cfg.Interval = DefaultInterval
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	return &Tracker{
		cfg: cfg,
		status: Status{
			State:       StateIdle,
			Supervision: loadSupervision(cfg.RunDir),
		},
	}
}

// RunDir is where the handoff files live. Empty means there is nowhere
// to signal.
func (t *Tracker) RunDir() string { return t.cfg.RunDir }

// WriteEngineArgs publishes engine_args for the wrapper's next launch.
func (t *Tracker) WriteEngineArgs(raw string) error {
	return WriteEngineArgs(t.cfg.RunDir, raw)
}

// Request bumps the generation and starts observing. It returns as soon
// as the file is on disk; the observation outlives the caller and lands
// in Status. Requesting again supersedes an in-flight observation, since
// the older one can no longer be attributed to a single generation.
func (t *Tracker) Request() (Receipt, error) {
	gen, err := Bump(t.cfg.RunDir)
	if err != nil {
		return Receipt{Supervision: t.Supervision()}, err
	}
	now := t.cfg.Now()

	t.mu.Lock()
	if t.cancel != nil {
		t.cancel()
		t.cancel = nil
	}
	t.status.Generation = gen
	t.status.RequestedAt = &now
	t.status.ObservedDownAt = nil
	t.status.ObservedBackAt = nil
	if t.cfg.Probe == nil || t.cfg.RunDir == "" {
		t.status.State = StateUnverified
	} else {
		t.status.State = StateWatching
	}
	receipt := Receipt{
		Signaled:    t.cfg.RunDir != "",
		Generation:  gen,
		Supervision: t.status.Supervision,
	}
	watch := t.status.State == StateWatching
	if watch {
		ctx, cancel := context.WithCancel(context.Background())
		t.cancel = cancel
		t.watched.Add(1)
		go func() {
			defer t.watched.Done()
			defer cancel()
			t.observe(ctx, gen)
		}()
	}
	t.mu.Unlock()

	// Outside the lock: this reaches code that has nothing to do with
	// the tracker, and holding the mutex through it would let an
	// observer's callback deadlock every reader of Status.
	if t.cfg.OnRequest != nil {
		t.cfg.OnRequest()
	}

	return receipt, nil
}

// Status returns the current picture.
func (t *Tracker) Status() Status {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.status
}

// Supervision returns the durable verdict on its own.
func (t *Tracker) Supervision() Supervision {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.status.Supervision
}

// Relaunching reports whether a requested relaunch is still outstanding —
// the signal went out and the engine has not been seen serving since.
//
// It answers a different question from Supervision. Supervision is about
// this deployment in general: will anything ever act on a signal here.
// This is about right now: is the engine between two lives. A phase, and
// anything polling one, needs the second.
func (t *Tracker) Relaunching() bool { return t.Status().Relaunching() }

// Relaunching is Tracker.Relaunching against a snapshot.
//
// The states that settle without the engine going away — no supervisor
// acted, or nothing could be verified — are deliberately not outstanding.
// A relaunch nothing performed leaves an engine that never stopped
// serving, and reporting it would hold a phase down over a healthy model.
func (s Status) Relaunching() bool {
	switch s.State {
	case StateWatching:
		return true
	case StateConfirmed:
		return s.ObservedBackAt == nil
	default:
		return false
	}
}

// Relaunch identifies the last relaunch the engine was seen to come back
// from, and is empty until there has been one. It changes exactly once per
// relaunch, which is what a consumer that has to re-read something the engine
// only reports at startup can follow.
//
// It is deliberately empty between the request and the engine answering
// again. A consumer that acted on the request alone would read the engine
// that is about to be replaced, and settle on its answer.
func (t *Tracker) Relaunch() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.status.ObservedBackAt == nil {
		return ""
	}
	return t.status.Generation + "@" + t.status.ObservedBackAt.UTC().Format(time.RFC3339Nano)
}

// Wait blocks until no observation is in flight. Tests use it; the
// production path never needs to, because an observation is bounded by
// Window and holds nothing a shutdown has to reclaim.
func (t *Tracker) Wait() { t.watched.Wait() }

// observe polls the engine for the transition that proves a supervisor
// acted on the signal.
//
// Which transition that is depends on where the engine started, and the
// baseline probe is what decides. An engine that is serving has to be seen
// going away. An engine that is already down has to be seen coming back —
// and that is the case an operator is most often in, because a wedged
// engine is the usual reason to ask for a relaunch at all. Reading the
// second as unverifiable, as this did until now, withheld the answer
// exactly when it was wanted and left supervision stuck at "unknown" for
// the whole life of the run dir.
//
// What the baseline still buys is the honesty it was added for: an engine
// that is down before the signal and down after it proves nothing, and
// settles unverified rather than confirmed.
func (t *Tracker) observe(ctx context.Context, gen string) {
	deadline := t.cfg.Now().Add(t.cfg.Window)

	down := !t.cfg.Probe(ctx)
	if down {
		start := t.cfg.Now()
		t.settle(gen, func(s *Status) { s.ObservedDownAt = &start })
	}
	confirmed := false

	for {
		if !t.sleep(ctx, t.cfg.Interval) {
			return
		}
		alive := t.cfg.Probe(ctx)
		now := t.cfg.Now()
		switch {
		case !alive && !down:
			down = true
			confirmed = true
			t.settle(gen, func(s *Status) {
				s.State = StateConfirmed
				s.ObservedDownAt = &now
				s.Supervision = SupervisionConfirmed
			})
			t.persistSupervision(gen, now)
			slog.Info("engine restart confirmed: engine went down after signal",
				"gen", gen)
		case alive && down:
			// Confirming here as well as on the dip is what covers the
			// engine that was already down: the rise is the only edge
			// that run will ever produce.
			t.settle(gen, func(s *Status) {
				s.State = StateConfirmed
				s.ObservedBackAt = &now
				s.Supervision = SupervisionConfirmed
			})
			if !confirmed {
				t.persistSupervision(gen, now)
			}
			slog.Info("engine back after restart", "gen", gen)
			return
		}
		if !now.Before(deadline) {
			switch {
			case !down:
				t.settle(gen, func(s *Status) { s.State = StateNoSupervisor })
				slog.Warn("engine restart signaled but the engine never went down; "+
					"no supervise loop is watching engine_restart",
					"gen", gen, "window", t.cfg.Window)
			case !confirmed:
				t.settle(gen, func(s *Status) { s.State = StateUnverified })
				slog.Warn("engine restart not verifiable: the engine was down when "+
					"signaled and has not come back",
					"gen", gen, "window", t.cfg.Window)
			}
			return
		}
	}
}

// settle applies mutate unless a newer generation has taken over, so a
// superseded observation cannot overwrite the current one's verdict.
func (t *Tracker) settle(gen string, mutate func(*Status)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.status.Generation != gen {
		return
	}
	mutate(&t.status)
}

func (t *Tracker) sleep(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// persistedState is the on-disk notebook. It is advisory: losing it costs
// a receipt its "confirmed" answer until the next observation, nothing more.
type persistedState struct {
	Supervised              bool      `json:"supervised"`
	LastConfirmedGeneration string    `json:"last_confirmed_generation,omitempty"`
	LastConfirmedAt         time.Time `json:"last_confirmed_at,omitempty"`
}

func loadSupervision(runDir string) Supervision {
	if runDir == "" {
		return SupervisionUnknown
	}
	data, err := os.ReadFile(filepath.Join(runDir, StateName))
	if err != nil {
		return SupervisionUnknown
	}
	var st persistedState
	if err := json.Unmarshal(data, &st); err != nil || !st.Supervised {
		return SupervisionUnknown
	}
	return SupervisionConfirmed
}

func (t *Tracker) persistSupervision(gen string, at time.Time) {
	if t.cfg.RunDir == "" {
		return
	}
	data, err := json.Marshal(persistedState{
		Supervised:              true,
		LastConfirmedGeneration: gen,
		LastConfirmedAt:         at,
	})
	if err != nil {
		return
	}
	if err := writeFileAtomic(t.cfg.RunDir, StateName, data); err != nil {
		slog.Warn("could not record engine supervision evidence", "err", err)
	}
}
