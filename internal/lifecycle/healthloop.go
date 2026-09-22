package lifecycle

import (
	"context"
	"log/slog"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/progress"
)

// healthProbeTimeout caps a single Adapter.Ready call. Pre-v1.0.5 the
// first probe ran with the parent context (no per-probe timeout) while
// every subsequent tick used WithTimeout. A wedged Ready call on
// startup could therefore hang the entire health loop indefinitely,
// keeping `m.lastReady` at its zero value forever -- the dataplane
// stayed 503 not because the engine was down, but because the FIRST
// probe never returned. Bound the first probe with the same timeout
// as the steady-state ones.
const healthProbeTimeout = 5 * time.Second

// defaultRelaunchProbeInterval is how often the engine is probed while a
// relaunch it was asked for is still outstanding.
//
// The steady-state interval is sized for an engine nobody is touching, and
// at 10s it is longer than a healthy relaunch: the wrapper stops the engine
// ~2s after the signal and a small model is back in under ten. So the whole
// dip could fall between two ticks and the phase never moved at all — a
// console that asked for a restart watched a row read `ready` from the click
// to the finish. This interval only applies while Tracker.Relaunching says
// there is something to catch, which is bounded by that tracker's own
// window, so the extra probes are spent on the seconds they can answer for.
const defaultRelaunchProbeInterval = time.Second

// defaultEngineDownGrace is how long the engine has to keep failing its
// probe before the health loop calls it down.
//
// The number is set by a restart, not by a crash. Editing a model card
// relaunches the engine subprocess through the wrapper, which stops it
// with ENGINE_STOP_GRACE_SECONDS (15 by default) of grace and then has to
// load the weights again — 15 to 25 seconds during which a perfectly
// healthy model fails every probe. A grace shorter than that window
// reports a failure after every successful edit, and a status that cries
// wolf on the happy path is worse than no status at all: nobody acts on
// the one time it is real.
const defaultEngineDownGrace = 60 * time.Second

// healthLoop ticks Adapter.Ready and caches the result on Manager.
// It runs concurrently with ensure() so the data plane can flip from
// 503 to 200 the moment the engine reports the model — even if ensure
// is still in PhaseDownload (which happens for resumes that finish
// before adapter.Register but after adapter has independently
// observed the freshly-written files).
//
// This loop writes exactly one phase transition in each direction:
// PhaseReady → PhaseDegraded when the engine has been failing for
// longer than EngineDownGrace, and back again when it recovers. It
// touches no other phase, so ensure() remains the only writer of the
// download and loading path. This loop was read-only originally, which
// left an engine that died after a restart indistinguishable from one
// that was serving.
//
// On the downward edge it may also ask the lifecycle loop for one ensure
// pass — see probeAndPublish for when and why. It does not run ensure
// itself; Run stays the single goroutine that does.
//
// Each probe is timed and observed into
// `llm_init_engine_probe_duration_seconds`. The boolean result feeds
// `llm_init_engine_ready`. Metrics calls are nil-safe so unit tests can
// omit Options.Metrics.
func (m *Manager) healthLoop(ctx context.Context) {
	// First probe is immediate so the cached value isn't zero for the
	// duration of the first tick interval. Wrap it in the same
	// per-probe timeout as the ticker branch so a wedged
	// Adapter.Ready on startup cannot hang the entire health loop
	// (see healthProbeTimeout godoc).
	probeCtx, cancel := context.WithTimeout(ctx, healthProbeTimeout)
	m.probeAndPublish(ctx, probeCtx)
	cancel()

	for {
		// The gap is chosen per iteration rather than fixed by a ticker,
		// because a relaunch needs a closer look than steady state does
		// and it is not known when the loop starts. See probeInterval.
		//
		// A timer per iteration rather than one Reset: the wake branch
		// leaves an armed timer behind, and a fresh one costs an
		// allocation against a loop that runs at most once a second.
		t := time.NewTimer(m.probeInterval())
		select {
		case <-ctx.Done():
			t.Stop()
			return
		case <-m.opts.RelaunchRequested:
			// Probing now is not what this is for — the engine is still
			// serving at the instant of the signal, and will be for the
			// couple of seconds the wrapper takes to notice. The point is
			// that this iteration ends, so the next gap is chosen with
			// RelaunchInFlight true. Without it a loop that had just armed
			// a ten-second wait would spend the entire relaunch asleep.
			t.Stop()
		case <-t.C:
		}
		probeCtx, cancel := context.WithTimeout(ctx, healthProbeTimeout)
		m.probeAndPublish(ctx, probeCtx)
		cancel()
	}
}

// probeInterval is the gap before the next probe: short while a relaunch is
// outstanding, the configured health interval otherwise.
func (m *Manager) probeInterval() time.Duration {
	if m.opts.RelaunchInFlight == nil || !m.opts.RelaunchInFlight() {
		return m.opts.HealthInterval
	}
	if d := m.opts.RelaunchProbeInterval; d > 0 && d < m.opts.HealthInterval {
		return d
	}
	return m.opts.HealthInterval
}

// probeAndPublish runs Adapter.Ready under probeCtx, caches the result,
// emits the engine_probe_duration_seconds + engine_ready metrics, and
// moves the phase between ready and degraded once the engine has been
// unhealthy for longer than the grace window.
//
// parentCtx is used only for done-channel checks; probeCtx carries the
// per-call timeout.
func (m *Manager) probeAndPublish(parentCtx, probeCtx context.Context) {
	if parentCtx.Err() != nil {
		return
	}
	engineKind := string(m.opts.Adapter.Kind())
	start := time.Now()
	rs, err := m.opts.Adapter.Ready(probeCtx)
	elapsed := time.Since(start).Seconds()
	if m.opts.Metrics != nil {
		m.opts.Metrics.EngineProbeDuration.WithLabelValues(engineKind, "/api/ready").Observe(elapsed)
	}

	healthy := err == nil && rs.Alive && rs.ModelExists
	if m.opts.Metrics != nil {
		val := 0.0
		if healthy {
			val = 1.0
		}
		m.opts.Metrics.EngineReady.WithLabelValues(engineKind).Set(val)
	}

	// A probe that returned an answer is cached even when the answer is
	// "not ready"; a probe that could not be made says nothing about the
	// engine, so the previous answer stands. That stickiness is why the
	// grace window below exists at all.
	if err == nil {
		m.lastReady.Store(&rs)
	}

	if healthy {
		m.unhealthySince.Store(nil)
		m.recoverFromEngineOutage()
		return
	}

	now := m.now()
	since := m.unhealthySince.Load()
	if since == nil {
		m.unhealthySince.Store(&now)
		since = &now
	}

	// A relaunch that was asked for explains this probe, and `loading` is
	// what a relaunch looks like from outside: the weights are on disk and
	// the engine is not serving yet. Reported on the first failing probe
	// rather than after the grace, because a healthy relaunch finishes
	// well inside that window — so waiting it out is exactly why a console
	// that asked for one used to see nothing move at all.
	if m.markEngineRelaunching() {
		slog.Info("engine relaunch in flight; phase moved to loading",
			slog.String("engine", engineKind))
	}

	if now.Sub(*since) < m.engineDownGrace() {
		return
	}
	if !m.markEngineDown(reasonFor(err)) {
		return
	}
	// The engine answered, it is up, and the model it was serving is
	// gone: an Ollama daemon that restarted has lost what Register
	// pushed into it, and nothing else will put it back. Re-running
	// ensure re-pulls or re-registers.
	//
	// Gated on the ready → degraded edge, so this loop asks once per
	// outage and cannot ask during download or loading, where a missing
	// model is the expected state. Gated on AliveBeforeBoot because that
	// is true only for the daemon-backed engine: the proxy adapters' own
	// Register is a no-op, and a pass that cannot fix anything would
	// only add a 503 window and a round trip to the registry.
	//
	// What happens after the ask is not this loop's business. A pass
	// that fails hands off to Run's own backoff; a pass that succeeds
	// while the daemon still reports nothing comes back here, and the
	// next edge is a whole grace window away.
	if err == nil && rs.Alive && !rs.ModelExists && m.opts.Adapter.AliveBeforeBoot() {
		slog.Warn("engine lost its model; re-running ensure to register it again",
			slog.String("engine", engineKind))
		m.requestReensure()
	}
}

// markEngineRelaunching moves a ready model to loading while a relaunch
// of the engine is outstanding. The bool reports whether this call is the
// one that moved it, so a caller logs once per relaunch instead of once
// per probe.
//
// Only PhaseReady is moved, for the same reason markEngineDown only moves
// it: a download or a boot load belongs to ensure(), and a relaunch cannot
// be happening during either.
//
// The window this opens is bounded by the grace, not by the relaunch. An
// engine that never comes back leaves the tracker reporting an outstanding
// relaunch forever, so markEngineDown has to be able to take this loading
// away again — otherwise a dead engine would sit in a phase that reads as
// progress, and every client polling it would keep polling fast.
func (m *Manager) markEngineRelaunching() bool {
	if m.opts.RelaunchInFlight == nil || !m.opts.RelaunchInFlight() {
		return false
	}
	moved := false
	m.opts.Manager.Update(func(s *progress.State) {
		if s.Phase != progress.PhaseReady {
			return
		}
		s.Phase = progress.PhaseLoading
		moved = true
	})
	if moved {
		m.relaunchLoading.Store(true)
	}
	return moved
}

// markEngineDown flips the cached liveness and moves a serving model to
// degraded. It is idempotent: a loop that keeps failing keeps calling it.
// The bool reports whether this call is the one that performed the
// transition, which is what makes the caller's recovery edge-triggered.
//
// Two phases are moved, and they are the two this loop wrote: PhaseReady,
// and the PhaseLoading it put there for a relaunch that has now outlasted
// the grace. Everything else is somebody else's: an engine that is not
// answering during a download or a boot load is the expected state of
// those phases, and overwriting them here would erase the progress the
// dashboard is showing and the failure ensure() already recorded.
func (m *Manager) markEngineDown(reason string) bool {
	if prev := m.lastReady.Load(); prev == nil || prev.Alive {
		down := adapter.ReadyState{Alive: false}
		if prev != nil {
			down.ModelExists = prev.ModelExists
		}
		m.lastReady.Store(&down)
	}
	moved := false
	m.opts.Manager.Update(func(s *progress.State) {
		if s.Phase != progress.PhaseReady && !m.ownsRelaunchLoading(s.Phase) {
			return
		}
		s.Phase = progress.PhaseDegraded
		s.LastError = reason
		moved = true
	})
	if moved {
		m.relaunchLoading.Store(false)
	}
	return moved
}

// recoverFromEngineOutage returns a model the engine had stopped serving
// to ready. The engine is answering again, and nothing else is watching
// for that: ensure() is parked in waitForRetryOrCtx and will not re-run on
// its own.
//
// Both phases it accepts are ones this loop wrote — degraded from an
// engine that went down, loading from a relaunch — so a boot load still
// reaches ready only through Run, after WaitAlive.
func (m *Manager) recoverFromEngineOutage() {
	moved := false
	m.opts.Manager.Update(func(s *progress.State) {
		if s.Phase != progress.PhaseDegraded && !m.ownsRelaunchLoading(s.Phase) {
			return
		}
		s.Phase = progress.PhaseReady
		s.LastError = ""
		moved = true
	})
	if moved {
		m.relaunchLoading.Store(false)
	}
}

// ownsRelaunchLoading reports whether this phase is the loading the health
// loop wrote for a relaunch, as opposed to the one ensure() writes while a
// model boots.
func (m *Manager) ownsRelaunchLoading(p progress.Phase) bool {
	return p == progress.PhaseLoading && m.relaunchLoading.Load()
}

func reasonFor(err error) string {
	if err != nil {
		return "engine health probe failed: " + err.Error()
	}
	return "engine health probe reports the model is not being served"
}

func (m *Manager) engineDownGrace() time.Duration {
	if m.opts.EngineDownGrace > 0 {
		return m.opts.EngineDownGrace
	}
	return defaultEngineDownGrace
}

// now is NowFunc with a fallback, because the probe-level tests build a
// Manager directly rather than through New.
func (m *Manager) now() time.Time {
	if m.opts.NowFunc != nil {
		return m.opts.NowFunc()
	}
	return time.Now()
}
