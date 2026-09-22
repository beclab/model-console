package lifecycle

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/progress"
)

// probeManager builds a Manager wired with only the fields probeAndPublish
// and healthLoop touch (Adapter, HealthInterval, Manager, lastReady). It
// bypasses New() on purpose so these probe-level tests don't drag in a
// downloader the health loop never reads. The progress manager is real
// because the loop now moves the phase between ready and degraded.
//
// The grace window is left at its production default so a test has to opt
// into degradation explicitly, the way probeManagerWithGrace does.
func probeManager(fa *fakeAdapter, interval time.Duration) *Manager {
	return probeManagerWithGrace(fa, interval, defaultEngineDownGrace)
}

func probeManagerWithGrace(fa *fakeAdapter, interval, grace time.Duration) *Manager {
	return &Manager{
		// The wake channel is what New() would have built. The loop can
		// ask the lifecycle loop for one ensure pass, and closing a nil
		// channel panics.
		retry: make(chan struct{}),
		opts: Options{
			Adapter:         fa,
			HealthInterval:  interval,
			EngineDownGrace: grace,
			Manager:         progress.New(time.Now()),
		},
	}
}

// readyManager is a probeManager whose model is already serving, which is
// the only phase the health loop is allowed to move.
func readyManager(fa *fakeAdapter, interval, grace time.Duration) *Manager {
	m := probeManagerWithGrace(fa, interval, grace)
	m.opts.Manager.Update(func(s *progress.State) { s.Phase = progress.PhaseReady })
	return m
}

func phaseOf(m *Manager) progress.Phase { return m.opts.Manager.Snapshot().Phase }

func TestProbeAndPublish(t *testing.T) {
	good := adapter.ReadyState{Alive: true, ModelExists: true}

	tests := []struct {
		name string
		// seed primes lastReady with a prior success before the probe
		// under test, so error/skip paths can assert the cache is sticky.
		seed        bool
		parentDone  bool
		probeFn     func(context.Context) (adapter.ReadyState, error)
		wantCalled  bool
		wantAlive   bool
		wantExists  bool
		wantHasPrev bool
	}{
		{
			name:        "success stores ready",
			probeFn:     func(context.Context) (adapter.ReadyState, error) { return good, nil },
			wantCalled:  true,
			wantAlive:   true,
			wantExists:  true,
			wantHasPrev: true,
		},
		{
			name:        "error preserves previous cached value",
			seed:        true,
			probeFn:     func(context.Context) (adapter.ReadyState, error) { return adapter.ReadyState{}, errors.New("boom") },
			wantCalled:  true,
			wantAlive:   true,
			wantExists:  true,
			wantHasPrev: true,
		},
		{
			name:        "cancelled parent skips probe entirely",
			seed:        true,
			parentDone:  true,
			probeFn:     func(context.Context) (adapter.ReadyState, error) { return good, nil },
			wantCalled:  false,
			wantAlive:   true,
			wantExists:  true,
			wantHasPrev: true,
		},
		{
			name: "probe bounded by probeCtx timeout",
			seed: true,
			probeFn: func(ctx context.Context) (adapter.ReadyState, error) {
				<-ctx.Done()
				return adapter.ReadyState{}, ctx.Err()
			},
			wantCalled:  true,
			wantAlive:   true,
			wantExists:  true,
			wantHasPrev: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Int32
			fa := &fakeAdapter{
				kind: "test",
				readyFn: func(ctx context.Context) (adapter.ReadyState, error) {
					calls.Add(1)
					return tc.probeFn(ctx)
				},
			}
			m := probeManager(fa, time.Hour)
			if tc.seed {
				m.probeAndPublishOnce(t, good)
			}

			parentCtx, cancelParent := context.WithCancel(context.Background())
			defer cancelParent()
			if tc.parentDone {
				cancelParent()
			}
			// A short probe timeout keeps the "bounded by probeCtx" case
			// fast instead of waiting the production 5s.
			probeCtx, cancelProbe := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancelProbe()

			before := calls.Load()
			m.probeAndPublish(parentCtx, probeCtx)
			gotCalled := calls.Load() > before
			if gotCalled != tc.wantCalled {
				t.Fatalf("Ready called = %v, want %v", gotCalled, tc.wantCalled)
			}

			rd := m.Readiness()
			if rs := m.lastReady.Load(); (rs != nil) != tc.wantHasPrev {
				t.Fatalf("lastReady set = %v, want %v", rs != nil, tc.wantHasPrev)
			}
			if rd.EngineAlive != tc.wantAlive || rd.ModelExists != tc.wantExists {
				t.Fatalf("cached probe = (alive=%v exists=%v), want (alive=%v exists=%v)",
					rd.EngineAlive, rd.ModelExists, tc.wantAlive, tc.wantExists)
			}
		})
	}
}

// probeAndPublishOnce seeds lastReady with a known-good probe result so
// subsequent error/skip paths have a prior value to preserve.
func (m *Manager) probeAndPublishOnce(t *testing.T, rs adapter.ReadyState) {
	t.Helper()
	orig := m.opts.Adapter
	m.opts.Adapter = &fakeAdapter{kind: "test", readyState: rs}
	m.probeAndPublish(context.Background(), context.Background())
	m.opts.Adapter = orig
}

func TestHealthLoop_FirstProbeBeforeFirstTick(t *testing.T) {
	// HealthInterval is an hour: if the cache flips ready, it can only be
	// the immediate first probe, never a ticker tick.
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			return adapter.ReadyState{Alive: true, ModelExists: true}, nil
		},
	}
	m := probeManager(fa, time.Hour)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { m.healthLoop(ctx); close(done) }()

	deadline := time.After(2 * time.Second)
	for {
		if rd := m.Readiness(); rd.EngineAlive && rd.ModelExists {
			break
		}
		select {
		case <-deadline:
			t.Fatal("first probe did not publish ready before first tick")
		case <-time.After(2 * time.Millisecond):
		}
	}

	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("healthLoop did not return after context cancel")
	}
}

func TestHealthLoop_StopsOnContextCancel(t *testing.T) {
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			return adapter.ReadyState{Alive: true, ModelExists: true}, nil
		},
	}
	m := probeManager(fa, 5*time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { m.healthLoop(ctx); close(done) }()

	time.Sleep(20 * time.Millisecond) // let a few ticks run
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("healthLoop did not return after context cancel")
	}
}

// A restart triggered by a card edit takes the engine away for 15 to 25
// seconds. Failures inside the grace window must change nothing, or every
// successful edit would be reported as a failure.
func TestHealthLoop_TransientErrorIsSticky(t *testing.T) {
	var n atomic.Int32
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			if n.Add(1) == 1 {
				return adapter.ReadyState{Alive: true, ModelExists: true}, nil
			}
			return adapter.ReadyState{}, errors.New("transient")
		},
	}
	m := readyManager(fa, 5*time.Millisecond, time.Hour)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() { m.healthLoop(ctx); close(done) }()

	// Wait until at least a few error probes have happened past the first.
	deadline := time.After(2 * time.Second)
	for n.Load() < 4 {
		select {
		case <-deadline:
			t.Fatal("not enough probes ran")
		case <-time.After(2 * time.Millisecond):
		}
	}
	cancel()
	<-done

	if rd := m.Readiness(); !rd.EngineAlive || !rd.ModelExists {
		t.Fatalf("transient errors flipped cached state to (alive=%v exists=%v), want sticky ready",
			rd.EngineAlive, rd.ModelExists)
	}
	if got := phaseOf(m); got != progress.PhaseReady {
		t.Fatalf("phase = %q, want ready — a failure inside the grace window degraded the model", got)
	}
}

// Past the grace window the same failures mean the engine did not come
// back. Before this, an engine that died during a restart was
// indistinguishable from one that was serving.
func TestHealthLoop_SustainedFailureDegrades(t *testing.T) {
	var n atomic.Int32
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			if n.Add(1) == 1 {
				return adapter.ReadyState{Alive: true, ModelExists: true}, nil
			}
			return adapter.ReadyState{}, errors.New("connection refused")
		},
	}
	m := readyManager(fa, 2*time.Millisecond, 10*time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseDegraded)
	if m.Readiness().EngineAlive {
		t.Error("a sustained failure left the cached liveness at alive")
	}
	if msg := m.opts.Manager.Snapshot().LastError; msg == "" {
		t.Error("degraded without recording why")
	}
}

// The lifecycle loop is parked in waitForRetryOrCtx by the time the engine
// comes back, so nothing else would ever undo the degradation.
func TestHealthLoop_RecoveryReturnsToReady(t *testing.T) {
	var down atomic.Bool
	down.Store(true)
	var n atomic.Int32
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			if n.Add(1) > 1 && down.Load() {
				return adapter.ReadyState{}, errors.New("connection refused")
			}
			return adapter.ReadyState{Alive: true, ModelExists: true}, nil
		},
	}
	m := readyManager(fa, 2*time.Millisecond, 10*time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseDegraded)
	down.Store(false)
	waitForPhase(t, m, progress.PhaseReady)

	if rd := m.Readiness(); !rd.EngineAlive || !rd.ModelExists {
		t.Errorf("recovered but cached probe = (alive=%v exists=%v)", rd.EngineAlive, rd.ModelExists)
	}
	if msg := m.opts.Manager.Snapshot().LastError; msg != "" {
		t.Errorf("recovered phase still advertises %q", msg)
	}
}

// relaunchManager is a readyManager that also knows whether the engine is
// being relaunched by its wrapper — the one fact a probe cannot supply.
func relaunchManager(fa *fakeAdapter, interval, grace time.Duration, inFlight *atomic.Bool) *Manager {
	m := readyManager(fa, interval, grace)
	m.opts.RelaunchInFlight = inFlight.Load
	return m
}

// flakyThenDown answers one healthy probe and then fails while down is
// set, which is the shape every relaunch test needs: start from a serving
// model, take the engine away, optionally give it back.
func flakyThenDown(down *atomic.Bool) *fakeAdapter {
	var n atomic.Int32
	return &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			if n.Add(1) > 1 && down.Load() {
				return adapter.ReadyState{}, errors.New("connection refused")
			}
			return adapter.ReadyState{Alive: true, ModelExists: true}, nil
		},
	}
}

// A relaunch finishes well inside the grace window, so the grace alone
// holds `ready` through the entire dip and lands back on `ready` — which
// is why a console that asked for a relaunch used to see nothing move.
// `loading` is what a relaunch actually looks like from outside: the
// weights are on disk and the engine is not serving.
func TestHealthLoop_RelaunchInFlightMovesReadyToLoading(t *testing.T) {
	var inFlight, down atomic.Bool
	inFlight.Store(true)
	down.Store(true)
	// An hour of grace, so only the relaunch can be what moved the phase.
	m := relaunchManager(flakyThenDown(&down), 2*time.Millisecond, time.Hour, &inFlight)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseLoading)
}

func TestHealthLoop_RelaunchedEngineReturnsToReady(t *testing.T) {
	var inFlight, down atomic.Bool
	inFlight.Store(true)
	down.Store(true)
	m := relaunchManager(flakyThenDown(&down), 2*time.Millisecond, time.Hour, &inFlight)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseLoading)
	down.Store(false)
	inFlight.Store(false)
	waitForPhase(t, m, progress.PhaseReady)

	if msg := m.opts.Manager.Snapshot().LastError; msg != "" {
		t.Errorf("a completed relaunch still advertises %q", msg)
	}
}

// The steady-state interval is longer than a healthy relaunch, so before
// the loop learned to look closer the whole dip could land between two
// ticks: the engine went away and came back, and the phase read `ready`
// from the click to the finish. Measured on a real deployment as an 8.4s
// outage against a 10s interval, with no phase movement logged at all.
//
// An hour of health interval makes this unambiguous. Nothing but the
// relaunch cadence can produce a second probe inside waitForPhase.
func TestHealthLoop_RelaunchIsCaughtInsideOneHealthInterval(t *testing.T) {
	var inFlight, down atomic.Bool
	inFlight.Store(true)
	down.Store(true)
	m := relaunchManager(flakyThenDown(&down), time.Hour, time.Hour, &inFlight)
	m.opts.RelaunchProbeInterval = 2 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseLoading)
}

// Learning about a relaunch is worth nothing if the loop is already
// asleep for longer than the relaunch will last. The gap is chosen at the
// end of the previous one, so a wait armed at the steady interval a moment
// before the signal covers the entire dip. Measured on a real deployment:
// the same restart reached `loading` in 0.8s when the loop happened to be
// on the fast cadence already, and 7.2s when it was not.
func TestHealthLoop_RelaunchSignalEndsTheCurrentWait(t *testing.T) {
	var inFlight, down atomic.Bool
	wake := make(chan struct{}, 1)
	// An hour of steady interval, so the only way a second probe happens
	// is the wake, and the only way a third happens is the cadence the
	// wake let it reconsider.
	m := relaunchManager(flakyThenDown(&down), time.Hour, time.Hour, &inFlight)
	m.opts.RelaunchProbeInterval = 2 * time.Millisecond
	m.opts.RelaunchRequested = wake

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	// The order a real request arrives in: the tracker starts answering
	// true, and only then does the wrapper get round to stopping the
	// engine. So the probe the wake prompts still finds a healthy engine.
	inFlight.Store(true)
	down.Store(true)
	wake <- struct{}{}

	waitForPhase(t, m, progress.PhaseLoading)
}

// Nothing wired the channel, which every unit test here and any embedder
// that does not run a wrapper looks like. A nil channel blocks forever, so
// this only works because it is one case of the select rather than a step.
func TestHealthLoop_RunsWithNoRelaunchSignal(t *testing.T) {
	var inFlight, down atomic.Bool
	inFlight.Store(true)
	down.Store(true)
	m := relaunchManager(flakyThenDown(&down), time.Hour, time.Hour, &inFlight)
	m.opts.RelaunchProbeInterval = 2 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseLoading)
}

// The closer look is for a relaunch and lasts as long as one. An engine
// nobody asked anything of is probed at the interval it was configured
// with, whatever RelaunchProbeInterval says.
func TestHealthLoop_SteadyStateKeepsItsOwnInterval(t *testing.T) {
	var inFlight, down atomic.Bool
	down.Store(true)
	// A grace of one millisecond: any second probe at all would degrade
	// this model, so a phase still reading ready is proof none ran.
	m := relaunchManager(flakyThenDown(&down), time.Hour, time.Millisecond, &inFlight)
	m.opts.RelaunchProbeInterval = 2 * time.Millisecond

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	time.Sleep(80 * time.Millisecond)
	if got := phaseOf(m); got != progress.PhaseReady {
		t.Fatalf("phase = %q, want ready — steady state probed at the relaunch cadence", got)
	}
}

func TestProbeInterval(t *testing.T) {
	const health = 10 * time.Second

	tests := []struct {
		name     string
		inFlight *bool
		relaunch time.Duration
		want     time.Duration
	}{
		{name: "no hook wired", relaunch: time.Second, want: health},
		{name: "nothing outstanding", inFlight: ptr(false), relaunch: time.Second, want: health},
		{name: "relaunch outstanding", inFlight: ptr(true), relaunch: time.Second, want: time.Second},
		// Both are floors on how often the engine is asked, so the
		// shorter one wins and a longer relaunch value means nothing.
		{name: "relaunch value is not shorter", inFlight: ptr(true), relaunch: time.Minute, want: health},
		{name: "relaunch value unset", inFlight: ptr(true), want: health},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &Manager{opts: Options{
				HealthInterval:        health,
				RelaunchProbeInterval: tc.relaunch,
			}}
			if tc.inFlight != nil {
				m.opts.RelaunchInFlight = func() bool { return *tc.inFlight }
			}
			if got := m.probeInterval(); got != tc.want {
				t.Fatalf("probeInterval() = %v, want %v", got, tc.want)
			}
		})
	}
}

func ptr[T any](v T) *T { return &v }

// The tracker goes on reporting an outstanding relaunch for an engine that
// never comes back, so the loading it opened has to be closed by the grace
// rather than by the relaunch. A phase that reads as progress forever
// would also pin every client polling it to its fast interval.
func TestHealthLoop_RelaunchThatNeverComesBackStillDegrades(t *testing.T) {
	var inFlight, down atomic.Bool
	inFlight.Store(true)
	down.Store(true)
	m := relaunchManager(flakyThenDown(&down), 2*time.Millisecond, 10*time.Millisecond, &inFlight)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseDegraded)
	if msg := m.opts.Manager.Snapshot().LastError; msg == "" {
		t.Error("degraded without recording why")
	}
}

// ensure() owns the loading a model boots through, and Run is what flips
// it to ready after WaitAlive. A health loop that moved it would call a
// model ready off a probe ensure had not finished acting on.
func TestHealthLoop_LeavesEnsureLoadingAlone(t *testing.T) {
	var inFlight atomic.Bool
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			return adapter.ReadyState{Alive: true, ModelExists: true}, nil
		},
	}
	m := relaunchManager(fa, 2*time.Millisecond, time.Hour, &inFlight)
	m.opts.Manager.Update(func(s *progress.State) { s.Phase = progress.PhaseLoading })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	time.Sleep(80 * time.Millisecond)
	if got := phaseOf(m); got != progress.PhaseLoading {
		t.Fatalf("phase = %q, want loading left to ensure", got)
	}
}

// An engine that is not answering during a download is the expected state
// of that phase, not a fault, and overwriting it would erase the progress
// the dashboard is showing.
func TestHealthLoop_LeavesNonReadyPhasesAlone(t *testing.T) {
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			return adapter.ReadyState{}, errors.New("not up yet")
		},
	}
	m := probeManagerWithGrace(fa, 2*time.Millisecond, time.Millisecond)
	m.opts.Manager.Update(func(s *progress.State) { s.Phase = progress.PhaseDownload })

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	time.Sleep(80 * time.Millisecond)
	if got := phaseOf(m); got != progress.PhaseDownload {
		t.Fatalf("phase = %q, want download left untouched", got)
	}
}

// An Ollama daemon that restarts answers again with an empty model list.
// Nothing else puts the model back: Run is parked in waitForRetryOrCtx,
// and with no periodic pass it stays there until someone asks.
func TestHealthLoop_LostModelAsksForOneReEnsure(t *testing.T) {
	var lost atomic.Bool
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			return adapter.ReadyState{Alive: true, ModelExists: !lost.Load()}, nil
		},
	}
	m := readyManager(fa, 2*time.Millisecond, 10*time.Millisecond)
	wake := wakeChan(m)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	// Let one healthy probe land so the loop starts from ready.
	waitForPhase(t, m, progress.PhaseReady)
	lost.Store(true)

	if !closedWithin(wake, 3*time.Second) {
		t.Fatal("daemon up with no model did not ask for a re-ensure")
	}
	waitForPhase(t, m, progress.PhaseDegraded)

	// The probe keeps failing, but the ready → degraded edge has already
	// been spent. A second wake would restart ensure under itself.
	if closedWithin(wakeChan(m), 100*time.Millisecond) {
		t.Fatal("re-ensure asked for more than once in a single outage")
	}
}

// The proxy engines load from disk and their Register is a no-op, so a
// pass cannot restore anything a missing model would need — it would only
// bounce /v1/* with 503 and call the registry for nothing.
func TestHealthLoop_LostModelDoesNotReEnsureProxyEngines(t *testing.T) {
	aliveBeforeBoot := false
	fa := &fakeAdapter{
		kind:            "test",
		aliveBeforeBoot: &aliveBeforeBoot,
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			return adapter.ReadyState{Alive: true, ModelExists: false}, nil
		},
	}
	m := readyManager(fa, 2*time.Millisecond, 5*time.Millisecond)
	wake := wakeChan(m)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseDegraded)
	if closedWithin(wake, 100*time.Millisecond) {
		t.Fatal("a proxy engine asked for a re-ensure that cannot fix it")
	}
}

// A probe that could not be made says nothing about the model. Re-running
// ensure against an unreachable daemon fails the pass and reports a
// download error over the real reason, which the probe already recorded.
func TestHealthLoop_UnreachableEngineDoesNotReEnsure(t *testing.T) {
	fa := &fakeAdapter{
		kind: "test",
		readyFn: func(context.Context) (adapter.ReadyState, error) {
			return adapter.ReadyState{}, errors.New("connection refused")
		},
	}
	m := readyManager(fa, 2*time.Millisecond, 5*time.Millisecond)
	wake := wakeChan(m)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go m.healthLoop(ctx)

	waitForPhase(t, m, progress.PhaseDegraded)
	if closedWithin(wake, 100*time.Millisecond) {
		t.Fatal("an unreachable engine asked for a re-ensure")
	}
}

// wakeChan returns the channel requestReensure would close. Held under the
// same lock Run reads it with.
func wakeChan(m *Manager) <-chan struct{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.retry
}

func closedWithin(ch <-chan struct{}, d time.Duration) bool {
	select {
	case <-ch:
		return true
	case <-time.After(d):
		return false
	}
}

func waitForPhase(t *testing.T, m *Manager, want progress.Phase) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if phaseOf(m) == want {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("phase = %q, want %q", phaseOf(m), want)
}
