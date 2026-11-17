package lifecycle

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/handoff"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/sentinel"
)

// fakeRelauncher settles every request to the state it is given, as a
// tracker does once it has watched the engine.
type fakeRelauncher struct {
	settle    handoff.State
	err       error
	onRequest func()
	mu        sync.Mutex
	requests  int
	status    handoff.Status
}

func (f *fakeRelauncher) Request() (handoff.Receipt, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests++
	if f.err != nil {
		return handoff.Receipt{}, f.err
	}
	if f.onRequest != nil && f.settle == handoff.StateConfirmed {
		f.onRequest()
	}
	gen := time.Now().String()
	f.status = handoff.Status{State: f.settle, Generation: gen}
	return handoff.Receipt{Signaled: true, Generation: gen}, nil
}

func (f *fakeRelauncher) Status() handoff.Status {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.status
}

func (f *fakeRelauncher) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.requests
}

func (f *fakeRelauncher) set(settle handoff.State, err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.settle, f.err = settle, err
}

// An engine that did not restart onto the new model still serves the
// previous one, so the pass fails instead of calling it ready. The run
// dir already names the new model, yet the next pass asks for the
// restart again, and is ready once it happens.
func TestRun_FailsUntilTheEngineRestartsOntoTheNewModel(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		settle handoff.State
		err    error
	}{
		{name: "no_supervisor", settle: handoff.StateNoSupervisor},
		{name: "request_failed", err: errors.New("run dir read-only")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			runDir := t.TempDir()
			mainPath := filepath.Join(t.TempDir(), "model.gguf")
			if err := os.WriteFile(mainPath, []byte("x"), 0o644); err != nil {
				t.Fatal(err)
			}
			if err := sentinel.Write(runDir, "/cache/snapshots/old/model.gguf"); err != nil {
				t.Fatal(err)
			}
			o := baseOptions(t, []config.ModelSource{
				hfSource("a/b", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "model.gguf"),
			})
			o.Config.Runtime.RunDir = runDir
			o.Config.Engine.Kind = config.EngineLlamaCpp
			o.HFRun = func(context.Context, hfwrap.Config, progress.Sink) (hfwrap.Result, error) {
				return hfwrap.Result{Status: "ok", Path: mainPath}, nil
			}
			rl := &fakeRelauncher{settle: tc.settle, err: tc.err}
			o.Relauncher = rl
			aliveBeforeBoot := false
			var waits atomic.Int32
			o.Adapter = &fakeAdapter{
				kind:            config.EngineLlamaCpp,
				aliveBeforeBoot: &aliveBeforeBoot,
				readyState:      adapter.ReadyState{Alive: true, ModelExists: true},
				waitFn: func(context.Context) error {
					waits.Add(1)
					return nil
				},
			}
			mgr, err := New(o)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go func() { _ = mgr.Run(ctx) }()

			waitPhase(t, o.Manager, progress.PhaseFailed)
			if st := o.Manager.Snapshot(); !strings.Contains(st.LastError, "still serves the previous model") {
				t.Errorf("last_error = %q, want it to say the previous model is served", st.LastError)
			}
			if n := waits.Load(); n != 0 {
				t.Errorf("WaitAlive ran %d times on the engine serving the previous model", n)
			}

			rl.set(handoff.StateConfirmed, nil)
			mgr.Retry()
			waitPhase(t, o.Manager, progress.PhaseReady)
			if got := rl.count(); got != 2 {
				t.Errorf("relaunch requests = %d, want the retry to ask again", got)
			}
		})
	}
}

// reportingAdapter is an engine that says which model file it loaded,
// as llama.cpp does in /props. An empty path is an engine that cannot
// answer, and so are the first failFirst queries, as /props answers 503
// while a model loads.
type reportingAdapter struct {
	*fakeAdapter
	mu        sync.Mutex
	served    string
	failFirst int
	queries   int
}

func (r *reportingAdapter) ServedModelPath(context.Context) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.queries++
	if r.queries <= r.failFirst {
		return "", errors.New("503 loading model")
	}
	if r.served == "" {
		return "", errors.New("engine down")
	}
	return r.served, nil
}

func (r *reportingAdapter) serve(path string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.served = path
}

// An engine that reports its model is asked rather than trusted to match
// the run dir. That holds when llm-init restarted after a failed engine
// restart (the run dir already names the new model) and when the engine
// was restarted by hand, and it catches an engine that came back on the
// previous model.
func TestRun_AsksTheEngineWhichModelItServes(t *testing.T) {
	t.Parallel()
	const oldPath = "/cache/hf/hub/models--a--b/snapshots/old/model.gguf"
	cases := []struct {
		name       string
		sentinel   string // "new" seeds the path this pass writes
		served     string // "new" is the path this pass writes
		settle     handoff.State
		loadsNew   bool // a confirmed restart comes back on the new model
		wantPhase  progress.Phase
		wantReqs   int
		wantErrHas string
	}{
		{name: "llm_init_restarted", sentinel: "new", served: oldPath, settle: handoff.StateConfirmed, loadsNew: true, wantPhase: progress.PhaseReady, wantReqs: 1},
		{name: "engine_already_on_new", sentinel: oldPath, served: "new", settle: handoff.StateConfirmed, wantPhase: progress.PhaseReady, wantReqs: 0},
		{name: "came_back_on_old", sentinel: oldPath, served: oldPath, settle: handoff.StateConfirmed, wantPhase: progress.PhaseFailed, wantReqs: 1, wantErrHas: "restarted on"},
		{name: "no_supervisor", sentinel: oldPath, served: oldPath, settle: handoff.StateNoSupervisor, wantPhase: progress.PhaseFailed, wantReqs: 1, wantErrHas: "still serves the previous model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			o, rl, ra, newPath := reportingSetup(t, tc.sentinel, tc.served, tc.settle, tc.loadsNew)
			mgr, err := New(o)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go func() { _ = mgr.Run(ctx) }()

			waitPhase(t, o.Manager, tc.wantPhase)
			if got := rl.count(); got != tc.wantReqs {
				t.Errorf("relaunch requests = %d, want %d", got, tc.wantReqs)
			}
			if tc.wantErrHas != "" && !strings.Contains(o.Manager.Snapshot().LastError, tc.wantErrHas) {
				t.Errorf("last_error = %q, want it to contain %q", o.Manager.Snapshot().LastError, tc.wantErrHas)
			}
			if tc.wantPhase != progress.PhaseFailed {
				return
			}
			// Restarted by hand onto the new model: the retry finds it
			// there and asks for nothing more.
			ra.serve(newPath)
			mgr.Retry()
			waitPhase(t, o.Manager, progress.PhaseReady)
			if got := rl.count(); got != tc.wantReqs {
				t.Errorf("relaunch requests after a restart by hand = %d, want still %d", got, tc.wantReqs)
			}
		})
	}
}

// The engine is checked once alive even when the pass asked for no
// restart. llm-init restarted while the engine was still loading the
// previous model onto a run dir that already named the new one, so the
// first queries fail and nothing looks changed; the engine that comes
// alive is on the previous model and is restarted within the same pass.
// An engine that never says is ready, since WaitAlive passed and a
// llama.cpp without /props would otherwise never be.
func TestRun_ChecksTheServedModelOnceAlive(t *testing.T) {
	t.Parallel()
	const oldPath = "/cache/hf/hub/models--a--b/snapshots/old/model.gguf"
	cases := []struct {
		name       string
		served     string
		failFirst  int
		loadsNew   bool
		wantPhase  progress.Phase
		wantReqs   int
		wantErrHas string
	}{
		{name: "loading_previous_when_llm_init_restarted", served: oldPath, failFirst: 2, loadsNew: true, wantPhase: progress.PhaseReady, wantReqs: 1},
		{name: "comes_back_on_previous", served: oldPath, failFirst: 2, wantPhase: progress.PhaseFailed, wantReqs: 1, wantErrHas: "restarted on"},
		{name: "never_says", served: "", wantPhase: progress.PhaseReady, wantReqs: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			o, rl, ra, _ := reportingSetup(t, "new", tc.served, handoff.StateConfirmed, tc.loadsNew)
			ra.failFirst = tc.failFirst
			mgr, err := New(o)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go func() { _ = mgr.Run(ctx) }()

			waitPhase(t, o.Manager, tc.wantPhase)
			if got := rl.count(); got != tc.wantReqs {
				t.Errorf("relaunch requests = %d, want %d", got, tc.wantReqs)
			}
			if tc.wantErrHas != "" && !strings.Contains(o.Manager.Snapshot().LastError, tc.wantErrHas) {
				t.Errorf("last_error = %q, want it to contain %q", o.Manager.Snapshot().LastError, tc.wantErrHas)
			}
		})
	}
}

func reportingSetup(t *testing.T, sentinelPath, served string, settle handoff.State, loadsNew bool) (Options, *fakeRelauncher, *reportingAdapter, string) {
	t.Helper()
	runDir := t.TempDir()
	newPath := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(newPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if sentinelPath == "new" {
		sentinelPath = newPath
	}
	if served == "new" {
		served = newPath
	}
	if err := sentinel.Write(runDir, sentinelPath); err != nil {
		t.Fatal(err)
	}
	o := baseOptions(t, []config.ModelSource{
		hfSource("a/b", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "model.gguf"),
	})
	o.Config.Runtime.RunDir = runDir
	o.Config.Engine.Kind = config.EngineLlamaCpp
	o.HFRun = func(context.Context, hfwrap.Config, progress.Sink) (hfwrap.Result, error) {
		return hfwrap.Result{Status: "ok", Path: newPath}, nil
	}
	aliveBeforeBoot := false
	ra := &reportingAdapter{
		fakeAdapter: &fakeAdapter{
			kind:            config.EngineLlamaCpp,
			aliveBeforeBoot: &aliveBeforeBoot,
			readyState:      adapter.ReadyState{Alive: true, ModelExists: true},
		},
		served: served,
	}
	o.Adapter = ra
	rl := &fakeRelauncher{settle: settle}
	if loadsNew {
		rl.onRequest = func() { ra.serve(newPath) }
	}
	o.Relauncher = rl
	return o, rl, ra, newPath
}

func TestSameModelFile(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		a, b string
		want bool
	}{
		{"/cache/hf/hub/models--o--r/snapshots/c1/m.gguf", "/cache/hf/hub/models--o--r/snapshots/c1/m.gguf", true},
		{"/root/.cache/huggingface/hub/models--o--r/snapshots/c1/m.gguf", "/cache/hf/hub/models--o--r/snapshots/c1/m.gguf", true},
		{"/cache/hf/hub/models--o--r/snapshots/c1/m.gguf", "/cache/hf/hub/models--o--r/snapshots/c2/m.gguf", false},
		{"/models/a/m.gguf", "/models/b/m.gguf", false},
		{"/models/a/./m.gguf", "/models/a/m.gguf", true},
	} {
		if got := sameModelFile(tc.a, tc.b); got != tc.want {
			t.Errorf("sameModelFile(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
	if !extrasOnlyChange("/m.gguf\n/old-mmproj.gguf", "/m.gguf\n/new-mmproj.gguf") {
		t.Error("a new projector under the same main model is an extras-only change")
	}
	if extrasOnlyChange("", "/m.gguf") || extrasOnlyChange("/a.gguf", "/b.gguf") {
		t.Error("a first install or a new main model is not an extras-only change")
	}
}

func waitPhase(t *testing.T, pm progress.Manager, want progress.Phase) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for pm.Snapshot().Phase != want {
		if time.Now().After(deadline) {
			t.Fatalf("phase = %v, want %v", pm.Snapshot().Phase, want)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// An upgrade leaves the previous model_path behind, and the engine that
// started on it serves the previous model. When the pass moves the run
// dir to a new model, the engine is asked to relaunch before the pass
// waits for it to come alive; when the pass lands on the same model, or
// there was no model before, it is not.
func TestRun_RelaunchesTheEngineWhenTheModelChanged(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		previous string // "" seeds no sentinel; "same" seeds the new path
		settle   handoff.State
		want     int
	}{
		{name: "upgrade", previous: "/cache/snapshots/old/model.gguf", settle: handoff.StateConfirmed, want: 1},
		{name: "upgrade_unverified", previous: "/cache/snapshots/old/model.gguf", settle: handoff.StateUnverified, want: 1},
		{name: "same_model", previous: "same", settle: handoff.StateConfirmed, want: 0},
		{name: "first_install", previous: "", settle: handoff.StateConfirmed, want: 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			runDir := t.TempDir()
			mainPath := filepath.Join(t.TempDir(), "model.gguf")
			if err := os.WriteFile(mainPath, []byte("x"), 0o644); err != nil {
				t.Fatal(err)
			}
			switch tc.previous {
			case "":
			case "same":
				if err := sentinel.Write(runDir, mainPath); err != nil {
					t.Fatal(err)
				}
			default:
				if err := sentinel.Write(runDir, tc.previous); err != nil {
					t.Fatal(err)
				}
			}

			o := baseOptions(t, []config.ModelSource{
				hfSource("a/b", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "model.gguf"),
			})
			o.Config.Runtime.RunDir = runDir
			o.Config.Engine.Kind = config.EngineLlamaCpp
			o.HFRun = func(context.Context, hfwrap.Config, progress.Sink) (hfwrap.Result, error) {
				return hfwrap.Result{Status: "ok", Path: mainPath}, nil
			}
			rl := &fakeRelauncher{settle: tc.settle}
			o.Relauncher = rl
			aliveBeforeBoot := false
			var requestsAtWait atomic.Int32
			requestsAtWait.Store(-1)
			o.Adapter = &fakeAdapter{
				kind:            config.EngineLlamaCpp,
				aliveBeforeBoot: &aliveBeforeBoot,
				readyState:      adapter.ReadyState{Alive: true, ModelExists: true},
				waitFn: func(context.Context) error {
					requestsAtWait.CompareAndSwap(-1, int32(rl.count()))
					return nil
				},
			}
			mgr, err := New(o)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go func() { _ = mgr.Run(ctx) }()

			deadline := time.Now().Add(2 * time.Second)
			for o.Manager.Snapshot().Phase != progress.PhaseReady {
				if time.Now().After(deadline) {
					t.Fatalf("phase = %v, want ready", o.Manager.Snapshot().Phase)
				}
				time.Sleep(5 * time.Millisecond)
			}
			if got := rl.count(); got != tc.want {
				t.Errorf("relaunch requests = %d, want %d", got, tc.want)
			}
			if got := int(requestsAtWait.Load()); got != tc.want {
				t.Errorf("relaunch requests before WaitAlive = %d, want %d", got, tc.want)
			}
		})
	}
}
