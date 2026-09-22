package handoff

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// scriptedProbe answers from a script, repeating the last answer once the
// script runs out. The first answer is the baseline taken before any
// polling, so a script reads as [at-request, then one per poll].
type scriptedProbe struct {
	mu      sync.Mutex
	answers []bool
	calls   int
}

func (p *scriptedProbe) probe(context.Context) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	i := p.calls
	p.calls++
	if i >= len(p.answers) {
		i = len(p.answers) - 1
	}
	return p.answers[i]
}

func testTracker(t *testing.T, answers []bool) (*Tracker, string) {
	t.Helper()
	dir := t.TempDir()
	var probe LiveProbe
	if answers != nil {
		p := &scriptedProbe{answers: answers}
		probe = p.probe
	}
	return NewTracker(Config{
		RunDir:   dir,
		Probe:    probe,
		Window:   500 * time.Millisecond,
		Interval: time.Millisecond,
	}), dir
}

func TestTracker_EngineDipConfirmsRestart(t *testing.T) {
	t.Parallel()
	tr, dir := testTracker(t, []bool{true, false, true})

	receipt, err := tr.Request()
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if !receipt.Signaled {
		t.Error("Signaled must be true once the generation file is written")
	}
	if receipt.Supervision != SupervisionUnknown {
		t.Errorf("Supervision = %q at first request, want unknown", receipt.Supervision)
	}
	if receipt.Generation == "" {
		t.Error("Generation must identify the request")
	}

	tr.Wait()
	got := tr.Status()
	if got.State != StateConfirmed {
		t.Fatalf("State = %q, want confirmed", got.State)
	}
	if got.Supervision != SupervisionConfirmed {
		t.Errorf("Supervision = %q after an observed dip, want confirmed", got.Supervision)
	}
	if got.ObservedDownAt == nil || got.ObservedBackAt == nil {
		t.Errorf("both ends of the dip should be recorded: %+v", got)
	}

	// The evidence outlives the process: a fresh tracker on the same run
	// dir must answer "confirmed" before it has signaled anything itself.
	data, err := os.ReadFile(filepath.Join(dir, StateName))
	if err != nil {
		t.Fatalf("state file: %v", err)
	}
	var st persistedState
	if err := json.Unmarshal(data, &st); err != nil {
		t.Fatalf("state file decode: %v", err)
	}
	if !st.Supervised || st.LastConfirmedGeneration != receipt.Generation {
		t.Errorf("state file = %+v, want supervised at gen %q", st, receipt.Generation)
	}
	if again := NewTracker(Config{RunDir: dir}).Supervision(); again != SupervisionConfirmed {
		t.Errorf("reloaded Supervision = %q, want confirmed", again)
	}
}

// Relaunch is what a consumer follows to re-read something the engine only
// reports when it starts. It has to be empty until the engine has actually
// come back: reading it between the request and the return would measure the
// engine that is being replaced, and settle on that answer.
func TestTracker_RelaunchNamesOnlyACompletedRelaunch(t *testing.T) {
	t.Parallel()
	// Two dips: up, down, back for the first relaunch, then up again as the
	// second request's baseline, down, back.
	tr, _ := testTracker(t, []bool{true, false, true, true, false, true})

	if got := tr.Relaunch(); got != "" {
		t.Errorf("Relaunch = %q before any request, want empty", got)
	}

	if _, err := tr.Request(); err != nil {
		t.Fatalf("Request: %v", err)
	}
	tr.Wait()

	back := tr.Relaunch()
	if back == "" {
		t.Fatal("Relaunch is empty after the engine was observed back")
	}
	if again := tr.Relaunch(); again != back {
		t.Errorf("Relaunch changed without a relaunch: %q then %q", back, again)
	}

	// A second relaunch is a second token, so a consumer that compares
	// against what it last saw re-reads exactly once more.
	if _, err := tr.Request(); err != nil {
		t.Fatalf("second Request: %v", err)
	}
	tr.Wait()
	if next := tr.Relaunch(); next == back || next == "" {
		t.Errorf("second Relaunch = %q, want a new non-empty token (first was %q)", next, back)
	}
}

func TestTracker_RelaunchIsEmptyWhileTheRestartIsOnlyRequested(t *testing.T) {
	t.Parallel()
	// No probe wired, so the request lands in StateUnverified and the
	// engine is never observed back.
	tr, _ := testTracker(t, nil)

	if _, err := tr.Request(); err != nil {
		t.Fatalf("Request: %v", err)
	}
	tr.Wait()
	if got := tr.Relaunch(); got != "" {
		t.Errorf("Relaunch = %q for a restart that was never confirmed, want empty", got)
	}
}

func TestTracker_EngineNeverDipsMeansNoSupervisor(t *testing.T) {
	t.Parallel()
	tr, _ := testTracker(t, []bool{true})

	if _, err := tr.Request(); err != nil {
		t.Fatalf("Request: %v", err)
	}
	tr.Wait()

	got := tr.Status()
	if got.State != StateNoSupervisor {
		t.Fatalf("State = %q, want no-supervisor", got.State)
	}
	if got.Supervision != SupervisionUnknown {
		t.Errorf("Supervision = %q; an unanswered signal is not evidence of a supervisor",
			got.Supervision)
	}
	if got.ObservedDownAt != nil {
		t.Errorf("ObservedDownAt set without a dip: %+v", got)
	}
}

func TestTracker_EngineDownAndStayingDownIsUnverifiable(t *testing.T) {
	t.Parallel()
	// Down before we ask and down for the whole window. Reading that as a
	// successful restart is the false success this receipt exists to
	// remove, so the verdict has to be "cannot tell".
	tr, dir := testTracker(t, []bool{false})

	if _, err := tr.Request(); err != nil {
		t.Fatalf("Request: %v", err)
	}
	tr.Wait()

	if got := tr.Status().State; got != StateUnverified {
		t.Fatalf("State = %q, want unverified", got)
	}
	if got := tr.Supervision(); got != SupervisionUnknown {
		t.Errorf("Supervision = %q, want unknown", got)
	}
	if _, err := os.Stat(filepath.Join(dir, StateName)); !os.IsNotExist(err) {
		t.Errorf("nothing was observed, so no evidence should be recorded (stat err=%v)", err)
	}
}

// A wedged engine is the usual reason to ask for a relaunch, so the case
// has to produce an answer rather than a shrug. There is no dip to see —
// the engine is already down — which leaves the rise as the only edge that
// run will ever produce, and only a supervisor produces it.
func TestTracker_EngineAlreadyDownConfirmsOnTheRise(t *testing.T) {
	t.Parallel()
	tr, dir := testTracker(t, []bool{false, false, true})

	receipt, err := tr.Request()
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if receipt.Supervision != SupervisionUnknown {
		t.Errorf("receipt Supervision = %q; the evidence cannot exist yet", receipt.Supervision)
	}
	tr.Wait()

	got := tr.Status()
	if got.State != StateConfirmed {
		t.Fatalf("State = %q, want confirmed", got.State)
	}
	if got.Supervision != SupervisionConfirmed {
		t.Errorf("Supervision = %q, want confirmed", got.Supervision)
	}
	if got.ObservedDownAt == nil {
		t.Error("ObservedDownAt should carry the moment we found it down")
	}
	if got.ObservedBackAt == nil {
		t.Fatal("ObservedBackAt should carry the rise that confirmed the relaunch")
	}
	if tr.Relaunch() == "" {
		t.Error("Relaunch should name a completed relaunch once the engine is back")
	}
	if _, err := os.Stat(filepath.Join(dir, StateName)); err != nil {
		t.Errorf("the rise is evidence and belongs on disk: %v", err)
	}
}

// Relaunching is what a phase is derived from, so the states that settle
// without the engine ever going away have to read as false. Reporting one
// of those would hold a phase down over a model that never stopped
// serving.
func TestStatus_RelaunchingCoversOnlyAnOutstandingRelaunch(t *testing.T) {
	t.Parallel()
	at := time.Now()

	cases := []struct {
		name string
		st   Status
		want bool
	}{
		{"idle", Status{State: StateIdle}, false},
		{"signaled, nothing observed yet", Status{State: StateWatching}, true},
		{"down, not back", Status{State: StateConfirmed, ObservedDownAt: &at}, true},
		{"down and back", Status{State: StateConfirmed, ObservedDownAt: &at, ObservedBackAt: &at}, false},
		{"nothing acted on the signal", Status{State: StateNoSupervisor}, false},
		{"cannot tell", Status{State: StateUnverified}, false},
	}
	for _, tc := range cases {
		if got := tc.st.Relaunching(); got != tc.want {
			t.Errorf("%s: Relaunching = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestTracker_NoProbeSignalsWithoutVerdict(t *testing.T) {
	t.Parallel()
	tr, dir := testTracker(t, nil)

	receipt, err := tr.Request()
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if !receipt.Signaled {
		t.Error("the signal still reaches disk without a probe to watch it")
	}
	if got := tr.Status().State; got != StateUnverified {
		t.Errorf("State = %q, want unverified", got)
	}
	if ReadGeneration(dir) != receipt.Generation {
		t.Errorf("generation on disk = %q, want %q", ReadGeneration(dir), receipt.Generation)
	}
}

// The health loop probes on its own schedule, and at ten seconds that
// schedule is longer than a healthy relaunch — so being told about one
// only when it next happens to look means the dip passes unseen. The hook
// has to fire while the relaunch is still outstanding, or the observer
// would reconsider its cadence and find nothing to reconsider for.
func TestTracker_RequestNotifiesAnObserverWhileRelaunching(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	relaunching := make(chan bool, 1)
	var tr *Tracker
	tr = NewTracker(Config{
		RunDir:    dir,
		Probe:     (&scriptedProbe{answers: []bool{true}}).probe,
		Window:    500 * time.Millisecond,
		Interval:  time.Millisecond,
		OnRequest: func() { relaunching <- tr.Relaunching() },
	})

	if _, err := tr.Request(); err != nil {
		t.Fatalf("Request: %v", err)
	}

	select {
	case got := <-relaunching:
		if !got {
			t.Error("the observer was notified before the relaunch was outstanding")
		}
	default:
		t.Fatal("Request did not notify the observer")
	}
	tr.Wait()
}

// A tracker with nowhere to signal cannot produce a relaunch, and telling
// an observer to go looking for one would only cost it a probe.
func TestTracker_FailedRequestNotifiesNobody(t *testing.T) {
	t.Parallel()
	blocked := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(blocked, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	var notified atomic.Bool
	tr := NewTracker(Config{RunDir: blocked, OnRequest: func() { notified.Store(true) }})

	if _, err := tr.Request(); err == nil {
		t.Fatal("Request must fail when the generation file cannot be written")
	}
	if notified.Load() {
		t.Error("a request that never landed notified an observer anyway")
	}
}

func TestTracker_UnwritableRunDirFailsLoudly(t *testing.T) {
	t.Parallel()
	// A regular file where the run dir should be: MkdirAll cannot make
	// it, so the signal never lands and the caller must hear about it
	// rather than be told the engine will relaunch.
	blocked := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(blocked, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	tr := NewTracker(Config{RunDir: blocked})

	receipt, err := tr.Request()
	if err == nil {
		t.Fatal("Request must fail when the generation file cannot be written")
	}
	if receipt.Signaled {
		t.Error("Signaled must stay false when the write failed")
	}
	if got := tr.Status().State; got != StateIdle {
		t.Errorf("State = %q; a failed request never started", got)
	}
}

func TestTracker_LaterRequestSupersedesTheObservation(t *testing.T) {
	t.Parallel()
	tr, _ := testTracker(t, []bool{true})

	first, err := tr.Request()
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	second, err := tr.Request()
	if err != nil {
		t.Fatalf("Request: %v", err)
	}
	if first.Generation == second.Generation {
		t.Fatal("each request needs its own generation")
	}
	tr.Wait()

	got := tr.Status()
	if got.Generation != second.Generation {
		t.Errorf("Generation = %q, want the newest request %q", got.Generation, second.Generation)
	}
	if got.State != StateNoSupervisor {
		t.Errorf("State = %q, want the newest observation's verdict", got.State)
	}
}

func TestWriteEngineArgs_EmptyIsDistinctFromMissing(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()

	if raw, missing := ReadEngineArgs(dir); !missing || raw != "" {
		t.Fatalf("before any write: raw=%q missing=%v, want missing", raw, missing)
	}
	if err := WriteEngineArgs(dir, ""); err != nil {
		t.Fatalf("WriteEngineArgs: %v", err)
	}
	// "Model Console says no flags" must not read as "Model Console has
	// not spoken": the wrapper falls back to the chart env on the latter.
	if raw, missing := ReadEngineArgs(dir); missing || raw != "" {
		t.Fatalf("after empty write: raw=%q missing=%v, want present and empty", raw, missing)
	}
}
