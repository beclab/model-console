package enginecapacity

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

// recorder collects what the reporter publishes.
type recorder struct {
	mu   sync.Mutex
	seen []Capacity
	err  error
}

func (r *recorder) record(c Capacity) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.err != nil {
		return false, r.err
	}
	r.seen = append(r.seen, c)
	return true, nil
}

func (r *recorder) all() []Capacity {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Capacity(nil), r.seen...)
}

func (r *recorder) waitFor(t *testing.T, n int) []Capacity {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if got := r.all(); len(got) >= n {
			return got
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d reading(s); got %d", n, len(r.all()))
	return nil
}

func llamacppTarget(url, args string) func() Options {
	return func() Options {
		parsed, _ := config.ParseEngineArgs(config.EngineLlamaCpp, args)
		return Options{
			Kind:      config.EngineLlamaCpp,
			Args:      parsed,
			EngineURL: url,
		}
	}
}

func TestReporter_PublishesOnceWhenReady(t *testing.T) {
	var probes atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		probes.Add(1)
		_, _ = w.Write([]byte(
			`{"total_slots": 2, "default_generation_settings": {"n_ctx": 6144}}`))
	}))
	defer srv.Close()

	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target:   llamacppTarget(srv.URL, "-c 12288 -np 2 -no-kvu"),
		Ready:    func() bool { return true },
		Record:   rec.record,
		Interval: time.Millisecond,
	}).Run(ctx)

	got := rec.waitFor(t, 1)
	if got[0].ContextSize != 6144 || got[0].MaxConcurrency != 2 {
		t.Errorf("published (%d, %d), want (6144, 2)",
			got[0].ContextSize, got[0].MaxConcurrency)
	}

	// A capacity is a startup decision, so a steady-state ready engine is
	// not re-probed. With a 1ms tick, a reporter that polled would have
	// run hundreds of times by now.
	time.Sleep(50 * time.Millisecond)
	if n := probes.Load(); n != 1 {
		t.Errorf("probed %d times while steadily ready, want 1", n)
	}
	if n := len(rec.all()); n != 1 {
		t.Errorf("published %d readings, want 1", n)
	}
}

// A relaunch is the one thing that changes capacity. A dip is one way to
// notice one -- see the two tests below for the ways that are not.
func TestReporter_ReprobesAfterAReadinessDip(t *testing.T) {
	var slots atomic.Int64
	slots.Store(2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(
			`{"total_slots": ` + strconv.Itoa(int(slots.Load())) +
				`, "default_generation_settings": {"n_ctx": 4096}}`))
	}))
	defer srv.Close()

	var ready atomic.Bool
	ready.Store(true)
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target:   llamacppTarget(srv.URL, "-c 8192 -np 2 -no-kvu"),
		Ready:    ready.Load,
		Record:   rec.record,
		Interval: time.Millisecond,
	}).Run(ctx)

	rec.waitFor(t, 1)

	// The engine restarts with different flags.
	ready.Store(false)
	slots.Store(4)
	time.Sleep(10 * time.Millisecond)
	ready.Store(true)

	got := rec.waitFor(t, 2)
	if got[1].MaxConcurrency != 4 {
		t.Errorf("second reading MaxConcurrency = %d, want 4", got[1].MaxConcurrency)
	}
}

// TestReporter_ReprobesWhenTheEngineIsRelaunchedInPlace is the regression
// that readiness alone cannot cover.
//
// The wrapper relaunches the engine inside the process, and the health loop
// holds its last verdict through a grace period longer than the relaunch
// takes. Readiness therefore never dips, and a reporter that waited for one
// left the capacity of the replaced engine on the card -- where Router reads
// it and admits requests against it. Note that the flags do not change here:
// the operator's edit was to something else, or the relaunch was the
// wrapper's own doing.
func TestReporter_ReprobesWhenTheEngineIsRelaunchedInPlace(t *testing.T) {
	var slots atomic.Int64
	slots.Store(1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(
			`{"total_slots": ` + strconv.Itoa(int(slots.Load())) +
				`, "default_generation_settings": {"n_ctx": 4096}}`))
	}))
	defer srv.Close()

	var relaunch atomic.Value
	relaunch.Store("")
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target: llamacppTarget(srv.URL, "-c 32768"),
		// Never false, which is the whole point.
		Ready:     func() bool { return true },
		Restarted: func() string { return relaunch.Load().(string) },
		Record:    rec.record,
		Interval:  time.Millisecond,
	}).Run(ctx)

	rec.waitFor(t, 1)

	// The engine goes away and comes back holding more slots. The tracker
	// sees it answering again and names the relaunch.
	slots.Store(8)
	relaunch.Store("7@2026-08-29T14:00:00Z")

	got := rec.waitFor(t, 2)
	if got[1].MaxConcurrency != 8 {
		t.Errorf("second reading MaxConcurrency = %d, want 8", got[1].MaxConcurrency)
	}
}

// A relaunch that was only requested is not one that happened. Probing then
// reads the engine that is about to be replaced and settles on its answer,
// which is the same staleness by a shorter route.
func TestReporter_IgnoresARelaunchThatHasNotComeBack(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(
			`{"total_slots": 1, "default_generation_settings": {"n_ctx": 4096}}`))
	}))
	defer srv.Close()

	var relaunch atomic.Value
	relaunch.Store("3@2026-08-29T13:00:00Z")
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target:    llamacppTarget(srv.URL, "-c 32768"),
		Ready:     func() bool { return true },
		Restarted: func() string { return relaunch.Load().(string) },
		Record:    rec.record,
		Interval:  time.Millisecond,
	}).Run(ctx)

	rec.waitFor(t, 1)

	// What the tracker reports between the request and the engine
	// answering again.
	relaunch.Store("")
	time.Sleep(30 * time.Millisecond)
	if n := len(rec.all()); n != 1 {
		t.Errorf("published %d readings, want 1: a requested relaunch is not evidence of one", n)
	}
}

// TestReporter_ReprobesWhenTheFlagsChange is the case where nothing is
// supervising the engine: the card was edited, the relaunch was signalled and
// never happened. The engine that is running is still the old one, so its
// measurement is honest -- but it was taken under different flags, and the
// reading has to say so rather than being left to describe an edit it predates.
func TestReporter_ReprobesWhenTheFlagsChange(t *testing.T) {
	var probes atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		probes.Add(1)
		_, _ = w.Write([]byte(
			`{"total_slots": 4, "default_generation_settings": {"n_ctx": 4096}}`))
	}))
	defer srv.Close()

	var args atomic.Value
	args.Store("-c 16384 -np 4 -no-kvu")
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target: func() Options {
			raw := args.Load().(string)
			parsed, _ := config.ParseEngineArgs(config.EngineLlamaCpp, raw)
			return Options{
				Kind:      config.EngineLlamaCpp,
				Args:      parsed,
				EngineURL: srv.URL,
			}
		},
		Ready:    func() bool { return true },
		Record:   rec.record,
		Interval: time.Millisecond,
	}).Run(ctx)

	rec.waitFor(t, 1)
	after := probes.Load()

	args.Store("-c 32768 -np 4 -no-kvu")

	rec.waitFor(t, 2)
	if probes.Load() <= after {
		t.Error("the engine was not asked again after the flags it was measured under changed")
	}
}

// An engine that answers /readyz before it publishes /props gets retried
// rather than pinned to a declaration.
func TestReporter_RetriesUntilAProbeAnswers(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = w.Write([]byte(
			`{"total_slots": 1, "default_generation_settings": {"n_ctx": 2048}}`))
	}))
	defer srv.Close()

	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target:   llamacppTarget(srv.URL, "-c 2048 -np 1"),
		Ready:    func() bool { return true },
		Record:   rec.record,
		Interval: time.Millisecond,
	}).Run(ctx)

	got := rec.waitFor(t, 1)
	// Nothing was published while the probe was failing: a declaration
	// Router would not be told was superseded is worse than a moment
	// without one.
	if got[0].Source != SourceLlamacppProps {
		t.Errorf("Source = %q, want the measurement", got[0].Source)
	}
	if len(got) != 1 {
		t.Errorf("published %d readings, want 1", len(got))
	}
}

// An endpoint that will never exist must not become a permanent poll, and
// must not leave the card without any capacity at all.
func TestReporter_GivesUpAndPublishesTheDeclaration(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target:   llamacppTarget(srv.URL, "-c 8192 -np 2 -no-kvu"),
		Ready:    func() bool { return true },
		Record:   rec.record,
		Interval: time.Millisecond,
	}).Run(ctx)

	got := rec.waitFor(t, 1)
	if got[0].Source != SourceEngineArgs {
		t.Errorf("Source = %q, want %q", got[0].Source, SourceEngineArgs)
	}
	if got[0].ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want the declared 4096", got[0].ContextSize)
	}

	time.Sleep(50 * time.Millisecond)
	if n := calls.Load(); n > retryLimit {
		t.Errorf("probed %d times, want at most %d", n, retryLimit)
	}
	if n := len(rec.all()); n != 1 {
		t.Errorf("published %d readings, want 1", n)
	}
}

func TestReporter_NonLLMConfigFallbackAndChangeReprobe(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	var configured atomic.Int64
	configured.Store(1)
	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target: func() Options {
			return Options{
				Kind:                     config.EngineRerank,
				EngineURL:                srv.URL,
				ConfiguredMaxConcurrency: int(configured.Load()),
			}
		},
		Ready:    func() bool { return true },
		Record:   rec.record,
		Interval: time.Millisecond,
	}).Run(ctx)

	got := rec.waitFor(t, 1)
	if got[0].Source != SourceEngineConfig || got[0].MaxConcurrency != 1 {
		t.Fatalf("first reading = %+v, want engine_config/1", got[0])
	}
	firstCalls := calls.Load()
	configured.Store(2)
	got = rec.waitFor(t, 2)
	if got[1].Source != SourceEngineConfig || got[1].MaxConcurrency != 2 {
		t.Fatalf("second reading = %+v, want engine_config/2", got[1])
	}
	if calls.Load() <= firstCalls {
		t.Fatal("configured capacity change did not re-arm the probe")
	}
}

func TestReporter_NotReadyProbesNothing(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
	}))
	defer srv.Close()

	rec := &recorder{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go (&Reporter{
		Target:   llamacppTarget(srv.URL, "-c 8192"),
		Ready:    func() bool { return false },
		Record:   rec.record,
		Interval: time.Millisecond,
	}).Run(ctx)

	time.Sleep(50 * time.Millisecond)
	if n := calls.Load(); n != 0 {
		t.Errorf("probed %d times while not ready, want 0", n)
	}
	if n := len(rec.all()); n != 0 {
		t.Errorf("published %d readings, want 0", n)
	}
}

// A publish that fails is a warning, not a stall: the reporter has to
// stay alive and keep following readiness.
func TestReporter_SurvivesARecordFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(
			`{"total_slots": 1, "default_generation_settings": {"n_ctx": 512}}`))
	}))
	defer srv.Close()

	rec := &recorder{err: context.DeadlineExceeded}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go func() {
		defer close(done)
		(&Reporter{
			Target:   llamacppTarget(srv.URL, "-c 512 -np 1"),
			Ready:    func() bool { return true },
			Record:   rec.record,
			Interval: time.Millisecond,
		}).Run(ctx)
	}()

	time.Sleep(20 * time.Millisecond)
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Run did not return after ctx cancel")
	}
}
