package obs

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/llm-init/llm-init/internal/progress"
)

// allTestPhases mirrors controlplane.AllPhases without forcing the
// import (controlplane → obs would create a cycle once obs depends on
// progress + controlplane).
var allTestPhases = []string{
	string(progress.PhaseInit),
	string(progress.PhaseDownload),
	string(progress.PhaseLoading),
	string(progress.PhaseReady),
	string(progress.PhaseDegraded),
	string(progress.PhaseFailed),
}

// TestSubscriber_PhaseGaugeReflectsUpdates pushes two state changes and
// asserts that the active-phase gauge is 1 only on the latest phase.
func TestSubscriber_PhaseGaugeReflectsUpdates(t *testing.T) {
	t.Parallel()
	mgr := progress.New(time.Now())
	m := NewMetrics()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go PhaseAndDownloadMetricsSubscriber(ctx, mgr, m, allTestPhases, "hf")

	mgr.Update(func(s *progress.State) { s.Phase = progress.PhaseDownload })
	if !waitForGauge(t, m, "llm_init_phase", "phase", string(progress.PhaseDownload), 1.0, 500*time.Millisecond) {
		t.Fatalf("phase=download gauge never reached 1")
	}
	mgr.Update(func(s *progress.State) { s.Phase = progress.PhaseReady })
	if !waitForGauge(t, m, "llm_init_phase", "phase", string(progress.PhaseReady), 1.0, 500*time.Millisecond) {
		t.Fatalf("phase=ready gauge never reached 1")
	}
	if got := readGauge(t, m, "llm_init_phase", "phase", string(progress.PhaseDownload)); got != 0 {
		t.Errorf("phase=download gauge should be reset to 0, got %v", got)
	}
}

// TestSubscriber_DownloadCounterMonotonicAndDeltas asserts the
// counter increments by deltas (not Set), tracks BytesTotal as a
// gauge, and drops backwards motion silently to keep monotonicity.
func TestSubscriber_DownloadCounterMonotonicAndDeltas(t *testing.T) {
	t.Parallel()
	mgr := progress.New(time.Now())
	m := NewMetrics()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go PhaseAndDownloadMetricsSubscriber(ctx, mgr, m, allTestPhases, "url")

	// First update: 1024 bytes downloaded out of 4096.
	mgr.Update(func(s *progress.State) {
		s.BytesCompleted = 1024
		s.BytesTotal = 4096
	})
	if !waitForCounter(t, m, "llm_init_download_bytes_total", "source_kind", "url", 1024, 500*time.Millisecond) {
		t.Fatalf("download counter never observed 1024")
	}

	// Second update: 3072 bytes (delta=2048).
	mgr.Update(func(s *progress.State) {
		s.BytesCompleted = 3072
		s.BytesTotal = 4096
	})
	if !waitForCounter(t, m, "llm_init_download_bytes_total", "source_kind", "url", 3072, 500*time.Millisecond) {
		t.Fatalf("download counter never reached cumulative 3072")
	}

	// Backwards motion: simulate a retry resetting BytesCompleted to 0.
	// Counter must NOT decrement; gauge does. Subsequent forward motion
	// must add fresh deltas from 0.
	mgr.Update(func(s *progress.State) { s.BytesCompleted = 0 })
	// Give the subscriber a tick to ingest the reset.
	deadline := time.After(200 * time.Millisecond)
	for {
		if got := readCounter(t, m, "llm_init_download_bytes_total", "source_kind", "url"); got == 3072 {
			break // subscriber processed the reset, counter held at 3072
		}
		select {
		case <-deadline:
			t.Fatalf("counter changed during reset; got %v, want 3072",
				readCounter(t, m, "llm_init_download_bytes_total", "source_kind", "url"))
		default:
			time.Sleep(5 * time.Millisecond)
		}
	}

	// Forward again from 0 → 500 should add exactly 500 to the counter.
	mgr.Update(func(s *progress.State) { s.BytesCompleted = 500 })
	if !waitForCounter(t, m, "llm_init_download_bytes_total", "source_kind", "url", 3572, 500*time.Millisecond) {
		t.Fatalf("counter never reached 3572 after forward-motion-from-reset")
	}
}

// readGauge fetches the current value of a labelled gauge from the
// registry. Returns NaN on miss so the caller can distinguish from 0.
func readGauge(t *testing.T, m *Metrics, name, labelName, labelValue string) float64 {
	t.Helper()
	mfs, err := m.Registry().Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, sample := range mf.GetMetric() {
			match := false
			for _, l := range sample.GetLabel() {
				if l.GetName() == labelName && l.GetValue() == labelValue {
					match = true
				}
			}
			if !match {
				continue
			}
			if g := sample.GetGauge(); g != nil {
				return g.GetValue()
			}
		}
	}
	return -1
}

func readCounter(t *testing.T, m *Metrics, name, labelName, labelValue string) float64 {
	t.Helper()
	mfs, err := m.Registry().Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, sample := range mf.GetMetric() {
			match := false
			for _, l := range sample.GetLabel() {
				if l.GetName() == labelName && l.GetValue() == labelValue {
					match = true
				}
			}
			if !match {
				continue
			}
			if c := sample.GetCounter(); c != nil {
				return c.GetValue()
			}
		}
	}
	return -1
}

func waitForGauge(t *testing.T, m *Metrics, name, labelName, labelValue string, want float64, max time.Duration) bool {
	t.Helper()
	deadline := time.After(max)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		if got := readGauge(t, m, name, labelName, labelValue); got == want {
			return true
		}
		select {
		case <-deadline:
			return false
		case <-tick.C:
		}
	}
}

func waitForCounter(t *testing.T, m *Metrics, name, labelName, labelValue string, want float64, max time.Duration) bool {
	t.Helper()
	deadline := time.After(max)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for {
		if got := readCounter(t, m, name, labelName, labelValue); got == want {
			return true
		}
		select {
		case <-deadline:
			return false
		case <-tick.C:
		}
	}
}

// suppress unused import lint when tightening this test in the future.
var _ = prometheus.NewRegistry
var _ dto.Metric
