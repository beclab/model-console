package obs

import (
	"context"

	"github.com/llm-init/llm-init/internal/progress"
)

// PhaseAndDownloadMetricsSubscriber attaches a goroutine to mgr.Subscribe()
// that mirrors every progress.State update into the corresponding
// Prometheus collectors. It is the single source of truth for
// `llm_init_phase`, `llm_init_download_bytes_total`,
// `llm_init_download_bytes_target`, and `llm_init_download_speed_bps`,
// which were declared by `NewMetrics` since v1.0 but had no producer
// — every collector reported zero in production. Pre-v1.0.3 only the
// initial PhaseInit gauge was set (by `controlplane.publishInitialMetrics`)
// and never updated again.
//
// The subscriber blocks on `mgr.Subscribe()` until ctx is done. It is
// safe to call once; passing the same Manager twice would create two
// subscribers and double-count `DownloadBytesTotal` deltas, which the
// caller is expected to avoid (in production main.go calls it exactly
// once).
//
// Counter-vs-Gauge note: `DownloadBytesTotal` is a Counter, so it must
// monotonically increase. progress.State.BytesCompleted can shrink
// (e.g. after a retry resets the baseline), so the subscriber tracks
// the last-seen value and only adds positive deltas. Decreases are
// recorded as a baseline reset (no Counter mutation).
//
// `sourceKind` is the label value applied to the counter; pass the
// primary source kind string ("hf" / "url" / "ollama" / "ollama-url"
// — the v1.1 ModelSource.kind enum) via cfg.PrimarySourceKind().
// Unknown source kinds receive an "unknown" label so the counter
// always has a label set.
func PhaseAndDownloadMetricsSubscriber(
	ctx context.Context,
	mgr progress.Manager,
	metrics *Metrics,
	allPhases []string,
	sourceKind string,
) {
	if mgr == nil || metrics == nil {
		return
	}
	if sourceKind == "" {
		sourceKind = "unknown"
	}
	ch, unsub := mgr.Subscribe()
	defer unsub()

	var lastBytesCompleted int64

	for {
		select {
		case <-ctx.Done():
			return
		case st, ok := <-ch:
			if !ok {
				return
			}
			metrics.SetPhase(string(st.Phase), allPhases)
			metrics.DownloadBytesTarget.Set(float64(st.BytesTotal))
			metrics.DownloadSpeedBPS.Set(st.SpeedBytesPerSec)

			delta := st.BytesCompleted - lastBytesCompleted
			if delta > 0 {
				metrics.DownloadBytesTotal.WithLabelValues(sourceKind).Add(float64(delta))
				lastBytesCompleted = st.BytesCompleted
			} else if delta < 0 {
				// BytesCompleted moved backwards (retry reset); rebase
				// without touching the Counter so monotonicity holds.
				lastBytesCompleted = st.BytesCompleted
			}
		}
	}
}
