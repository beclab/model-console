package obs

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/llm-init/llm-init/internal/version"
)

// Prometheus label NAMES (not values) used across the metrics struct.
// Pulled out as named constants so a label rename is a one-site edit
// and the goconst v2 sweep doesn't fire on the 4-way duplication of
// "engine_kind". labelResult appears twice (verify + diag.gpu
// counters); other label names ("route", "status", "reason", "phase",
// "endpoint", ...) appear <= 2 times in this file and a
// constant-per-string would obscure rather than clarify.
const (
	labelEngineKind = "engine_kind"
	labelResult     = "result"
)

// Metrics owns every Prometheus collector llm-init exports.
//
// The set is intentionally not fixed-size: the v1.0.x line started at
// 11 collectors and grew through v1.0.3 (download metrics) and v1.0.5
// (SSE / retry / proxy / transport observability). Don't bake a count
// into doc strings -- TestNewMetrics_RegistersAll is what holds the
// declared surface and the registry together.
//
// All metrics are registered on a private prometheus.Registry to keep
// llm-init's data isolated from any host process that might also expose
// /metrics. Handler() returns the http.Handler to mount on /metrics.
type Metrics struct {
	reg *prometheus.Registry

	Phase                *prometheus.GaugeVec   // labels: phase
	DownloadBytesTotal   *prometheus.CounterVec // labels: source_kind
	DownloadBytesTarget  prometheus.Gauge       // no labels
	DownloadSpeedBPS     prometheus.Gauge       // no labels
	DownloadRetriesTotal *prometheus.CounterVec // labels: source_kind, error_code
	// EngineProbeDuration measures Adapter.Ready probe latency. It
	// replaced the misleadingly-named llm_init_engine_rtt_seconds,
	// which was hard-removed in v1.1.0 (see CHANGELOG.md).
	EngineProbeDuration *prometheus.HistogramVec // labels: engine_kind, endpoint
	EngineReady         *prometheus.GaugeVec     // labels: engine_kind
	VerifyTotal         *prometheus.CounterVec   // labels: result, trigger
	DataPlaneRequests   *prometheus.CounterVec   // labels: route, status
	DataPlaneLatency    *prometheus.HistogramVec // labels: route
	BuildInfo           *prometheus.GaugeVec     // labels: version, commit

	// v1.0.5: /api/retry rate-limit telemetry. Bumps every time the
	// token bucket rejects a retry call so operators can correlate
	// "ensure looks slow" with "retry storm at the front door".
	RetryThrottledTotal prometheus.Counter // no labels

	// v1.0.5: ReverseProxy panic recovery. Pre-fix a panic in the
	// proxy's Director / ModifyResponse / transport goroutines would
	// surface only as a logged stack and a closed client connection;
	// operators saw "EOF" with no actionable root cause. The
	// recover() wrapper now bumps this counter so an alert can fire
	// when an engine is repeatedly tripping the proxy.
	ProxyPanicsTotal *prometheus.CounterVec // labels: engine_kind

	// v1.0.5: transport-level retry counter, the Prometheus side of
	// the State.TransportRetries split (see plan #12). The fetch
	// inner retry loop -- the one that handles transient I/O errors
	// inside a single download -- bumps this every time a chunk
	// retries; lifecycle-level retries (operator-driven /api/retry)
	// stay on DownloadRetriesTotal. The kind label discriminates
	// {http_5xx, network, timeout, eof} so dashboards can tell a
	// flapping upstream from a saturated network without parsing the
	// debug log.
	TransportRetriesTotal *prometheus.CounterVec // labels: kind

	// DiagGPUCallsTotal{result}: every GET /api/diag/gpu call. Result
	// label is "ok" | "error" so a sudden burst of error means the
	// engine became unreachable (not a misconfigured client).
	DiagGPUCallsTotal *prometheus.CounterVec

	// EngineSpecFetchTotal{result}: every attempt to read the engine's
	// own /api/engine-spec. The relay is deliberately silent towards
	// callers -- a catalog must not fail because the engine is briefly
	// unreachable -- which left an operator unable to tell "the engine
	// declares nothing" from "we could not ask it". The result label
	// separates the two.
	EngineSpecFetchTotal *prometheus.CounterVec

	// EngineSpecRowsDroppedTotal{reason}: engine-declared endpoints this
	// relay refused. A capability that fails validation disappears from
	// the catalog entirely, and the gateway downstream then answers
	// "operation not supported" for a route the engine really serves.
	// The reason label names the field that failed.
	EngineSpecRowsDroppedTotal *prometheus.CounterVec
}

// NewMetrics creates and registers every collector listed on the
// Metrics struct.
//
// The returned *Metrics holds the only references to the underlying
// vectors; callers mutate them via the exported fields (e.g.
// `m.Phase.WithLabelValues("init").Set(1)`).
func NewMetrics() *Metrics {
	r := prometheus.NewRegistry()
	m := &Metrics{
		reg: r,
		Phase: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "llm_init_phase",
			Help: "Current Phase: 1 for the active phase, 0 for the others.",
		}, []string{"phase"}),
		DownloadBytesTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_download_bytes_total",
			Help: "Total bytes downloaded so far.",
		}, []string{"source_kind"}),
		DownloadBytesTarget: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "llm_init_download_bytes_target",
			Help: "Target total bytes for the current download.",
		}),
		DownloadSpeedBPS: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "llm_init_download_speed_bps",
			Help: "Exponential moving average of download speed (bytes/sec).",
		}),
		DownloadRetriesTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_download_retries_total",
			Help: "Total number of download retries.",
		}, []string{"source_kind", "error_code"}),
		EngineProbeDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "llm_init_engine_probe_duration_seconds",
			Help:    "Adapter.Ready probe duration in seconds. Replaces llm_init_engine_rtt_seconds (the old name was misleading -- it never measured chat/completions latency, only Ready probes).",
			Buckets: prometheus.ExponentialBuckets(0.005, 2, 12), // 5ms..~10s
		}, []string{labelEngineKind, "endpoint"}),
		EngineReady: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "llm_init_engine_ready",
			Help: "Engine readiness: 1 ready, 0 not ready.",
		}, []string{labelEngineKind}),
		VerifyTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_verify_total",
			Help: "Verify runs labelled by result and trigger.",
		}, []string{labelResult, "trigger"}),
		DataPlaneRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_dataplane_requests_total",
			Help: "Data plane requests counted by route and HTTP status.",
		}, []string{"route", "status"}),
		DataPlaneLatency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "llm_init_dataplane_request_duration_seconds",
			Help:    "Data plane request duration in seconds.",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 14), // 1ms..~16s
		}, []string{"route"}),
		BuildInfo: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "llm_init_build_info",
			Help: "Static build metadata. Always 1.",
		}, []string{"version", "commit"}),
		RetryThrottledTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "llm_init_retry_throttled_total",
			Help: "Total POST /api/retry calls rejected by the token-bucket rate limit.",
		}),
		ProxyPanicsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_proxy_panics_total",
			Help: "Panics recovered inside the reverse-proxy ServeHTTP wrapper.",
		}, []string{labelEngineKind}),
		TransportRetriesTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_transport_retries_total",
			Help: "Transport-level retries inside the fetch inner loop. Distinct from llm_init_download_retries_total which counts lifecycle-level retries (operator /api/retry, repair scheduling).",
		}, []string{"kind"}),
		DiagGPUCallsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_diag_gpu_calls_total",
			Help: "Total GET /api/diag/gpu calls, labelled by result (ok|error).",
		}, []string{labelResult}),
		EngineSpecFetchTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_engine_spec_fetch_total",
			Help: "Reads of the engine's /api/engine-spec, labelled by result (ok|unreachable|http_status|decode_error|invalid_spec|no_endpoints).",
		}, []string{labelResult}),
		EngineSpecRowsDroppedTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_init_engine_spec_rows_dropped_total",
			Help: "Engine-declared endpoints refused by the relay, labelled by the field that failed validation.",
		}, []string{"reason"}),
	}

	r.MustRegister(
		m.Phase,
		m.DownloadBytesTotal,
		m.DownloadBytesTarget,
		m.DownloadSpeedBPS,
		m.DownloadRetriesTotal,
		m.EngineProbeDuration,
		m.EngineReady,
		m.VerifyTotal,
		m.DataPlaneRequests,
		m.DataPlaneLatency,
		m.BuildInfo,
		m.RetryThrottledTotal,
		m.ProxyPanicsTotal,
		m.TransportRetriesTotal,
		m.DiagGPUCallsTotal,
		m.EngineSpecFetchTotal,
		m.EngineSpecRowsDroppedTotal,
	)
	return m
}

// LogDeprecations emits one warn-level slog message per metric we still
// register but plan to remove, giving operators who scrape /metrics and
// pin Grafana queries a one-shot heads-up in the logs without reading the
// changelog. Called once from cmd/llm-init.
//
// As of v1.1.0 there are no deprecated metrics: llm_init_engine_rtt_seconds
// (deprecated v1.0.5) was hard-removed per the P1-5 decision (2026-05-16).
//
// New deprecations land here as additional slog.Warn calls (one per
// metric). Don't fold multiple deprecations into a single Warn -- the
// per-metric structure makes it grep-able from log aggregators.
func LogDeprecations() {}

// SetBuildInfo records the static build labels. Idempotent.
func (m *Metrics) SetBuildInfo(v version.Info) {
	m.BuildInfo.WithLabelValues(v.Version, v.Commit).Set(1)
}

// SetPhase ensures only the supplied phase is set to 1, with all known
// phases reset to 0. Callers must use the same string set used by the
// progress.Phase constants.
func (m *Metrics) SetPhase(active string, all []string) {
	for _, p := range all {
		val := 0.0
		if p == active {
			val = 1.0
		}
		m.Phase.WithLabelValues(p).Set(val)
	}
}

// Registry exposes the underlying registry; mostly useful for tests that
// want to gather samples directly.
func (m *Metrics) Registry() *prometheus.Registry { return m.reg }

// Handler returns an http.Handler that serves the standard Prometheus text
// format for the metrics in this Metrics instance.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.reg, promhttp.HandlerOpts{})
}
