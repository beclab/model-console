package obs

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/version"
)

// TestNewMetrics_RegistersAll asserts every collector listed on the
// Metrics struct is wired up and exported through the registry. The
// test's `want` set is the operator-facing /metrics surface, so
// registering a collector without listing it here is the regression
// this guards against.
//
// Renamed from the v1.0.x-era "RegistersAllEleven" -- the count
// became stale at v1.0.3 (download retries) and again at v1.0.5
// (SSE / retry / proxy / transport observability). The set is no
// longer fixed-size, so the name no longer claims one.
func TestNewMetrics_RegistersAll(t *testing.T) {
	t.Parallel()
	m := NewMetrics()
	// Sanity-gather the empty registry first: any registration-time
	// duplicate or labelling error would surface here, before we
	// pollute the failure with bad sample data.
	if _, err := m.Registry().Gather(); err != nil {
		t.Fatalf("Gather: %v", err)
	}

	// Touch each metric so it shows up in Gather (counters and histograms
	// without any observed sample are not exported).
	m.Phase.WithLabelValues("init").Set(1)
	m.DownloadBytesTotal.WithLabelValues("hf").Add(1)
	m.DownloadBytesTarget.Set(1)
	m.DownloadSpeedBPS.Set(1)
	m.DownloadRetriesTotal.WithLabelValues("hf", "network_timeout").Inc()
	m.EngineProbeDuration.WithLabelValues("vllm", "/api/ready").Observe(0.1)
	m.EngineReady.WithLabelValues("vllm").Set(1)
	m.VerifyTotal.WithLabelValues("ok", "startup").Inc()
	m.DataPlaneRequests.WithLabelValues("/v1/chat/completions", "200").Inc()
	m.DataPlaneLatency.WithLabelValues("/v1/chat/completions").Observe(0.1)
	m.RetryThrottledTotal.Inc()
	m.ProxyPanicsTotal.WithLabelValues("vllm").Inc()
	m.TransportRetriesTotal.WithLabelValues("network_timeout").Inc()
	m.DiagGPUCallsTotal.WithLabelValues("ok").Inc()
	m.SetBuildInfo(version.Info{Version: "v1.2.3", Commit: "abc"})

	families, err := m.Registry().Gather()
	if err != nil {
		t.Fatalf("Gather 2: %v", err)
	}

	want := map[string]bool{
		"llm_init_phase":                              false,
		"llm_init_download_bytes_total":               false,
		"llm_init_download_bytes_target":              false,
		"llm_init_download_speed_bps":                 false,
		"llm_init_download_retries_total":             false,
		"llm_init_engine_probe_duration_seconds":      false, // v1.0.5 canonical
		"llm_init_engine_ready":                       false,
		"llm_init_verify_total":                       false,
		"llm_init_dataplane_requests_total":           false,
		"llm_init_dataplane_request_duration_seconds": false,
		"llm_init_build_info":                         false,
		"llm_init_retry_throttled_total":              false, // v1.0.5
		"llm_init_proxy_panics_total":                 false, // v1.0.5
		"llm_init_transport_retries_total":            false, // v1.0.5
		"llm_init_diag_gpu_calls_total":               false, // v1.0.7
	}
	for _, f := range families {
		if _, ok := want[f.GetName()]; ok {
			want[f.GetName()] = true
		}
	}
	for name, present := range want {
		if !present {
			t.Errorf("metric %q not exposed", name)
		}
	}
}

func TestMetrics_HandlerExposesPrometheusFormat(t *testing.T) {
	t.Parallel()
	m := NewMetrics()
	m.SetBuildInfo(version.Info{Version: "v1.2.3", Commit: "deadbee"})
	m.Phase.WithLabelValues("init").Set(1)
	// Touch label-bearing counter / histogram so they show up in Gather.
	m.DownloadBytesTotal.WithLabelValues("hf").Add(0)
	m.EngineProbeDuration.WithLabelValues("vllm", "/api/ready").Observe(0.01)

	req := httptest.NewRequest("GET", "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	if rr.Code != 200 {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	body := rr.Body.String()
	mustContain := []string{
		"# TYPE llm_init_phase gauge",
		"# TYPE llm_init_download_bytes_total counter",
		"# TYPE llm_init_engine_probe_duration_seconds histogram",
		"# TYPE llm_init_build_info gauge",
		`llm_init_phase{phase="init"} 1`,
		`llm_init_build_info{commit="deadbee",version="v1.2.3"} 1`,
	}
	for _, want := range mustContain {
		if !strings.Contains(body, want) {
			t.Errorf("body missing %q\n--- body ---\n%s", want, body)
		}
	}
}

func TestSetPhase_OnlyOneActive(t *testing.T) {
	t.Parallel()
	m := NewMetrics()
	all := []string{"init", "download", "ready"}
	m.SetPhase("download", all)

	req := httptest.NewRequest("GET", "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)
	body := rr.Body.String()

	for _, p := range all {
		want := `llm_init_phase{phase="` + p + `"}`
		if !strings.Contains(body, want) {
			t.Errorf("missing line for phase %s", p)
		}
	}
	if !strings.Contains(body, `llm_init_phase{phase="download"} 1`) {
		t.Errorf("download phase not 1: %s", body)
	}
	if !strings.Contains(body, `llm_init_phase{phase="init"} 0`) {
		t.Errorf("init phase not 0: %s", body)
	}

	// Switching phase clears the previous one.
	m.SetPhase("ready", all)
	rr = httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)
	body = rr.Body.String()
	if !strings.Contains(body, `llm_init_phase{phase="ready"} 1`) {
		t.Errorf("ready not 1 after switch: %s", body)
	}
	if !strings.Contains(body, `llm_init_phase{phase="download"} 0`) {
		t.Errorf("download not reset to 0: %s", body)
	}
}

func TestNewMetrics_TwiceIndependent(t *testing.T) {
	t.Parallel()
	// Each NewMetrics owns its own registry, so creating two does not
	// trigger duplicate-collector panics.
	a := NewMetrics()
	b := NewMetrics()
	if a.reg == b.reg {
		t.Error("registries must be independent")
	}
}
