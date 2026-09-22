package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
)

// newDiagAdapter is a small helper that builds a proxy.Adapter. The
// proxy package's other tests use newProxy() which doesn't expose
// Engine fields directly, so we duplicate the constructor shape here
// to keep the diag tests self-contained. EngineNativeStats now reaches
// out to the engine for every kind (vLLM /metrics, SGLang /server_info,
// llama.cpp /metrics best-effort), so most tests set Engine.URL to an
// httptest server; an unset URL defaults to a refusing port.
func newDiagAdapter(t *testing.T, cfg config.Config, kind config.EngineKind) *Adapter {
	t.Helper()
	if cfg.Engine.URL == "" {
		cfg.Engine.URL = "http://127.0.0.1:1"
	}
	cfg.Engine.Kind = kind
	a, err := NewAdapter(cfg, kind)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	return a
}

// promServer stages a /metrics endpoint returning the given body and
// 404s everything else. Used by the proxy diag tests as a stand-in for
// the engine's Prometheus exposition.
func promServer(t *testing.T, metricsBody string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			_, _ = w.Write([]byte(metricsBody))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestVLLMNativeStats_FullByDefault: cache_config_info reports
// cpu_offload_gb=0 -> full GPU; gpu_memory_utilization and the live
// kv_cache_usage_perc gauge are surfaced.
func TestVLLMNativeStats_FullByDefault(t *testing.T) {
	t.Parallel()
	srv := promServer(t, `# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{cpu_offload_gb="0",gpu_memory_utilization="0.92"} 1
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.3
`)
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineVLLM, URL: srv.URL},
	}, config.EngineVLLM)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want %q", got.GPU.Mode, adapter.GPUModeFull)
	}
	if got.Source != "vllm_metrics" {
		t.Errorf("Source = %q want vllm_metrics", got.Source)
	}
	if got.GPU.CPUOffloadGB != nil {
		t.Errorf("CPUOffloadGB should be nil when full; got %d", *got.GPU.CPUOffloadGB)
	}
	if got.GPU.GPUMemoryUtilization == nil || *got.GPU.GPUMemoryUtilization != 0.92 {
		t.Errorf("GPUMemoryUtilization = %v, want *0.92", got.GPU.GPUMemoryUtilization)
	}
	if got.GPU.KVCacheUsagePerc == nil || *got.GPU.KVCacheUsagePerc != 0.3 {
		t.Errorf("KVCacheUsagePerc = %v, want *0.3", got.GPU.KVCacheUsagePerc)
	}
}

// TestVLLMNativeStats_PartialWhenOffloadSet: positive cpu_offload_gb
// in cache_config_info -> partial; cpu_offload_gb is surfaced.
func TestVLLMNativeStats_PartialWhenOffloadSet(t *testing.T) {
	t.Parallel()
	srv := promServer(t, `# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{cpu_offload_gb="4",gpu_memory_utilization="0.9"} 1
`)
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineVLLM, URL: srv.URL},
	}, config.EngineVLLM)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModePartial {
		t.Errorf("Mode = %q, want %q", got.GPU.Mode, adapter.GPUModePartial)
	}
	if got.GPU.CPUOffloadGB == nil || *got.GPU.CPUOffloadGB != 4 {
		t.Errorf("CPUOffloadGB = %v, want *4", got.GPU.CPUOffloadGB)
	}
}

// TestVLLMNativeStats_UnreachableErrors: /metrics is vLLM's primary
// GPU signal, so a refused connection must propagate as an error (the
// diag handler renders it 503), not a silently-degraded answer.
func TestVLLMNativeStats_UnreachableErrors(t *testing.T) {
	t.Parallel()
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineVLLM, URL: "http://127.0.0.1:1"},
	}, config.EngineVLLM)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := a.EngineNativeStats(ctx); err == nil {
		t.Error("expected error when /metrics is unreachable")
	}
}

// TestLlamacppNativeStats_FullOnContract: with ENGINE_ARGS requesting
// full GPU placement (n_gpu_layers=all) the diag reports full.
func TestLlamacppNativeStats_FullOnContract(t *testing.T) {
	t.Parallel()
	// Serve a reachable /metrics (no kv gauge) so the best-effort
	// fetch succeeds and adds no soft warning; the contract path is
	// what this test asserts.
	srv := promServer(t, "# no metrics of interest\n")
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{
			Kind: config.EngineLlamaCpp,
			URL:  srv.URL,
			Args: config.EngineArgs{Known: map[string]string{"n_gpu_layers": "all"}},
		},
	}, config.EngineLlamaCpp)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want %q", got.GPU.Mode, adapter.GPUModeFull)
	}
	if got.Source != "llamacpp_fit_off_alive" {
		t.Errorf("Source = %q", got.Source)
	}
	if len(got.Warnings) != 0 {
		t.Errorf("unexpected warnings: %v", got.Warnings)
	}
}

// TestLlamacppNativeStats_KVUsageFromMetrics: with the contract met
// and /metrics reachable, kv_cache_usage_ratio is surfaced as
// kv_cache_usage_perc without flipping the residency answer.
func TestLlamacppNativeStats_KVUsageFromMetrics(t *testing.T) {
	t.Parallel()
	srv := promServer(t, "# TYPE llamacpp:kv_cache_usage_ratio gauge\nllamacpp:kv_cache_usage_ratio 0.55\n")
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{
			Kind: config.EngineLlamaCpp,
			URL:  srv.URL,
			Args: config.EngineArgs{Known: map[string]string{"n_gpu_layers": "all"}},
		},
	}, config.EngineLlamaCpp)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want full", got.GPU.Mode)
	}
	if got.Source != "llamacpp_fit_off_alive" {
		t.Errorf("Source = %q", got.Source)
	}
	if got.GPU.KVCacheUsagePerc == nil || *got.GPU.KVCacheUsagePerc != 0.55 {
		t.Errorf("KVCacheUsagePerc = %v, want *0.55", got.GPU.KVCacheUsagePerc)
	}
	if len(got.Warnings) != 0 {
		t.Errorf("unexpected warnings: %v", got.Warnings)
	}
}

// TestLlamacppNativeStats_FullWithPositiveNGL: -ngl 32 also counts
// as full (the operator explicitly asked for 32 GPU layers).
func TestLlamacppNativeStats_FullWithPositiveNGL(t *testing.T) {
	t.Parallel()
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{
			Kind: config.EngineLlamaCpp,
			Args: config.EngineArgs{Known: map[string]string{"n_gpu_layers": "32"}},
		},
	}, config.EngineLlamaCpp)

	got, _ := a.EngineNativeStats(context.Background())
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want %q (-ngl 32 + --fit off should be full)",
			got.GPU.Mode, adapter.GPUModeFull)
	}
}

// TestLlamacppNativeStats_UnknownOnContractViolation covers the
// n_gpu_layers violation paths: a value that can't be parsed and a
// zero placement. Both flip Mode to unknown and emit a warning.
func TestLlamacppNativeStats_UnknownOnContractViolation(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		ngl     string
		wantMsg string
	}{
		{"ngl_nonsense", "half", "n_gpu_layers"},
		{"ngl_zero", "0", "n_gpu_layers"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			a := newDiagAdapter(t, config.Config{
				Engine: config.Engine{
					Kind: config.EngineLlamaCpp,
					Args: config.EngineArgs{Known: map[string]string{"n_gpu_layers": tc.ngl}},
				},
			}, config.EngineLlamaCpp)
			got, err := a.EngineNativeStats(context.Background())
			if err != nil {
				t.Fatalf("EngineNativeStats: %v", err)
			}
			if got.GPU.Mode != adapter.GPUModeUnknown {
				t.Errorf("Mode = %q, want %q (contract violated)",
					got.GPU.Mode, adapter.GPUModeUnknown)
			}
			if got.Source != "unavailable" {
				t.Errorf("Source = %q, want unavailable", got.Source)
			}
			if len(got.Warnings) == 0 {
				t.Errorf("expected warnings; got none")
			}
			joined := strings.Join(got.Warnings, "|")
			if !strings.Contains(joined, tc.wantMsg) {
				t.Errorf("warnings %q lack expected mention of %q", joined, tc.wantMsg)
			}
		})
	}
}

// TestSGLangNativeStats_FullViaServerInfo verifies the canonical
// /server_info path: the response body is forwarded verbatim into
// Payload, cpu_offload_gb=0 yields full.
func TestSGLangNativeStats_FullViaServerInfo(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/server_info":
			_, _ = w.Write([]byte(`{
				"model_path": "/models/q",
				"dtype": "bfloat16",
				"cpu_offload_gb": 0,
				"mem_fraction_static": 0.9
			}`))
		case "/metrics":
			_, _ = w.Write([]byte("# TYPE sglang:kv_cache_usage_perc gauge\nsglang:kv_cache_usage_perc 0.42\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineSGLang, URL: srv.URL},
	}, config.EngineSGLang)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	got, err := a.EngineNativeStats(ctx)
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want %q", got.GPU.Mode, adapter.GPUModeFull)
	}
	if got.Source != "sglang_server_info" {
		t.Errorf("Source = %q", got.Source)
	}
	// mem_fraction_static maps to GPUMemoryUtilization; /metrics kv usage layered on top.
	if got.GPU.GPUMemoryUtilization == nil || *got.GPU.GPUMemoryUtilization != 0.9 {
		t.Errorf("GPUMemoryUtilization = %v, want *0.9", got.GPU.GPUMemoryUtilization)
	}
	if got.GPU.KVCacheUsagePerc == nil || *got.GPU.KVCacheUsagePerc != 0.42 {
		t.Errorf("KVCacheUsagePerc = %v, want *0.42", got.GPU.KVCacheUsagePerc)
	}
	// Original /server_info payload preserved (dtype, mem_fraction_static).
	if !strings.Contains(string(got.Payload), `"dtype": "bfloat16"`) {
		t.Errorf("Payload lost dtype field: %s", got.Payload)
	}
}

// TestSGLangNativeStats_MetricsMissDegradesSoft: /server_info answers
// but /metrics is absent (operator forgot --enable-metrics). kv usage
// is omitted with a warning; the residency answer still stands.
func TestSGLangNativeStats_MetricsMissDegradesSoft(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/server_info" {
			_, _ = w.Write([]byte(`{"cpu_offload_gb": 0}`))
			return
		}
		http.NotFound(w, r) // /metrics -> 404
	}))
	defer srv.Close()

	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineSGLang, URL: srv.URL},
	}, config.EngineSGLang)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want full (server_info still authoritative)", got.GPU.Mode)
	}
	if got.GPU.KVCacheUsagePerc != nil {
		t.Errorf("KVCacheUsagePerc should be nil when /metrics missing")
	}
	if !strings.Contains(strings.Join(got.Warnings, "|"), "/metrics unavailable") {
		t.Errorf("expected /metrics-unavailable warning; got %v", got.Warnings)
	}
}

// TestSGLangNativeStats_FallsBackToGetServerInfo: 404 on
// /server_info -> retry on /get_server_info with the same body.
func TestSGLangNativeStats_FallsBackToGetServerInfo(t *testing.T) {
	t.Parallel()
	var serverInfoHits, getServerInfoHits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/server_info":
			serverInfoHits++
			http.NotFound(w, r)
		case "/get_server_info":
			getServerInfoHits++
			_, _ = w.Write([]byte(`{"cpu_offload_gb": 8}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineSGLang, URL: srv.URL},
	}, config.EngineSGLang)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	got, err := a.EngineNativeStats(ctx)
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if serverInfoHits == 0 || getServerInfoHits == 0 {
		t.Errorf("expected both endpoints hit; server_info=%d get_server_info=%d",
			serverInfoHits, getServerInfoHits)
	}
	if got.Source != "sglang_get_server_info" {
		t.Errorf("Source = %q, want fallback tag", got.Source)
	}
	if got.GPU.Mode != adapter.GPUModePartial {
		t.Errorf("Mode = %q, want partial (cpu_offload_gb=8)", got.GPU.Mode)
	}
	if got.GPU.CPUOffloadGB == nil || *got.GPU.CPUOffloadGB != 8 {
		t.Errorf("CPUOffloadGB = %v, want *8", got.GPU.CPUOffloadGB)
	}
}

// TestSGLangNativeStats_CtxCancelledPropagates: when the upstream
// is unreachable, EngineNativeStats returns an error (not a
// silently-degraded "unknown") so callers can render a 503 envelope.
func TestSGLangNativeStats_CtxCancelledPropagates(t *testing.T) {
	t.Parallel()
	a := newDiagAdapter(t, config.Config{
		// Bind to a port that refuses connects so the GET fails.
		Engine: config.Engine{Kind: config.EngineSGLang, URL: "http://127.0.0.1:1"},
	}, config.EngineSGLang)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := a.EngineNativeStats(ctx); err == nil {
		t.Error("expected error when upstream is unreachable")
	}
}

// TestSGLangNativeStats_MissingFieldUnknown: unusual SGLang build
// that doesn't expose cpu_offload_gb -> unknown + warning.
func TestSGLangNativeStats_MissingFieldUnknown(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model_path": "/x"}`)) // no cpu_offload_gb
	}))
	defer srv.Close()

	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineSGLang, URL: srv.URL},
	}, config.EngineSGLang)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeUnknown {
		t.Errorf("Mode = %q, want %q", got.GPU.Mode, adapter.GPUModeUnknown)
	}
	if len(got.Warnings) == 0 {
		t.Errorf("expected warning when cpu_offload_gb missing")
	}
}

// TestProxyMetricsStats_FromGenericGauges: gpu_* gauges give full + VRAM.
func TestProxyMetricsStats_FromGenericGauges(t *testing.T) {
	t.Parallel()
	srv := promServer(t, `# TYPE gpu_present gauge
gpu_present 1
# TYPE gpu_mem_used_bytes gauge
gpu_mem_used_bytes 4184000000
# TYPE gpu_mem_total_bytes gauge
gpu_mem_total_bytes 12884901888
# TYPE gpu_util_ratio gauge
gpu_util_ratio 0.0
`)
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: srv.URL},
	}, config.EngineAudio)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.Source != "engine_metrics" {
		t.Errorf("Source = %q, want engine_metrics", got.Source)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want full", got.GPU.Mode)
	}
	if got.GPU.VRAMBytes == nil || *got.GPU.VRAMBytes != 4184000000 {
		t.Errorf("VRAMBytes = %v, want *4184000000", got.GPU.VRAMBytes)
	}
	if got.GPU.GPUMemoryUtilization == nil {
		t.Fatalf("GPUMemoryUtilization is nil; want used/total")
	}
	if r := *got.GPU.GPUMemoryUtilization; r < 0.32 || r > 0.33 {
		t.Errorf("GPUMemoryUtilization = %v, want ~0.3247", r)
	}
}

// TestProxyMetricsStats_CPUOnlyWhenNoDevice: gpu_present=0 means cpu_only.
func TestProxyMetricsStats_CPUOnlyWhenNoDevice(t *testing.T) {
	t.Parallel()
	srv := promServer(t, `# TYPE gpu_present gauge
gpu_present 0
# TYPE gpu_mem_used_bytes gauge
gpu_mem_used_bytes 0
# TYPE gpu_mem_total_bytes gauge
gpu_mem_total_bytes 0
`)
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: srv.URL},
	}, config.EngineAudio)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeCPUOnly {
		t.Errorf("Mode = %q, want cpu_only", got.GPU.Mode)
	}
	if got.GPU.VRAMBytes != nil {
		t.Errorf("VRAMBytes should be nil on cpu_only; got %d", *got.GPU.VRAMBytes)
	}
}

// TestProxyMetricsStats_FallsBackToVLLM: no gpu_*, so vLLM metrics are used.
func TestProxyMetricsStats_FallsBackToVLLM(t *testing.T) {
	t.Parallel()
	srv := promServer(t, `# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{cpu_offload_gb="0",gpu_memory_utilization="0.45"} 1
`)
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: srv.URL},
	}, config.EngineAudio)

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.Source != "vllm_metrics" {
		t.Errorf("Source = %q, want vllm_metrics (fallback)", got.Source)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want full", got.GPU.Mode)
	}
	if got.GPU.GPUMemoryUtilization == nil || *got.GPU.GPUMemoryUtilization != 0.45 {
		t.Errorf("GPUMemoryUtilization = %v, want *0.45", got.GPU.GPUMemoryUtilization)
	}
}

// TestProxyMetricsStats_MetricsUnreachableDegradesSoft: unknown + warning.
func TestProxyMetricsStats_MetricsUnreachableDegradesSoft(t *testing.T) {
	t.Parallel()
	a := newDiagAdapter(t, config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: "http://127.0.0.1:1"},
	}, config.EngineAudio)

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	got, err := a.EngineNativeStats(ctx)
	if err != nil {
		t.Fatalf("expected nil error (soft degrade); got %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeUnknown {
		t.Errorf("Mode = %q, want unknown", got.GPU.Mode)
	}
	if got.Source != "unavailable" {
		t.Errorf("Source = %q, want unavailable", got.Source)
	}
	if len(got.Warnings) == 0 {
		t.Errorf("expected a warning when /metrics unreachable")
	}
}
