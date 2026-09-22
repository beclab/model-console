package diag

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/obs"
)

// newTestServer builds a Handler + httptest.Server pair with the
// supplied stubAdapter. The mux is the same pattern production uses,
// so route-collision regressions surface in tests.
func newTestServer(t *testing.T, stub adapter.Adapter) (*httptest.Server, *Handler) {
	t.Helper()
	mux := http.NewServeMux()
	h := &Handler{
		Adapter: stub,
		Config: config.Config{
			Engine: config.Engine{Kind: config.EngineVLLM, URL: "http://x:1"},
			Model:  config.Model{Name: "test-model"},
		},
		Metrics: obs.NewMetrics(),
		Now:     fixedNow,
	}
	Mount(mux, h)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, h
}

func TestHandler_DiagGPU_HappyPath(t *testing.T) {
	t.Parallel()
	stub := &stubAdapter{
		kind: config.EngineVLLM,
		stats: adapter.NativeStats{
			Source: "vllm_metrics",
			GPU:    adapter.GPUResidencyHints{Mode: adapter.GPUModeFull},
		},
	}
	srv, _ := newTestServer(t, stub)
	resp, err := http.Get(srv.URL + "/api/diag/gpu")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var got GPUReport
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.SchemaVersion != SchemaVersion {
		t.Errorf("SchemaVersion = %d", got.SchemaVersion)
	}
	if got.GPU.Source != "vllm_metrics" {
		t.Errorf("Source = %q want vllm_metrics", got.GPU.Source)
	}
}

// TestHandler_DiagGPU_EngineNativeOptIn asserts the raw engine_native
// dump is omitted by default and only returned with the explicit
// ?include_engine_native=true opt-in.
func TestHandler_DiagGPU_EngineNativeOptIn(t *testing.T) {
	t.Parallel()
	stub := &stubAdapter{
		kind: config.EngineVLLM,
		stats: adapter.NativeStats{
			Source:  "vllm_metrics",
			Payload: []byte(`{"secret_engine_internals":true}`),
			GPU:     adapter.GPUResidencyHints{Mode: adapter.GPUModeFull},
		},
	}
	srv, _ := newTestServer(t, stub)

	get := func(t *testing.T, url string) GPUReport {
		t.Helper()
		resp, err := http.Get(url)
		if err != nil {
			t.Fatalf("GET: %v", err)
		}
		defer resp.Body.Close()
		var got GPUReport
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return got
	}

	if got := get(t, srv.URL+"/api/diag/gpu"); len(got.EngineNative) != 0 {
		t.Errorf("default response leaked engine_native: %s", got.EngineNative)
	}
	if got := get(t, srv.URL+"/api/diag/gpu?include_engine_native=true"); len(got.EngineNative) == 0 {
		t.Error("opt-in response missing engine_native")
	}
	if got := get(t, srv.URL+"/api/diag/gpu?include_engine_native=0"); len(got.EngineNative) != 0 {
		t.Errorf("falsey opt-in still returned engine_native: %s", got.EngineNative)
	}
}

func TestHandler_DiagGPU_EngineUnreachable_503(t *testing.T) {
	t.Parallel()
	stub := &stubAdapter{
		kind: config.EngineVLLM,
		err:  errors.New("dial tcp 127.0.0.1:1: connection refused"),
	}
	srv, _ := newTestServer(t, stub)
	resp, err := http.Get(srv.URL + "/api/diag/gpu")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"code":"engine_unreachable"`) {
		t.Errorf("body lacks code: %s", body)
	}
}

// TestHandler_Mount_OnlyGPU pins the post-removal diag surface: the
// perf endpoints are gone and Mount must not resurrect them.
func TestHandler_Mount_OnlyGPU(t *testing.T) {
	t.Parallel()
	stub := &stubAdapter{kind: config.EngineVLLM}
	srv, _ := newTestServer(t, stub)

	post, err := http.Post(srv.URL+"/api/diag/perf", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("POST perf: %v", err)
	}
	defer post.Body.Close()
	if post.StatusCode != http.StatusNotFound {
		t.Errorf("POST /api/diag/perf status = %d, want 404", post.StatusCode)
	}

	last, err := http.Get(srv.URL + "/api/diag/perf/last")
	if err != nil {
		t.Fatalf("GET perf/last: %v", err)
	}
	defer last.Body.Close()
	if last.StatusCode != http.StatusNotFound {
		t.Errorf("GET /api/diag/perf/last status = %d, want 404", last.StatusCode)
	}
}
