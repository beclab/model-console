package ollama

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
)

// psServer stages a small /api/ps + /api/tags fixture for diag tests.
// We always answer /api/tags with a plausible list because the real
// adapter doesn't call it from EngineNativeStats — but if a future
// refactor does, the fixture won't be the bottleneck.
type psServer struct {
	srv  *httptest.Server
	body string // raw /api/ps body
	code int    // /api/ps status code (default 200)
}

func newPSServer(t *testing.T, body string, code int) *psServer {
	t.Helper()
	if code == 0 {
		code = http.StatusOK
	}
	ps := &psServer{body: body, code: code}
	ps.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/ps":
			w.WriteHeader(ps.code)
			_, _ = w.Write([]byte(ps.body))
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(ps.srv.Close)
	return ps
}

func newDiagAdapterWith(t *testing.T, ps *psServer, modelName string) *Adapter {
	t.Helper()
	return NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: ps.srv.URL},
		Model:  config.Model{Name: modelName},
		Sources: []config.ModelSource{{
			Index:     1,
			Kind:      config.KindOllama,
			Role:      config.RoleMain,
			OllamaTag: modelName,
		}},
	})
}

// TestOllamaNativeStats_FullWhenSizeVRAMEqualsSize: the green path.
func TestOllamaNativeStats_FullWhenSizeVRAMEqualsSize(t *testing.T) {
	t.Parallel()
	ps := newPSServer(t, `{"models":[{
		"name": "qwen2.5:7b",
		"model": "qwen2.5:7b",
		"size": 5368709120,
		"size_vram": 5368709120
	}]}`, 0)
	a := newDiagAdapterWith(t, ps, "qwen2.5:7b")

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if got.Source != "ollama_ps" {
		t.Errorf("Source = %q, want ollama_ps", got.Source)
	}
	if got.GPU.Mode != adapter.GPUModeFull {
		t.Errorf("Mode = %q, want %q", got.GPU.Mode, adapter.GPUModeFull)
	}
	if got.GPU.VRAMBytes == nil || got.GPU.ModelBytes == nil {
		t.Errorf("VRAMBytes/ModelBytes should be populated; got %+v", got.GPU)
	}
	if len(got.Warnings) != 0 {
		t.Errorf("unexpected warnings: %v", got.Warnings)
	}
}

// TestOllamaNativeStats_PartialWhenVRAMSmaller: the documented split
// case (size_vram < size).
func TestOllamaNativeStats_PartialWhenVRAMSmaller(t *testing.T) {
	t.Parallel()
	ps := newPSServer(t, `{"models":[{
		"name": "qwen2.5:7b",
		"model": "qwen2.5:7b",
		"size": 5000000000,
		"size_vram": 3000000000
	}]}`, 0)
	a := newDiagAdapterWith(t, ps, "qwen2.5:7b")

	got, _ := a.EngineNativeStats(context.Background())
	if got.GPU.Mode != adapter.GPUModePartial {
		t.Errorf("Mode = %q, want partial", got.GPU.Mode)
	}
}

// TestOllamaNativeStats_CPUOnlyWhenVRAMZero.
func TestOllamaNativeStats_CPUOnlyWhenVRAMZero(t *testing.T) {
	t.Parallel()
	ps := newPSServer(t, `{"models":[{
		"name": "qwen2.5:7b",
		"model": "qwen2.5:7b",
		"size": 5000000000,
		"size_vram": 0
	}]}`, 0)
	a := newDiagAdapterWith(t, ps, "qwen2.5:7b")

	got, _ := a.EngineNativeStats(context.Background())
	if got.GPU.Mode != adapter.GPUModeCPUOnly {
		t.Errorf("Mode = %q, want cpu_only", got.GPU.Mode)
	}
}

// TestOllamaNativeStats_ModelNotResident: /api/ps returned an empty
// (or differently-named) list, which the daemon does whenever the
// model has been freed by KEEP_ALIVE expiry. We surface cpu_only +
// a warning so dashboards know to send a warm-up call.
func TestOllamaNativeStats_ModelNotResident(t *testing.T) {
	t.Parallel()
	ps := newPSServer(t, `{"models":[]}`, 0)
	a := newDiagAdapterWith(t, ps, "qwen2.5:7b")

	got, _ := a.EngineNativeStats(context.Background())
	if got.GPU.Mode != adapter.GPUModeCPUOnly {
		t.Errorf("Mode = %q, want cpu_only", got.GPU.Mode)
	}
	if len(got.Warnings) == 0 {
		t.Errorf("expected warning about not-resident model")
	}
	if !strings.Contains(strings.Join(got.Warnings, "|"), "not currently resident") {
		t.Errorf("warning missing 'not currently resident': %v", got.Warnings)
	}
}

// TestOllamaNativeStats_OldEngine404: Ollama <0.1.31 returns 404 on
// /api/ps; we degrade to unknown + warning rather than failing.
func TestOllamaNativeStats_OldEngine404(t *testing.T) {
	t.Parallel()
	ps := newPSServer(t, `not found`, http.StatusNotFound)
	a := newDiagAdapterWith(t, ps, "qwen2.5:7b")

	got, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("404 should not propagate as error; got: %v", err)
	}
	if got.GPU.Mode != adapter.GPUModeUnknown {
		t.Errorf("Mode = %q, want unknown", got.GPU.Mode)
	}
	if got.Source != "unavailable" {
		t.Errorf("Source = %q, want unavailable", got.Source)
	}
	if len(got.Warnings) == 0 {
		t.Errorf("expected version-warning")
	}
}

// TestOllamaNativeStats_TransportErrorPropagates: connection refused
// is a real engine-down error, NOT a "give us unknown" path.
func TestOllamaNativeStats_TransportErrorPropagates(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: "http://127.0.0.1:1"},
		Model:  config.Model{Name: "m"},
		Sources: []config.ModelSource{{
			Index:     1,
			Kind:      config.KindOllama,
			Role:      config.RoleMain,
			OllamaTag: "m",
		}},
	})
	if _, err := a.EngineNativeStats(context.Background()); err == nil {
		t.Error("expected transport error")
	}
}
