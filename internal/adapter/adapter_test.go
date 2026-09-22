package adapter

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

// fakeAdapter is a no-op stand-in used to prove the interface can be
// implemented from outside the package without circular deps. Real
// adapters live in internal/adapter/{ollama,proxy} and have their own
// table-driven tests; this fake exists only to exercise the interface
// shape and the helpers in this package.
type fakeAdapter struct {
	kind config.EngineKind
}

func (f *fakeAdapter) Kind() config.EngineKind         { return f.kind }
func (f *fakeAdapter) WaitAlive(context.Context) error { return nil }
func (f *fakeAdapter) AliveBeforeBoot() bool           { return true }
func (f *fakeAdapter) Ready(context.Context) (ReadyState, error) {
	return ReadyState{Alive: true, ModelExists: true}, nil
}
func (f *fakeAdapter) OpenAIHandler(config.Config) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}
func (f *fakeAdapter) AnthropicHandler(config.Config) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}
func (f *fakeAdapter) EngineNativeStats(context.Context) (NativeStats, error) {
	return NativeStats{Source: "test", GPU: GPUResidencyHints{Mode: GPUModeUnknown}}, nil
}

func TestAdapter_InterfaceImplementable(t *testing.T) {
	t.Parallel()
	var a Adapter = &fakeAdapter{kind: config.EngineOllama}
	if a.Kind() != config.EngineOllama {
		t.Errorf("Kind() = %v", a.Kind())
	}
	rs, err := a.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready: %v", err)
	}
	if !rs.Alive || !rs.ModelExists {
		t.Errorf("ReadyState = %+v", rs)
	}
}

func TestAdapter_OpenAIHandlerServes(t *testing.T) {
	t.Parallel()
	a := &fakeAdapter{kind: config.EngineVLLM}
	h := a.OpenAIHandler(config.Config{})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d", rec.Code)
	}
}
