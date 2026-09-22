package nullengine

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
)

func TestAdapter_NoOpSurface(t *testing.T) {
	a := New()

	if a.Kind() != "" {
		t.Errorf("Kind = %q, want empty", a.Kind())
	}
	if a.AliveBeforeBoot() {
		t.Error("AliveBeforeBoot should be false")
	}
	if err := a.WaitAlive(context.Background()); err != nil {
		t.Errorf("WaitAlive = %v, want nil", err)
	}
	// There is no engine to make files visible to, so this adapter
	// must not claim to install anything -- a caller that resolved an
	// installer off it would be handed one that does nothing.
	if _, ok := any(a).(adapter.ModelInstaller); ok {
		t.Error("nullengine implements ModelInstaller; it has no engine to install into")
	}

	rs, err := a.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready: %v", err)
	}
	if !rs.Alive || !rs.ModelExists {
		t.Errorf("Ready = %+v, want Alive && ModelExists", rs)
	}

	stats, err := a.EngineNativeStats(context.Background())
	if err != nil {
		t.Fatalf("EngineNativeStats: %v", err)
	}
	if stats.GPU.Mode != "unknown" {
		t.Errorf("GPU.Mode = %q, want unknown", stats.GPU.Mode)
	}
}

func TestAdapter_HandlersReturn501(t *testing.T) {
	a := New()
	for name, h := range map[string]http.Handler{
		"openai":    a.OpenAIHandler(config.Config{}),
		"anthropic": a.AnthropicHandler(config.Config{}),
	} {
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
		if rec.Code != http.StatusNotImplemented {
			t.Errorf("%s handler status = %d, want 501", name, rec.Code)
		}
	}
}
