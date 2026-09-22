package ollama

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

func TestModels_OnlyConfiguredModelExposed(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:7b","modified_at":"2024-01-01T00:00:00Z"},{"name":"other"}]}`))
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "qwen2.5"},
	})
	rec := httptest.NewRecorder()
	a.handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	data, _ := out["data"].([]any)
	if len(data) != 1 {
		t.Fatalf("expect 1 model, got %d", len(data))
	}
	first := data[0].(map[string]any)
	if first["id"] != "qwen2.5" {
		t.Errorf("id=%v", first["id"])
	}
	if first["owned_by"] != "llm-init" {
		t.Errorf("owned_by=%v", first["owned_by"])
	}
	if first["created"] == float64(0) {
		t.Errorf("created should be populated from /api/tags modified_at")
	}
}

func TestModels_DaemonDown(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: "http://127.0.0.1:1"},
		Model:  config.Model{Name: "m"},
	})
	rec := httptest.NewRecorder()
	a.handleModels(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	data := out["data"].([]any)
	if len(data) != 1 {
		t.Fatalf("data len = %d", len(data))
	}
	if data[0].(map[string]any)["id"] != "m" {
		t.Errorf("id=%v", data[0])
	}
}
