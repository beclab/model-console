package ollama

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

func TestResponses_RewritesModelToUpstream(t *testing.T) {
	t.Parallel()
	var gotModel string
	fake := &fakeOllama{
		responses: func(w http.ResponseWriter, r *http.Request) {
			body, _ := io.ReadAll(r.Body)
			var obj map[string]any
			_ = json.Unmarshal(body, &obj)
			gotModel, _ = obj["model"].(string)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"r1","object":"response","status":"completed","output":[]}`))
		},
	}
	a, _ := newTestAdapter(t, fake, config.Config{
		Model: config.Model{Name: "client-alias"},
		Sources: []config.ModelSource{{
			Index:     1,
			Kind:      config.KindOllama,
			Role:      config.RoleMain,
			OllamaTag: "upstream-tag",
		}},
	})
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"client-alias","input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if gotModel != "upstream-tag" {
		t.Errorf("upstream model=%q want upstream-tag", gotModel)
	}
}

func TestResponses_AdmissionTooLarge(t *testing.T) {
	t.Parallel()
	a, _ := newTestAdapter(t, &fakeOllama{}, config.Config{Model: config.Model{Name: "m"}})
	req := httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"m","input":"hi"}`))
	req.ContentLength = maxResponsesBody + 1
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status=%d want 413", rec.Code)
	}
}
