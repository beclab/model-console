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

func TestEmbed_PrimaryAPIBatch(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			http.NotFound(w, r)
			return
		}
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		inputs := req["input"].([]any)
		if len(inputs) != 2 {
			t.Errorf("len(input)=%d", len(inputs))
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": [][]float64{{0.1, 0.2}, {0.3, 0.4}},
		})
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "embed"},
	})

	body := []byte(`{"input":["hello","world"]}`)
	rec := httptest.NewRecorder()
	a.handleEmbed(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(string(body))))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	data := out["data"].([]any)
	if len(data) != 2 {
		t.Fatalf("data len=%d", len(data))
	}
	first := data[0].(map[string]any)
	if first["index"] != float64(0) {
		t.Errorf("index = %v", first["index"])
	}
	emb := first["embedding"].([]any)
	if len(emb) != 2 || emb[0] != 0.1 {
		t.Errorf("embedding=%v", emb)
	}
}

func TestEmbed_FallbackToLegacy(t *testing.T) {
	t.Parallel()
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/embed":
			http.Error(w, "not found", http.StatusNotFound)
		case "/api/embeddings":
			hits++
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"embedding": []float64{float64(hits), float64(hits) + 0.5},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "e"},
	})
	rec := httptest.NewRecorder()
	a.handleEmbed(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(`{"input":["one","two"]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if hits != 2 {
		t.Errorf("legacy hits=%d (expect one per input)", hits)
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	data := out["data"].([]any)
	if len(data) != 2 {
		t.Fatalf("data len=%d", len(data))
	}
}

func TestEmbed_StringInput(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &req)
		// /api/embed receives an array even for a single string input.
		if got := req["input"]; got != nil {
			if arr, ok := got.([]any); !ok || len(arr) != 1 {
				t.Errorf("input shape = %v", got)
			}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"embeddings": [][]float64{{1, 2, 3}},
		})
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "e"},
	})
	rec := httptest.NewRecorder()
	a.handleEmbed(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings",
		strings.NewReader(`{"input":"hello"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
}

func TestEmbed_BadInputs(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: "http://x"}, Model: config.Model{Name: "e"}})
	cases := []string{`not json`, `{}`, `{"input":""}`, `{"input":null}`}
	for _, c := range cases {
		rec := httptest.NewRecorder()
		a.handleEmbed(rec, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(c)))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("input=%q got status=%d", c, rec.Code)
		}
	}
}

func TestNormalizeEmbedInput(t *testing.T) {
	t.Parallel()
	if got := normalizeEmbedInput("hi"); len(got) != 1 || got[0] != "hi" {
		t.Errorf("string: %v", got)
	}
	if got := normalizeEmbedInput([]any{"a", "b", 7}); len(got) != 3 || got[2] != "7" {
		t.Errorf("mixed: %v", got)
	}
	if got := normalizeEmbedInput(nil); got != nil {
		t.Errorf("nil: %v", got)
	}
	if got := normalizeEmbedInput(""); got != nil {
		t.Errorf("empty string: %v", got)
	}
}
