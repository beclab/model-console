package ollama

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

type completionFake struct {
	frames    []map[string]any
	gotBody   map[string]any
	notStream bool
}

func startCompletionFake(t *testing.T, f *completionFake) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &f.gotBody)
		if f.notStream {
			if len(f.frames) > 0 {
				bb, _ := json.Marshal(f.frames[0])
				_, _ = w.Write(bb)
			}
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		for _, fr := range f.frames {
			bb, _ := json.Marshal(fr)
			_, _ = w.Write(bb)
			_, _ = w.Write([]byte("\n"))
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func TestCompletion_StreamingPromptForwarded(t *testing.T) {
	t.Parallel()
	f := &completionFake{frames: []map[string]any{
		{"response": "Pa", "done": false},
		{"response": "ris", "done": true, "prompt_eval_count": 4, "eval_count": 2},
	}}
	url := startCompletionFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: url},
		Model:  config.Model{Name: "m"},
	})
	body := []byte(`{"prompt":"Capital of France is","stream":true,"max_tokens":50,"stop":["\\n"]}`)
	rec := httptest.NewRecorder()
	a.handleCompletion(rec, httptest.NewRequest(http.MethodPost, "/v1/completions",
		bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"text":"Pa"`) {
		t.Errorf("missing first chunk")
	}
	if !strings.Contains(rec.Body.String(), `[DONE]`) {
		t.Errorf("missing [DONE]")
	}
	if f.gotBody["prompt"] != "Capital of France is" {
		t.Errorf("prompt = %v", f.gotBody["prompt"])
	}
	opts := f.gotBody["options"].(map[string]any)
	if opts["num_predict"] != float64(50) {
		t.Errorf("num_predict=%v", opts["num_predict"])
	}
}

func TestCompletion_NonStreaming(t *testing.T) {
	t.Parallel()
	f := &completionFake{
		notStream: true,
		frames: []map[string]any{{
			"response": "answer", "done": true,
			"prompt_eval_count": 5, "eval_count": 2,
		}},
	}
	url := startCompletionFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: url}, Model: config.Model{Name: "m"},
	})
	rec := httptest.NewRecorder()
	a.handleCompletion(rec, httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"prompt":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	choice := out["choices"].([]any)[0].(map[string]any)
	if choice["text"] != "answer" {
		t.Errorf("text=%v", choice["text"])
	}
}

func TestCompletion_BadJSON(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: "http://x"}, Model: config.Model{Name: "m"}})
	rec := httptest.NewRecorder()
	a.handleCompletion(rec, httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader("garbage")))
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d", rec.Code)
	}
}

func TestLastGenerateFrame(t *testing.T) {
	t.Parallel()
	data := []byte(`{"response":"a","done":false}` + "\n" +
		`{"response":"b","done":true,"eval_count":7}` + "\n")
	got := lastGenerateFrame(data)
	if got.Response != "b" || !got.Done || got.EvalCount != 7 {
		t.Errorf("got=%+v", got)
	}
}
