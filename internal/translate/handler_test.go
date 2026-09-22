package translate

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMount_LanguagesAndTranslate(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	fakeChat := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "Hello"}},
			},
		})
	})
	Mount(mux, Options{
		Completer: Completer{
			Handler:   fakeChat,
			ModelName: "translate-model",
		},
	})

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/languages", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("languages status=%d body=%s", rec.Code, rec.Body.String())
	}
	var langs languagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &langs); err != nil {
		t.Fatal(err)
	}
	if len(langs.Languages) != 54 {
		t.Fatalf("languages len=%d want 54", len(langs.Languages))
	}
	if len(langs.Pairs) != 102 {
		t.Fatalf("pairs len=%d want 102", len(langs.Pairs))
	}

	body := `{"from":"zh-CN","to":"en","text":"你好","html":false}`
	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/translate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("translate status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out translateResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Result != "Hello" {
		t.Fatalf("result=%q", out.Result)
	}

	batchBody := `{"from":"auto","to":"en","texts":["你好","世界"]}`
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/translate/batch", strings.NewReader(batchBody))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("batch status=%d body=%s", rec.Code, rec.Body.String())
	}
	var batch batchResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &batch); err != nil {
		t.Fatal(err)
	}
	if len(batch.Results) != 2 || batch.Results[0] != "Hello" {
		t.Fatalf("batch=%+v", batch)
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/translate/batch",
		strings.NewReader(`{"from":"auto","to":"en","texts":[]}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("empty batch status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestTranslate_WrapRejects(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Wrap: func(http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "not ready", http.StatusServiceUnavailable)
			})
		},
		Completer: Completer{
			Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				t.Fatal("should not reach chat")
			}),
			ModelName: "m",
		},
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/translate",
		bytes.NewReader([]byte(`{"from":"auto","to":"en","text":"hi"}`)))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestTranslate_Validation(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Completer: Completer{
			Handler:   http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
			ModelName: "m",
		},
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/translate",
		strings.NewReader(`{"from":"auto","to":"xx","text":"hi"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestTranslate_NoHTMLEscape(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Completer: Completer{
			Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"choices": []map[string]any{
						{"message": map[string]any{"content": "a <b>tag</b> & more"}},
					},
				})
			}),
			ModelName: "m",
		},
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/translate",
		strings.NewReader(`{"from":"auto","to":"en","text":"x"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if strings.Contains(rec.Body.String(), `\u003c`) {
		t.Fatalf("result HTML-escaped: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `<b>tag</b>`) {
		t.Fatalf("want raw tags in body: %s", rec.Body.String())
	}
}

func TestTranslate_BatchTooLarge(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Completer: Completer{
			Handler:   http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
			ModelName: "m",
		},
	})
	texts := make([]string, maxBatchTexts+1)
	for i := range texts {
		texts[i] = "x"
	}
	body, _ := json.Marshal(map[string]any{"from": "auto", "to": "en", "texts": texts})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/translate/batch", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestLanguages_NoWrap(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Wrap: func(http.Handler) http.Handler {
			return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "blocked", http.StatusServiceUnavailable)
			})
		},
		Completer: Completer{Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}), ModelName: "m"},
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/languages", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("languages must not use Wrap; status=%d", rec.Code)
	}
}
