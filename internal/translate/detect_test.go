package translate

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDetectPureCJK(t *testing.T) {
	t.Parallel()
	cases := []struct {
		text string
		want string
		ok   bool
	}{
		{"你好世界", langZhHans, true},
		{"こんにちは", "ja", true},
		{"안녕하세요", "ko", true},
		{"Hello 你好", "", false},
		{"", "", false},
	}
	for _, tc := range cases {
		got, ok := detectPureCJK(tc.text)
		if ok != tc.ok || got != tc.want {
			t.Errorf("detectPureCJK(%q) = %q,%v; want %q,%v", tc.text, got, ok, tc.want, tc.ok)
		}
	}
}

func TestDetect_CJKWithoutLLM(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Completer: Completer{
			Handler: http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				t.Fatal("CJK detect should not call chat")
			}),
			ModelName: "m",
		},
	})
	body := `{"text":"今天天气真好"}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/detect", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out detectResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Language != langZhHans {
		t.Fatalf("language=%q", out.Language)
	}
	if out.Confidence != nil {
		t.Fatal("confidence must be omitted without minConfidence")
	}
}

func TestDetect_MinConfidence(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Completer: Completer{
			Handler:   http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}),
			ModelName: "m",
		},
	})
	body := `{"text":"你好","minConfidence":0.5}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/detect", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out detectResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Language != langZhHans || out.Confidence == nil || *out.Confidence != 1.0 {
		t.Fatalf("out=%+v", out)
	}

	// Threshold above heuristic confidence → empty language, confidence kept.
	body = `{"text":"你好","minConfidence":1.1}`
	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/detect", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if out.Language != "" || out.Confidence == nil || *out.Confidence != 1.0 {
		t.Fatalf("below-threshold out=%+v conf=%v", out, out.Confidence)
	}
}

func TestDetect_LLMFallback(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Completer: Completer{
			Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"choices": []map[string]any{
						{"message": map[string]any{"content": "en"}},
					},
				})
			}),
			ModelName: "m",
		},
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/detect",
		bytes.NewReader([]byte(`{"text":"Bonjour tout le monde"}`)))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out detectResponse
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if out.Language != "en" {
		// model stub always returns en; real model would return fr
		t.Fatalf("language=%q want en (stub)", out.Language)
	}
}

func TestParseDetectOutput(t *testing.T) {
	t.Parallel()
	h := &Handler{catalog: BuiltinCatalog()}
	if got := h.parseDetectOutput(langZhHans); got != langZhHans {
		t.Fatalf("got %q", got)
	}
	if got := h.parseDetectOutput("  en\n"); got != "en" {
		t.Fatalf("got %q", got)
	}
	if got := h.parseDetectOutput("Chinese"); got != langZhHans {
		t.Fatalf("got %q", got)
	}
}
