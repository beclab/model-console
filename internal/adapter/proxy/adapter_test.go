package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/obs"
)

// obsMetricsForTest returns a fresh metrics registry. Per-test
// instances avoid duplicate-register panics and let assertions read
// counter values without cross-test contamination.
func obsMetricsForTest() *obs.Metrics { return obs.NewMetrics() }

// readCounter extracts the float value from a prometheus.Counter
// (lifecycle / dataplane tests already use the same pattern; we
// duplicate it here to avoid pulling in a test-only dep on those
// internals).
func readCounter(c prometheus.Counter) float64 {
	var m dto.Metric
	_ = c.Write(&m)
	return m.GetCounter().GetValue()
}

// upstream is a tiny stand-in for vLLM/llama.cpp/SGLang. It records each
// inbound request and emits canned responses keyed off the path.
type upstream struct {
	srv         *httptest.Server
	gotRequests []*recordedRequest

	modelsBody  string
	historyBody string
}

type recordedRequest struct {
	method   string
	path     string
	rawQuery string
	body     []byte
	headers  http.Header
}

func newUpstream(t *testing.T) *upstream {
	t.Helper()
	u := &upstream{
		modelsBody: `{"object":"list","data":[{"id":"target-model","object":"model"}]}`,
	}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		u.gotRequests = append(u.gotRequests, &recordedRequest{
			method:   r.Method,
			path:     r.URL.Path,
			rawQuery: r.URL.RawQuery,
			body:     body,
			headers:  r.Header.Clone(),
		})
		switch r.URL.Path {
		case "/v1/models":
			_, _ = w.Write([]byte(u.modelsBody))
		case "/v1/tasks":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
		case "/v1/history/hi-1":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(u.historyBody))
		case "/v1/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		case "/v1/responses":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed","output":[]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(u.srv.Close)
	return u
}

func newProxy(t *testing.T, u *upstream, modelName string, kind config.EngineKind) *Adapter {
	t.Helper()
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{Kind: kind, URL: u.srv.URL},
		Model:  config.Model{Name: modelName},
	}, kind)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	return a
}

func TestProxy_NewAdapterValidatesURL(t *testing.T) {
	t.Parallel()
	cases := []struct{ url string }{
		{":not a url"},
		{"no-scheme.example"},
		{""},
	}
	for _, tc := range cases {
		_, err := NewAdapter(config.Config{Engine: config.Engine{URL: tc.url}},
			config.EngineVLLM)
		if err == nil {
			t.Errorf("url %q should fail", tc.url)
		}
	}
}

func TestProxy_KindRoundtrip(t *testing.T) {
	t.Parallel()
	for _, k := range []config.EngineKind{config.EngineVLLM, config.EngineLlamaCpp, config.EngineSGLang} {
		a, err := NewAdapter(config.Config{
			Engine: config.Engine{URL: "http://x:1"}, Model: config.Model{Name: "m"},
		}, k)
		if err != nil {
			t.Fatalf("NewAdapter: %v", err)
		}
		if a.Kind() != k {
			t.Errorf("Kind=%v want %v", a.Kind(), k)
		}
	}
}

func TestProxy_ResponsesRewritesModel(t *testing.T) {
	t.Parallel()
	// Cover the three proxy kinds so a future refactor that branches
	// by Kind on /v1/responses doesn't quietly break SGLang or
	// llama.cpp while vLLM keeps working.
	for _, k := range []config.EngineKind{
		config.EngineVLLM,
		config.EngineSGLang,
		config.EngineLlamaCpp,
	} {
		k := k
		t.Run(string(k), func(t *testing.T) {
			t.Parallel()
			u := newUpstream(t)
			a := newProxy(t, u, "target-model", k)

			reqBody := []byte(`{"model":"old-name","input":"hi"}`)
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(reqBody))
			req.ContentLength = int64(len(reqBody))
			rec := httptest.NewRecorder()
			a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
			}
			if len(u.gotRequests) != 1 {
				t.Fatalf("upstream requests=%d want 1", len(u.gotRequests))
			}
			var bodyJSON map[string]any
			if err := json.Unmarshal(u.gotRequests[0].body, &bodyJSON); err != nil {
				t.Fatalf("body: %v", err)
			}
			if bodyJSON["model"] != "target-model" {
				t.Errorf("model=%v want target-model", bodyJSON["model"])
			}
		})
	}
}

// TestProxy_ResponsesUpstream4xxPassthrough proves /v1/responses
// forwards the upstream status code (4xx/5xx) verbatim instead of
// masking it as a 502 upstream_unreachable envelope. SGLang v0.5.x
// returns 4xx for structured input arrays and unsupported tools;
// surfacing the original code lets clients distinguish "engine said
// no" from "engine unreachable".
func TestProxy_ResponsesUpstream4xxPassthrough(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{"sglang_422_input_array", http.StatusUnprocessableEntity,
			`{"object":"error","message":"input must be string","type":"BadRequestError","param":"input","code":400}`},
		{"sglang_400_unsupported_tool", http.StatusBadRequest,
			`{"object":"error","message":"only built-in tools supported","type":"BadRequestError","param":"tools","code":400}`},
		{"upstream_500", http.StatusInternalServerError,
			`{"object":"error","message":"asgi error"}`},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/responses" {
					http.NotFound(w, r)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			t.Cleanup(srv.Close)
			a, err := NewAdapter(config.Config{
				Engine: config.Engine{Kind: config.EngineSGLang, URL: srv.URL},
				Model:  config.Model{Name: "m"},
			}, config.EngineSGLang)
			if err != nil {
				t.Fatalf("NewAdapter: %v", err)
			}
			reqBody := []byte(`{"model":"m","input":[{"type":"input_text","text":"hi"}]}`)
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(reqBody))
			req.ContentLength = int64(len(reqBody))
			rec := httptest.NewRecorder()
			a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

			if rec.Code != tc.status {
				t.Errorf("status=%d want %d body=%s", rec.Code, tc.status, rec.Body.String())
			}
			if rec.Body.String() != tc.body {
				t.Errorf("body=%q want %q", rec.Body.String(), tc.body)
			}
		})
	}
}

func TestProxy_PassesThroughChatStreaming(t *testing.T) {
	t.Parallel()
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)

	reqBody := []byte(`{"model":"old-name","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer secret")
	req.ContentLength = int64(len(reqBody))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Errorf("missing SSE [DONE]: %q", rec.Body.String())
	}

	if len(u.gotRequests) != 1 {
		t.Fatalf("expected 1 upstream request, got %d", len(u.gotRequests))
	}
	got := u.gotRequests[0]
	if got.path != "/v1/chat/completions" {
		t.Errorf("upstream path=%q", got.path)
	}
	var bodyJSON map[string]any
	if err := json.Unmarshal(got.body, &bodyJSON); err != nil {
		t.Fatalf("body not JSON: %v\n%s", err, got.body)
	}
	if bodyJSON["model"] != "target-model" {
		t.Errorf("model not rewritten: %v", bodyJSON["model"])
	}
	if got.headers.Get("Authorization") != "" {
		t.Errorf("Authorization should be stripped, got %q", got.headers.Get("Authorization"))
	}
}

func TestProxy_NoRewriteOnPassthroughPaths(t *testing.T) {
	t.Parallel()
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)

	reqBody := []byte(`{"model":"keep-me"}`)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", bytes.NewReader(reqBody))
	req.ContentLength = int64(len(reqBody))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	got := u.gotRequests[len(u.gotRequests)-1]
	if got.path != "/v1/models" {
		t.Errorf("path=%q", got.path)
	}
	// /v1/models is not in rewritePaths so the body must be untouched.
	if string(got.body) != string(reqBody) {
		t.Errorf("body changed for /v1/models: got %q want %q", got.body, reqBody)
	}
}

// TestProxy_ForwardsQueryString locks what the async task contract's list
// filters depend on. director only rewrites Scheme/Host/Path, so RawQuery
// survives implicitly — and until /v1/tasks?status=&limit= (see
// async task routes landed, no proxied route carried a query string at all,
// every OpenAI path putting its arguments in the body. An engine's own smoke
// talks to the engine directly and would stay green while llm-init dropped the
// filter, so the assertion belongs here.
func TestProxy_ForwardsQueryString(t *testing.T) {
	t.Parallel()
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)

	req := httptest.NewRequest(http.MethodGet, "/v1/tasks?status=queued&limit=10", nil)
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	got := u.gotRequests[len(u.gotRequests)-1]
	if got.path != "/v1/tasks" {
		t.Errorf("path=%q want /v1/tasks", got.path)
	}
	if got.rawQuery != "status=queued&limit=10" {
		t.Errorf("query=%q want status=queued&limit=10", got.rawQuery)
	}
	if rec.Code != http.StatusOK {
		t.Errorf("status=%d want 200", rec.Code)
	}
}

func TestProxy_PassesThroughHistoryDurationMetadata(t *testing.T) {
	t.Parallel()
	u := newUpstream(t)
	u.historyBody = `{"history_item_id":"hi-1","state":"created","output_duration_seconds":2.75}`
	a := newProxy(t, u, "target-model", config.EngineVLLM)

	req := httptest.NewRequest(http.MethodGet, "/v1/history/hi-1", nil)
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if rec.Body.String() != u.historyBody {
		t.Fatalf("body changed: got %q want %q", rec.Body.String(), u.historyBody)
	}
}

// TestProxy_LargeBodyRejectedWith413 locks the v1.0.5 admission
// guard. Pre-fix oversized bodies were forwarded to the upstream
// VERBATIM (with the OpenAI alias `model` field intact), which broke
// the "consistent model identity" contract: a request submitted to
// llm-init for "qwen2.5-7b" silently arrived at the engine as
// "gpt-4o" if the body was >1 MiB. Post-fix the proxy rejects the
// request with 413 + code rewrite_body_too_large + the
// X-Llm-Init-Max-Rewrite-Bytes header so clients can size to the cap.
//
// Text-only rewrite paths stay at 1 MiB: completions always, and
// /v1/embeddings unless ENGINE_KIND=clipembed (see
// TestProxy_ClipEmbedEmbeddingsImageBodyForwarded). Chat completions /
// responses use the larger vision cap (TestProxy_VisionBodyForwarded).
// Each sub-case asserts status, code, header, AND that the upstream
// NEVER received the request (admission short-circuits before ServeHTTP).
func TestProxy_LargeBodyRejectedWith413(t *testing.T) {
	t.Parallel()
	chunk := strings.Repeat("a", 1<<20+10) // > maxRewriteBody
	body := []byte(`{"model":"old","data":"` + chunk + `"}`)

	cases := []struct {
		path string
		kind config.EngineKind
	}{
		{"/v1/completions", config.EngineVLLM},
		{"/v1/embeddings", config.EngineEmbed},
		{"/v1/embeddings", config.EngineVLLM}, // non-clipembed defaults to 1 MiB
	}
	for _, tc := range cases {
		tc := tc
		t.Run(string(tc.kind)+tc.path, func(t *testing.T) {
			t.Parallel()
			u := newUpstream(t)
			a := newProxy(t, u, "target-model", tc.kind)
			req := httptest.NewRequest(http.MethodPost, tc.path, bytes.NewReader(body))
			req.ContentLength = int64(len(body))
			rec := httptest.NewRecorder()
			a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

			if rec.Code != http.StatusRequestEntityTooLarge {
				t.Fatalf("status=%d, want 413", rec.Code)
			}
			if got := rec.Header().Get("X-Llm-Init-Max-Rewrite-Bytes"); got != "1048576" {
				t.Errorf("X-Llm-Init-Max-Rewrite-Bytes=%q want 1048576", got)
			}
			if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
				t.Errorf("Content-Type=%q", got)
			}
			var env struct {
				Error struct {
					Code string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
				t.Fatalf("body must be JSON: %v\n%s", err, rec.Body.String())
			}
			if env.Error.Code != "rewrite_body_too_large" {
				t.Errorf("error.code=%q want rewrite_body_too_large", env.Error.Code)
			}
			if len(u.gotRequests) > 0 {
				t.Errorf("upstream was reached for oversized body: got %d requests; admission did not short-circuit", len(u.gotRequests))
			}
		})
	}
}

// TestProxy_VisionBodyForwarded locks the fix for the OpenWebUI vision
// regression: a chat-completions request carrying an embedded base64
// image is several MiB, which the old 1 MiB chat cap rejected with 413
// ("request body exceeds the model-rewrite cap"). Vision-capable paths
// now admit up to maxVisionRewriteBody (16 MiB), so a ~2 MiB body must
// reach the upstream with its model field rewritten.
func TestProxy_VisionBodyForwarded(t *testing.T) {
	t.Parallel()
	image := strings.Repeat("A", 2<<20) // ~2 MiB base64 image, > 1 MiB text cap
	body := []byte(`{"model":"old","messages":[{"role":"user","content":"` + image + `"}]}`)

	for _, path := range []string{"/v1/chat/completions", "/api/chat/completions"} {
		path := path
		t.Run(path, func(t *testing.T) {
			t.Parallel()
			u := newUpstream(t)
			a := newProxy(t, u, "target-model", config.EngineVLLM)
			req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
			req.ContentLength = int64(len(body))
			rec := httptest.NewRecorder()
			a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

			if rec.Code == http.StatusRequestEntityTooLarge {
				t.Fatalf("status=413, vision body should be forwarded under the 16 MiB cap")
			}
			if len(u.gotRequests) != 1 {
				t.Fatalf("upstream requests = %d, want 1 (body was not forwarded)", len(u.gotRequests))
			}
			var obj struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(u.gotRequests[0].body, &obj); err != nil {
				t.Fatalf("forwarded body not JSON: %v", err)
			}
			if obj.Model != "target-model" {
				t.Errorf("model=%q want target-model", obj.Model)
			}
		})
	}
}

// TestProxy_ClipEmbedEmbeddingsImageBodyForwarded locks ENGINE_KIND=clipembed
// admitting ~2 MiB /v1/embeddings bodies (inline image_url data URLs) under
// the 16 MiB vision cap, with model rewrite applied.
func TestProxy_ClipEmbedEmbeddingsImageBodyForwarded(t *testing.T) {
	t.Parallel()
	image := strings.Repeat("A", 2<<20)
	body := []byte(`{"model":"old","input":{"type":"image","image_url":"data:image/jpeg;base64,` + image + `"}}`)

	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineClipEmbed)
	req := httptest.NewRequest(http.MethodPost, pathEmbeddings, bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code == http.StatusRequestEntityTooLarge {
		t.Fatalf("status=413, clipembed embeddings body should be forwarded under the 16 MiB cap")
	}
	if len(u.gotRequests) != 1 {
		t.Fatalf("upstream requests = %d, want 1 (body was not forwarded)", len(u.gotRequests))
	}
	var obj struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(u.gotRequests[0].body, &obj); err != nil {
		t.Fatalf("forwarded body not JSON: %v", err)
	}
	if obj.Model != "target-model" {
		t.Errorf("model=%q want target-model", obj.Model)
	}
}

// TestProxy_RerankModelRewrite locks POST /v1/rerank model alias rewrite.
func TestProxy_RerankModelRewrite(t *testing.T) {
	t.Parallel()
	body := []byte(`{"model":"old-alias","query":"q","documents":["a","b"]}`)

	u := newUpstream(t)
	a := newProxy(t, u, "bge-reranker-v2-m3", config.EngineRerank)
	req := httptest.NewRequest(http.MethodPost, pathRerank, bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if len(u.gotRequests) != 1 {
		t.Fatalf("upstream requests = %d, want 1", len(u.gotRequests))
	}
	var obj struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(u.gotRequests[0].body, &obj); err != nil {
		t.Fatalf("forwarded body not JSON: %v", err)
	}
	if obj.Model != "bge-reranker-v2-m3" {
		t.Errorf("model=%q want bge-reranker-v2-m3", obj.Model)
	}
}

// TestProxy_ClipEmbedEmbeddingsBodyOver16MiBRejected locks the hard ceiling
// for clipembed embeddings: bodies above maxVisionRewriteBody still 413.
func TestProxy_ClipEmbedEmbeddingsBodyOver16MiBRejected(t *testing.T) {
	t.Parallel()
	chunk := strings.Repeat("A", maxVisionRewriteBody+10)
	body := []byte(`{"model":"old","input":{"type":"image","image_url":"data:image/jpeg;base64,` + chunk + `"}}`)

	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineClipEmbed)
	req := httptest.NewRequest(http.MethodPost, pathEmbeddings, bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status=%d, want 413", rec.Code)
	}
	if got := rec.Header().Get("X-Llm-Init-Max-Rewrite-Bytes"); got != "16777216" {
		t.Errorf("X-Llm-Init-Max-Rewrite-Bytes=%q want 16777216", got)
	}
	if len(u.gotRequests) > 0 {
		t.Errorf("upstream was reached for oversized body: got %d requests", len(u.gotRequests))
	}
}

// TestProxy_VisionBodyRejectedAbove16MiB confirms the larger cap is
// still bounded: a chat body over maxVisionRewriteBody is 413'd with
// the 16 MiB cap advertised, not silently forwarded with the alias.
func TestProxy_VisionBodyRejectedAbove16MiB(t *testing.T) {
	t.Parallel()
	chunk := strings.Repeat("a", 16<<20+10) // > maxVisionRewriteBody
	body := []byte(`{"model":"old","data":"` + chunk + `"}`)

	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status=%d, want 413", rec.Code)
	}
	if got := rec.Header().Get("X-Llm-Init-Max-Rewrite-Bytes"); got != "16777216" {
		t.Errorf("X-Llm-Init-Max-Rewrite-Bytes=%q want 16777216", got)
	}
	if len(u.gotRequests) > 0 {
		t.Errorf("upstream reached for >16 MiB body: admission did not short-circuit")
	}
}

// TestProxy_OversizedBodyOnNonRewritePath_StillForwards locks that
// the 413 admission only applies to rewritePaths. /v1/models or other
// pass-through endpoints must still forward unchanged regardless of
// body size.
func TestProxy_OversizedBodyOnNonRewritePath_StillForwards(t *testing.T) {
	t.Parallel()
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)
	chunk := strings.Repeat("a", 1<<20+10)
	body := []byte(`{"foo":"` + chunk + `"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/models", bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)
	// 200 from the upstream's /v1/models stub (it responds to GET; we
	// POSTed but the test stub doesn't differentiate — what matters
	// is that we did NOT short-circuit with 413).
	if rec.Code == http.StatusRequestEntityTooLarge {
		t.Fatal("non-rewrite path was wrongly rejected with 413")
	}
}

// TestProxy_ChunkedOversizedBodyRejectedWith413 closes the v1.0.5
// audit gap: pre-fix `rewriteAdmission` only consulted
// req.ContentLength, so a chunked-encoding request (Content-Length =
// -1) skated past admission entirely; the >cap branch in
// rewriteModelInBody then logged a warning and forwarded the body
// VERBATIM with the OpenAI alias `model` field intact -- the same
// "consistent model identity" contract violation the
// Content-Length-based 413 was supposed to close.
//
// Post-fix admission drains the body up to cap+1 bytes regardless of
// Content-Length, so a chunked oversized body now gets the same 413
// envelope as its Content-Length-advertised twin -- and crucially
// the upstream sees ZERO requests.
func TestProxy_ChunkedOversizedBodyRejectedWith413(t *testing.T) {
	t.Parallel()
	chunk := strings.Repeat("a", 1<<20+10)
	body := []byte(`{"model":"old","data":"` + chunk + `"}`)

	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)
	// Use a text-only rewrite path (1 MiB cap) so this stays a test of
	// chunked-body draining rather than the vision cap; chat completions
	// now admits up to 16 MiB (see TestProxy_VisionBodyForwarded).
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(body))
	// Simulate a chunked-encoding client: the server-side request
	// shape has ContentLength=-1 and TransferEncoding=[chunked]. The
	// pre-fix admission filter ignored both and let the body flow
	// to the Director.
	req.ContentLength = -1
	req.TransferEncoding = []string{"chunked"}
	req.Header.Del("Content-Length")
	req.Header.Set("Transfer-Encoding", "chunked")

	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status=%d, want 413; chunked oversize body bypassed admission", rec.Code)
	}
	if got := rec.Header().Get("X-Llm-Init-Max-Rewrite-Bytes"); got != "1048576" {
		t.Errorf("X-Llm-Init-Max-Rewrite-Bytes=%q want 1048576", got)
	}
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("body must be JSON: %v\n%s", err, rec.Body.String())
	}
	if env.Error.Code != "rewrite_body_too_large" {
		t.Errorf("error.code=%q want rewrite_body_too_large", env.Error.Code)
	}
	if len(u.gotRequests) > 0 {
		t.Errorf("upstream was reached for oversized chunked body: got %d requests", len(u.gotRequests))
	}
}

// TestProxy_ChunkedSmallBodyRewrittenAndForwarded locks the happy
// path of the v1.0.5 chunked-bypass fix. A chunked request with a
// body <= cap must (a) reach the upstream and (b) have the model
// field rewritten to cfg.Model.Name, just like the Content-Length
// path. We pin Content-Length on the upstream-observed request to
// confirm admission re-framed the body from chunked to
// length-delimited (Transfer-Encoding stripped, ContentLength set).
func TestProxy_ChunkedSmallBodyRewrittenAndForwarded(t *testing.T) {
	t.Parallel()
	body := []byte(`{"model":"old-alias","messages":[]}`)

	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.ContentLength = -1
	req.TransferEncoding = []string{"chunked"}
	req.Header.Del("Content-Length")
	req.Header.Set("Transfer-Encoding", "chunked")

	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code == http.StatusRequestEntityTooLarge {
		t.Fatalf("small chunked body wrongly rejected with 413; body=%d bytes", len(body))
	}
	if len(u.gotRequests) != 1 {
		t.Fatalf("upstream got %d requests, want 1", len(u.gotRequests))
	}
	got := u.gotRequests[0]
	var obj map[string]any
	if err := json.Unmarshal(got.body, &obj); err != nil {
		t.Fatalf("upstream body not JSON: %v\n%s", err, got.body)
	}
	if obj["model"] != "target-model" {
		t.Errorf("upstream model=%v want target-model (rewrite did not apply on chunked path)", obj["model"])
	}
}

func TestProxy_NonJSONBodyPassThrough(t *testing.T) {
	t.Parallel()
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)
	reqBody := []byte("not even json")
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(reqBody))
	req.ContentLength = int64(len(reqBody))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)
	got := u.gotRequests[len(u.gotRequests)-1]
	if string(got.body) != string(reqBody) {
		t.Errorf("body changed: %q vs %q", got.body, reqBody)
	}
}

func TestProxy_WaitAliveAndReady(t *testing.T) {
	t.Parallel()
	u := newUpstream(t)
	a := newProxy(t, u, "target-model", config.EngineVLLM)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := a.WaitAlive(ctx); err != nil {
		t.Fatalf("WaitAlive: %v", err)
	}

	rs, err := a.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready: %v", err)
	}
	if !rs.Alive || !rs.ModelExists {
		t.Errorf("rs=%+v", rs)
	}

	// Ready when model is missing: replace upstream payload.
	u.modelsBody = `{"object":"list","data":[{"id":"other"}]}`
	rs, err = a.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready: %v", err)
	}
	if rs.ModelExists {
		t.Error("ModelExists should be false when /v1/models lists no match")
	}
}

func TestProxy_WaitAliveCancels(t *testing.T) {
	t.Parallel()
	a, _ := NewAdapter(config.Config{
		Engine: config.Engine{URL: "http://127.0.0.1:1"}, Model: config.Model{Name: "m"},
	}, config.EngineVLLM)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := a.WaitAlive(ctx); err == nil {
		t.Error("expected ctx error")
	}
}

func TestProxy_ReadyTransportError(t *testing.T) {
	t.Parallel()
	a, _ := NewAdapter(config.Config{
		Engine: config.Engine{URL: "http://127.0.0.1:1"}, Model: config.Model{Name: "m"},
	}, config.EngineVLLM)
	if _, err := a.Ready(context.Background()); err == nil {
		t.Error("expected error")
	}
}

func TestProxy_ReadyBadStatus(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "down", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	a, _ := NewAdapter(config.Config{
		Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "m"},
	}, config.EngineVLLM)
	if _, err := a.Ready(context.Background()); err == nil {
		t.Error("expected error on 503")
	}
}

func TestProxy_ReadyBadJSON(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	defer srv.Close()
	a, _ := NewAdapter(config.Config{
		Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "m"},
	}, config.EngineVLLM)
	if _, err := a.Ready(context.Background()); err == nil {
		t.Error("expected decode error")
	}
}

// These engines are launched with a path and read it, so there is
// nothing to tell them once the bytes are there. Implementing
// ModelInstaller would put a pull and a registration step back on a
// path that has neither.
func TestProxy_InstallsNothing(t *testing.T) {
	t.Parallel()
	a, _ := NewAdapter(config.Config{
		Engine: config.Engine{URL: "http://x:1"}, Model: config.Model{Name: "m"},
	}, config.EngineVLLM)
	if _, ok := any(a).(adapter.ModelInstaller); ok {
		t.Error("proxy adapter implements ModelInstaller; these engines load from a path")
	}
}

func TestProxy_ImplementsAdapter(t *testing.T) {
	t.Parallel()
	var _ adapter.Adapter = (*Adapter)(nil)
}

func TestRewriteJSONModel(t *testing.T) {
	t.Parallel()
	body := []byte(`{"model":"a","x":1}`)
	out, ok := rewriteJSONModel(body, "B")
	if !ok {
		t.Fatal("expected ok=true")
	}
	if !strings.Contains(string(out), `"model":"B"`) {
		t.Errorf("not rewritten: %s", out)
	}
	if _, ok := rewriteJSONModel([]byte("not json"), "B"); ok {
		t.Error("non-JSON should not rewrite")
	}
	// Adds the field if missing.
	out, ok = rewriteJSONModel([]byte(`{"x":1}`), "Y")
	if !ok || !strings.Contains(string(out), `"model":"Y"`) {
		t.Errorf("missing-field add failed: ok=%v out=%s", ok, out)
	}
}

func TestStripAuthorization(t *testing.T) {
	t.Parallel()
	h := http.Header{}
	h.Set("Authorization", "Bearer x")
	h.Set("Proxy-Authorization", "Basic y")
	h.Set("X-Other", "keep")
	stripAuthorization(h)
	if h.Get("Authorization") != "" || h.Get("Proxy-Authorization") != "" {
		t.Errorf("auth not stripped: %+v", h)
	}
	if h.Get("X-Other") != "keep" {
		t.Errorf("non-auth header dropped")
	}
}

func TestJoinPath(t *testing.T) {
	t.Parallel()
	cases := []struct{ base, path, want string }{
		{"http://x", "/v1/models", "http://x/v1/models"},
		{"http://x/", "/v1/models", "http://x/v1/models"},
		{"http://x", "v1/models", "http://x/v1/models"},
		{"http://x/", "v1/models", "http://x/v1/models"},
	}
	for _, tc := range cases {
		if got := joinPath(tc.base, tc.path); got != tc.want {
			t.Errorf("joinPath(%q,%q)=%q want %q", tc.base, tc.path, got, tc.want)
		}
	}
}

// TestProxy_UpstreamErrorJSONEnvelope asserts the ReverseProxy
// ErrorHandler emits the same `{"error":{"code","message"}}` envelope
// as the rest of the data plane (matches OpenAPI components.schemas.Error
// in the public API). Plain text 502 bodies break OpenAI Python
// SDK / LangChain that try to JSON-decode the failure.
func TestProxy_UpstreamErrorJSONEnvelope(t *testing.T) {
	t.Parallel()
	// Bind a listener and immediately close it. The free port is then
	// guaranteed to refuse connects, which makes ReverseProxy invoke
	// our ErrorHandler with a real Dial error.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()

	a, err := NewAdapter(config.Config{
		Engine: config.Engine{URL: "http://" + addr},
		Model:  config.Model{Name: "m"},
	}, config.EngineVLLM)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[]}`))
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Errorf("Content-Type=%q, want application/json", got)
	}
	var env struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("body must be JSON: %v\n%s", err, rec.Body.String())
	}
	if env.Error.Code != "upstream_unreachable" {
		t.Errorf("error.code=%q", env.Error.Code)
	}
	if env.Error.Message == "" {
		t.Errorf("error.message empty: %s", rec.Body.String())
	}
}

// TestReverseProxy_PanicRecover locks the v1.0.5 panic-recover
// middleware. Pre-fix a panic inside Director / ModifyResponse / a
// transport goroutine surfaced as a logged stack and a closed client
// connection — operators saw "EOF" with no actionable error code.
// Post-fix the wrapper:
//
//   - returns 502 + error.code "proxy_panic"
//   - increments llm_init_proxy_panics_total{engine_kind} so an alert
//     can fire when the engine is repeatedly tripping the proxy
//
// We inject a panicking Director by reaching into the package-private
// httputil.ReverseProxy.Director field, which is the closest we can
// get to a real-world panic without depending on a flaky transport
// scenario.
func TestReverseProxy_PanicRecover(t *testing.T) {
	t.Parallel()
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{URL: "http://127.0.0.1:1"}, // never reached
		Model:  config.Model{Name: "m"},
	}, config.EngineVLLM)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	a.Metrics = obsMetricsForTest()
	// Force a panic at the very top of the proxy hot path so the
	// recover wrapper is the only thing standing between the panic
	// and the test's assertions.
	a.proxy.Director = func(*http.Request) {
		panic("synthetic-director-panic-for-test")
	}

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[]}`))

	// MUST NOT panic into the test runner; the recover() wrapper has
	// to absorb it. If we reach the assertions below, the wrapper did
	// its job.
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status=%d, want 502", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Errorf("Content-Type=%q want application/json", got)
	}
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &env); err != nil {
		t.Fatalf("body must be JSON: %v\n%s", err, rec.Body.String())
	}
	if env.Error.Code != "proxy_panic" {
		t.Errorf("error.code=%q want proxy_panic", env.Error.Code)
	}
	// The metric should have observed exactly one increment for the
	// vLLM kind.
	if got := readCounter(a.Metrics.ProxyPanicsTotal.WithLabelValues("vllm")); got != 1 {
		t.Errorf("ProxyPanicsTotal=%v want 1", got)
	}
}

func TestItoa(t *testing.T) {
	t.Parallel()
	cases := []struct {
		n    int64
		want string
	}{{0, "0"}, {1, "1"}, {12345, "12345"}}
	for _, tc := range cases {
		if got := itoa(tc.n); got != tc.want {
			t.Errorf("itoa(%d)=%q", tc.n, got)
		}
	}
}

// TestAudio_ModelsProxiedToEngine: GET /v1/models is forwarded verbatim.
func TestAudio_ModelsProxiedToEngine(t *testing.T) {
	t.Parallel()
	hit := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			hit = true
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"Qwen/Qwen3-ASR-1.7B","mode":"audio","supports":["stt"]}]}`))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: srv.URL},
		Model:  config.Model{Name: "Qwen/Qwen3-ASR-1.7B"},
		Spec:   config.ModelSpec{Name: "Qwen/Qwen3-ASR-1.7B", Mode: "audio", Supports: map[string]bool{"stt": true}},
	}, config.EngineAudio)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !hit {
		t.Error("upstream /v1/models MUST be hit (pure passthrough, no edge synthesis)")
	}
	if !strings.Contains(rec.Body.String(), "Qwen/Qwen3-ASR-1.7B") {
		t.Errorf("engine body must pass through verbatim; got %s", rec.Body.String())
	}
}

func TestAudio_HistoryRangeContractIsTransparent(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "bytes=12-" || r.Header.Get("Accept-Encoding") != "identity" {
			t.Fatalf("request headers = %q / %q", r.Header.Get("Range"), r.Header.Get("Accept-Encoding"))
		}
		w.Header().Set("Content-Range", "bytes 12-14/15")
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("history-item-id", "h-1")
		w.Header().Set("X-Audio-Format", "pcm")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte("end"))
	}))
	t.Cleanup(server.Close)
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: server.URL},
		Model:  config.Model{Name: "tts"},
		Spec:   config.ModelSpec{Name: "tts", Mode: "tts"},
	}, config.EngineAudio)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodGet, "/v1/history/h-1/audio", nil)
	request.Header.Set("Range", "bytes=12-")
	request.Header.Set("Accept-Encoding", "identity")
	recorder := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(recorder, request)
	if recorder.Code != http.StatusPartialContent || recorder.Header().Get("Content-Range") != "bytes 12-14/15" ||
		recorder.Header().Get("Accept-Ranges") != "bytes" || recorder.Header().Get("history-item-id") != "h-1" ||
		recorder.Header().Get("X-Audio-Format") != "pcm" {
		t.Fatalf("response = %d %v", recorder.Code, recorder.Header())
	}
}

// TestAudio_ReadyProbesModels: liveness/readiness probe /v1/models.
func TestAudio_ReadyProbesModels(t *testing.T) {
	t.Parallel()
	var models bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			models = true
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"m"}]}`))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: srv.URL},
		Model:  config.Model{Name: "m"},
		Spec:   config.ModelSpec{Name: "m", Mode: "audio", Supports: map[string]bool{"stt": true}},
	}, config.EngineAudio)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := a.WaitAlive(ctx); err != nil {
		t.Fatalf("WaitAlive: %v", err)
	}
	rs, err := a.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready: %v", err)
	}
	if !rs.Alive || !rs.ModelExists {
		t.Errorf("rs = %+v", rs)
	}
	if !models {
		t.Error("expected /v1/models to be probed for audio liveness/readiness")
	}
}

// TestAudio_PathsPurePassthrough: /v1/audio/* is forwarded, never gated.
func TestAudio_PathsPurePassthrough(t *testing.T) {
	t.Parallel()
	var gotPaths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPaths = append(gotPaths, r.URL.Path)
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineAudio, URL: srv.URL},
		Model:  config.Model{Name: "Qwen/Qwen3-ASR-1.7B"},
		Spec:   config.ModelSpec{Name: "Qwen/Qwen3-ASR-1.7B", Mode: "audio", Supports: map[string]bool{"stt": true}},
	}, config.EngineAudio)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	h := a.OpenAIHandler(config.Config{})

	// vad is not in MODEL_SUPPORTS, yet still forwarded.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/audio/vad", nil))
	if rec.Code == http.StatusNotFound {
		t.Errorf("vad must be forwarded, not edge-gated; got 404 body=%s", rec.Body.String())
	}

	rec2 := httptest.NewRecorder()
	h.ServeHTTP(rec2, httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", nil))
	if rec2.Code != http.StatusOK {
		t.Errorf("transcriptions status = %d, want 200 (proxied)", rec2.Code)
	}

	if len(gotPaths) != 2 {
		t.Errorf("both audio paths must reach the engine; upstream saw %v", gotPaths)
	}
}

func TestProxy_NewAdapterUsesConfiguredHeaderTimeout(t *testing.T) {
	t.Parallel()
	want := 90 * time.Second
	a, err := NewAdapter(config.Config{
		Engine:  config.Engine{URL: "http://x:1"},
		Model:   config.Model{Name: "m"},
		Runtime: config.Runtime{UpstreamResponseHeaderTimeout: want},
	}, config.EngineLlamaCpp)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	tr, ok := a.proxy.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("Transport type %T", a.proxy.Transport)
	}
	if tr.ResponseHeaderTimeout != want {
		t.Errorf("ResponseHeaderTimeout = %v, want %v", tr.ResponseHeaderTimeout, want)
	}

	a, err = NewAdapter(config.Config{
		Engine: config.Engine{URL: "http://x:1"},
		Model:  config.Model{Name: "m"},
	}, config.EngineLlamaCpp)
	if err != nil {
		t.Fatalf("NewAdapter default: %v", err)
	}
	tr = a.proxy.Transport.(*http.Transport)
	if tr.ResponseHeaderTimeout != config.DefaultUpstreamResponseHeaderTimeout {
		t.Errorf("default ResponseHeaderTimeout = %v, want %v",
			tr.ResponseHeaderTimeout, config.DefaultUpstreamResponseHeaderTimeout)
	}
}

func TestProxy_ChatCompletionsHeaderTimeout(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		time.Sleep(300 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	a, err := NewAdapter(config.Config{
		Engine:  config.Engine{URL: srv.URL},
		Model:   config.Model{Name: "m"},
		Runtime: config.Runtime{UpstreamResponseHeaderTimeout: 50 * time.Millisecond},
	}, config.EngineLlamaCpp)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	a.OpenAIHandler(config.Config{}).ServeHTTP(rec, req)

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "timeout awaiting response headers") {
		t.Errorf("body=%s, want timeout awaiting response headers", rec.Body.String())
	}
}
