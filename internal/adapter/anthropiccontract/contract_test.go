package anthropiccontract_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter/ollama"
	"github.com/llm-init/llm-init/internal/adapter/proxy"
	"github.com/llm-init/llm-init/internal/config"
)

// modelAlias is the OpenAI-facing name both adapters advertise; the
// contract requires it (not the upstream tag) to appear on the
// Anthropic response.
const modelAlias = "qwen-served"

// engine names the two /v1/messages implementations under test.
type engine struct {
	name string
	// handler builds the AnthropicHandler wired to an upstream fake
	// that serves the canned non-stream / stream payloads below.
	handler func(t *testing.T, stream bool) http.Handler
}

// engines returns the proxy (OpenAI-shaped) and ollama (NDJSON-shaped)
// adapters, each pointed at an upstream httptest fake that returns the
// SAME assistant turn ("hi back", 5 prompt / 2 completion tokens) in
// its native wire shape. Any divergence in the Anthropic output is a
// contract break.
func engines() []engine {
	return []engine{
		{name: "proxy/vllm", handler: newProxyHandler},
		{name: "ollama", handler: newOllamaHandler},
	}
}

func newProxyHandler(t *testing.T, stream bool) http.Handler {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("proxy upstream hit unexpected path %q", r.URL.Path)
		}
		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, line := range openAIStreamFrames {
				_, _ = w.Write([]byte(line))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
			return
		}
		_, _ = w.Write([]byte(openAINonStream))
	}))
	t.Cleanup(srv.Close)
	a, err := proxy.NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineVLLM, URL: srv.URL},
		Model:  config.Model{Name: modelAlias},
	}, config.EngineVLLM)
	if err != nil {
		t.Fatalf("proxy.NewAdapter: %v", err)
	}
	return a.AnthropicHandler(config.Config{})
}

func newOllamaHandler(t *testing.T, stream bool) http.Handler {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("ollama upstream hit unexpected path %q", r.URL.Path)
		}
		if stream {
			for _, line := range ollamaStreamFrames {
				_, _ = w.Write([]byte(line))
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
			return
		}
		_, _ = w.Write([]byte(ollamaNonStream))
	}))
	t.Cleanup(srv.Close)
	a := ollama.NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: srv.URL},
		Model:  config.Model{Name: modelAlias},
	})
	return a.AnthropicHandler(config.Config{})
}

// Canned upstream payloads: both express the same assistant turn
// ("hi back", prompt=5, completion=2 tokens, natural stop).
const (
	openAINonStream = `{
		"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"upstream-tag",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi back"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
	}`
	ollamaNonStream = `{"model":"upstream-tag","message":{"role":"assistant","content":"hi back"},"done":true,"prompt_eval_count":5,"eval_count":2}`
)

var (
	openAIStreamFrames = []string{
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" back\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
		"data: [DONE]\n\n",
	}
	ollamaStreamFrames = []string{
		`{"message":{"role":"assistant","content":"hi"},"done":false}` + "\n",
		`{"message":{"role":"assistant","content":" back"},"done":false}` + "\n",
		`{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":5,"eval_count":2}` + "\n",
	}
)

func postMessages(t *testing.T, h http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	h.ServeHTTP(rec, req)
	return rec
}

func baseRequest() map[string]any {
	return map[string]any{
		// A deliberately different model name: the contract is that
		// both adapters rewrite the response model to the local alias,
		// never echoing the client-supplied name or the upstream tag.
		"model":      "claude-3-5-sonnet",
		"max_tokens": 64,
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
		},
	}
}

// TestContract_NonStreaming locks the non-streaming MessagesResponse
// shape: both engines must emit identical type/role/model/content/
// stop_reason/usage for the same canned upstream turn.
func TestContract_NonStreaming(t *testing.T) {
	type result struct {
		Type       string `json:"type"`
		Role       string `json:"role"`
		Model      string `json:"model"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"usage"`
	}

	var got []result
	for _, e := range engines() {
		h := e.handler(t, false)
		rec := postMessages(t, h, "/v1/messages", baseRequest())
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status = %d, body = %s", e.name, rec.Code, rec.Body.String())
		}
		var r result
		if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
			t.Fatalf("%s: decode response: %v (%s)", e.name, err, rec.Body.String())
		}
		// Per-engine invariants.
		if r.Type != "message" || r.Role != "assistant" {
			t.Errorf("%s: type/role = %q/%q, want message/assistant", e.name, r.Type, r.Role)
		}
		if r.Model != modelAlias {
			t.Errorf("%s: model = %q, want alias %q (response must not echo upstream tag or client name)", e.name, r.Model, modelAlias)
		}
		if len(r.Content) != 1 || r.Content[0].Type != "text" || r.Content[0].Text != "hi back" {
			t.Errorf("%s: content = %+v, want single text block %q", e.name, r.Content, "hi back")
		}
		if r.StopReason != "end_turn" {
			t.Errorf("%s: stop_reason = %q, want end_turn", e.name, r.StopReason)
		}
		if r.Usage.InputTokens != 5 || r.Usage.OutputTokens != 2 {
			t.Errorf("%s: usage = %d/%d, want 5/2", e.name, r.Usage.InputTokens, r.Usage.OutputTokens)
		}
		got = append(got, r)
	}

	// Cross-engine equivalence: ID is the only field allowed to differ
	// (it embeds a timestamp), so compare everything else field-by-field.
	a, b := got[0], got[1]
	if a.Type != b.Type || a.Role != b.Role || a.Model != b.Model ||
		a.StopReason != b.StopReason || a.Usage != b.Usage {
		t.Errorf("non-stream contract diverges: proxy=%+v ollama=%+v", a, b)
	}
	if len(a.Content) != len(b.Content) || a.Content[0] != b.Content[0] {
		t.Errorf("non-stream content diverges: proxy=%+v ollama=%+v", a.Content, b.Content)
	}
}

// TestContract_Streaming locks the SSE event sequence both engines must
// produce: message_start, content_block_start, content_block_delta(s),
// content_block_stop, message_delta, message_stop — with the deltas
// concatenating to the full text and the final message_delta carrying
// the settled usage + stop_reason.
func TestContract_Streaming(t *testing.T) {
	type parsed struct {
		eventOrder []string
		text       string
		stopReason string
		inputTok   int64
		outputTok  int64
		model      string
	}

	parse := func(t *testing.T, raw string) parsed {
		t.Helper()
		var p parsed
		for _, block := range strings.Split(raw, "\n\n") {
			block = strings.TrimSpace(block)
			if block == "" {
				continue
			}
			var event, data string
			for _, line := range strings.Split(block, "\n") {
				if v, ok := strings.CutPrefix(line, "event: "); ok {
					event = strings.TrimSpace(v)
				}
				if v, ok := strings.CutPrefix(line, "data: "); ok {
					data = v
				}
			}
			if event == "" {
				continue
			}
			p.eventOrder = append(p.eventOrder, event)
			switch event {
			case "message_start":
				var m struct {
					Message struct {
						Model string `json:"model"`
					} `json:"message"`
				}
				_ = json.Unmarshal([]byte(data), &m)
				p.model = m.Message.Model
			case "content_block_delta":
				var d struct {
					Delta struct {
						Type string `json:"type"`
						Text string `json:"text"`
					} `json:"delta"`
				}
				_ = json.Unmarshal([]byte(data), &d)
				if d.Delta.Type == "text_delta" {
					p.text += d.Delta.Text
				}
			case "message_delta":
				var d struct {
					Delta struct {
						StopReason string `json:"stop_reason"`
					} `json:"delta"`
					Usage struct {
						InputTokens  int64 `json:"input_tokens"`
						OutputTokens int64 `json:"output_tokens"`
					} `json:"usage"`
				}
				_ = json.Unmarshal([]byte(data), &d)
				p.stopReason = d.Delta.StopReason
				p.inputTok = d.Usage.InputTokens
				p.outputTok = d.Usage.OutputTokens
			}
		}
		return p
	}

	wantOrder := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	var results []parsed
	for _, e := range engines() {
		h := e.handler(t, true)
		rec := postMessages(t, h, "/v1/messages", withStream(baseRequest()))
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status = %d", e.name, rec.Code)
		}
		p := parse(t, rec.Body.String())
		if !equalStrings(p.eventOrder, wantOrder) {
			t.Errorf("%s: event order = %v, want %v", e.name, p.eventOrder, wantOrder)
		}
		if p.text != "hi back" {
			t.Errorf("%s: concatenated text = %q, want %q", e.name, p.text, "hi back")
		}
		if p.model != modelAlias {
			t.Errorf("%s: message_start model = %q, want %q", e.name, p.model, modelAlias)
		}
		if p.stopReason != "end_turn" {
			t.Errorf("%s: stop_reason = %q, want end_turn", e.name, p.stopReason)
		}
		if p.inputTok != 5 || p.outputTok != 2 {
			t.Errorf("%s: usage = %d/%d, want 5/2", e.name, p.inputTok, p.outputTok)
		}
		results = append(results, p)
	}

	a, b := results[0], results[1]
	if !equalStrings(a.eventOrder, b.eventOrder) {
		t.Errorf("stream event order diverges: proxy=%v ollama=%v", a.eventOrder, b.eventOrder)
	}
	if a.text != b.text || a.stopReason != b.stopReason ||
		a.inputTok != b.inputTok || a.outputTok != b.outputTok || a.model != b.model {
		t.Errorf("stream contract diverges: proxy=%+v ollama=%+v", a, b)
	}
}

// TestContract_RequestValidation locks the shared rejection rules: both
// adapters must reject the same malformed requests with 400 +
// invalid_request_error before contacting any upstream.
func TestContract_RequestValidation(t *testing.T) {
	cases := []struct {
		name string
		body map[string]any
	}{
		{"missing_max_tokens", map[string]any{"model": "x", "messages": []any{map[string]any{"role": "user", "content": "hi"}}}},
		{"zero_max_tokens", map[string]any{"model": "x", "max_tokens": 0, "messages": []any{map[string]any{"role": "user", "content": "hi"}}}},
		{"empty_messages", map[string]any{"model": "x", "max_tokens": 8, "messages": []any{}}},
	}
	for _, e := range engines() {
		for _, tc := range cases {
			h := e.handler(t, false)
			rec := postMessages(t, h, "/v1/messages", tc.body)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("%s/%s: status = %d, want 400", e.name, tc.name, rec.Code)
			}
			if !strings.Contains(rec.Body.String(), "invalid_request_error") {
				t.Errorf("%s/%s: body = %s, want invalid_request_error", e.name, tc.name, rec.Body.String())
			}
		}
	}
}

// TestContract_CountTokens locks the count_tokens envelope + heuristic
// header. Both adapters share the chars/4 estimator, so identical input
// must yield identical input_tokens and the same advisory header.
func TestContract_CountTokens(t *testing.T) {
	body := map[string]any{
		"model": "x",
		"messages": []any{
			map[string]any{"role": "user", "content": "hello world this is a token count test"},
		},
	}
	var counts []int64
	for _, e := range engines() {
		h := e.handler(t, false)
		rec := postMessages(t, h, "/v1/messages/count_tokens", body)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status = %d", e.name, rec.Code)
		}
		if got := rec.Header().Get("X-Llm-Init-Token-Estimate"); got != "heuristic" {
			t.Errorf("%s: estimate header = %q, want heuristic", e.name, got)
		}
		var r struct {
			InputTokens int64 `json:"input_tokens"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &r); err != nil {
			t.Fatalf("%s: decode: %v", e.name, err)
		}
		counts = append(counts, r.InputTokens)
	}
	if counts[0] != counts[1] {
		t.Errorf("count_tokens diverges: proxy=%d ollama=%d", counts[0], counts[1])
	}
}

func withStream(m map[string]any) map[string]any {
	m["stream"] = true
	return m
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
