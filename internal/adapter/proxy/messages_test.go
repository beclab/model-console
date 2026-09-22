package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/anthropic"
	"github.com/llm-init/llm-init/internal/config"
)

// messagesUpstream is a focused httptest fake that captures the
// /v1/chat/completions body and replies with a canned OpenAI response.
// Simpler than the package's `upstream` because /v1/messages doesn't
// need to share the routing scaffolding.
type messagesUpstream struct {
	srv         *httptest.Server
	gotBody     map[string]any
	gotPath     string
	respStatus  int
	respBody    string
	respHeaders map[string]string
	// streamLines, when non-nil, makes the fake reply text/event-stream
	// with one frame per line. respBody is ignored in that case.
	streamLines []string
}

func startMessagesUpstream(t *testing.T) *messagesUpstream {
	t.Helper()
	u := &messagesUpstream{}
	u.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u.gotPath = r.URL.Path
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &u.gotBody)
		for k, v := range u.respHeaders {
			w.Header().Set(k, v)
		}
		if u.respStatus != 0 {
			w.WriteHeader(u.respStatus)
		}
		if len(u.streamLines) > 0 {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, line := range u.streamLines {
				_, _ = io.WriteString(w, line)
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
			return
		}
		if u.respBody != "" {
			_, _ = io.WriteString(w, u.respBody)
			return
		}
		// Default: a healthy single-choice OpenAI response.
		_, _ = io.WriteString(w, `{
            "id":"chatcmpl-1","object":"chat.completion","created":1,"model":"target",
            "choices":[{"index":0,"message":{"role":"assistant","content":"hi back"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
        }`)
	}))
	t.Cleanup(u.srv.Close)
	return u
}

func newMessagesProxy(t *testing.T, u *messagesUpstream) *Adapter {
	t.Helper()
	a, err := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineVLLM, URL: u.srv.URL},
		Model:  config.Model{Name: "qwen-served"},
	}, config.EngineVLLM)
	if err != nil {
		t.Fatalf("NewAdapter: %v", err)
	}
	return a
}

func postMessagesProxy(t *testing.T, a *Adapter, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
	a.handleMessages(rec, req)
	return rec
}

func TestProxyMessages_HappyPath(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "hello"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if u.gotPath != "/v1/chat/completions" {
		t.Errorf("upstream path = %q want /v1/chat/completions", u.gotPath)
	}
	if u.gotBody["model"] != "qwen-served" {
		t.Errorf("upstream model = %v want qwen-served (alias)", u.gotBody["model"])
	}
	if u.gotBody["max_tokens"] != float64(64) {
		t.Errorf("max_tokens forwarded wrong: %v", u.gotBody["max_tokens"])
	}

	var resp anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v body=%s", err, rec.Body.String())
	}
	if resp.Type != "message" {
		t.Errorf("type=%q", resp.Type)
	}
	if resp.Model != "qwen-served" {
		t.Errorf("model alias = %q", resp.Model)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text != "hi back" {
		t.Errorf("content=%+v", resp.Content)
	}
	if resp.StopReason != anthropic.StopReasonEndTurn {
		t.Errorf("stop_reason=%q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 2 {
		t.Errorf("usage=%+v", resp.Usage)
	}
}

func TestProxyMessages_SystemAndStopSequences(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	_ = postMessagesProxy(t, a, map[string]any{
		"model":          "x",
		"max_tokens":     10,
		"system":         "you are helpful",
		"stop_sequences": []string{"<eos>", "STOP"},
		"messages":       []map[string]any{{"role": "user", "content": "hi"}},
	})

	msgs, _ := u.gotBody["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("expected system+user, got %d: %+v", len(msgs), msgs)
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "you are helpful" {
		t.Errorf("system prepend wrong: %+v", first)
	}
	stop, _ := u.gotBody["stop"].([]any)
	if len(stop) != 2 || stop[0] != "<eos>" {
		t.Errorf("stop sequences = %+v", stop)
	}
}

func TestProxyMessages_FinishReasonLengthMapsToMaxTokens(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	u.respBody = `{
        "id":"x","object":"chat.completion","created":1,"model":"m",
        "choices":[{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"length"}],
        "usage":{"prompt_tokens":1,"completion_tokens":10}
    }`
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	var resp anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != anthropic.StopReasonMaxTokens {
		t.Errorf("stop_reason=%q want max_tokens", resp.StopReason)
	}
}

func TestProxyMessages_TopKForwarded(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	temp := 0.5
	topK := 40
	req := anthropic.MessagesRequest{
		Model:       "m",
		MaxTokens:   10,
		Temperature: &temp,
		TopK:        &topK,
		Messages:    []anthropic.Message{{Role: "user", Content: anthropic.Content{Plain: "hi"}}},
	}
	b, _ := json.Marshal(req)
	rec := httptest.NewRecorder()
	a.handleMessages(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if u.gotBody["temperature"] != 0.5 {
		t.Errorf("temperature=%v", u.gotBody["temperature"])
	}
	if u.gotBody["top_k"] != float64(40) {
		t.Errorf("top_k=%v", u.gotBody["top_k"])
	}
}

func TestProxyMessages_UpstreamError(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	u.respStatus = 500
	u.respBody = `{"error":"boom"}`
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"api_error"`) {
		t.Errorf("expected api_error: %s", rec.Body.String())
	}
}

func TestProxyMessages_MissingMaxTokens(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"invalid_request_error"`) {
		t.Errorf("expected invalid_request_error: %s", rec.Body.String())
	}
}

func TestProxyMessages_BodyTooLarge(t *testing.T) {
	// Intentionally NOT t.Parallel(): mutates the package-level
	// MessagesBodyCap. Other tests in this file run with t.Parallel()
	// and would observe the small 256-byte cap mid-flight if this one
	// raced with them.
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	prev := MessagesBodyCap
	MessagesBodyCap = 256
	t.Cleanup(func() { MessagesBodyCap = prev })

	huge := strings.Repeat("a", 1024)
	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": huge}},
	})
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestProxyMessages_InvalidJSON(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("not-json"))
	a.handleMessages(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d", rec.Code)
	}
}

func TestProxyMessages_StreamingHappyPath(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	u.streamLines = []string{
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}` + "\n\n",
		`data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}` + "\n\n",
		`data: {"choices":[{"index":0,"delta":{"content":" there"}}]}` + "\n\n",
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}` + "\n\n",
		"data: [DONE]\n\n",
	}
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 100,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"stream":     true,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type=%q", ct)
	}
	out := rec.Body.String()
	// Forwarded body must have stream:true
	if u.gotBody["stream"] != true {
		t.Errorf("upstream stream flag=%v", u.gotBody["stream"])
	}
	for _, want := range []string{
		"event: message_start\n",
		"event: content_block_start\n",
		`"text":"Hi"`,
		`"text":" there"`,
		"event: content_block_stop\n",
		`"stop_reason":"end_turn"`,
		`"input_tokens":7`,
		`"output_tokens":2`,
		"event: message_stop\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in stream output", want)
		}
	}
}

func TestProxyMessages_StreamingLengthMapsToMaxTokens(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	u.streamLines = []string{
		`data: {"choices":[{"delta":{"content":"x"}}]}` + "\n\n",
		`data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":10}}` + "\n\n",
		"data: [DONE]\n\n",
	}
	a := newMessagesProxy(t, u)
	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"stream":     true,
	})
	if !strings.Contains(rec.Body.String(), `"stop_reason":"max_tokens"`) {
		t.Errorf("expected stop_reason=max_tokens: %s", rec.Body.String())
	}
}

func TestProxyMessages_ToolsTranslated(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "weather?"}},
		"tools": []map[string]any{{
			"name":         "get_weather",
			"description":  "Get the weather",
			"input_schema": map[string]any{"type": "object"},
		}},
		"tool_choice": map[string]any{"type": "tool", "name": "get_weather"},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	tools, _ := u.gotBody["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 forwarded tool, got %d: %+v", len(tools), u.gotBody["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("type=%v", tool["type"])
	}
	tc, _ := u.gotBody["tool_choice"].(map[string]any)
	if tc["type"] != "function" {
		t.Errorf("tool_choice.type=%v", tc["type"])
	}
}

func TestProxyMessages_ResponseToolUse(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	u.respBody = `{
        "id":"x","object":"chat.completion","created":1,"model":"m",
        "choices":[{"index":0,"message":{
            "role":"assistant","content":null,
            "tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"SF\"}"}}]
        },"finish_reason":"tool_calls"}],
        "usage":{"prompt_tokens":10,"completion_tokens":3}
    }`
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "weather?"}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != anthropic.StopReasonToolUse {
		t.Errorf("stop_reason=%q", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != anthropic.BlockToolUse {
		t.Fatalf("content=%+v", resp.Content)
	}
	if resp.Content[0].ID != "call_1" || resp.Content[0].Name != "get_weather" {
		t.Errorf("tool_use=%+v", resp.Content[0])
	}
}

func TestProxyMessages_ToolResultFannedOut(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages": []any{
			map[string]any{"role": "user", "content": "weather?"},
			map[string]any{"role": "assistant", "content": []map[string]any{
				{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": map[string]any{"location": "SF"}},
			}},
			map[string]any{"role": "user", "content": []map[string]any{
				{"type": "tool_result", "tool_use_id": "call_1", "content": "72F"},
			}},
		},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := u.gotBody["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d: %+v", len(msgs), msgs)
	}
	asst, _ := msgs[1].(map[string]any)
	if asst["role"] != "assistant" {
		t.Errorf("msgs[1] role=%v", asst["role"])
	}
	tool, _ := msgs[2].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_1" {
		t.Errorf("tool msg=%+v", tool)
	}
}

func TestProxyMessages_StreamingToolUse(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	u.streamLines = []string{
		// vLLM-style streaming: first frame announces the tool call
		// (id + name), subsequent frames stream the args incrementally.
		`data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}` + "\n\n",
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}` + "\n\n",
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}` + "\n\n",
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"SF\"}"}}]}}]}` + "\n\n",
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":4}}` + "\n\n",
		"data: [DONE]\n\n",
	}
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"stream":     true,
		"messages":   []map[string]any{{"role": "user", "content": "weather?"}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	for _, want := range []string{
		`"type":"tool_use"`,
		`"name":"get_weather"`,
		`"id":"call_1"`,
		`"type":"input_json_delta"`,
		`{\"location\":\"SF\"}`,
		`"stop_reason":"tool_use"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in stream output", want)
		}
	}
}

func TestProxyMessages_VisionBase64(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "what's in this?"},
				{"type": "image", "source": map[string]any{
					"type": "base64", "media_type": "image/jpeg", "data": "iVBOR",
				}},
			},
		}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := u.gotBody["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 user message, got %d", len(msgs))
	}
	m, _ := msgs[0].(map[string]any)
	content, ok := m["content"].([]any)
	if !ok {
		t.Fatalf("expected typed content array, got %T: %#v", m["content"], m["content"])
	}
	if len(content) != 2 {
		t.Fatalf("expected text + image parts, got %d: %+v", len(content), content)
	}
	imgPart, _ := content[1].(map[string]any)
	if imgPart["type"] != "image_url" {
		t.Errorf("image part type=%v", imgPart["type"])
	}
	imgURL, _ := imgPart["image_url"].(map[string]any)
	if imgURL["url"] != "data:image/jpeg;base64,iVBOR" {
		t.Errorf("image url=%v", imgURL["url"])
	}
}

func TestProxyMessages_VisionURL(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "image", "source": map[string]any{"type": "url", "url": "https://x.com/a.jpg"}},
			},
		}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := u.gotBody["messages"].([]any)
	m, _ := msgs[0].(map[string]any)
	content, _ := m["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("expected 1 image part, got %d", len(content))
	}
	part, _ := content[0].(map[string]any)
	urlObj, _ := part["image_url"].(map[string]any)
	if urlObj["url"] != "https://x.com/a.jpg" {
		t.Errorf("url passthrough failed: %v", urlObj["url"])
	}
}

func TestProxyMessages_TextOnlyStaysString(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	rec := postMessagesProxy(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 5,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	msgs, _ := u.gotBody["messages"].([]any)
	m, _ := msgs[0].(map[string]any)
	if _, isStr := m["content"].(string); !isStr {
		t.Errorf("text-only must stay as string content, got %T: %#v", m["content"], m["content"])
	}
}

func TestProxyCountTokens_HappyPath(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)
	body, _ := json.Marshal(map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hello world"}}, // 11 chars
	})
	rec := httptest.NewRecorder()
	a.handleCountTokens(rec, httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if hdr := rec.Header().Get("X-Llm-Init-Token-Estimate"); hdr != "heuristic" {
		t.Errorf("header=%q", hdr)
	}
	var resp anthropic.CountTokensResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.InputTokens != 3 {
		t.Errorf("input_tokens=%d", resp.InputTokens)
	}
}

func TestProxyCountTokens_RoutedViaAnthropicHandler(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)
	h := a.AnthropicHandler(a.cfg)
	body, _ := json.Marshal(map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "x"}},
	})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Errorf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestProxyMessages_AnthropicHandlerWiring(t *testing.T) {
	t.Parallel()
	u := startMessagesUpstream(t)
	a := newMessagesProxy(t, u)

	h := a.AnthropicHandler(a.cfg)
	if h == nil {
		t.Fatal("AnthropicHandler returned nil")
	}
	rec := httptest.NewRecorder()
	body, _ := json.Marshal(map[string]any{
		"model":      "m",
		"max_tokens": 5,
		"messages":   []map[string]any{{"role": "user", "content": "x"}},
	})
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Errorf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}
