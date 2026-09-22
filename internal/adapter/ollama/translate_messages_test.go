package ollama

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

// messagesFake captures the /api/chat request body and replies with a
// pre-canned NDJSON body (one frame, done=true) so the non-streaming
// path has something to decode. The shape mirrors chatFake but with
// fewer knobs because Stage 1 only exercises the buffered path.
type messagesFake struct {
	gotBody    map[string]any
	respBody   map[string]any
	respStatus int
	// streamFrames, when non-nil, makes the fake reply NDJSON of one
	// JSON object per frame. respBody is ignored in that case.
	streamFrames []map[string]any
}

func startMessagesFake(t *testing.T, f *messagesFake) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[]}`))
		case "/api/chat":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &f.gotBody)
			if f.respStatus != 0 {
				w.WriteHeader(f.respStatus)
				return
			}
			if len(f.streamFrames) > 0 {
				w.Header().Set("Content-Type", "application/x-ndjson")
				for _, fr := range f.streamFrames {
					b2, _ := json.Marshal(fr)
					_, _ = w.Write(b2)
					_, _ = w.Write([]byte("\n"))
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
				}
				return
			}
			if f.respBody == nil {
				f.respBody = map[string]any{
					"message":           map[string]any{"role": "assistant", "content": "hi back"},
					"done":              true,
					"prompt_eval_count": 5,
					"eval_count":        2,
				}
			}
			b2, _ := json.Marshal(f.respBody)
			_, _ = w.Write(b2)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func newMessagesTestAdapter(t *testing.T, f *messagesFake) *Adapter {
	t.Helper()
	url := startMessagesFake(t, f)
	cfg := config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "qwen2.5"},
	}
	return NewAdapter(cfg)
}

func postMessages(t *testing.T, a *Adapter, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
	a.handleMessages(rec, req)
	return rec
}

func TestMessages_HappyPath_Text(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "claude-3-5-sonnet-20241022",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "hello"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type=%q", ct)
	}

	var resp anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v body=%s", err, rec.Body.String())
	}
	if resp.Type != "message" || resp.Role != anthropic.RoleAssistant {
		t.Errorf("envelope wrong: type=%q role=%q", resp.Type, resp.Role)
	}
	if resp.Model != "qwen2.5" {
		t.Errorf("model alias = %q (want client-facing alias)", resp.Model)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != anthropic.BlockText || resp.Content[0].Text != "hi back" {
		t.Errorf("content blocks: %+v", resp.Content)
	}
	if resp.StopReason != anthropic.StopReasonEndTurn {
		t.Errorf("stop_reason = %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestMessages_SystemPromptPrepended(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "x",
		"max_tokens": 10,
		"system":     "you are concise",
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := f.gotBody["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("expected system+user messages, got %d: %+v", len(msgs), msgs)
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "you are concise" {
		t.Errorf("system prepend wrong: %+v", first)
	}
}

func TestMessages_UpstreamModelRewritten(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)
	a.cfg.Sources = []config.ModelSource{{
		Index:     1,
		Kind:      config.KindOllama,
		Role:      config.RoleMain,
		OllamaTag: "qwen2.5:7b-instruct", // alias vs daemon tag
	}}

	_ = postMessages(t, a, map[string]any{
		"model":      "claude-3-5-sonnet",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if got := f.gotBody["model"]; got != "qwen2.5:7b-instruct" {
		t.Errorf("upstream model = %v want qwen2.5:7b-instruct", got)
	}
}

func TestMessages_OptionsForwarded(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)
	a.cfg.Engine.Args = config.EngineArgs{Known: map[string]string{
		"num_ctx":        "8192",
		"repeat_penalty": "1.1",
		"keep_alive":     "10m",
	}}

	temp := 0.7
	topP := 0.9
	topK := 40
	req := anthropic.MessagesRequest{
		Model:         "m",
		MaxTokens:     200,
		Messages:      []anthropic.Message{{Role: "user", Content: anthropic.Content{Plain: "hi"}}},
		Temperature:   &temp,
		TopP:          &topP,
		TopK:          &topK,
		StopSequences: []string{"<eos>"},
	}
	b, _ := json.Marshal(req)
	rec := httptest.NewRecorder()
	a.handleMessages(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	opts, _ := f.gotBody["options"].(map[string]any)
	if opts["temperature"] != 0.7 {
		t.Errorf("temperature=%v", opts["temperature"])
	}
	if opts["top_p"] != 0.9 {
		t.Errorf("top_p=%v", opts["top_p"])
	}
	if opts["top_k"] != float64(40) {
		t.Errorf("top_k=%v", opts["top_k"])
	}
	if opts["num_predict"] != float64(200) {
		t.Errorf("num_predict=%v", opts["num_predict"])
	}
	if opts["num_ctx"] != float64(8192) {
		t.Errorf("num_ctx=%v", opts["num_ctx"])
	}
	if opts["repeat_penalty"] != 1.1 {
		t.Errorf("repeat_penalty=%v", opts["repeat_penalty"])
	}
	stop, _ := opts["stop"].([]any)
	if len(stop) != 1 || stop[0] != "<eos>" {
		t.Errorf("stop=%v", opts["stop"])
	}
	if f.gotBody["keep_alive"] != "10m" {
		t.Errorf("keep_alive=%v", f.gotBody["keep_alive"])
	}
}

// Note: tools are NOT stripped any more (Stage 3 / Track R.3 added
// real translation). See TestMessages_ToolsForwarded for the
// positive coverage.

func TestMessages_BlockContentFlattened(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "part1 "},
				{"type": "text", "text": "part2"},
				{"type": "image", "source": map[string]any{"type": "base64", "data": "x"}}, // skipped Stage 1
			},
		}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := f.gotBody["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	m, _ := msgs[0].(map[string]any)
	if m["content"] != "part1 part2" {
		t.Errorf("flatten=%v", m["content"])
	}
}

func TestMessages_MaxTokensReached_StopReason(t *testing.T) {
	t.Parallel()
	f := &messagesFake{respBody: map[string]any{
		"message":           map[string]any{"role": "assistant", "content": "..."},
		"done":              true,
		"prompt_eval_count": 5,
		"eval_count":        10, // matches max_tokens below
	}}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	var resp anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.StopReason != anthropic.StopReasonMaxTokens {
		t.Errorf("stop_reason = %q want max_tokens", resp.StopReason)
	}
}

func TestMessages_InvalidJSON(t *testing.T) {
	t.Parallel()
	a := newMessagesTestAdapter(t, &messagesFake{})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("not-json"))
	a.handleMessages(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"invalid_request_error"`) {
		t.Errorf("expected invalid_request_error: %s", rec.Body.String())
	}
}

func TestMessages_MissingMaxTokens(t *testing.T) {
	t.Parallel()
	a := newMessagesTestAdapter(t, &messagesFake{})
	rec := postMessages(t, a, map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d", rec.Code)
	}
}

func TestMessages_EmptyMessages(t *testing.T) {
	t.Parallel()
	a := newMessagesTestAdapter(t, &messagesFake{})
	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{},
	})
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d", rec.Code)
	}
}

func TestMessages_UpstreamError(t *testing.T) {
	t.Parallel()
	f := &messagesFake{respStatus: 500}
	a := newMessagesTestAdapter(t, f)
	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status=%d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"api_error"`) {
		t.Errorf("expected api_error: %s", rec.Body.String())
	}
}

func TestMessages_StreamingHappyPath(t *testing.T) {
	t.Parallel()
	f := &messagesFake{streamFrames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "content": "Hello"}, "done": false},
		{"message": map[string]any{"role": "assistant", "content": " world"}, "done": false},
		{"message": map[string]any{"role": "assistant", "content": "!"}, "done": true,
			"prompt_eval_count": 11, "eval_count": 3},
	}}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "claude-3-5-sonnet",
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
	// Must start with message_start.
	if !strings.HasPrefix(out, "event: message_start\ndata: ") {
		t.Errorf("missing leading message_start: head=%q", out[:min(80, len(out))])
	}
	// Must contain content_block_start.
	if !strings.Contains(out, "event: content_block_start\n") {
		t.Errorf("missing content_block_start: %s", out)
	}
	// Must contain text_delta frames.
	for _, want := range []string{`"text":"Hello"`, `"text":" world"`, `"text":"!"`} {
		if !strings.Contains(out, want) {
			t.Errorf("missing delta %s: %s", want, out)
		}
	}
	// content_block_stop must precede message_delta.
	stopIdx := strings.Index(out, "event: content_block_stop\n")
	deltaIdx := strings.Index(out, "event: message_delta\n")
	if stopIdx < 0 || deltaIdx < 0 || stopIdx > deltaIdx {
		t.Errorf("event order wrong: stop=%d delta=%d", stopIdx, deltaIdx)
	}
	// Final message_stop.
	if !strings.HasSuffix(strings.TrimRight(out, "\n "), `"type":"message_stop"}`) {
		// Look for the substring instead — some payloads include
		// other JSON keys before. Just require its presence.
		if !strings.Contains(out, `"type":"message_stop"`) {
			t.Errorf("missing message_stop: %s", out)
		}
	}
	// message_delta must carry the usage and stop_reason.
	if !strings.Contains(out, `"stop_reason":"end_turn"`) {
		t.Errorf("missing stop_reason end_turn: %s", out)
	}
	if !strings.Contains(out, `"input_tokens":11`) {
		t.Errorf("missing input_tokens 11 in message_delta: %s", out)
	}
	if !strings.Contains(out, `"output_tokens":3`) {
		t.Errorf("missing output_tokens 3: %s", out)
	}
}

func TestMessages_StreamingMaxTokensStopReason(t *testing.T) {
	t.Parallel()
	f := &messagesFake{streamFrames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "content": "abc"}, "done": false},
		{"message": map[string]any{"role": "assistant", "content": ""}, "done": true,
			"prompt_eval_count": 1, "eval_count": 5},
	}}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 5,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"stream":     true,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"stop_reason":"max_tokens"`) {
		t.Errorf("expected stop_reason=max_tokens, got: %s", rec.Body.String())
	}
}

func TestMessages_ToolsForwarded(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "what's the weather?"}},
		"tools": []map[string]any{{
			"name":        "get_weather",
			"description": "Get the weather",
			"input_schema": map[string]any{
				"type":       "object",
				"properties": map[string]any{"location": map[string]any{"type": "string"}},
				"required":   []string{"location"},
			},
		}},
		"tool_choice": map[string]any{"type": "auto"},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	tools, _ := f.gotBody["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 forwarded tool, got %d: %+v", len(tools), f.gotBody["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("type=%v", tool["type"])
	}
	fn, _ := tool["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("function.name=%v", fn["name"])
	}
	if f.gotBody["tool_choice"] != "auto" {
		t.Errorf("tool_choice=%v", f.gotBody["tool_choice"])
	}
}

func TestMessages_ResponseToolUse(t *testing.T) {
	t.Parallel()
	f := &messagesFake{respBody: map[string]any{
		"message": map[string]any{
			"role":    "assistant",
			"content": "",
			"tool_calls": []map[string]any{{
				"id":   "call_1",
				"type": "function",
				"function": map[string]any{
					"name":      "get_weather",
					"arguments": map[string]any{"location": "SF"},
				},
			}},
		},
		"done":              true,
		"prompt_eval_count": 10,
		"eval_count":        5,
	}}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages":   []map[string]any{{"role": "user", "content": "weather?"}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp anthropic.MessagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v body=%s", err, rec.Body.String())
	}
	if resp.StopReason != anthropic.StopReasonToolUse {
		t.Errorf("stop_reason=%q want tool_use", resp.StopReason)
	}
	// Should have a single tool_use content block (empty text was
	// dropped).
	if len(resp.Content) != 1 || resp.Content[0].Type != anthropic.BlockToolUse {
		t.Fatalf("content blocks=%+v", resp.Content)
	}
	if resp.Content[0].ID != "call_1" || resp.Content[0].Name != "get_weather" {
		t.Errorf("tool_use=%+v", resp.Content[0])
	}
}

func TestMessages_ToolResultFannedOut(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
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
	msgs, _ := f.gotBody["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("expected user/assistant/tool, got %d: %+v", len(msgs), msgs)
	}
	// msgs[1] is the assistant with tool_calls.
	asst, _ := msgs[1].(map[string]any)
	if asst["role"] != "assistant" {
		t.Errorf("msgs[1] role=%v", asst["role"])
	}
	tc, _ := asst["tool_calls"].([]any)
	if len(tc) != 1 {
		t.Errorf("tool_calls=%+v", tc)
	}
	// msgs[2] is the tool response.
	tool, _ := msgs[2].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "call_1" || tool["content"] != "72F" {
		t.Errorf("tool msg=%+v", tool)
	}
	fn, _ := tc[0].(map[string]any)[openaiKeyFunction].(map[string]any)
	if _, ok := fn[openaiKeyArguments].(map[string]any); !ok {
		t.Errorf("assistant tool arguments must be object for Ollama, got %T", fn[openaiKeyArguments])
	}
}

func TestMessages_StreamingToolUse(t *testing.T) {
	t.Parallel()
	f := &messagesFake{streamFrames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "content": "Checking..."}, "done": false},
		{
			"message": map[string]any{
				"role":    "assistant",
				"content": "",
				"tool_calls": []map[string]any{{
					"id": "call_1", "type": "function",
					"function": map[string]any{"name": "get_weather", "arguments": map[string]any{"location": "SF"}},
				}},
			},
			"done":              true,
			"prompt_eval_count": 11,
			"eval_count":        2,
		},
	}}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
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
		`"text":"Checking..."`,
		`"type":"tool_use"`,
		`"name":"get_weather"`,
		`"type":"input_json_delta"`,
		`"stop_reason":"tool_use"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in stream output", want)
		}
	}
	// Tool block must appear at index 1 (text was index 0).
	if !strings.Contains(out, `"index":1`) {
		t.Errorf("expected index:1 for tool block: %s", out)
	}
}

func TestMessages_VisionBase64Forwarded(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "what's in this image?"},
				{"type": "image", "source": map[string]any{
					"type": "base64", "media_type": "image/png", "data": "iVBOR",
				}},
			},
		}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := f.gotBody["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 user message, got %d", len(msgs))
	}
	m, _ := msgs[0].(map[string]any)
	if m["content"] != "what's in this image?" {
		t.Errorf("content=%v", m["content"])
	}
	imgs, _ := m["images"].([]any)
	if len(imgs) != 1 || imgs[0] != "iVBOR" {
		t.Errorf("images=%+v", imgs)
	}
}

func TestMessages_VisionURLDropped(t *testing.T) {
	t.Parallel()
	f := &messagesFake{}
	a := newMessagesTestAdapter(t, f)

	rec := postMessages(t, a, map[string]any{
		"model":      "m",
		"max_tokens": 64,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": "describe"},
				{"type": "image", "source": map[string]any{"type": "url", "url": "https://x.com/a.jpg"}},
			},
		}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := f.gotBody["messages"].([]any)
	m, _ := msgs[0].(map[string]any)
	if _, hasImages := m["images"]; hasImages {
		t.Errorf("URL image must not produce images[]: %+v", m)
	}
}

func TestCountTokens_HappyPath(t *testing.T) {
	t.Parallel()
	a := newMessagesTestAdapter(t, &messagesFake{})
	body, _ := json.Marshal(map[string]any{
		"model":    "m",
		"messages": []map[string]any{{"role": "user", "content": "hello world"}}, // 11
	})
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewReader(body))
	a.handleCountTokens(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if hdr := rec.Header().Get("X-Llm-Init-Token-Estimate"); hdr != "heuristic" {
		t.Errorf("missing heuristic header: %q", hdr)
	}
	var resp anthropic.CountTokensResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.InputTokens != 3 {
		t.Errorf("input_tokens=%d", resp.InputTokens)
	}
}

func TestCountTokens_RoutedViaAnthropicHandler(t *testing.T) {
	t.Parallel()
	a := newMessagesTestAdapter(t, &messagesFake{})
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

func TestMessages_BodyTooLarge(t *testing.T) {
	// Intentionally NOT t.Parallel(): mutates the package-level
	// MessagesBodyCap. Other tests in this file run with t.Parallel()
	// and would observe the small 256-byte cap mid-flight if this one
	// raced with them.
	a := newMessagesTestAdapter(t, &messagesFake{})

	prev := MessagesBodyCap
	MessagesBodyCap = 256
	t.Cleanup(func() { MessagesBodyCap = prev })

	huge := strings.Repeat("a", 1024)
	body := map[string]any{
		"model":      "m",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": huge}},
	}
	b, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
	a.handleMessages(rec, req)
	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}
