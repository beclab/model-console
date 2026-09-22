package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

func newTestAdapter(t *testing.T, fake *fakeOllama, cfg config.Config) (*Adapter, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(fake.handler())
	t.Cleanup(srv.Close)
	cfg.Engine.URL = srv.URL
	cfg.Engine.Kind = config.EngineOllama
	if cfg.Model.Name == "" {
		cfg.Model.Name = "qwen2.5"
	}
	a := NewAdapter(cfg)
	return a, srv
}

// chatHandler returns an httptest fake whose /api/chat hook delegates to fn.
func chatHandler(fn func(w http.ResponseWriter, r *http.Request)) *fakeOllama {
	return &fakeOllama{
		// We re-use the existing serveMux from fakeOllama.handler(), which
		// only knows the /api/chat path through this raw addition.
	}
}

// chatFakeServer returns an httptest server that captures the JSON body
// the adapter forwarded and replies with `frames` as NDJSON. It also
// answers /api/tags so WaitAlive / models translator stays happy.
type chatFake struct {
	frames     []map[string]any
	gotBody    map[string]any
	notStream  bool
	respStatus int
}

func startChatFake(t *testing.T, f *chatFake) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[]}`))
			return
		case "/api/chat":
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &f.gotBody)
			if f.respStatus != 0 {
				w.WriteHeader(f.respStatus)
				return
			}
			if f.notStream {
				if len(f.frames) > 0 {
					b, _ := json.Marshal(f.frames[0])
					_, _ = w.Write(b)
				}
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			for _, fr := range f.frames {
				b, _ := json.Marshal(fr)
				_, _ = w.Write(b)
				_, _ = w.Write([]byte("\n"))
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func TestChat_StreamingHappyPath(t *testing.T) {
	t.Parallel()
	f := &chatFake{frames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "content": "Hello"}, "done": false},
		{"message": map[string]any{"role": "assistant", "content": " world"}, "done": false},
		{"message": map[string]any{"role": "assistant", "content": "!"}, "done": true,
			"prompt_eval_count": 12, "eval_count": 3},
	}}
	url := startChatFake(t, f)
	cfg := config.Config{
		Engine: config.Engine{
			Kind: config.EngineOllama, URL: url,
			Args: config.EngineArgs{Known: map[string]string{
				"num_ctx":        "8192",
				"repeat_penalty": "1.1",
				"repeat_last_n":  "64",
				"keep_alive":     "10m",
			}},
		},
		Model: config.Model{Name: "qwen2.5"},
	}
	a := NewAdapter(cfg)

	body := map[string]any{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"stream":      true,
		"temperature": 0.7,
		"max_tokens":  100,
		"stop":        "<eos>",
	}
	bodyJSON, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(bodyJSON))
	a.handleChat(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type=%q", got)
	}
	out := rec.Body.String()
	if !strings.Contains(out, "data: ") {
		t.Errorf("missing SSE prefix: %q", out)
	}
	if !strings.Contains(out, "Hello") || !strings.Contains(out, "world") {
		t.Errorf("missing content frames: %q", out)
	}
	if !strings.HasSuffix(strings.TrimRight(out, "\n"), "data: [DONE]") {
		t.Errorf("missing [DONE] terminator at end: %q", out[len(out)-50:])
	}

	// Verify upstream body translation: model rewritten, options flattened,
	// keep_alive forwarded.
	if f.gotBody["model"] != "qwen2.5" {
		t.Errorf("model = %v", f.gotBody["model"])
	}
	opts, _ := f.gotBody["options"].(map[string]any)
	if opts["num_ctx"] != float64(8192) {
		t.Errorf("num_ctx=%v", opts["num_ctx"])
	}
	if opts["repeat_penalty"] != 1.1 {
		t.Errorf("repeat_penalty=%v", opts["repeat_penalty"])
	}
	if opts["num_predict"] != float64(100) {
		t.Errorf("num_predict=%v", opts["num_predict"])
	}
	if opts["temperature"] != 0.7 {
		t.Errorf("temperature=%v", opts["temperature"])
	}
	stops, _ := opts["stop"].([]any)
	if len(stops) != 1 || stops[0] != "<eos>" {
		t.Errorf("stop=%v", opts["stop"])
	}
	if f.gotBody["keep_alive"] != "10m" {
		t.Errorf("keep_alive=%v", f.gotBody["keep_alive"])
	}
}

func TestChat_ForwardsToolsAndToolChoice(t *testing.T) {
	t.Parallel()
	f := &chatFake{notStream: true, frames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "content": "ok"}, "done": true},
	}}
	url := startChatFake(t, f)
	cfg := config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "gemma4-26b"},
	}
	a := NewAdapter(cfg)

	body := map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "get_weather",
				"description": "weather",
				"parameters":  map[string]any{"type": "object"},
			},
		}},
		"tool_choice": "auto",
	}
	bodyJSON, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyJSON))
	a.handleChat(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	tools, ok := f.gotBody["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools=%v, want one forwarded tool", f.gotBody["tools"])
	}
	if f.gotBody["tool_choice"] != "auto" {
		t.Errorf("tool_choice=%v", f.gotBody["tool_choice"])
	}
}

func TestChat_BlockContentFlattened(t *testing.T) {
	t.Parallel()
	f := &chatFake{notStream: true, frames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "content": "ok"}, "done": true},
	}}
	url := startChatFake(t, f)
	cfg := config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "gemma4-26b"},
	}
	a := NewAdapter(cfg)

	body := map[string]any{
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{openaiKeyType: openaiBlockText, openaiKeyText: "part1 "},
				{openaiKeyType: openaiBlockInputText, openaiKeyText: "part2"},
			},
		}},
	}
	bodyJSON, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyJSON))
	a.handleChat(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, ok := f.gotBody["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("messages=%v", f.gotBody["messages"])
	}
	m, _ := msgs[0].(map[string]any)
	if m["content"] != "part1 part2" {
		t.Errorf("flatten=%v", m["content"])
	}
}

func TestChat_StreamingToolCalls(t *testing.T) {
	t.Parallel()
	f := &chatFake{frames: []map[string]any{
		{"message": map[string]any{
			"role": "assistant", "content": "calling",
			"tool_calls": []map[string]any{{
				"id": "call_1", "type": "function",
				"function": map[string]any{"name": "get_weather", "arguments": `{"city":"Paris"}`},
			}},
		}, "done": true},
	}}
	url := startChatFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "gemma4-26b"},
	})

	body := map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "weather?"}},
		"stream":   true,
	}
	bodyJSON, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyJSON)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var chunks []OpenAIChatResponse
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var c OpenAIChatResponse
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &c); err != nil {
			t.Fatalf("decode chunk: %v", err)
		}
		chunks = append(chunks, c)
	}
	if len(chunks) < 2 {
		t.Fatalf("chunks=%d, want tool_calls + stop", len(chunks))
	}
	toolChunk := chunks[len(chunks)-2]
	if len(toolChunk.Choices[0].Delta.ToolCalls) != 1 {
		t.Fatalf("tool_calls=%+v", toolChunk.Choices[0].Delta.ToolCalls)
	}
	if toolChunk.Choices[0].Delta.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool name=%q", toolChunk.Choices[0].Delta.ToolCalls[0].Function.Name)
	}
	if chunks[len(chunks)-1].Choices[0].FinishReason != finishReasonToolCalls {
		t.Errorf("finish_reason=%q", chunks[len(chunks)-1].Choices[0].FinishReason)
	}
}

func TestChat_ToolCallArgumentsObjectNormalized(t *testing.T) {
	t.Parallel()
	f := &chatFake{notStream: true, frames: []map[string]any{
		{"message": map[string]any{
			"role": "assistant", "content": "",
			"tool_calls": []map[string]any{{
				"id": "call_1", "type": "function",
				"function": map[string]any{
					"name":      "get_weather",
					"arguments": map[string]any{"city": "北京"},
				},
			}},
		}, "done": true, "prompt_eval_count": 10, "eval_count": 5},
	}}
	url := startChatFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "gemma4-26b"},
	})

	body := map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "北京未来一周天气"}},
	}
	bodyJSON, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyJSON)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var wire struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					Function struct {
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &wire); err != nil {
		t.Fatalf("decode: %v", err)
	}
	args := wire.Choices[0].Message.ToolCalls[0].Function.Arguments
	var parsed map[string]any
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("arguments not valid JSON object string: %q err=%v", args, err)
	}
	if parsed["city"] != "北京" {
		t.Errorf("city=%v", parsed["city"])
	}
}

func TestChat_MultiTurnToolReplayArgumentsObject(t *testing.T) {
	t.Parallel()
	f := &chatFake{notStream: true, frames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "content": "晴，25°C"}, "done": true},
	}}
	url := startChatFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "gemma4-26b"},
	})

	body := map[string]any{
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name": "get_weather",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
				},
			},
		}},
		"messages": []map[string]any{
			{"role": "user", "content": "北京未来一周天气"},
			{"role": "assistant", "content": "", "tool_calls": []map[string]any{{
				"id": "call_1", "type": "function",
				"function": map[string]any{
					"name":      "get_weather",
					"arguments": `{"city":"北京"}`,
				},
			}}},
			{"role": "tool", "tool_call_id": "call_1", "content": "晴，25°C"},
		},
	}
	bodyJSON, _ := json.Marshal(body)
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(bodyJSON)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	msgs, _ := f.gotBody["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages=%v", f.gotBody["messages"])
	}
	asst, _ := msgs[1].(map[string]any)
	tcs, _ := asst[openaiKeyToolCalls].([]any)
	if len(tcs) != 1 {
		t.Fatalf("tool_calls=%v", asst[openaiKeyToolCalls])
	}
	tc, _ := tcs[0].(map[string]any)
	fn, _ := tc[openaiKeyFunction].(map[string]any)
	args, ok := fn[openaiKeyArguments].(map[string]any)
	if !ok {
		t.Fatalf("arguments must be object for Ollama, got %T: %v", fn[openaiKeyArguments], fn[openaiKeyArguments])
	}
	if args["city"] != "北京" {
		t.Errorf("city=%v", args["city"])
	}
}

// TestChat_StreamingFlushesIncrementally proves the adapter forwards SSE
// frames as soon as upstream produces NDJSON, rather than buffering the
// whole stream. We arrange an upstream server that flushes frame 1, then
// blocks on a channel before sending frame 2, and a real HTTP client (not
// httptest.NewRecorder, which can mask flush semantics) reading from the
// adapter's handler. The client must observe the first SSE frame *before*
// it releases the channel that lets upstream emit the second frame —
// otherwise time-to-first-token would be the time of the whole stream,
// silently breaking the v1.0 latency contract.
func TestChat_StreamingFlushesIncrementally(t *testing.T) {
	t.Parallel()
	allowSecond := make(chan struct{})

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[]}`))
			return
		case "/api/chat":
			w.Header().Set("Content-Type", "application/x-ndjson")
			flusher, _ := w.(http.Flusher)
			frame := func(content string, done bool) {
				m := map[string]any{
					"message": map[string]any{"role": "assistant", "content": content},
					"done":    done,
				}
				b, _ := json.Marshal(m)
				_, _ = w.Write(b)
				_, _ = w.Write([]byte("\n"))
				if flusher != nil {
					flusher.Flush()
				}
			}
			frame("first", false)
			select {
			case <-allowSecond:
			case <-r.Context().Done():
				return
			}
			frame("second", false)
			frame("", true)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(upstream.Close)

	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: upstream.URL},
		Model:  config.Model{Name: "qwen2.5"},
	})

	// Wrap the adapter handler in its own HTTP server so the client uses
	// real network I/O. httptest.NewRecorder buffers everything until the
	// handler returns, hiding the flush behaviour we want to verify.
	front := httptest.NewServer(http.HandlerFunc(a.handleChat))
	t.Cleanup(front.Close)

	body := strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	req, err := http.NewRequest(http.MethodPost, front.URL+"/v1/chat/completions", body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}

	// Read until we see "first" content. Use a per-line scanner with a
	// bounded deadline; if the adapter buffers the whole stream this
	// times out without ever seeing the first frame.
	type readResult struct {
		seenFirst bool
		err       error
	}
	gotFirst := make(chan readResult, 1)
	reader := bufio.NewReader(resp.Body)
	go func() {
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			line, err := reader.ReadString('\n')
			if err != nil && err != io.EOF {
				gotFirst <- readResult{err: err}
				return
			}
			if strings.Contains(line, "first") {
				gotFirst <- readResult{seenFirst: true}
				return
			}
			if err == io.EOF {
				gotFirst <- readResult{err: fmt.Errorf("EOF before first frame")}
				return
			}
		}
		gotFirst <- readResult{err: fmt.Errorf("timeout before first frame")}
	}()

	select {
	case r := <-gotFirst:
		if r.err != nil {
			t.Fatalf("did not see first SSE frame: %v", r.err)
		}
		if !r.seenFirst {
			t.Fatal("scanner returned without first frame")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first SSE frame (adapter may be buffering)")
	}

	// Now release the upstream and drain to [DONE].
	close(allowSecond)
	rest, err := io.ReadAll(reader)
	if err != nil && !errors.Is(err, io.EOF) {
		t.Fatalf("drain: %v", err)
	}
	if !strings.Contains(string(rest), "second") {
		t.Errorf("missing second frame: %q", rest)
	}
	if !strings.Contains(string(rest), "[DONE]") {
		t.Errorf("missing [DONE] terminator: %q", rest)
	}
}

// TestChat_TranslatesAliasToOllamaTag is the load-bearing assertion for
// the v1.0.1 OLLAMA_MODEL split: clients address the OpenAI-facing alias
// (Model.Name) while the daemon receives the canonical Ollama Library
// tag (Source.OllamaModel). The response object's "model" field carries
// the alias back to the client unchanged so SDKs that pin on it (OpenAI
// Python client) keep working.
func TestChat_TranslatesAliasToOllamaTag(t *testing.T) {
	t.Parallel()
	f := &chatFake{
		notStream: true,
		frames: []map[string]any{{
			"message":           map[string]any{"role": "assistant", "content": "ok"},
			"done":              true,
			"prompt_eval_count": 1,
			"eval_count":        1,
		}},
	}
	url := startChatFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "qwen-alias"},
		Sources: []config.ModelSource{{
			Index:     1,
			Kind:      config.KindOllama,
			Role:      config.RoleMain,
			OllamaTag: "qwen2.5:7b-instruct",
		}},
	})
	body := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}

	if got := f.gotBody["model"]; got != "qwen2.5:7b-instruct" {
		t.Errorf("upstream model = %v, want %q", got, "qwen2.5:7b-instruct")
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got := resp["model"]; got != "qwen-alias" {
		t.Errorf("response model = %v, want %q (alias must be preserved)",
			got, "qwen-alias")
	}
}

func TestChat_NonStreaming(t *testing.T) {
	t.Parallel()
	f := &chatFake{
		notStream: true,
		frames: []map[string]any{{
			"message":           map[string]any{"role": "assistant", "content": "answer"},
			"done":              true,
			"prompt_eval_count": 5,
			"eval_count":        2,
		}},
	}
	url := startChatFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "m"},
	})
	body := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var out map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	choices, _ := out["choices"].([]any)
	if len(choices) != 1 {
		t.Fatalf("choices=%v", choices)
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "answer" {
		t.Errorf("content=%v", msg["content"])
	}
	usage := out["usage"].(map[string]any)
	if usage["total_tokens"] != float64(7) {
		t.Errorf("total_tokens=%v", usage["total_tokens"])
	}
}

func TestChat_BadJSON(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: "http://x"}, Model: config.Model{Name: "m"}})
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader("not json")))
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d", rec.Code)
	}
}

func TestChat_UpstreamUnreachable(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{
		Engine: config.Engine{URL: srv.URL},
		Model:  config.Model{Name: "m"},
	})
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[]}`)))
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

// TestChat_ThinkTogglePassthrough exercises every wire-level convention
// OpenAI clients use to toggle a reasoning model's thinking mode. All
// four (top-level `think`, `chat_template_kwargs.enable_thinking`,
// `extra_body.chat_template_kwargs.enable_thinking`, `reasoning_effort`)
// must resolve into Ollama's native top-level `think` field on the
// outbound /api/chat body — otherwise OpenWebUI's "Thinking: off"
// toggle is silently dropped and the model stays in default-think mode.
func TestChat_ThinkTogglePassthrough(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		request map[string]any
		want    any // nil means: must NOT appear in outbound body
	}{
		{
			name: "top_level_think_false",
			request: map[string]any{
				"messages": []map[string]string{{"role": "user", "content": "hi"}},
				"think":    false,
			},
			want: false,
		},
		{
			name: "top_level_think_true",
			request: map[string]any{
				"messages": []map[string]string{{"role": "user", "content": "hi"}},
				"think":    true,
			},
			want: true,
		},
		{
			name: "chat_template_kwargs_enable_thinking_false",
			request: map[string]any{
				"messages": []map[string]string{{"role": "user", "content": "hi"}},
				"chat_template_kwargs": map[string]any{
					"enable_thinking": false,
				},
			},
			want: false,
		},
		{
			name: "extra_body_chat_template_kwargs_enable_thinking_false",
			request: map[string]any{
				"messages": []map[string]string{{"role": "user", "content": "hi"}},
				"extra_body": map[string]any{
					"chat_template_kwargs": map[string]any{
						"enable_thinking": false,
					},
				},
			},
			want: false,
		},
		{
			name: "reasoning_effort_none_maps_to_false",
			request: map[string]any{
				"messages":         []map[string]string{{"role": "user", "content": "hi"}},
				"reasoning_effort": "none",
			},
			want: false,
		},
		{
			name: "reasoning_effort_low_maps_to_level",
			request: map[string]any{
				"messages":         []map[string]string{{"role": "user", "content": "hi"}},
				"reasoning_effort": "low",
			},
			want: "low",
		},
		{
			name: "reasoning_effort_max_maps_to_level",
			request: map[string]any{
				"messages":         []map[string]string{{"role": "user", "content": "hi"}},
				"reasoning_effort": "max",
			},
			want: "max",
		},
		{
			// OpenAI has this level, Ollama does not. The client asked to
			// think, so we keep thinking on rather than dropping it.
			name: "reasoning_effort_unknown_level_falls_back_to_true",
			request: map[string]any{
				"messages":         []map[string]string{{"role": "user", "content": "hi"}},
				"reasoning_effort": "minimal",
			},
			want: true,
		},
		{
			name: "no_toggle_means_field_absent",
			request: map[string]any{
				"messages": []map[string]string{{"role": "user", "content": "hi"}},
			},
			want: nil,
		},
		{
			name: "top_level_think_wins_over_chat_template_kwargs",
			request: map[string]any{
				"messages": []map[string]string{{"role": "user", "content": "hi"}},
				"think":    false,
				"chat_template_kwargs": map[string]any{
					"enable_thinking": true,
				},
			},
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := translateChatRequest(tc.request, config.Config{
				Model: config.Model{Name: "qwen3"},
			})
			actual, present := got["think"]
			switch tc.want {
			case nil:
				if present {
					t.Errorf("think should be absent, got %v", actual)
				}
			default:
				if !present {
					t.Errorf("think missing from outbound body")
				}
				if actual != tc.want {
					t.Errorf("think = %v, want %v", actual, tc.want)
				}
			}
		})
	}
}

// TestResolveThinkPreference is a focused unit test on the helper, complementing
// the integration test above with explicit (value, ok) assertions including
// negative cases that the integration test only verifies indirectly.
func TestResolveThinkPreference(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		in      map[string]any
		wantVal any
		wantOk  bool
	}{
		{"empty", map[string]any{}, nil, false},
		{"think_not_bool", map[string]any{"think": "false"}, nil, false},
		{"effort_empty_string", map[string]any{"reasoning_effort": ""}, nil, false},
		{"effort_whitespace_off", map[string]any{"reasoning_effort": "  off  "}, false, true},
		{"effort_disabled_uppercase", map[string]any{"reasoning_effort": "DISABLED"}, false, true},
		{"effort_none_is_the_off_switch", map[string]any{"reasoning_effort": "none"}, false, true},
		{"effort_high", map[string]any{"reasoning_effort": "high"}, "high", true},
		{"effort_medium_uppercase_normalized", map[string]any{"reasoning_effort": "MEDIUM"}, "medium", true},
		{"effort_whitespace_level_trimmed", map[string]any{"reasoning_effort": " low "}, "low", true},
		{"effort_unknown_level", map[string]any{"reasoning_effort": "xhigh"}, true, true},
		{"ctk_present_but_no_enable_thinking", map[string]any{
			"chat_template_kwargs": map[string]any{"other_kwarg": true},
		}, nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gotVal, gotOk := resolveThinkPreference(tc.in)
			if gotOk != tc.wantOk || gotVal != tc.wantVal {
				t.Errorf("got=(%v,%v) want=(%v,%v)", gotVal, gotOk, tc.wantVal, tc.wantOk)
			}
		})
	}
}

// TestChat_ThinkLevelDowngradedOnBadRequest covers the daemon that
// predates the thinking levels: it answers `"think":"high"` with 400, and
// the adapter has to retry with the boolean rather than failing a request
// the engine is perfectly able to serve.
func TestChat_ThinkLevelDowngradedOnBadRequest(t *testing.T) {
	t.Parallel()
	var (
		mu     sync.Mutex
		bodies []map[string]any
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		b, _ := io.ReadAll(r.Body)
		var got map[string]any
		_ = json.Unmarshal(b, &got)
		mu.Lock()
		bodies = append(bodies, got)
		mu.Unlock()
		if _, isLevel := got["think"].(string); isLevel {
			http.Error(w, `{"error":"invalid think value"}`, http.StatusBadRequest)
			return
		}
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"ok"},"done":true}`))
	}))
	t.Cleanup(srv.Close)

	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: srv.URL},
		Model:  config.Model{Name: "gpt-oss"},
	})
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 2 {
		t.Fatalf("upstream calls=%d, want 2 (level then downgrade)", len(bodies))
	}
	if got := bodies[0]["think"]; got != "high" {
		t.Errorf("first attempt think=%v, want %q", got, "high")
	}
	if got := bodies[1]["think"]; got != true {
		t.Errorf("retry think=%v, want true", got)
	}
	// The retry must carry the rest of the request unchanged.
	if msgs, _ := bodies[1]["messages"].([]any); len(msgs) != 1 {
		t.Errorf("retry messages=%v", bodies[1]["messages"])
	}
}

// TestDowngradeThinkLevel pins the trigger to exactly one situation. A 400
// on a body that never asked for a level, or any other status, means
// something else is wrong with the request, and retrying would replace one
// error with another while charging the user a second round trip.
func TestDowngradeThinkLevel(t *testing.T) {
	t.Parallel()
	badRequest := &StatusError{Path: "/api/chat", Status: http.StatusBadRequest, Body: "nope"}
	cases := []struct {
		name string
		err  error
		body map[string]any
		want bool
	}{
		{"level_and_400", badRequest, map[string]any{keyThink: "high"}, true},
		{"level_but_500", &StatusError{Path: "/api/chat", Status: 500}, map[string]any{keyThink: "high"}, false},
		{"bool_and_400", badRequest, map[string]any{keyThink: true}, false},
		{"absent_and_400", badRequest, map[string]any{}, false},
		{"transport_error", errors.New("connection refused"), map[string]any{keyThink: "high"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, ok := downgradeThinkLevel(tc.err, tc.body)
			if ok != tc.want {
				t.Fatalf("ok=%v want %v", ok, tc.want)
			}
			if !ok {
				return
			}
			var decoded map[string]any
			if err := json.Unmarshal(got, &decoded); err != nil {
				t.Fatalf("retry body not JSON: %v", err)
			}
			if decoded[keyThink] != true {
				t.Errorf("retry think=%v, want true", decoded[keyThink])
			}
			// The original body must not be mutated: the caller still
			// needs it to describe what was actually attempted.
			if tc.body[keyThink] != "high" {
				t.Errorf("source body mutated: think=%v", tc.body[keyThink])
			}
		})
	}
}

// TestChat_StreamingThinkingFrames is the regression test for the
// "OpenWebUI sees infinite empty deltas" report. Pre-fix, Ollama frames
// shaped {"message":{"thinking":"...","content":""}} were decoded into
// a struct that had no Thinking field, so the adapter forwarded them
// as role-only deltas. The fix surfaces them via delta.reasoning_content
// and drops frames that are genuinely empty (no content AND no thinking).
func TestChat_StreamingThinkingFrames(t *testing.T) {
	t.Parallel()
	f := &chatFake{frames: []map[string]any{
		{"message": map[string]any{"role": "assistant", "thinking": "Let me ", "content": ""}, "done": false},
		{"message": map[string]any{"role": "assistant", "thinking": "think.", "content": ""}, "done": false},
		// Genuinely empty heartbeat frame — must be dropped, not forwarded.
		{"message": map[string]any{"role": "assistant", "thinking": "", "content": ""}, "done": false},
		{"message": map[string]any{"role": "assistant", "content": "Paris"}, "done": false},
		{"message": map[string]any{"role": "assistant", "content": "."}, "done": true,
			"prompt_eval_count": 5, "eval_count": 2},
	}}
	url := startChatFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "qwen3.5-0.8b"},
	})
	body := []byte(`{"messages":[{"role":"user","content":"capital of france?"}],"stream":true}`)
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}

	// Parse every "data: {...}" line into a typed chunk and inspect deltas.
	var chunks []OpenAIChatResponse
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		var c OpenAIChatResponse
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("decode chunk %q: %v", payload, err)
		}
		chunks = append(chunks, c)
	}

	// Expectations:
	//   chunks[0]: role="assistant", reasoning_content="Let me " (first emit carries role)
	//   chunks[1]: reasoning_content="think." (no role)
	//   chunks[2]: content="Paris" (no role; empty-empty heartbeat dropped)
	//   chunks[3]: content="."
	//   chunks[4]: empty delta + finish_reason="stop"
	if len(chunks) != 5 {
		t.Fatalf("got %d chunks, want 5; raw=%s", len(chunks), rec.Body.String())
	}
	d0 := chunks[0].Choices[0].Delta
	if d0.Role != roleAssistant || d0.ReasoningContent != "Let me " || d0.Content != "" {
		t.Errorf("chunks[0].delta = %+v", d0)
	}
	d1 := chunks[1].Choices[0].Delta
	if d1.Role != "" || d1.ReasoningContent != "think." {
		t.Errorf("chunks[1].delta = %+v (role must be empty on non-first frames)", d1)
	}
	d2 := chunks[2].Choices[0].Delta
	if d2.Role != "" || d2.Content != "Paris" || d2.ReasoningContent != "" {
		t.Errorf("chunks[2].delta = %+v", d2)
	}
	if chunks[4].Choices[0].FinishReason != finishReasonStop {
		t.Errorf("chunks[4].finish_reason = %q", chunks[4].Choices[0].FinishReason)
	}
	if d := chunks[4].Choices[0].Delta; d == nil || d.Role != "" || d.Content != "" || d.ReasoningContent != "" {
		t.Errorf("chunks[4].delta should be empty, got %+v", d)
	}

	// Verify the dropped empty heartbeat did NOT manifest as a role-only chunk.
	for i, c := range chunks {
		d := c.Choices[0].Delta
		if d == nil {
			continue
		}
		isFinal := c.Choices[0].FinishReason != ""
		if !isFinal && d.Content == "" && d.ReasoningContent == "" && d.Role == "" {
			t.Errorf("chunks[%d] is an empty heartbeat that should have been dropped: %+v", i, d)
		}
	}
}

// TestChat_NonStreamingSurfacesThinking proves the buffered path also
// carries the thinking trace through as reasoning_content, not only the
// streaming path.
func TestChat_NonStreamingSurfacesThinking(t *testing.T) {
	t.Parallel()
	f := &chatFake{
		notStream: true,
		frames: []map[string]any{{
			"message": map[string]any{
				"role":     "assistant",
				"content":  "Paris.",
				"thinking": "The capital of France is well-known.",
			},
			"done":              true,
			"prompt_eval_count": 5,
			"eval_count":        2,
		}},
	}
	url := startChatFake(t, f)
	a := NewAdapter(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: url},
		Model:  config.Model{Name: "qwen3.5-0.8b"},
	})
	body := []byte(`{"messages":[{"role":"user","content":"capital of france?"}]}`)
	rec := httptest.NewRecorder()
	a.handleChat(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d", rec.Code)
	}
	var resp OpenAIChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	msg := resp.Choices[0].Message
	if msg.Content != "Paris." {
		t.Errorf("content = %q", msg.Content)
	}
	if msg.ReasoningContent != "The capital of France is well-known." {
		t.Errorf("reasoning_content = %q", msg.ReasoningContent)
	}
}

// TestAggregateNDJSONFrames covers the rare fallback where Ollama emits
// NDJSON despite stream=false. Pre-fix, collectChatResponse called
// lastNDJSONFrame and the user-facing content was empty (the terminal
// frame's content is "" by convention). The aggregating helper sums
// content and thinking across frames and adopts the last frame's
// scalar metadata.
func TestAggregateNDJSONFrames(t *testing.T) {
	t.Parallel()
	data := []byte(`{"message":{"role":"assistant","thinking":"think1","content":"a"},"done":false}` + "\n" +
		`{"message":{"role":"assistant","thinking":"think2","content":"b"},"done":false}` + "\n" +
		`{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":7,"eval_count":3}` + "\n")
	got := aggregateNDJSONFrames(data)
	if got.Message.Content != "ab" {
		t.Errorf("content aggregation: got %q want %q", got.Message.Content, "ab")
	}
	if got.Message.Thinking != "think1think2" {
		t.Errorf("thinking aggregation: got %q", got.Message.Thinking)
	}
	if !got.Done {
		t.Errorf("Done should be true (taken from terminal frame)")
	}
	if got.PromptEvalCount != 7 || got.EvalCount != 3 {
		t.Errorf("usage = (%d,%d)", got.PromptEvalCount, got.EvalCount)
	}
}

func TestNormalizeStop(t *testing.T) {
	t.Parallel()
	if got := normalizeStop("a"); len(got) != 1 || got[0] != "a" {
		t.Errorf("string: %v", got)
	}
	if got := normalizeStop([]any{"a", "b"}); len(got) != 2 {
		t.Errorf("array: %v", got)
	}
	if got := normalizeStop(42); got != nil {
		t.Errorf("int: %v", got)
	}
}

func TestLastNDJSONFrame(t *testing.T) {
	t.Parallel()
	data := []byte(`{"message":{"role":"assistant","content":"a"},"done":false}` + "\n" +
		`{"message":{"role":"assistant","content":"b"},"done":true,"eval_count":2}` + "\n")
	got := lastNDJSONFrame(data)
	if got.Message.Content != "b" || !got.Done || got.EvalCount != 2 {
		t.Errorf("got=%+v", got)
	}
}

func TestHandlerRouter(t *testing.T) {
	t.Parallel()
	a, _ := newTestAdapter(t, &fakeOllama{}, config.Config{
		Model: config.Model{Name: "m"},
	})
	h := a.OpenAIHandler(config.Config{})

	// /v1/responses is proxied to the daemon's OpenAI surface.
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"m","input":"hi"}`)))
	if rec.Code != http.StatusOK {
		t.Errorf("/v1/responses status=%d", rec.Code)
	}
	// /api/chat/completions alias is mounted (will 502 because the fake
	// doesn't have /api/chat hooked).
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/api/chat/completions",
		strings.NewReader(`{"messages":[]}`)))
	if rec.Code == http.StatusNotFound {
		t.Errorf("alias should be mounted, got 404")
	}
	_ = chatHandler
	_ = context.Background
}

// BenchmarkChatTranslate measures translateChatRequest on a typical
// inbound payload: messages array, sampling knobs, plus inference-time
// overrides from Config. The translation is on the hot path of every
// /v1/chat/completions request, so even sub-microsecond regressions
// show up at high QPS. Pure CPU, no allocations outside the function
// being benchmarked.
//
// Run locally with: go test -bench=BenchmarkChatTranslate -benchtime=3s ./internal/adapter/ollama/
func BenchmarkChatTranslate(b *testing.B) {
	openai := map[string]any{
		"messages": []map[string]string{
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "What is the capital of France?"},
		},
		"stream":            true,
		"temperature":       0.7,
		"top_p":             0.95,
		"max_tokens":        100,
		"frequency_penalty": 0.0,
		"presence_penalty":  0.0,
		"seed":              42,
		"stop":              []any{"<eos>"},
	}
	cfg := config.Config{
		Engine: config.Engine{
			Kind: config.EngineOllama, URL: "http://x",
			Args: config.EngineArgs{Known: map[string]string{
				"num_ctx":        "8192",
				"repeat_penalty": "1.1",
				"repeat_last_n":  "64",
				"keep_alive":     "10m",
			}},
		},
		Model: config.Model{Name: "qwen2.5", Type: config.ModelChat},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = translateChatRequest(openai, cfg)
	}
}
