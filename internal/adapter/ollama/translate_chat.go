package ollama

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/reasoning"
)

// handleChat is the /v1/chat/completions translator. The wire-level
// behaviour is a field map plus an NDJSON → SSE conversion. The
// implementation strategy:
//
//  1. read & decode the OpenAI body into a generic map so we can preserve
//     fields we don't translate (response_format etc. log as warnings,
//     but don't fail the request);
//  2. build the Ollama /api/chat body in-place;
//  3. POST the body; if stream==true read NDJSON line-by-line and emit
//     SSE chunks; otherwise read the single JSON response and convert
//     to OpenAI shape.
//
// Tools / tool_choice are forwarded unchanged when present — Ollama
// 0.1.45+ accepts OpenAI-shape tools on /api/chat (same as /v1/messages).
func (a *Adapter) handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1 MiB body cap
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "bad_request", "read body: "+err.Error())
		return
	}
	defer r.Body.Close()

	var openaiReq map[string]any
	if err := json.Unmarshal(body, &openaiReq); err != nil {
		writeJSONError(w, http.StatusBadRequest, "bad_request", "invalid JSON: "+err.Error())
		return
	}

	stream, _ := openaiReq["stream"].(bool)
	ollamaReq := translateChatRequest(openaiReq, a.cfg)

	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "internal_error",
			"marshal upstream: "+err.Error())
		return
	}

	resp, err := a.client.PostJSON(r.Context(), "/api/chat", reqBody)
	if err != nil {
		// A daemon predating the thinking levels rejects `"think":"low"`
		// with 400. Retry once with the boolean rather than failing a
		// request the engine can serve, losing only the level.
		if retryBody, ok := downgradeThinkLevel(err, ollamaReq); ok {
			resp, err = a.client.PostJSON(r.Context(), "/api/chat", retryBody)
		}
	}
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "adapter_unreachable", err.Error())
		return
	}
	defer resp.Body.Close()

	if stream {
		streamChatResponse(w, resp.Body, a.cfg.Model.Name)
		return
	}
	collectChatResponse(w, resp.Body, a.cfg.Model.Name)
}

// translateChatRequest maps OpenAI body fields onto the Ollama /api/chat
// JSON. The output map is intentionally minimal — only fields we know
// Ollama understands — so unknown OpenAI fields don't leak through and
// confuse the daemon.
//
// The "model" field is the upstream daemon tag (cfg.PrimaryOllamaTag(),
// which walks Sources for the ollama:// entry and falls back to
// MODEL_NAME); the alias the client sent stays opaque to the daemon
// and is restored on the response side.
func translateChatRequest(openai map[string]any, cfg config.Config) map[string]any {
	upstreamName := cfg.PrimaryOllamaTag()
	out := map[string]any{
		keyModel:  upstreamName,
		keyStream: false,
	}
	if v, ok := openai[keyStream].(bool); ok {
		out[keyStream] = v
	}
	if v, ok := openai["messages"]; ok {
		out["messages"] = normalizeOpenAIChatMessages(v)
	}

	if v, ok := openai["tools"]; ok {
		out["tools"] = v
	}
	if v, ok := openai["tool_choice"]; ok {
		out["tool_choice"] = v
	}

	options := map[string]any{}
	mapNumberOption(options, openai, "temperature", "temperature")
	mapNumberOption(options, openai, "top_p", "top_p")
	mapNumberOption(options, openai, "max_tokens", "num_predict")
	mapNumberOption(options, openai, "frequency_penalty", "frequency_penalty")
	mapNumberOption(options, openai, "presence_penalty", "presence_penalty")
	mapNumberOption(options, openai, "seed", "seed")

	// stop may be string or array; Ollama always wants array.
	if stop, ok := openai["stop"]; ok {
		options["stop"] = normalizeStop(stop)
	}

	applyInferenceInjection(out, options, cfg)
	if len(options) > 0 {
		out["options"] = options
	}

	// Reasoning/thinking toggle. OpenAI clients address this knob through
	// several conventions; we resolve them all into Ollama's native
	// top-level `think`.
	//
	// Priority — highest wins, mirroring how operators expect
	// "more-specific overrides less-specific" to behave:
	//
	//	1. top-level "think"               (ollama-native passthrough)
	//	2. top-level "chat_template_kwargs.enable_thinking"
	//	                                   (vLLM/SGLang Qwen-3 hint)
	//	3. "extra_body.chat_template_kwargs.enable_thinking"
	//	                                   (OpenAI Python SDK wrapping
	//	                                    of the same hint)
	//	4. "reasoning_effort"              (OpenAI o-series)
	//
	// Without this translation, OpenWebUI's "Thinking: off" toggle is
	// silently dropped (the field is not on the previous whitelist),
	// the model stays in default thinking mode, and the upstream emits
	// `message.thinking="…", message.content=""` frames that look like
	// an infinite stream of empty deltas to the client. See the
	// matching response-side fix in streamChatResponse.
	if think, ok := resolveThinkPreference(openai); ok {
		out[keyThink] = think
	}

	return out
}

// keyThink is Ollama's native reasoning field on /api/chat. Its value is
// either a boolean or a level string; see resolveThinkPreference.
const keyThink = "think"

// thinkLevels are the level strings Ollama's native `think` accepts,
// spelled the same as OpenAI's reasoning_effort. `none` is deliberately
// absent: /api/chat rejects it with 400, and the off switch on that
// endpoint is `think:false`.
var thinkLevels = map[string]struct{}{
	reasoning.EffortLow:    {},
	reasoning.EffortMedium: {},
	reasoning.EffortHigh:   {},
	reasoning.EffortMax:    {},
}

// resolveThinkPreference scans the OpenAI request body for any of the
// reasoning-toggle conventions clients use in the wild and returns the
// value for Ollama's native `think` plus an "ok" indicating whether any
// was found.
//
// The value is a bool for the toggle conventions and a level string when
// the client sent a reasoning_effort Ollama also has a level for. The
// levels matter: gpt-oss reads low/medium/high as three different trace
// lengths and cannot be turned off at all, so collapsing them into
// `true` discards the only control that model has.
//
// Returning (nil, false) when no toggle is present is deliberate:
// callers must not infer a default. The default lives in the model's
// own chat template; injecting `think:false` unconditionally would
// override deliberately-thinking deployments and silently change
// behaviour for clients that haven't opted in.
func resolveThinkPreference(openai map[string]any) (any, bool) {
	if v, ok := openai[keyThink].(bool); ok {
		return v, true
	}
	if ctk, ok := openai["chat_template_kwargs"].(map[string]any); ok {
		if v, ok := ctk["enable_thinking"].(bool); ok {
			return v, true
		}
	}
	if eb, ok := openai["extra_body"].(map[string]any); ok {
		if ctk, ok := eb["chat_template_kwargs"].(map[string]any); ok {
			if v, ok := ctk["enable_thinking"].(bool); ok {
				return v, true
			}
		}
	}
	if eff, ok := openai[reasoning.Param].(string); ok {
		norm := reasoning.Normalize(eff)
		switch {
		case norm == "":
			// Field present but empty: treat as not-set.
		case reasoning.IsOff(norm):
			return false, true
		case isThinkLevel(norm):
			return norm, true
		default:
			// A level this daemon has no spelling for (OpenAI's
			// "minimal", llama.cpp's "xhigh"). The client asked to
			// think; honour that much rather than dropping it.
			return true, true
		}
	}
	return nil, false
}

// isThinkLevel reports whether norm is a level Ollama's native `think`
// accepts verbatim.
func isThinkLevel(norm string) bool {
	_, ok := thinkLevels[norm]
	return ok
}

// downgradeThinkLevel rebuilds the outbound body with a boolean `think`
// when err is a 400 and the body carried a level string. Returns ok=false
// for every other failure, which the caller surfaces unchanged.
//
// The narrow trigger is the point: a 400 from a daemon that does
// understand levels means something else about the request is wrong, and
// retrying it with a boolean would only replace one error with another.
// This costs one wasted round trip on such requests and buys thinking
// levels on daemons that predate them, with no version probe to keep in
// sync with upstream.
func downgradeThinkLevel(err error, body map[string]any) ([]byte, bool) {
	var statusErr *StatusError
	if !errors.As(err, &statusErr) || statusErr.Status != http.StatusBadRequest {
		return nil, false
	}
	if _, isLevel := body[keyThink].(string); !isLevel {
		return nil, false
	}
	retry := make(map[string]any, len(body))
	for k, v := range body {
		retry[k] = v
	}
	retry[keyThink] = true
	out, marshalErr := json.Marshal(retry)
	if marshalErr != nil {
		return nil, false
	}
	return out, true
}

// mapNumberOption forwards a numeric field into the Ollama options map
// only if the source contains it AND the value is a JSON number.
func mapNumberOption(out, src map[string]any, srcKey, dstKey string) {
	v, ok := src[srcKey]
	if !ok || v == nil {
		return
	}
	if _, ok := v.(float64); ok {
		out[dstKey] = v
	}
}

// normalizeStop reshapes the OpenAI "stop" field (string | []string | nil)
// into the array form Ollama expects.
func normalizeStop(v any) []any {
	switch s := v.(type) {
	case string:
		return []any{s}
	case []any:
		return s
	default:
		return nil
	}
}

// streamChatResponse pipes Ollama's NDJSON output to the client as SSE:
// one chunk per Ollama frame, "data: [DONE]\n\n" after the terminal frame.
//
// Wire-level contract (matches OpenAI streaming exactly so existing
// clients — OpenAI Python SDK, OpenWebUI, LibreChat — don't need
// special-casing):
//
//   - The very first emitted delta carries `role:"assistant"` plus
//     whatever payload (content / reasoning_content) the underlying
//     frame had. Subsequent deltas omit role.
//   - A frame whose `content` AND `thinking` are both empty AND which
//     does not carry `done:true` is dropped. Without this drop a
//     thinking-capable model in default-think mode floods the client
//     with role-only chunks (the symptom seen in the OpenWebUI hang
//     report — see `Thinking` field comment on ollamaChatChunk).
//   - The terminal frame emits an empty `delta` and
//     `finish_reason:"stop"`, then "data: [DONE]\n\n".
func streamChatResponse(w http.ResponseWriter, body io.Reader, modelName string) {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	if flusher != nil {
		flusher.Flush()
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 4<<20) // raise to 4 MiB; long lines from Ollama do happen
	emit := func(choice OpenAIChatChoice) {
		writeSSEChunk(w, OpenAIChatResponse{
			ID:      id,
			Object:  objectChatCompletionChunk,
			Created: created,
			Model:   modelName,
			Choices: []OpenAIChatChoice{choice},
		})
		if flusher != nil {
			flusher.Flush()
		}
	}
	roleEmitted := false
	var bufferedTools []ollamaToolCall
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var fr ollamaChatChunk
		if err := json.Unmarshal(line, &fr); err != nil {
			continue
		}

		content := fr.Message.Content
		thinking := fr.Message.Thinking
		if len(fr.Message.ToolCalls) > 0 {
			bufferedTools = append(bufferedTools, fr.Message.ToolCalls...)
		}

		// First: if this frame has any payload, flush it as a content
		// delta. We must NOT bundle finish_reason with content in the
		// same chunk — the OpenAI streaming contract requires content
		// deltas and the terminal stop chunk to be separate frames,
		// and bundling has historically tripped OpenAI Python clients.
		if content != "" || thinking != "" {
			delta := &OpenAIChatMessage{
				Content:          content,
				ReasoningContent: thinking,
			}
			if !roleEmitted {
				delta.Role = roleAssistant
				roleEmitted = true
			}
			emit(OpenAIChatChoice{Index: 0, Delta: delta})
		}
		// Second: if this frame closed the stream, emit tool_calls (when
		// present) then the terminal stop chunk and exit.
		if fr.Done {
			if len(bufferedTools) > 0 {
				delta := &OpenAIChatMessage{ToolCalls: openAIWireToolCalls(bufferedTools)}
				if !roleEmitted {
					delta.Role = roleAssistant
				}
				emit(OpenAIChatChoice{Index: 0, Delta: delta})
			}
			finish := finishReasonStop
			if len(bufferedTools) > 0 {
				finish = finishReasonToolCalls
			}
			emit(OpenAIChatChoice{
				Index:        0,
				Delta:        &OpenAIChatMessage{},
				FinishReason: finish,
			})
			break
		}
		// Pure heartbeat / role-only frames (no payload, not done)
		// fall through the two branches above and are silently
		// dropped. Forwarding them would manifest as a stream of
		// empty deltas client side; OpenAI's streaming contract
		// has no notion of a "keepalive" chunk.
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

// collectChatResponse buffers a non-streaming Ollama response and emits
// a single OpenAI-shaped JSON body. Thinking-capable models produce a
// `message.thinking` field alongside `message.content`; both are
// surfaced (the thinking trace becomes `message.reasoning_content`)
// so clients can render reasoning panes independently.
func collectChatResponse(w http.ResponseWriter, body io.Reader, modelName string) {
	data, err := io.ReadAll(io.LimitReader(body, 8<<20))
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "adapter_5xx",
			"read upstream: "+err.Error())
		return
	}
	var fr ollamaChatChunk
	if err := json.Unmarshal(data, &fr); err != nil {
		// Some Ollama versions still emit NDJSON even when stream=false:
		// aggregate every frame's content + thinking and adopt the
		// terminal frame's usage counters. The pre-fix path called
		// lastNDJSONFrame, which silently truncated to the (empty)
		// done frame.
		fr = aggregateNDJSONFrames(data)
	}
	resp := OpenAIChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  objectChatCompletion,
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []OpenAIChatChoice{
			{
				Index: 0,
				Message: &OpenAIChatMessage{
					Role:             roleAssistant,
					Content:          fr.Message.Content,
					ReasoningContent: fr.Message.Thinking,
					ToolCalls:        openAIWireToolCalls(fr.Message.ToolCalls),
				},
				FinishReason: chatFinishReason(fr.Message.ToolCalls),
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     fr.PromptEvalCount,
			CompletionTokens: fr.EvalCount,
			TotalTokens:      fr.PromptEvalCount + fr.EvalCount,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func writeSSEChunk(w io.Writer, chunk any) {
	b, err := json.Marshal(chunk)
	if err != nil {
		return
	}
	_, _ = io.WriteString(w, "data: ")
	_, _ = w.Write(b)
	_, _ = io.WriteString(w, "\n\n")
}

// ollamaChatChunk is the subset of /api/chat output we use.
type ollamaChatChunk struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role    string `json:"role"`
		Content string `json:"content"`
		// Thinking carries the model's reasoning trace for
		// thinking-capable models (Qwen-3, DeepSeek-R1, ...) when
		// the request did NOT set `think:false`. Ollama 0.24+
		// emits it alongside (or before) `content`; pre-1.0.10
		// llm-init lacked this field and the JSON decoder silently
		// dropped it, so the adapter saw `content=""` frames and
		// forwarded them as role-only deltas — manifesting client
		// side as "an infinite stream of empty SSE chunks". Now
		// surfaced as `delta.reasoning_content` on the OpenAI side.
		Thinking string `json:"thinking,omitempty"`
		// ToolCalls is populated by Ollama 0.1.45+ when the model
		// invokes a tool. The shape is identical to OpenAI's
		// tool_calls (id / function {name, arguments}). Older
		// daemons omit it; downstream consumers must treat nil
		// and an empty slice equivalently.
		ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done            bool  `json:"done"`
	PromptEvalCount int64 `json:"prompt_eval_count"`
	EvalCount       int64 `json:"eval_count"`
}

// ollamaToolCall mirrors the OpenAI tool_call shape Ollama emits.
// Ollama 0.1.45 release notes describe the field as "OpenAI-shape
// passthrough"; we decode into the same wire types and pass them
// through to anthropic.ToolCallsToBlocks.
type ollamaToolCall struct {
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name string `json:"name"`
		// Arguments is a JSON object OR a serialised JSON string
		// depending on daemon version. Decode into json.RawMessage
		// so we can re-emit it without losing fidelity.
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

// lastNDJSONFrame extracts the last well-formed JSON object from a
// possibly-NDJSON byte buffer. It is the fallback path for non-streaming
// requests where the daemon happens to send a stream-shaped body.
func lastNDJSONFrame(data []byte) ollamaChatChunk {
	var out ollamaChatChunk
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var fr ollamaChatChunk
		if err := json.Unmarshal([]byte(line), &fr); err == nil {
			out = fr
		}
	}
	return out
}

// aggregateNDJSONFrames sums Message.{Content,Thinking} across all
// frames in a possibly-NDJSON byte buffer and adopts the last frame's
// scalar metadata (Done, PromptEvalCount, EvalCount). It exists for
// the non-streaming path's fallback: if Ollama violates `stream:false`
// and emits NDJSON anyway (observed on some 0.x point releases), the
// per-frame content/thinking deltas must be concatenated rather than
// truncated to the empty done-frame's payload. lastNDJSONFrame is
// preserved for callers that only need the terminal frame's metadata.
func aggregateNDJSONFrames(data []byte) ollamaChatChunk {
	var out ollamaChatChunk
	var content, thinking strings.Builder
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var fr ollamaChatChunk
		if err := json.Unmarshal([]byte(line), &fr); err != nil {
			continue
		}
		content.WriteString(fr.Message.Content)
		thinking.WriteString(fr.Message.Thinking)
		if len(fr.Message.ToolCalls) > 0 {
			out.Message.ToolCalls = fr.Message.ToolCalls
		}
		// Terminal frame supplies the canonical metadata.
		out.Done = fr.Done
		out.Model = fr.Model
		out.CreatedAt = fr.CreatedAt
		out.PromptEvalCount = fr.PromptEvalCount
		out.EvalCount = fr.EvalCount
	}
	out.Message.Role = roleAssistant
	out.Message.Content = content.String()
	out.Message.Thinking = thinking.String()
	return out
}

func chatFinishReason(toolCalls []ollamaToolCall) string {
	if len(toolCalls) > 0 {
		return finishReasonToolCalls
	}
	return finishReasonStop
}
