package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/anthropic"
	"github.com/llm-init/llm-init/internal/config"
)

// messagesBodyCap mirrors the ollama package's cap (16 MiB) so the
// /v1/messages contract is identical across engines. Operators can
// override via `MESSAGES_BODY_CAP`; see the `MessagesBodyCap` package
// variable.
const messagesBodyCap = 16 << 20

// MessagesBodyCap is overridable from main.go after config.Load. See
// the ollama equivalent for the rationale.
var MessagesBodyCap int64 = messagesBodyCap

// Wire-shape JSON keys used by the OpenAI request marshaler. Named
// constants keep goconst quiet about cross-file repetition of these
// well-known field names.
const (
	keyModel     = "model"
	keyMessages  = "messages"
	keyMaxTokens = "max_tokens"
	keyContent   = "content"
	keyRole      = "role"
	keyMessage   = "message"
	keyStream    = "stream"
)

// pathModels is the models-list route probed for liveness/readiness.
const pathModels = "/v1/models"

// AnthropicHandler returns the http.Handler dataplane mounts at
// POST /v1/messages and POST /v1/messages/count_tokens. The
// implementation:
//
//  1. Decodes the Anthropic request.
//  2. Translates to an OpenAI /v1/chat/completions body.
//  3. Sends it directly to the upstream (we do NOT use the existing
//     httputil.ReverseProxy because the response needs reshaping
//     back into Anthropic shape; ReverseProxy is byte-passthrough).
//  4. Buffers (non-stream) or parses SSE (stream) the upstream
//     OpenAI response and emits Anthropic shape to w.
//
// Every proxy backend (vLLM, llama.cpp, SGLang) speaks OpenAI's chat
// completions, so step 3 is identical across them.
//
// /v1/messages/count_tokens is served from the same handler tree
// via a heuristic estimator (chars / 4); the response carries an
// `X-Llm-Init-Token-Estimate: heuristic` header so clients can
// observe the approximation. A future stage may switch to upstream
// /tokenize where available.
func (a *Adapter) AnthropicHandler(_ config.Config) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", a.handleMessages)
	mux.HandleFunc("POST /v1/messages/count_tokens", a.handleCountTokens)
	return mux
}

// handleCountTokens mirrors the Ollama equivalent. We keep the
// duplication local (rather than extracting to a shared helper)
// because future stages may add backend-specific tokenize support
// here (POST {ENGINE_URL}/tokenize on vLLM/SGLang/llama.cpp), and
// the per-adapter file is the natural home for that divergence.
func (a *Adapter) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, MessagesBodyCap+1))
	if err != nil {
		anthropic.WriteError(w, http.StatusBadRequest, anthropic.ErrInvalidRequest,
			"read body: "+err.Error())
		return
	}
	defer r.Body.Close()
	if int64(len(body)) > MessagesBodyCap {
		anthropic.WriteError(w, http.StatusRequestEntityTooLarge, anthropic.ErrInvalidRequest,
			fmt.Sprintf("request body exceeds messages cap (%d bytes)", MessagesBodyCap))
		return
	}
	var req anthropic.CountTokensRequest
	if err := json.Unmarshal(body, &req); err != nil {
		anthropic.WriteError(w, http.StatusBadRequest, anthropic.ErrInvalidRequest,
			"invalid JSON: "+err.Error())
		return
	}
	w.Header().Set(headerContentType, contentTypeJSON)
	w.Header().Set("X-Llm-Init-Token-Estimate", "heuristic")
	_ = json.NewEncoder(w).Encode(anthropic.CountTokensResponse{
		InputTokens: anthropic.EstimateTokens(req),
	})
}

func (a *Adapter) handleMessages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, MessagesBodyCap+1))
	if err != nil {
		anthropic.WriteError(w, http.StatusBadRequest, anthropic.ErrInvalidRequest,
			"read body: "+err.Error())
		return
	}
	defer r.Body.Close()
	if int64(len(body)) > MessagesBodyCap {
		anthropic.WriteError(w, http.StatusRequestEntityTooLarge, anthropic.ErrInvalidRequest,
			fmt.Sprintf("request body exceeds messages cap (%d bytes)", MessagesBodyCap))
		return
	}

	var areq anthropic.MessagesRequest
	if err := json.Unmarshal(body, &areq); err != nil {
		anthropic.WriteError(w, http.StatusBadRequest, anthropic.ErrInvalidRequest,
			"invalid JSON: "+err.Error())
		return
	}
	if areq.MaxTokens <= 0 {
		anthropic.WriteError(w, http.StatusBadRequest, anthropic.ErrInvalidRequest,
			"max_tokens is required and must be > 0")
		return
	}
	if len(areq.Messages) == 0 {
		anthropic.WriteError(w, http.StatusBadRequest, anthropic.ErrInvalidRequest,
			"messages must not be empty")
		return
	}

	openaiBody, err := translateMessagesToOpenAI(areq, a.cfg.Model.Name)
	if err != nil {
		anthropic.WriteError(w, http.StatusInternalServerError, anthropic.ErrAPIError,
			"marshal upstream: "+err.Error())
		return
	}

	upURL := joinPath(a.cfg.Engine.URL, pathChatCompletions)
	upReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upURL, bytes.NewReader(openaiBody))
	if err != nil {
		anthropic.WriteError(w, http.StatusInternalServerError, anthropic.ErrAPIError,
			"build upstream request: "+err.Error())
		return
	}
	upReq.Header.Set(headerContentType, contentTypeJSON)
	if areq.Stream {
		upReq.Header.Set("Accept", contentTypeSSE)
	}

	// Use a long-timeout client distinct from a.httpClient (5s) — chat
	// completions can take minutes to stream. We construct it here
	// rather than carrying another field on Adapter because the
	// transport is fairly cheap to set up and tests appreciate the
	// locality.
	hc := newMessagesHTTPClient(a.cfg.Runtime.ResponseHeaderTimeout())
	upResp, err := hc.Do(upReq)
	if err != nil {
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			"upstream unreachable: "+err.Error())
		return
	}
	defer upResp.Body.Close()
	if upResp.StatusCode >= 400 {
		raw, _ := io.ReadAll(io.LimitReader(upResp.Body, 4<<10))
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			fmt.Sprintf("upstream /v1/chat/completions status %d: %s", upResp.StatusCode, string(raw)))
		return
	}

	if areq.Stream {
		streamMessagesResponse(w, upResp.Body, areq, a.cfg.Model.Name)
		return
	}
	collectMessagesResponse(w, upResp.Body, areq, a.cfg.Model.Name)
}

// translateMessagesToOpenAI builds an OpenAI /v1/chat/completions body
// from an Anthropic MessagesRequest. The output is JSON bytes ready
// to ship upstream.
//
// Mapping rules (Stage 1):
//
//   - system → first messages entry with role=system
//   - messages content text → string content (flattened)
//   - temperature / top_p / max_tokens / stop_sequences → direct
//   - top_k forwarded as `top_k` (vLLM/SGLang/llama.cpp all accept
//     it as an OpenAI extension; vanilla OpenAI ignores unknown keys)
//   - tools / tool_choice stripped + slog.Warn (Stage 3 wires real
//     translation)
//   - model rewritten to cfg.Model.Name (the upstream alias) so the
//     engine sees the name it advertises via /v1/models
//
// We deliberately produce a fresh map and json.Marshal it rather than
// rewriting in-place — the original Anthropic body has fields the
// OpenAI side wouldn't understand (system, stop_sequences, top_k
// nested under different paths in some clients), and a clean rebuild
// is easier to audit than a partial transformation.
func translateMessagesToOpenAI(req anthropic.MessagesRequest, modelAlias string) ([]byte, error) {
	out := map[string]any{
		keyModel:     modelAlias,
		keyStream:    req.Stream,
		keyMaxTokens: req.MaxTokens,
	}
	out[keyMessages] = buildUpstreamMessages(req)

	if req.Temperature != nil {
		out["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		out["top_p"] = *req.TopP
	}
	if req.TopK != nil {
		// OpenAI vanilla ignores top_k; vLLM / SGLang / llama.cpp
		// accept it as an extension. Forward unconditionally —
		// safer than per-engine branching here, and the cost is one
		// JSON field on engines that ignore it.
		out["top_k"] = *req.TopK
	}
	if len(req.StopSequences) > 0 {
		out["stop"] = req.StopSequences
	}

	if openaiTools := anthropic.ToolsToOpenAI(req.Tools); openaiTools != nil {
		out["tools"] = openaiTools
	}
	if tc := anthropic.ToolChoiceToOpenAI(req.ToolChoice); tc != nil {
		out["tool_choice"] = tc
	}

	return json.Marshal(out)
}

// buildUpstreamMessages composes the OpenAI messages array, including
// system prefix, user/assistant turns, and inline Anthropic
// tool_result blocks expanded into role:"tool" messages. Mirrors the
// equivalent in internal/adapter/ollama/translate_messages.go; we
// keep them separate so each adapter can deviate independently if a
// backend introduces a non-OpenAI quirk.
func buildUpstreamMessages(req anthropic.MessagesRequest) []map[string]any {
	msgs := make([]map[string]any, 0, len(req.Messages)+1)
	if sys := req.System.Text; sys != "" {
		msgs = append(msgs, map[string]any{keyRole: anthropic.RoleSystem, keyContent: sys})
	}
	for _, m := range req.Messages {
		switch m.Role {
		case anthropic.RoleUser:
			toolMsgs, residual := anthropic.ToolResultsFromMessage(m)
			msgs = append(msgs, toolMsgs...)
			if len(residual) > 0 || len(toolMsgs) == 0 {
				// Stage 4 / Track R.4: emit OpenAI's typed
				// multimodal content array when image blocks are
				// present; fall back to a plain string when the
				// turn is text-only. OpenAIContentFromBlocks
				// returns the appropriate shape.
				content := anthropic.OpenAIContentFromBlocks(residual)
				if content == nil {
					content = ""
				}
				msgs = append(msgs, map[string]any{
					keyRole:    anthropic.RoleUser,
					keyContent: content,
				})
			}
		case anthropic.RoleAssistant:
			content, toolCalls := anthropic.AssistantToolUseToOpenAI(m)
			am := map[string]any{
				keyRole:    anthropic.RoleAssistant,
				keyContent: content,
			}
			if len(toolCalls) > 0 {
				am["tool_calls"] = toolCalls
			}
			msgs = append(msgs, am)
		default:
			msgs = append(msgs, map[string]any{
				keyRole:    m.Role,
				keyContent: m.Content.Text(),
			})
		}
	}
	return msgs
}

// openAIChatResponse is the subset of the OpenAI /v1/chat/completions
// non-streaming response we consume. Mirrors the ollama package's
// OpenAIChatResponse but in this package to avoid an import dependency.
type openAIChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role      string                     `json:"role"`
			Content   string                     `json:"content"`
			ToolCalls []anthropic.OpenAIToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

// openAIStreamChunk is the per-frame shape of an OpenAI streaming
// SSE body. We need delta.content, choices[0].finish_reason, usage
// (vLLM / SGLang emit on the LAST chunk when the client requested
// usage; llama.cpp typically omits it), and delta.tool_calls
// (streamed incrementally — see openAIStreamToolCall below).
type openAIStreamChunk struct {
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role      string                 `json:"role"`
			Content   string                 `json:"content"`
			ToolCalls []openAIStreamToolCall `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

// openAIStreamToolCall is the per-chunk delta shape for tool_calls.
// vLLM / SGLang follow OpenAI's spec: the first frame for a given
// index carries `id` + `function.name` + an empty `function.arguments`
// fragment; subsequent frames carry argument-string fragments that
// the client concatenates into a complete JSON object string.
// Frames may interleave indices for parallel tool calls.
type openAIStreamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
}

// streamMessagesResponse parses an OpenAI-shaped upstream SSE stream
// and translates each frame into the Anthropic event sequence
// (message_start / content_block_start / content_block_delta+ /
// content_block_stop / message_delta / message_stop).
//
// OpenAI SSE shape (one event per HTTP chunk):
//
//	data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}],"usage":null}\n\n
//	data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}\n\n
//	data: [DONE]\n\n
//
// We keep the input_tokens / output_tokens null in message_start
// (upstream doesn't tell us until the final frame) and emit the
// settled counts in message_delta. The same shape Ollama uses.
func streamMessagesResponse(w http.ResponseWriter, body io.Reader, req anthropic.MessagesRequest, modelAlias string) {
	sse := anthropic.NewWriter(w)
	msgID := anthropic.NewMessageID(time.Now().UnixNano())

	_ = sse.Emit(anthropic.EventMessageStart, anthropic.MessageStartEvent{
		Type: anthropic.EventMessageStart,
		Message: anthropic.MessageStartBody{
			ID:      msgID,
			Type:    keyMessage,
			Role:    anthropic.RoleAssistant,
			Content: []anthropic.ContentBlock{},
			Model:   modelAlias,
			Usage:   anthropic.Usage{InputTokens: 0, OutputTokens: 0},
		},
	})

	_ = sse.Emit(anthropic.EventContentBlockStart, anthropic.ContentBlockStartEvent{
		Type:         anthropic.EventContentBlockStart,
		Index:        0,
		ContentBlock: anthropic.ContentBlock{Type: anthropic.BlockText, Text: ""},
	})

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 4<<20)

	var (
		inputTokens  int64
		outputTokens int64
		stopReason   = anthropic.StopReasonEndTurn
		sawDone      bool
		// Buffered tool calls keyed by the upstream's index field.
		// We accumulate id/name on the first frame and concatenate
		// argument fragments across subsequent frames. Emitted as
		// Anthropic tool_use blocks AFTER the text content block.
		toolBuf = map[int]*streamingToolCall{}
	)

	const sseDataPrefix = "data: "

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, sseDataPrefix) {
			// Blank line separator between SSE events, OR an
			// `event: ...` line some engines emit. Either way we
			// skip; only `data: ` lines carry the JSON payload.
			continue
		}
		payload := strings.TrimPrefix(line, sseDataPrefix)
		payload = strings.TrimSpace(payload)
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			// Malformed frame; skip. Some engines occasionally
			// emit non-JSON keepalive frames (`: keep-alive`)
			// which scrolled in through TrimPrefix.
			continue
		}
		if len(chunk.Choices) > 0 {
			ch := chunk.Choices[0]
			if ch.Delta.Content != "" {
				_ = sse.Emit(anthropic.EventContentBlockDelta, anthropic.ContentBlockDeltaEvent{
					Type:  anthropic.EventContentBlockDelta,
					Index: 0,
					Delta: anthropic.BlockDelta{
						Type: anthropic.DeltaText,
						Text: ch.Delta.Content,
					},
				})
			}
			for _, tc := range ch.Delta.ToolCalls {
				accumulateToolCall(toolBuf, tc)
			}
			if ch.FinishReason != "" {
				stopReason = anthropic.MapFinishReason(ch.FinishReason)
			}
		}
		if chunk.Usage != nil {
			inputTokens = chunk.Usage.PromptTokens
			outputTokens = chunk.Usage.CompletionTokens
		}
	}
	if err := scanner.Err(); err != nil {
		_ = sse.EmitError(anthropic.ErrAPIError, "upstream stream error: "+err.Error())
		return
	}
	if !sawDone {
		slog.Warn("proxy: OpenAI SSE stream ended without [DONE]",
			slog.String("model", modelAlias))
	}

	// req.MaxTokens lets us upgrade end_turn to max_tokens when the
	// upstream omits finish_reason but we did hit the cap. Cheap and
	// matches the non-streaming heuristic.
	if stopReason == anthropic.StopReasonEndTurn && req.MaxTokens > 0 &&
		outputTokens > 0 && outputTokens >= int64(req.MaxTokens) {
		stopReason = anthropic.StopReasonMaxTokens
	}

	_ = sse.Emit(anthropic.EventContentBlockStop, anthropic.ContentBlockStopEvent{
		Type:  anthropic.EventContentBlockStop,
		Index: 0,
	})

	if len(toolBuf) > 0 {
		stopReason = anthropic.StopReasonToolUse
		emitBufferedToolCalls(sse, toolBuf, 1)
	}

	_ = sse.Emit(anthropic.EventMessageDelta, anthropic.MessageDeltaEvent{
		Type: anthropic.EventMessageDelta,
		Delta: anthropic.MessageDeltaUp{
			StopReason:   stopReason,
			StopSequence: nil,
		},
		Usage: anthropic.Usage{InputTokens: inputTokens, OutputTokens: outputTokens},
	})
	_ = sse.Emit(anthropic.EventMessageStop, anthropic.MessageStopEvent{
		Type: anthropic.EventMessageStop,
	})
}

// collectMessagesResponse buffers the upstream non-streaming OpenAI
// response and emits a single Anthropic MessagesResponse to w.
func collectMessagesResponse(w http.ResponseWriter, body io.Reader, _ anthropic.MessagesRequest, modelAlias string) {
	data, err := io.ReadAll(io.LimitReader(body, 16<<20))
	if err != nil {
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			"read upstream: "+err.Error())
		return
	}
	var oresp openAIChatResponse
	if err := json.Unmarshal(data, &oresp); err != nil {
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			"decode upstream: "+err.Error())
		return
	}
	if len(oresp.Choices) == 0 {
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			"upstream returned no choices")
		return
	}
	choice := oresp.Choices[0]

	// Compose content blocks: text first (when non-empty), then any
	// tool_use blocks derived from the OpenAI tool_calls. The
	// canonical finish_reason mapping is applied separately so a
	// tool_calls choice with no text still emits stop_reason=tool_use.
	var blocks []anthropic.ContentBlock
	if choice.Message.Content != "" {
		blocks = append(blocks, anthropic.ContentBlock{
			Type: anthropic.BlockText,
			Text: choice.Message.Content,
		})
	}
	if len(choice.Message.ToolCalls) > 0 {
		blocks = append(blocks, anthropic.ToolCallsToBlocks(choice.Message.ToolCalls)...)
	}
	if len(blocks) == 0 {
		blocks = []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: ""}}
	}

	resp := anthropic.MessagesResponse{
		ID:           anthropic.NewMessageID(time.Now().UnixNano()),
		Type:         keyMessage,
		Role:         anthropic.RoleAssistant,
		Model:        modelAlias,
		Content:      blocks,
		StopReason:   anthropic.MapFinishReason(choice.FinishReason),
		StopSequence: nil,
		Usage: anthropic.Usage{
			InputTokens:  oresp.Usage.PromptTokens,
			OutputTokens: oresp.Usage.CompletionTokens,
		},
	}
	w.Header().Set(headerContentType, contentTypeJSON)
	_ = json.NewEncoder(w).Encode(resp)
}

// streamingToolCall is the per-tool accumulator for streaming
// tool_calls. id and name appear on the first delta for a given
// index; arguments are concatenated across subsequent deltas.
type streamingToolCall struct {
	ID   string
	Name string
	Args strings.Builder
}

// accumulateToolCall merges one OpenAI-stream tool_call delta into
// the per-index accumulator. OpenAI's protocol guarantees:
//   - First delta for index N carries id (always) and function.name
//     (usually); function.arguments may be the empty string or the
//     first fragment.
//   - Subsequent deltas for index N carry only function.arguments
//     fragments; id / name MAY repeat but MUST not change.
//
// We tolerate both repeats and missing fields defensively because
// vLLM's protocol-level emitter has shipped versions that violate
// the spec in minor ways (notably: emitting id on every chunk).
func accumulateToolCall(buf map[int]*streamingToolCall, tc openAIStreamToolCall) {
	st, ok := buf[tc.Index]
	if !ok {
		st = &streamingToolCall{}
		buf[tc.Index] = st
	}
	if tc.ID != "" {
		st.ID = tc.ID
	}
	if tc.Function.Name != "" {
		st.Name = tc.Function.Name
	}
	if tc.Function.Arguments != "" {
		st.Args.WriteString(tc.Function.Arguments)
	}
}

// emitBufferedToolCalls writes one Anthropic tool_use block per
// accumulated streaming tool call. Order is by ascending index so the
// client sees calls in the same order the upstream emitted them
// across parallel tool invocations.
//
// We emit the whole accumulated args as a single input_json_delta
// rather than re-streaming the original fragments because the
// Anthropic protocol does not require frame-by-frame fidelity — it
// requires that the concatenation of partial_json across the deltas
// equals the final arguments JSON. One delta with the full payload
// satisfies that contract and avoids us having to buffer the
// per-frame fragments separately for re-emission.
func emitBufferedToolCalls(sse *anthropic.Writer, buf map[int]*streamingToolCall, startIndex int) {
	indices := make([]int, 0, len(buf))
	for k := range buf {
		indices = append(indices, k)
	}
	sortInts(indices)
	for offset, k := range indices {
		st := buf[k]
		idx := startIndex + offset
		_ = sse.Emit(anthropic.EventContentBlockStart, anthropic.ContentBlockStartEvent{
			Type:  anthropic.EventContentBlockStart,
			Index: idx,
			ContentBlock: anthropic.ContentBlock{
				Type:  anthropic.BlockToolUse,
				ID:    st.ID,
				Name:  st.Name,
				Input: json.RawMessage("{}"),
			},
		})
		args := st.Args.String()
		if args == "" {
			args = "{}"
		}
		_ = sse.Emit(anthropic.EventContentBlockDelta, anthropic.ContentBlockDeltaEvent{
			Type:  anthropic.EventContentBlockDelta,
			Index: idx,
			Delta: anthropic.BlockDelta{
				Type:        anthropic.DeltaInputJSON,
				PartialJSON: args,
			},
		})
		_ = sse.Emit(anthropic.EventContentBlockStop, anthropic.ContentBlockStopEvent{
			Type:  anthropic.EventContentBlockStop,
			Index: idx,
		})
	}
}

// sortInts is a small insertion sort over the per-index list (always
// tiny in practice — the number of parallel tool calls is at most
// a handful). Avoids a `sort` import for the only call site that
// needs sorting in this file.
func sortInts(xs []int) {
	for i := 1; i < len(xs); i++ {
		v := xs[i]
		j := i - 1
		for j >= 0 && xs[j] > v {
			xs[j+1] = xs[j]
			j--
		}
		xs[j+1] = v
	}
}

// newMessagesHTTPClient builds a long-timeout *http.Client suitable
// for streaming chat completions. Distinct from a.httpClient (which
// has a 5s timeout used by WaitAlive/Ready) because chat completions
// can stream for minutes — a 5s cap would kill every real request.
// headerTimeout is UPSTREAM_RESPONSE_HEADER_TIMEOUT (default 5m).
func newMessagesHTTPClient(headerTimeout time.Duration) *http.Client {
	if headerTimeout <= 0 {
		headerTimeout = config.DefaultUpstreamResponseHeaderTimeout
	}
	return &http.Client{
		Timeout: 30 * time.Minute,
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			IdleConnTimeout:       90 * time.Second,
			ResponseHeaderTimeout: headerTimeout,
		},
	}
}
