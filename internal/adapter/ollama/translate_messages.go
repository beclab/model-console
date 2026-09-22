package ollama

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/llm-init/llm-init/internal/anthropic"
	"github.com/llm-init/llm-init/internal/config"
)

// messagesBodyCap is the maximum request body size for /v1/messages.
// Anthropic requests are larger than OpenAI's because:
//
//   - the `system` field can carry multi-block content (multi-paragraph
//     instructions are common);
//   - tool definitions (Stage 3) ship their full JSON schema in-band;
//   - vision blocks (Stage 4) carry base64-encoded images.
//
// 16 MiB is the same default Anthropic's own ingress accepts for a
// single /v1/messages call. Operators can override via the
// `MESSAGES_BODY_CAP` env (config.Runtime.MessagesBodyCap, defaulting
// to this constant when unset). Hardcoding rather than reading the
// config every request keeps the hot path branch-free for the common
// case; the env knob is plumbed through a package variable below.
const messagesBodyCap = 16 << 20 // 16 MiB

// MessagesBodyCap is overridable from main.go after config.Load — set
// to cfg.Runtime.MessagesBodyCap when non-zero. Package variable
// rather than passing through the Adapter struct because the cap is
// process-wide and changing it per-adapter would invite a class of
// "different cap on different /v1 path" bugs we currently don't have.
var MessagesBodyCap int64 = messagesBodyCap

// Wire-shape JSON keys used by the messages translator. Pulled out as
// named constants so goconst stops flagging duplicates and the
// per-file references stay searchable.
const (
	keyContent = "content"
	keyRole    = "role"
	keyOptions = "options"
)

// handleCountTokens serves POST /v1/messages/count_tokens. The
// handler decodes the Anthropic CountTokensRequest, runs the chars/4
// heuristic via anthropic.EstimateTokens, and returns the
// canonical {"input_tokens": N} envelope. Because the estimate is
// approximate, the response carries an `X-Llm-Init-Token-Estimate:
// heuristic` header so client SDKs can decide whether to compensate
// (e.g. by reserving an extra 10% headroom on a max_tokens budget).
// A future stage may swap the heuristic for upstream /tokenize.
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
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Llm-Init-Token-Estimate", "heuristic")
	_ = json.NewEncoder(w).Encode(anthropic.CountTokensResponse{
		InputTokens: anthropic.EstimateTokens(req),
	})
}

// handleMessages serves POST /v1/messages by decoding the Anthropic
// request, translating it to an Ollama /api/chat request, and
// translating the response back. Stage 1 covers non-streaming text;
// Stage 2 wires the streaming branch into streamMessagesResponse.
//
// Validation:
//   - max_tokens > 0 (Anthropic requires this; we mirror)
//   - messages != empty (Anthropic returns invalid_request_error)
//   - body within messagesBodyCap
//
// All error envelopes use anthropic.WriteError so anthropic-sdk-python
// / anthropic-sdk-typescript / Claude Code parse them cleanly.
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

	ollamaReq := translateMessagesRequest(areq, a.cfg)
	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		anthropic.WriteError(w, http.StatusInternalServerError, anthropic.ErrAPIError,
			"marshal upstream: "+err.Error())
		return
	}

	resp, err := a.client.PostJSON(r.Context(), "/api/chat", reqBody)
	if err != nil {
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			"upstream unreachable: "+err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		// Surface upstream-side errors as api_error with the body
		// bytes preserved so operators can diagnose. We deliberately
		// keep the status code as 502 (gateway-side) rather than
		// echo upstream's 4xx — upstream errors on the Ollama path
		// reflect llm-init misconfiguration, not client error.
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			fmt.Sprintf("upstream /api/chat status %d: %s", resp.StatusCode, string(raw)))
		return
	}

	if areq.Stream {
		streamMessagesResponse(w, resp.Body, areq, a.cfg.Model.Name)
		return
	}
	collectMessagesResponse(w, resp.Body, areq, a.cfg.Model.Name)
}

// translateMessagesRequest maps an Anthropic MessagesRequest onto an
// Ollama /api/chat body. The output is a fresh map (not a mutation of
// the input) so the inbound struct stays clean for any downstream
// logging.
//
// Mapping rules (Stage 1):
//
//   - system → messages[0] with role="system" when non-empty
//   - messages content → flattened text (multi-modal blocks
//     are dropped at this stage; Stage 4 introduces image handling)
//   - temperature / top_p / top_k → options.{temperature, top_p, top_k}
//   - max_tokens → options.num_predict
//   - stop_sequences → options.stop
//   - tools / tool_choice → forwarded unchanged (Ollama /api/chat)
//   - applyInferenceInjection mirrors translateChatRequest's
//     ENGINE_ARGS-driven num_ctx / repeat_* / keep_alive injection.
//
// The model field is the upstream daemon tag (cfg.PrimaryOllamaTag()
// = ollama:// source tag with MODEL_NAME fallback) — the Anthropic-side
// alias never leaks to the daemon, exactly like /v1/chat/completions
// handling.
func translateMessagesRequest(req anthropic.MessagesRequest, cfg config.Config) map[string]any {
	upstreamName := UpstreamModel(cfg)
	out := map[string]any{
		keyModel:  upstreamName,
		keyStream: req.Stream,
	}

	// Build the messages array. System (when present) is prepended as
	// the first entry; user/assistant turns follow in order. Each
	// turn's content is flattened to text — image / tool_* blocks are
	// dropped here in Stage 1.
	msgs := buildUpstreamMessages(req)
	out["messages"] = msgs

	// Options block carries the sampling / decoding knobs Ollama
	// understands. Only set the keys that were actually provided so
	// Ollama's per-model defaults still apply for fields the client
	// didn't override.
	options := map[string]any{}
	if req.Temperature != nil {
		options["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		options["top_p"] = *req.TopP
	}
	if req.TopK != nil {
		options["top_k"] = *req.TopK
	}
	if req.MaxTokens > 0 {
		options["num_predict"] = req.MaxTokens
	}
	if len(req.StopSequences) > 0 {
		options["stop"] = req.StopSequences
	}
	applyInferenceInjection(out, options, cfg)
	if len(options) > 0 {
		out[keyOptions] = options
	}

	// Tools translation (Stage 3 / Track R.3). Ollama 0.1.45+
	// accepts OpenAI-shape tools on /api/chat; we forward the
	// converted array unchanged. Operators on older daemons may
	// observe tools being ignored — Ollama silently drops unknown
	// fields rather than erroring, so the failure mode is at worst
	// "the model ignores the tools" rather than a 500.
	if openaiTools := anthropic.ToolsToOpenAI(req.Tools); openaiTools != nil {
		out["tools"] = openaiTools
	}
	if tc := anthropic.ToolChoiceToOpenAI(req.ToolChoice); tc != nil {
		out["tool_choice"] = tc
	}

	return out
}

// buildUpstreamMessages constructs the Ollama /api/chat messages
// array, including the system prefix, user/assistant turns, and
// inline tool_result blocks fanned out into role:"tool" messages.
//
// Anthropic puts tool_result inside a user-role Message's content[];
// OpenAI / Ollama require a separate role:"tool" message per call.
// We split each user message into a leading tool messages run + a
// residual user message carrying the non-tool content. This
// preserves message order so the model sees the conversation as the
// client intended.
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
				msgs = append(msgs, buildUserMessage(m, residual))
			}
		case anthropic.RoleAssistant:
			content, toolCalls := anthropic.AssistantToolUseToOpenAI(m)
			am := map[string]any{
				keyRole:    anthropic.RoleAssistant,
				keyContent: content,
			}
			if len(toolCalls) > 0 {
				am[openaiKeyToolCalls] = normalizeToolCallsForOllama(toolCalls)
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

// buildUserMessage emits one user-role Ollama message with the text
// content flattened from `residual` (the non-tool_result blocks) and
// any base64 image blocks attached via the `images: [...]` field
// Ollama's /api/chat accepts.
//
// Image handling (Stage 4 / Track R.4):
//   - base64 source -> appended to images[]
//   - url    source -> dropped + slog.Warn (Ollama does not accept
//     remote URLs; fetching client-side could land
//     in a future stage but invites SSRF risk; we
//     defer that decision).
func buildUserMessage(_ anthropic.Message, residual []anthropic.ContentBlock) map[string]any {
	var text string
	for _, blk := range residual {
		if blk.Type == anthropic.BlockText {
			text += blk.Text
		}
	}
	msg := map[string]any{
		keyRole:    anthropic.RoleUser,
		keyContent: text,
	}
	images, droppedURLs := anthropic.OllamaImagesFromBlocks(residual)
	if len(images) > 0 {
		msg["images"] = images
	}
	if droppedURLs > 0 {
		slog.Warn("ollama: dropped url-source image blocks (Ollama needs base64)",
			slog.Int("dropped_count", droppedURLs))
	}
	return msg
}

// streamMessagesResponse pipes Ollama's NDJSON output to the client
// as the Anthropic SSE event sequence:
//
//	event: message_start
//	  data: {message: {... usage: input_tokens:N, output_tokens:0}}
//	event: content_block_start
//	  data: {index:0, content_block: {type:text, text:""}}
//	(for each token chunk:)
//	  event: content_block_delta
//	    data: {index:0, delta:{type:text_delta, text:"..."}}
//	event: content_block_stop
//	  data: {index:0}
//	event: message_delta
//	  data: {delta:{stop_reason:"end_turn", stop_sequence:null},
//	         usage:{output_tokens:N}}
//	event: message_stop
//	  data: {}
//
// Ollama emits NDJSON: one JSON object per line, with `done:true` on
// the terminal frame and prompt_eval_count / eval_count populated
// there. The first frame's prompt_eval_count is often 0 — the daemon
// reports it on the done-frame instead — so we defer the
// message_start usage population until we see the terminal frame.
// Anthropic clients accept input_tokens being 0 on message_start as
// long as it's settled by the final message_delta. (Verified against
// anthropic-sdk-python's MessageStreamManager which sums per-event.)
//
// Mid-stream upstream errors (scanner error, malformed JSON beyond
// the first frame) emit an `error` event and stop. Some SDKs
// terminate the iterator on `event: error`; emitting one gives them
// a clean signal instead of an abrupt EOF.
func streamMessagesResponse(w http.ResponseWriter, body io.Reader, req anthropic.MessagesRequest, modelAlias string) {
	sse := anthropic.NewWriter(w)
	msgID := anthropic.NewMessageID(time.Now().UnixNano())

	// message_start with empty content + zero usage. We don't know
	// the input_tokens yet (Ollama reports prompt_eval_count on the
	// done-frame). Anthropic's wire shape accepts a placeholder
	// here, and the final message_delta usage settles the real
	// number.
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

	// content_block_start (index 0, text block). Stage 1/2 only
	// emits one block; Stage 3 (tools) introduces additional indices.
	_ = sse.Emit(anthropic.EventContentBlockStart, anthropic.ContentBlockStartEvent{
		Type:         anthropic.EventContentBlockStart,
		Index:        0,
		ContentBlock: anthropic.ContentBlock{Type: anthropic.BlockText, Text: ""},
	})

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 4<<20)

	var (
		inputTokens   int64
		outputTokens  int64
		stopReason    = anthropic.StopReasonEndTurn
		streamErr     error
		sawTerminator bool
		bufferedTools []ollamaToolCall
	)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var fr ollamaChatChunk
		if err := json.Unmarshal(line, &fr); err != nil {
			// Skip malformed frames; Ollama occasionally emits
			// blank lines or partial pings between frames. A
			// hard-fail here would surface false errors mid-stream.
			continue
		}
		// Emit a text delta when the frame carries new content. The
		// done-frame typically carries empty content + done:true; we
		// must NOT emit a delta with empty text (Anthropic SDKs
		// special-case empty deltas as a transport hiccup).
		if fr.Message.Content != "" {
			_ = sse.Emit(anthropic.EventContentBlockDelta, anthropic.ContentBlockDeltaEvent{
				Type:  anthropic.EventContentBlockDelta,
				Index: 0,
				Delta: anthropic.BlockDelta{
					Type: anthropic.DeltaText,
					Text: fr.Message.Content,
				},
			})
		}
		// Buffer tool calls. Ollama 0.1.45 emits the full tool_calls
		// array on the terminal frame (no per-token partials), so
		// buffering is essentially a no-op aside from accumulating
		// multiple calls. Stage 3 emits them as content_block_start
		// + input_json_delta + content_block_stop AFTER the text
		// block (index 1+). Per-token streaming of arguments would
		// need Ollama 0.2+ behaviour we don't yet rely on.
		bufferedTools = append(bufferedTools, fr.Message.ToolCalls...)
		if fr.Done {
			inputTokens = fr.PromptEvalCount
			outputTokens = fr.EvalCount
			if req.MaxTokens > 0 && outputTokens >= int64(req.MaxTokens) {
				stopReason = anthropic.StopReasonMaxTokens
			}
			sawTerminator = true
			break
		}
	}
	if err := scanner.Err(); err != nil {
		streamErr = err
	}

	if streamErr != nil {
		_ = sse.EmitError(anthropic.ErrAPIError, "upstream stream error: "+streamErr.Error())
		return
	}
	if !sawTerminator {
		// Upstream closed the stream without a done-frame. Emit
		// content_block_stop + a message_delta with end_turn
		// (best-guess) so the SDK sees a complete event sequence
		// rather than an open block.
		slog.Warn("ollama: NDJSON stream ended without done-frame",
			slog.String("model", modelAlias))
	}

	_ = sse.Emit(anthropic.EventContentBlockStop, anthropic.ContentBlockStopEvent{
		Type:  anthropic.EventContentBlockStop,
		Index: 0,
	})

	// Emit any buffered tool_calls as additional content blocks
	// AFTER the text block. Index numbering continues from 1 since
	// the text block occupied index 0.
	if len(bufferedTools) > 0 {
		stopReason = anthropic.StopReasonToolUse
		emitOllamaToolUseBlocks(sse, bufferedTools, 1)
	}
	_ = sse.Emit(anthropic.EventMessageDelta, anthropic.MessageDeltaEvent{
		Type: anthropic.EventMessageDelta,
		Delta: anthropic.MessageDeltaUp{
			StopReason:   stopReason,
			StopSequence: nil,
		},
		// Anthropic's message_delta usage carries the cumulative
		// counts; clients sum them with the input_tokens from
		// message_start to compute the billing total. We populate
		// both fields here because the message_start placeholder
		// reported zero — letting clients see the real number on
		// the closing event.
		Usage: anthropic.Usage{InputTokens: inputTokens, OutputTokens: outputTokens},
	})
	_ = sse.Emit(anthropic.EventMessageStop, anthropic.MessageStopEvent{
		Type: anthropic.EventMessageStop,
	})
}

// collectMessagesResponse buffers a non-streaming Ollama response and
// emits a single MessagesResponse. Stage 2 introduces an analogue for
// streaming.
func collectMessagesResponse(w http.ResponseWriter, body io.Reader, req anthropic.MessagesRequest, modelAlias string) {
	data, err := io.ReadAll(io.LimitReader(body, 16<<20))
	if err != nil {
		anthropic.WriteError(w, http.StatusBadGateway, anthropic.ErrAPIError,
			"read upstream: "+err.Error())
		return
	}
	var fr ollamaChatChunk
	if err := json.Unmarshal(data, &fr); err != nil {
		fr = lastNDJSONFrame(data)
	}

	stopReason := anthropic.StopReasonEndTurn
	// Heuristic: if eval_count >= max_tokens (or the daemon emitted
	// non-empty stop sequence info), prefer max_tokens. Ollama
	// doesn't report a finish_reason, only `done: true`, so we have
	// to infer. err on the side of end_turn (Anthropic's friendliest
	// default) and only switch when num_predict matches eval_count.
	if req.MaxTokens > 0 && fr.EvalCount > 0 && fr.EvalCount >= int64(req.MaxTokens) {
		stopReason = anthropic.StopReasonMaxTokens
	}

	// Compose the response content blocks. Text always comes first;
	// each tool_call from the Ollama frame turns into a tool_use
	// block (Stage 3 / Track R.3). Anthropic SDKs accept zero or
	// more blocks of either type in any order; we order text-then-
	// tool because that's what most clients render correctly.
	blocks := buildResponseBlocks(fr)
	if len(blocks) == 0 {
		blocks = []anthropic.ContentBlock{{Type: anthropic.BlockText, Text: ""}}
	}
	if hasToolUse(blocks) {
		stopReason = anthropic.StopReasonToolUse
	}

	resp := anthropic.MessagesResponse{
		ID:           anthropic.NewMessageID(time.Now().UnixNano()),
		Type:         keyMessage,
		Role:         anthropic.RoleAssistant,
		Model:        modelAlias,
		Content:      blocks,
		StopReason:   stopReason,
		StopSequence: nil,
		Usage: anthropic.Usage{
			InputTokens:  fr.PromptEvalCount,
			OutputTokens: fr.EvalCount,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// buildResponseBlocks composes Anthropic content blocks from one
// Ollama /api/chat frame. Text and tool_use are interleaved with text
// first, then any tool_calls. Empty text is dropped (no zero-length
// text block) unless there are no tools either, in which case the
// caller synthesises an empty text block to keep MessagesResponse
// schema-valid.
func buildResponseBlocks(fr ollamaChatChunk) []anthropic.ContentBlock {
	var blocks []anthropic.ContentBlock
	if fr.Message.Content != "" {
		blocks = append(blocks, anthropic.ContentBlock{
			Type: anthropic.BlockText,
			Text: fr.Message.Content,
		})
	}
	if len(fr.Message.ToolCalls) > 0 {
		calls := make([]anthropic.OpenAIToolCall, 0, len(fr.Message.ToolCalls))
		for _, tc := range fr.Message.ToolCalls {
			calls = append(calls, anthropic.OpenAIToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{
					Name:      tc.Function.Name,
					Arguments: string(tc.Function.Arguments),
				},
			})
		}
		blocks = append(blocks, anthropic.ToolCallsToBlocks(calls)...)
	}
	return blocks
}

func hasToolUse(blocks []anthropic.ContentBlock) bool {
	for _, b := range blocks {
		if b.Type == anthropic.BlockToolUse {
			return true
		}
	}
	return false
}

// emitOllamaToolUseBlocks emits one tool_use content block per
// buffered Ollama tool call (content_block_start +
// content_block_delta with the full JSON args + content_block_stop).
// startIndex is the first block index to use; subsequent blocks
// increment from there. We chose the "buffered single delta" wire
// shape over multi-chunk input_json_delta because Ollama's daemon
// itself does not stream tool argument tokens — sending one delta
// with the complete payload matches anthropic-sdk-python's
// JSONDecoder state machine (it accumulates until the block stops).
func emitOllamaToolUseBlocks(sse *anthropic.Writer, calls []ollamaToolCall, startIndex int) {
	for i, tc := range calls {
		idx := startIndex + i
		_ = sse.Emit(anthropic.EventContentBlockStart, anthropic.ContentBlockStartEvent{
			Type:  anthropic.EventContentBlockStart,
			Index: idx,
			ContentBlock: anthropic.ContentBlock{
				Type:  anthropic.BlockToolUse,
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage("{}"),
			},
		})
		args := string(tc.Function.Arguments)
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
