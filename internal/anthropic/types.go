package anthropic

import (
	"encoding/json"
	"strings"
)

// ────────────────────────────────────────────────────────────────────────
// Request types
// ────────────────────────────────────────────────────────────────────────

// MessagesRequest is the inbound /v1/messages body. Every adapter
// decodes into this type; backend-specific marshaling diverges from
// here. Field comments document each member's role and how it maps to
// backend-native fields where relevant.
type MessagesRequest struct {
	// Model is the model identifier the client asked for. llm-init
	// does NOT validate this against MODEL_NAME / OLLAMA_MODEL — chat
	// completions don't either (see proxy/rewrite.go) and a hard
	// match would break a class of clients that always send
	// "claude-3-5-sonnet-20241022" regardless of upstream. Adapters
	// rewrite it to the upstream tag before forwarding.
	Model string `json:"model"`

	// MaxTokens is REQUIRED by Anthropic's API. We honour the
	// requirement at the adapter layer — handlers return 400 +
	// `invalid_request_error` when missing or non-positive.
	MaxTokens int `json:"max_tokens"`

	// Messages is the conversation so far. Each entry's Content is
	// either a string (text-only shorthand) or an array of typed
	// blocks (text / image / tool_use / tool_result). We unmarshal
	// into a Content wrapper that handles both shapes.
	Messages []Message `json:"messages"`

	// System is Anthropic's top-level system prompt. The OpenAI
	// equivalent is messages[0] with role=system. Adapters prepend
	// System (when non-empty) as the first message in their upstream
	// payload. Spec allows either a plain string OR an array of
	// content blocks (multimodal system); we accept both via the
	// SystemPrompt wrapper.
	System SystemPrompt `json:"system,omitempty"`

	// Stream controls SSE streaming. Adapters route to the streaming
	// translator when true.
	Stream bool `json:"stream,omitempty"`

	// Temperature 0.0–1.0 (Anthropic), maps directly to upstream
	// temperature for OpenAI-shape engines. Anthropic clients
	// occasionally send 0.0–2.0 (OpenAI range); adapters clamp to
	// the backend's accepted range.
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP nucleus-sampling threshold.
	TopP *float64 `json:"top_p,omitempty"`

	// TopK is Anthropic-specific (OpenAI doesn't expose it). Ollama
	// understands `top_k` natively; vLLM / SGLang / llama.cpp accept
	// it via OpenAI extensions on most builds. Adapters forward
	// when present and rely on the backend to ignore unknown keys.
	TopK *int `json:"top_k,omitempty"`

	// StopSequences is Anthropic's name for OpenAI's `stop`. We pass
	// the array as-is to upstreams.
	StopSequences []string `json:"stop_sequences,omitempty"`

	// Metadata is Anthropic-specific request bookkeeping (user_id
	// etc.). We do NOT forward it upstream because the backends
	// don't know what to do with it; logged at debug level so
	// operators can observe its presence without leaking PII.
	Metadata json.RawMessage `json:"metadata,omitempty"`

	// Tools enables function-calling. Schema is Anthropic-native
	// (input_schema rather than OpenAI's parameters). Stage 3
	// (Track R.3) implements translation; Stage 1 strips with a
	// log line on Ollama for parity with /v1/chat/completions.
	Tools []Tool `json:"tools,omitempty"`

	// ToolChoice mirrors Anthropic's tool_choice. Either an object
	// (`{"type":"auto"}` / `{"type":"any"}` / `{"type":"tool","name":...}`)
	// or, on older clients, a bare string. Stage 3 translates;
	// Stage 1 strips.
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
}

// SystemPrompt accepts both wire shapes Anthropic allows: a plain
// string OR an array of content blocks. The text() helper flattens to
// a single concatenated string for backends that expect one
// system-role message (Ollama, OpenAI vanilla).
type SystemPrompt struct {
	// Raw retains the original JSON for adapters that want to
	// inspect the typed shape (e.g. forward image blocks).
	Raw json.RawMessage
	// Text is the flattened text content — empty when the raw value
	// is null / missing / empty array.
	Text string
}

// UnmarshalJSON decodes both "string" and `[{"type":"text","text":...}, ...]`
// forms into SystemPrompt. Unknown block types are silently skipped at
// flatten time; the Raw field preserves them for adapters that care.
func (s *SystemPrompt) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	s.Raw = append(s.Raw[:0], data...)
	// Try string first (the common case).
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		s.Text = str
		return nil
	}
	// Fall back to []ContentBlock; concat text blocks.
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err != nil {
		return err
	}
	var b strings.Builder
	for _, blk := range blocks {
		if blk.Type == BlockText {
			b.WriteString(blk.Text)
		}
	}
	s.Text = b.String()
	return nil
}

// MarshalJSON emits the original Raw form when present; otherwise the
// flattened text. Keeps round-trip behaviour stable for tests.
func (s SystemPrompt) MarshalJSON() ([]byte, error) {
	if len(s.Raw) > 0 {
		return s.Raw, nil
	}
	if s.Text == "" {
		return []byte("null"), nil
	}
	return json.Marshal(s.Text)
}

// Message is one entry in MessagesRequest.Messages. Role is "user" or
// "assistant" (no "system" — see SystemPrompt); Content holds either
// a string or an array of typed blocks.
type Message struct {
	Role    string  `json:"role"`
	Content Content `json:"content"`
}

// ────────────────────────────────────────────────────────────────────────
// Content blocks
// ────────────────────────────────────────────────────────────────────────

// Anthropic content-block type identifiers. These are wire constants
// — clients and adapters MUST emit and accept these exact strings.
const (
	BlockText       = "text"
	BlockImage      = "image"
	BlockToolUse    = "tool_use"
	BlockToolResult = "tool_result"
)

// Anthropic role values.
const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	// RoleSystem is not a valid Anthropic Message.Role value
	// (system prompts live at top-level via SystemPrompt) but
	// adapters need it when synthesising the OpenAI / Ollama
	// upstream payload (which DOES use a system-role message).
	RoleSystem = "system"
	// RoleTool is OpenAI's role for tool_result replay messages.
	// Anthropic encodes tool results as content blocks inside a
	// user-role message; we expand them into role:"tool" messages
	// before forwarding to the engine.
	RoleTool = "tool"
)

// Content carries either a plain string (Anthropic's shorthand for a
// single text block) or an array of ContentBlock. Adapters convert to
// backend-native shape via Blocks() which always returns the
// canonical block array form regardless of which wire shape the
// client used.
type Content struct {
	// Plain is non-empty when the client sent "content": "<string>".
	// In that case Blocks() returns a single text block synthesised
	// on demand.
	Plain string
	// blocks is the typed form when the client sent the array
	// shape. Internal because callers should go through Blocks() to
	// get the canonical view.
	blocks []ContentBlock
}

// UnmarshalJSON accepts both wire shapes (string or block array).
// Unknown block types are preserved as ContentBlock with Type set to
// the unknown value — adapters can decide whether to strip, error,
// or forward.
func (c *Content) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	if data[0] == '"' {
		return json.Unmarshal(data, &c.Plain)
	}
	return json.Unmarshal(data, &c.blocks)
}

// MarshalJSON round-trips: emits the array form when blocks were
// supplied, else the plain string. Synthesises an empty array (not
// null) when both are zero so the upstream sees a parseable shape.
func (c Content) MarshalJSON() ([]byte, error) {
	if len(c.blocks) > 0 {
		return json.Marshal(c.blocks)
	}
	return json.Marshal(c.Plain)
}

// Blocks returns the canonical block array view. A string-only
// Content yields one synthesised text block; an empty Content yields
// nil. Adapters that don't care about typed content can call Text()
// instead.
func (c Content) Blocks() []ContentBlock {
	if len(c.blocks) > 0 {
		return c.blocks
	}
	if c.Plain != "" {
		return []ContentBlock{{Type: BlockText, Text: c.Plain}}
	}
	return nil
}

// Text concatenates every text block (in order) into a single string.
// Image / tool_use / tool_result blocks are skipped — callers that
// need them should iterate Blocks() directly. Useful for text-only
// backends that have no notion of multimodal content.
func (c Content) Text() string {
	if c.Plain != "" && len(c.blocks) == 0 {
		return c.Plain
	}
	var b strings.Builder
	for _, blk := range c.blocks {
		if blk.Type == BlockText {
			b.WriteString(blk.Text)
		}
	}
	return b.String()
}

// SetBlocks replaces the typed-array form. Used by response builders
// that always emit the array shape.
func (c *Content) SetBlocks(blocks []ContentBlock) {
	c.blocks = blocks
	c.Plain = ""
}

// ContentBlock is one element of a typed Content array. Fields are
// populated based on Type; everything is omitempty so a text block
// doesn't leak image-only fields and vice versa.
//
// We keep the type loose (a single struct with all possible fields)
// rather than a sum-type interface because:
//   - adapters frequently switch on Type and reach into one or two
//     fields, which is more ergonomic than a type-switch wrapper;
//   - JSON encoding stays trivial without a custom MarshalJSON;
//   - adding a new block type is a single field addition.
type ContentBlock struct {
	Type string `json:"type"`

	// Text block fields.
	Text string `json:"text,omitempty"`

	// Image block fields (Stage 4 / Track R.4). Source carries the
	// image bytes; "type" inside the source is either "base64" or
	// "url" (Anthropic only added URL sources in mid-2024).
	Source *ImageSource `json:"source,omitempty"`

	// Tool-use block fields (Stage 3 / Track R.3): assistant
	// signals "call this tool with these inputs".
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// Tool-result block fields (Stage 3 / Track R.3): user replays
	// the tool's output back to the model. Pointer-typed so that
	// non-tool_result blocks marshal without an empty "content"
	// field; json's omitempty does not omit struct values (only
	// pointers, slices, etc.) — see the encoding/json docs.
	ToolUseID string   `json:"tool_use_id,omitempty"`
	Content   *Content `json:"content,omitempty"`
	IsError   bool     `json:"is_error,omitempty"`
}

// ImageSource describes an Anthropic image block's source. Anthropic
// supports two variants:
//
//   - base64: `{"type":"base64","media_type":"image/png","data":"..."}`
//   - url:    `{"type":"url","url":"https://..."}`
//
// Adapters that don't have an explicit base64 channel (Ollama uses
// `images: [base64]` directly) ignore MediaType.
type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

// ────────────────────────────────────────────────────────────────────────
// Tools
// ────────────────────────────────────────────────────────────────────────

// Tool is one Anthropic tool definition. Schema differs from OpenAI's
// in that the parameter schema is `input_schema` rather than nested
// inside `function.parameters`. Translation lives in
// internal/adapter/<engine>/translate_messages.go in Stage 3.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

// ────────────────────────────────────────────────────────────────────────
// Response types
// ────────────────────────────────────────────────────────────────────────

// MessagesResponse is the non-streaming /v1/messages response body.
// Streaming responses synthesise this incrementally through the SSE
// event sequence (see events.go).
type MessagesResponse struct {
	// ID is `msg_<random>`. We synthesise it per-request because no
	// backend emits an Anthropic-shaped ID.
	ID string `json:"id"`
	// Type is always "message" for /v1/messages responses.
	Type string `json:"type"`
	// Role is always "assistant" in responses.
	Role string `json:"role"`
	// Content is the assistant's response as a block array. Even
	// text-only responses use the array shape — Anthropic's spec
	// does not honour the string shorthand on responses.
	Content []ContentBlock `json:"content"`
	// Model echoes the client-facing alias (cfg.Model.Name), NOT
	// the upstream tag, so a client sees the name it asked for.
	Model string `json:"model"`
	// StopReason is one of the values in stopReason* constants
	// below. Adapters translate from backend-native finish_reason.
	StopReason string `json:"stop_reason,omitempty"`
	// StopSequence is the literal stop_sequence string that
	// triggered stoppage, when StopReason == "stop_sequence".
	// Pointer-typed so the JSON emits `null` when not set (which
	// is Anthropic's convention even when stop_reason is
	// "end_turn" or "max_tokens").
	StopSequence *string `json:"stop_sequence"`
	// Usage carries Anthropic's input_tokens / output_tokens
	// pair. We map backend counts (Ollama prompt_eval_count /
	// eval_count; OpenAI prompt_tokens / completion_tokens)
	// directly without normalisation — the same tokens count
	// twice on Anthropic's billing endpoint, so 1:1 mirroring
	// preserves the operator's existing dashboards.
	Usage Usage `json:"usage"`
}

// Usage is Anthropic's input/output token pair.
type Usage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

// stop_reason values. Anthropic's spec enumerates these exactly; an
// invalid value confuses some SDKs (anthropic-sdk-python's
// MessageStopEvent has them as a Literal type).
const (
	StopReasonEndTurn      = "end_turn"
	StopReasonMaxTokens    = "max_tokens"
	StopReasonStopSequence = "stop_sequence"
	StopReasonToolUse      = "tool_use"
)

// MapFinishReason translates a backend-native finish_reason into the
// Anthropic stop_reason enum. Unknown / empty values default to
// end_turn because that's the friendliest behaviour for SDKs (some
// abort hard on a missing stop_reason; "end_turn" is always safe to
// pretend on a clean exit).
//
// Mapping table:
//
//	OpenAI / vllm / sglang / llama.cpp    Anthropic
//	---------------------------------     -------------
//	"stop"                                end_turn
//	"length"                              max_tokens
//	"tool_calls" / "function_call"        tool_use
//	"content_filter"                      end_turn (no Anthropic
//	                                      equivalent; clients see
//	                                      the truncated content
//	                                      without a strict marker)
//	"" / unknown                          end_turn
//
// Ollama only emits `done: true` (no finish_reason), so adapters
// pass an empty string and get end_turn — the right default since
// Ollama's done-flag is the equivalent of a graceful stop.
func MapFinishReason(finish string) string {
	switch finish {
	case "stop", "":
		return StopReasonEndTurn
	case "length":
		return StopReasonMaxTokens
	case "tool_calls", "function_call":
		return StopReasonToolUse
	default:
		return StopReasonEndTurn
	}
}
