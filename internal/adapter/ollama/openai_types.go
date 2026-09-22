// Package ollama: OpenAI / Ollama wire types.
//
// This file holds the typed JSON envelopes the four translate_*.go files
// emit. Pre-v1.0.5 they assembled responses with `map[string]any` literals
// — readable but sprayed dozens of duplicate string literals (`"model"`,
// `"object"`, `"choices"`, ...) across the package. golangci-lint v2's
// goconst defaults flag those as duplicates; a typed-struct shape
// expresses the contract more clearly AND silences the linter without a
// per-file exclusion sprawl. See `.golangci.yml` header for the migration
// rationale.
//
// Field tags use `omitempty` consistently with the previous map behaviour:
// pre-fix, missing keys were absent from the encoded JSON; the typed
// version preserves that wire-level identity. Tests in
// `translate_*_test.go` assert on JSON bytes, not Go shape, so the
// before/after JSON is byte-for-byte identical.

package ollama

// ──────────────────────────────────────────────────────────────────────
// Shared
// ──────────────────────────────────────────────────────────────────────

// OpenAIUsage matches OpenAI's usage envelope. Embeddings only populate
// PromptTokens + TotalTokens; chat / completion populate all three.
type OpenAIUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens,omitempty"`
	TotalTokens      int64 `json:"total_tokens"`
}

// ──────────────────────────────────────────────────────────────────────
// Chat (/v1/chat/completions)
// ──────────────────────────────────────────────────────────────────────

// OpenAIChatMessage is one entry in a chat request's `messages` or in a
// non-streaming response's `message`/`delta`. Pointer-typed fields below
// (Message, Delta) let us preserve the "message vs delta" wire-level
// distinction depending on whether the response is non-streaming or a
// stream chunk.
//
// ReasoningContent is the DeepSeek-R1 / Qwen-3 / OpenWebUI convention
// for carrying the model's "thinking" tokens separately from the
// user-visible Content. Pre-1.0.10 llm-init had no such field and
// silently dropped Ollama's native `message.thinking` output, which
// caused thinking models to look like they were emitting an infinite
// stream of empty deltas (the role-only chunks observed in the
// "OpenWebUI hangs on Qwen3" report). Carrying it separately rather
// than concatenating into Content preserves the OpenWebUI side's
// ability to render the reasoning pane independently.
type OpenAIChatMessage struct {
	Role             string               `json:"role,omitempty"`
	Content          string               `json:"content,omitempty"`
	ReasoningContent string               `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIWireToolCall `json:"tool_calls,omitempty"`
}

// openAIWireToolCall is the OpenAI /v1/chat/completions tool_call shape.
// function.arguments MUST be a JSON-encoded string, not an object.
// Ollama often emits arguments as a JSON object; normalize before emit.
type openAIWireToolCall struct {
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// OpenAIChatChoice is one entry in a `choices` array. Streaming chunks
// emit `delta` with no `message`; non-streaming responses emit `message`
// with no `delta`. The terminal stream chunk emits an empty `delta` and
// a non-nil `FinishReason`.
type OpenAIChatChoice struct {
	Index        int                `json:"index"`
	Message      *OpenAIChatMessage `json:"message,omitempty"`
	Delta        *OpenAIChatMessage `json:"delta,omitempty"`
	FinishReason string             `json:"finish_reason,omitempty"`
}

// OpenAIChatResponse is the non-streaming /v1/chat/completions body and
// also the per-frame shape of a streaming SSE chunk (object switches
// between "chat.completion" and "chat.completion.chunk").
type OpenAIChatResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []OpenAIChatChoice `json:"choices"`
	Usage   *OpenAIUsage       `json:"usage,omitempty"`
}

// ──────────────────────────────────────────────────────────────────────
// Completion (/v1/completions)
// ──────────────────────────────────────────────────────────────────────

// OpenAICompletionChoice is one entry in a /v1/completions choices array.
// Both streaming and non-streaming variants share this shape; streaming
// sends Text per chunk, non-streaming sends a single buffered Text.
type OpenAICompletionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// OpenAICompletionResponse is the non-streaming /v1/completions body and
// also the per-frame shape of a streaming SSE chunk. Object is always
// "text_completion" in both cases.
type OpenAICompletionResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []OpenAICompletionChoice `json:"choices"`
	Usage   *OpenAIUsage             `json:"usage,omitempty"`
}

// ──────────────────────────────────────────────────────────────────────
// Embeddings (/v1/embeddings)
// ──────────────────────────────────────────────────────────────────────

// OpenAIEmbeddingDatum is one element of /v1/embeddings `data` array.
type OpenAIEmbeddingDatum struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

// OpenAIEmbeddingResponse is the /v1/embeddings response envelope.
// PromptTokens is set to 0 because Ollama's /api/embed endpoint does not
// report token counts; we preserve the field so OpenAI clients that
// inspect it don't NPE on a missing key.
type OpenAIEmbeddingResponse struct {
	Object string                 `json:"object"`
	Model  string                 `json:"model"`
	Data   []OpenAIEmbeddingDatum `json:"data"`
	Usage  OpenAIUsage            `json:"usage"`
}

// ──────────────────────────────────────────────────────────────────────
// Models (/v1/models)
// ──────────────────────────────────────────────────────────────────────

// OpenAIModelDatum is one entry in /v1/models data. We always emit
// exactly one — the alias the operator configured llm-init with.
type OpenAIModelDatum struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// OpenAIModelsResponse is the /v1/models envelope.
type OpenAIModelsResponse struct {
	Object string             `json:"object"`
	Data   []OpenAIModelDatum `json:"data"`
}

// ──────────────────────────────────────────────────────────────────────
// Wire-level constants for upstream-Ollama field names.
//
// These are not in the OpenAI surface — they're the upstream-side
// outbound request shapes. Pulling them into named constants gives the
// translate_*.go files something to refer to instead of repeating the
// raw literal across request marshalers.
// ──────────────────────────────────────────────────────────────────────

const (
	// OpenAI response envelope object values, per the OpenAI API spec.
	objectChatCompletion      = "chat.completion"
	objectChatCompletionChunk = "chat.completion.chunk"
	objectTextCompletion      = "text_completion"
	objectList                = "list"
	objectModel               = "model"
	objectEmbedding           = "embedding"

	// JSON key names shared between the Ollama-shape request marshalers
	// (translate_*.go writes these into `map[string]any`) and the
	// upstream response unmarshalers (client.go reads them via the same
	// key on inbound JSON). Same-string-different-semantics from the
	// `object*` enum above is intentional: goconst v2 collapses all
	// occurrences of `"model"` regardless of role, so we share one
	// constant for the literal and let the call site disambiguate.
	keyModel   = "model"
	keyStream  = "stream"
	keyMessage = "message"

	// finishReasonStop / finishReasonToolCalls mirror OpenAI chat
	// completion finish_reason values Ollama 0.1.45+ can emit.
	finishReasonStop      = "stop"
	finishReasonToolCalls = "tool_calls"

	// roleAssistant is the only role value llm-init sets in chat
	// responses; user/system roles only appear inbound (in requests).
	roleAssistant = "assistant"

	// ownerLLMInit is the value advertised in /v1/models `owned_by`.
	// Operators routinely filter on this to identify llm-init-served
	// models in mixed-stack dashboards; do not change without a major.
	ownerLLMInit = "llm-init"
)
