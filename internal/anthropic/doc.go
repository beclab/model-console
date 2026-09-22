// Package anthropic holds the wire-level types and shared helpers for
// Anthropic's Messages API (`POST /v1/messages` and
// `POST /v1/messages/count_tokens`). The package is pure types + small
// pure functions — no HTTP I/O, no upstream calls — so it can be
// imported by every adapter without dragging engine-specific code into
// the dependency graph.
//
// Translation strategy:
//
//   - Each adapter implements AnthropicHandler(cfg) http.Handler.
//     The adapter is responsible for taking an Anthropic-shaped
//     request and producing an Anthropic-shaped response, doing
//     whatever backend-specific translation is needed under the
//     hood. This is the "per-backend translator" model the user
//     selected in plan-mode confirmation — the alternative
//     ("Anthropic→OpenAI→backend" single transcoder) was rejected
//     to avoid double translation on the Ollama path and to make
//     the wire shape per-engine controllable.
//
//   - This package owns Request / Response / Content / SSE event
//     types, the stop-reason translation table, and the SSE event
//     encoder. It does NOT own backend-specific mapping logic
//     (e.g. Anthropic.system → Ollama messages[0]) — those live
//     in internal/adapter/<engine>/translate_messages.go.
//
// Spec references:
//
//   - https://docs.anthropic.com/en/api/messages (request/response)
//   - https://docs.anthropic.com/en/api/messages-streaming (SSE)
//   - https://docs.anthropic.com/en/api/messages-count-tokens
//
// Versioning: clients set the `anthropic-version` header (e.g.
// "2023-06-01"). llm-init accepts any value (or none) — see the
// `acceptVersion` helper for the loose policy and the rationale.
package anthropic
