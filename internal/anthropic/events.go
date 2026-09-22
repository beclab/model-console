package anthropic

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Anthropic SSE event-type identifiers. Wire constants — clients pin
// on these exact strings (anthropic-sdk-python's
// `RawMessageStreamEvent` type alias is a literal union of these
// values). The streaming protocol fires them in a fixed sequence:
//
//	message_start
//	  ↓
//	(for each output content block: content_block_start
//	                                → content_block_delta*
//	                                → content_block_stop)
//	  ↓
//	message_delta   (carries final stop_reason + cumulative usage)
//	  ↓
//	message_stop
//
// `ping` events may interleave anywhere for keepalive (Anthropic
// sends one every ~15s). `error` events terminate the stream.
const (
	EventMessageStart      = "message_start"
	EventContentBlockStart = "content_block_start"
	EventContentBlockDelta = "content_block_delta"
	EventContentBlockStop  = "content_block_stop"
	EventMessageDelta      = "message_delta"
	EventMessageStop       = "message_stop"
	EventPing              = "ping"
	EventError             = "error"
)

// Delta block type identifiers used in content_block_delta events.
const (
	DeltaText      = "text_delta"
	DeltaInputJSON = "input_json_delta" // Stage 3: streaming tool_use input
)

// ────────────────────────────────────────────────────────────────────────
// Event payload types
// ────────────────────────────────────────────────────────────────────────

// MessageStartEvent is the payload of the `message_start` SSE event.
// The nested Message mirrors a MessagesResponse but with empty
// Content and usually empty Usage (Anthropic emits initial Usage
// {input_tokens, output_tokens:0}). Adapters MAY populate
// input_tokens here when the backend reports it early; otherwise
// defer to the final message_delta.
type MessageStartEvent struct {
	Type    string           `json:"type"`
	Message MessageStartBody `json:"message"`
}

// MessageStartBody mirrors MessagesResponse but with Content omitted
// at start (Anthropic emits `"content": []` literally). Separate
// type so we can lock the empty-array shape vs the response's
// populated array.
type MessageStartBody struct {
	ID           string         `json:"id"`
	Type         string         `json:"type"`
	Role         string         `json:"role"`
	Content      []ContentBlock `json:"content"` // always empty at start
	Model        string         `json:"model"`
	StopReason   *string        `json:"stop_reason"`   // null at start
	StopSequence *string        `json:"stop_sequence"` // null at start
	Usage        Usage          `json:"usage"`
}

// ContentBlockStartEvent fires once per output content block. Index
// is monotonic from 0; Anthropic guarantees no holes. The nested
// ContentBlock matches the eventual final block's Type but with
// empty Text / Input (deltas fill it).
type ContentBlockStartEvent struct {
	Type         string       `json:"type"`
	Index        int          `json:"index"`
	ContentBlock ContentBlock `json:"content_block"`
}

// ContentBlockDeltaEvent carries an incremental text or tool_use
// input chunk. The Delta object's own Type identifies the delta
// flavour (text_delta vs input_json_delta).
type ContentBlockDeltaEvent struct {
	Type  string     `json:"type"`
	Index int        `json:"index"`
	Delta BlockDelta `json:"delta"`
}

// BlockDelta is the inner `delta` payload of a content_block_delta
// event. Text is populated on text deltas; PartialJSON is populated
// on tool_use streaming input deltas (Stage 3).
type BlockDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}

// ContentBlockStopEvent fires once per output content block, after
// all its deltas. Index matches the start event's index.
type ContentBlockStopEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

// MessageDeltaEvent carries the final stop_reason and cumulative
// output_tokens. input_tokens is unchanged from message_start so
// usually omitted here; Anthropic still emits it on some
// transitions, but we keep it omitempty to match the most common
// shape observed in the wild.
type MessageDeltaEvent struct {
	Type  string         `json:"type"`
	Delta MessageDeltaUp `json:"delta"`
	Usage Usage          `json:"usage"`
}

// MessageDeltaUp is the inner `delta` of a message_delta event.
// Anthropic spec: stop_reason and stop_sequence appear here, not on
// the outer event. Pointer to *string so we can emit `null` when no
// sequence triggered the stop — anthropic-sdk-python pins on the
// shape and chokes on a missing key.
type MessageDeltaUp struct {
	StopReason   string  `json:"stop_reason,omitempty"`
	StopSequence *string `json:"stop_sequence"`
}

// MessageStopEvent fires last (terminal). No payload beyond Type.
type MessageStopEvent struct {
	Type string `json:"type"`
}

// PingEvent is a no-op keepalive Anthropic emits every ~15s. We can
// emit it during long-running upstreams that hold tokens to keep
// load-balancers from killing idle SSE connections.
type PingEvent struct {
	Type string `json:"type"`
}

// ErrorEvent terminates a stream with an error envelope. The Error
// field mirrors the non-streaming error body so SDKs can use one
// parser for both paths.
type ErrorEvent struct {
	Type  string `json:"type"`
	Error Error  `json:"error"`
}

// Error is Anthropic's error envelope: `{"type":"<code>","message":...}`.
// The Type field uses Anthropic's vocabulary, not HTTP status codes
// (invalid_request_error / authentication_error / permission_error /
// not_found_error / rate_limit_error / api_error / overloaded_error).
type Error struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Anthropic error type identifiers. We use a subset — the rest
// (rate_limit_error, billing_error, permission_error etc.) don't
// have llm-init equivalents and would mislead clients into retrying
// against something that won't fix it.
const (
	ErrInvalidRequest = "invalid_request_error"
	ErrNotFound       = "not_found_error"
	ErrAPIError       = "api_error"
	ErrOverloaded     = "overloaded_error"
)

// ────────────────────────────────────────────────────────────────────────
// SSE writer
// ────────────────────────────────────────────────────────────────────────

// Writer is a thin SSE encoder wired to an http.ResponseWriter. It
// handles header setup (one-time), event framing, and per-event
// flush so the client sees tokens immediately rather than at OS
// socket-buffer boundaries.
//
// The zero value is not usable; construct via NewWriter. The Writer
// is NOT safe for concurrent use by multiple goroutines — each
// stream uses one Writer driven from one goroutine.
type Writer struct {
	w       http.ResponseWriter
	flusher http.Flusher
	started bool
}

// NewWriter wraps w. Headers are NOT written until the first event
// is emitted; this lets handlers fail early with a JSON error envelope
// (200 OK + SSE headers committed would otherwise prevent the
// 4xx/5xx envelope from being usable).
func NewWriter(w http.ResponseWriter) *Writer {
	f, _ := w.(http.Flusher)
	return &Writer{w: w, flusher: f}
}

// Emit writes one SSE event with the canonical two-line frame:
//
//	event: <eventType>\n
//	data: <json-encoded payload>\n
//	\n
//
// Anthropic's spec REQUIRES the `event:` line (OpenAI's `data: ` only
// shape is rejected by anthropic-sdk-python's parser). On flushable
// writers Emit calls Flush after writing so the client sees the
// frame in real time.
//
// First call writes headers (`Content-Type: text/event-stream` +
// `Cache-Control: no-cache` + `Connection: keep-alive`) and a 200
// status; subsequent calls only emit the frame.
func (w *Writer) Emit(eventType string, payload any) error {
	if !w.started {
		w.w.Header().Set("Content-Type", "text/event-stream")
		w.w.Header().Set("Cache-Control", "no-cache")
		w.w.Header().Set("Connection", "keep-alive")
		w.w.WriteHeader(http.StatusOK)
		w.started = true
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("anthropic: marshal %s event: %w", eventType, err)
	}
	if _, err := fmt.Fprintf(w.w, "event: %s\ndata: ", eventType); err != nil {
		return err
	}
	if _, err := w.w.Write(data); err != nil {
		return err
	}
	if _, err := io.WriteString(w.w, "\n\n"); err != nil {
		return err
	}
	if w.flusher != nil {
		w.flusher.Flush()
	}
	return nil
}

// EmitError writes a final `error` event and flushes. Streams should
// call this in lieu of any further Emit calls when an upstream
// failure is observed mid-stream.
func (w *Writer) EmitError(errType, message string) error {
	return w.Emit(EventError, ErrorEvent{
		Type:  EventError,
		Error: Error{Type: errType, Message: message},
	})
}

// ────────────────────────────────────────────────────────────────────────
// Non-streaming error envelope helpers
// ────────────────────────────────────────────────────────────────────────

// WriteError emits the non-streaming Anthropic error envelope to w:
//
//	{"type":"error","error":{"type":"<code>","message":"<msg>"}}
//
// with the supplied HTTP status. Status mapping follows Anthropic's
// own table (invalid_request_error → 400, not_found_error → 404,
// api_error → 500, overloaded_error → 529). Callers pass the status
// explicitly because the same Anthropic error code can occur at
// different HTTP statuses depending on context.
// errorEnvelopeKey is the literal "type" field name used in both the
// streaming and non-streaming Anthropic error envelopes. Pulled out
// as a named constant because the same string appears throughout
// events.go in event payloads — goconst flags the repetition.
const errorEnvelopeKey = "type"

// WriteError emits the non-streaming Anthropic error envelope. See
// the helper section header above for the wire shape and status
// mapping rationale.
func WriteError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		errorEnvelopeKey: EventError,
		"error":          Error{Type: errType, Message: message},
	})
}

// ────────────────────────────────────────────────────────────────────────
// Misc helpers
// ────────────────────────────────────────────────────────────────────────

// NewMessageID returns an Anthropic-shaped message ID. Format:
// `msg_<hex>` where the hex string is short and request-unique.
// Adapters MUST NOT generate IDs themselves — collisions across
// engines (Ollama vs vLLM) would break log correlation. One helper
// here keeps the format pinned.
func NewMessageID(seed int64) string {
	return fmt.Sprintf("msg_01%016x", uint64(seed))
}
