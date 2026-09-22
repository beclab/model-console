// Package proxy implements the Adapter for vLLM, llama.cpp and SGLang
// engines. The three are nearly indistinguishable from llm-init's
// perspective: each speaks the OpenAI HTTP surface natively, so the
// adapter is a httputil.ReverseProxy that
//
//  1. rewrites the JSON "model" field on /v1/{chat/completions,
//     completions, embeddings, rerank, responses} to whatever cfg.Model.Name is, and
//  2. strips the inbound Authorization header before forwarding.
//
// Differences between the three engines (e.g. SGLang's tokenize / Lua
// extensions) are intentionally not surfaced — they live on paths the
// proxy passes through verbatim.
package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/llm-init/llm-init/internal/config"
)

// maxRewriteBody is the cap for text-only rewrite paths (completions,
// and /v1/embeddings unless ENGINE_KIND=clipembed). 1 MiB is plenty for
// prompt text; oversized bodies are rejected with 413 rather than
// buffered into memory.
const maxRewriteBody = 1 << 20 // 1 MiB

// maxVisionRewriteBody is the cap for paths that can legitimately carry
// inline base64 image/audio (vision) payloads: chat completions (incl.
// the /api/chat/completions OpenWebUI alias), responses, and
// /v1/embeddings when ENGINE_KIND=clipembed (multimodal CLIP image
// data URLs). A single VLM/clip request with an embedded image blows
// past the 1 MiB text cap, so these admit up to 16 MiB — matching the
// ollama / anthropic message caps so the same image works regardless
// of which engine is behind us.
const maxVisionRewriteBody = 16 << 20 // 16 MiB

// HTTP header names used across the proxy. Net/http canonicalises
// these on Set/Get, so the literal-vs-constant decision is purely
// goconst hygiene; named constants keep rewrite.go and adapter.go
// (which both write Content-Type on error envelopes) in sync without
// the linter complaining about the cross-file repetition.
const (
	headerContentType   = "Content-Type"
	headerContentLength = "Content-Length"
	headerAuthorization = "Authorization"

	contentTypeJSON = "application/json"
	contentTypeSSE  = "text/event-stream"

	// JSON envelope keys for the error responses emitted by both
	// rewrite.go (4xx/413 admission errors) and adapter.go (502/500
	// upstream / panic errors). Shape is the unified error envelope
	// used everywhere; controlplane/handlers.go has the same
	// shape via the errorBody / errorPayload structs. Per-package
	// constants because the two packages would otherwise need a
	// shared internal/errs package for what is fundamentally just
	// three field names.
	jsonKeyError   = "error"
	jsonKeyCode    = "code"
	jsonKeyMessage = "message"

	// Error code values emitted by the rewrite admission path. These
	// are operator-visible (clients alert on the code), so changing
	// them is a wire-shape change.
	errCodeRewriteBodyTooLarge = "rewrite_body_too_large"

	// pathChatCompletions / pathCompletions / pathEmbeddings are the
	// OpenAI request paths whose JSON body the proxy rewrites. Pulled
	// out as constants so rewritePaths and the alias / route lookups
	// share one source of truth.
	pathChatCompletions = "/v1/chat/completions"
	pathCompletions     = "/v1/completions"
	pathEmbeddings      = "/v1/embeddings"
	pathRerank          = "/v1/rerank"
	pathResponses       = "/v1/responses"

	// pathAPIChatCompletions is OpenWebUI's chat alias;
	// it carries the same vision payloads as /v1/chat/completions.
	pathAPIChatCompletions = "/api/chat/completions"
)

// rewritePaths is the set of OpenAI request paths whose JSON body we may
// need to rewrite. /v1/models and /api/* are passed through as-is.
var rewritePaths = map[string]struct{}{
	pathChatCompletions:    {},
	pathCompletions:        {},
	pathEmbeddings:         {},
	pathRerank:             {},
	pathResponses:          {},
	pathAPIChatCompletions: {},
}

// rewriteBodyCap returns the admission cap for path + ENGINE_KIND.
// Vision-capable paths (chat completions + responses) always get the
// larger cap so embedded images are forwarded instead of 413'd.
//
// /v1/embeddings depends on kind (not MODEL_MODE):
//   - clipembed: 16 MiB (inline image_url data URLs are first-class)
//   - embed and every other kind: 1 MiB (text-only embedding)
func rewriteBodyCap(path string, kind config.EngineKind) int {
	switch path {
	case pathChatCompletions, pathAPIChatCompletions, pathResponses:
		return maxVisionRewriteBody
	case pathEmbeddings:
		if kind == config.EngineClipEmbed {
			return maxVisionRewriteBody
		}
		return maxRewriteBody
	default:
		return maxRewriteBody
	}
}

// rewriteModelInBody mutates req.Body in place, replacing the top-level
// JSON "model" field with modelName. If the body is not JSON or has no
// "model" field, the body is restored intact.
//
// Pre-v1.0.5 a body exceeding maxRewriteBody was passed through
// verbatim — this LEAKED the OpenAI alias `model` field upstream
// (e.g. a request to llm-init for "qwen2.5-7b" would forward "gpt-4o"
// to the engine if the body was big enough), which silently broke the
// "consistent model identity" contract.
//
// The right precondition is "reject oversized bodies before they reach
// the proxy hot path" — see rewriteAdmission below, called by the
// OpenAIHandler middleware. As of v1.0.5 admission consumes the body
// for both Content-Length and chunked-encoding requests, so by the
// time the Director runs req.Body is either an in-memory bytes.Reader
// of the buffered (and length-bounded) payload or untouched (non-
// rewrite path). A length check is retained as a defence-in-depth
// invariant; if it ever trips, that's a bug in the admission filter
// upstream of us, not a runtime fallback path.
//
// req.ContentLength and the Content-Length header are kept synchronised
// after rewrite so upstream parsers don't deadlock waiting for bytes
// that never arrive.
func rewriteModelInBody(req *http.Request, modelName string, kind config.EngineKind) {
	if _, ok := rewritePaths[req.URL.Path]; !ok {
		return
	}
	if req.Body == nil {
		return
	}

	cap := rewriteBodyCap(req.URL.Path, kind)
	body, err := io.ReadAll(io.LimitReader(req.Body, int64(cap)+1))
	_ = req.Body.Close()
	if err != nil {
		// Restore an empty body so the upstream gets a valid (empty)
		// request rather than a partially-consumed stream.
		req.Body = io.NopCloser(bytes.NewReader(nil))
		req.ContentLength = 0
		return
	}
	if len(body) > cap {
		// This branch is now an invariant: rewriteAdmission has already
		// drained chunked bodies and rejected oversized ones with 413
		// before we get here. Hitting this means a future refactor
		// reintroduced the bypass. Refuse to leak the alias upstream;
		// truncate to an empty body so the engine returns a clean
		// (parseable) error rather than the silently-aliased pass-
		// through pre-v1.0.5 shipped.
		slog.Error("proxy: body exceeded cap inside Director despite admission filter",
			slog.String("path", req.URL.Path),
			slog.Int("read", len(body)),
			slog.Int("cap", cap))
		req.Body = io.NopCloser(bytes.NewReader(nil))
		req.ContentLength = 0
		return
	}

	rewritten, ok := rewriteJSONModel(body, modelName)
	if !ok {
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
		return
	}
	req.Body = io.NopCloser(bytes.NewReader(rewritten))
	req.ContentLength = int64(len(rewritten))
	req.Header.Set("Content-Length", itoa(int64(len(rewritten))))
}

// rewriteAdmission returns reject=true (and writes a 4xx response) when
// the inbound request cannot proceed to the proxy hot path, and otherwise
// the payload it buffered — nil for a path whose body is passed through
// untouched. Handing the bytes back rather than making the caller read
// req.Body again is what keeps the request-parameter log line free: the
// body is already in memory here, and re-reading it would copy up to
// 16 MiB of a vision payload for one line of text.
//
// Two reject reasons:
//
//   - 413 Payload Too Large: body is bigger than rewriteBodyCap(path, kind).
//     This previously fired only on Content-Length; v1.0.5 closes the
//     chunked-encoding bypass by also reading up to cap+1 bytes from
//     the body when ContentLength is -1 (chunked) or 0-but-streamed.
//   - 400 Bad Request: body read failed mid-stream (timeout, abort).
//     Less common but the same envelope contract applies.
//
// On the success path admission REPLACES req.Body with a bytes.Reader
// over the already-drained payload and sets req.ContentLength to the
// real read length. The Director then reads from memory; there is no
// second wire read, so a chunked body cannot grow past the cap after
// the check passes.
//
// reject=true ⇒ caller must NOT continue the request; the response has
// already been written.
//
// Header semantics for 413:
//
//   - Status: 413 Payload Too Large.
//   - X-Llm-Init-Max-Rewrite-Bytes: the path/kind cap so a client can
//     size payloads (1 MiB text vs 16 MiB vision/clipembed embeddings).
//   - Body: the standard {"error":{"code":"rewrite_body_too_large",
//     "message":...}} envelope so OpenAI Python / LangChain /
//     OpenWebUI parse the failure cleanly.
func rewriteAdmission(w http.ResponseWriter, req *http.Request, kind config.EngineKind) (body []byte, reject bool) {
	if _, ok := rewritePaths[req.URL.Path]; !ok {
		return nil, false
	}
	cap := rewriteBodyCap(req.URL.Path, kind)
	// Fast-path Content-Length reject — don't touch the body if the
	// client already advertised an oversized payload.
	if req.ContentLength > int64(cap) {
		writeRewriteTooLarge(w, req, req.ContentLength, cap)
		return nil, true
	}
	if req.Body == nil {
		return nil, false
	}
	// Drain the body up to cap+1 bytes. This handles three cases:
	//   1. Content-Length set, within cap: read returns the body.
	//   2. Chunked / unknown length: read returns the actual bytes
	//      streamed; if it exceeds the cap we 413.
	//   3. Lying Content-Length (header says small, body is huge):
	//      same as (2).
	// Any of these reaches the same enforcement point, which is the
	// gap closed in v1.0.5: previously chunked > cap silently passed
	// through verbatim with the OpenAI alias model field unrewritten.
	buf, err := io.ReadAll(io.LimitReader(req.Body, int64(cap)+1))
	_ = req.Body.Close()
	if err != nil {
		w.Header().Set(headerContentType, contentTypeJSON)
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			jsonKeyError: map[string]string{
				jsonKeyCode:    "request_body_read_failed",
				jsonKeyMessage: "request body could not be read fully; client likely aborted",
			},
		})
		slog.Warn("proxy: failed to read rewrite body",
			slog.String("path", req.URL.Path),
			slog.String("err", err.Error()))
		return nil, true
	}
	if len(buf) > cap {
		writeRewriteTooLarge(w, req, int64(len(buf)), cap)
		return nil, true
	}
	// Replace the body with a bytes.Reader so the Director (and
	// any retry inside http.RoundTrip) reads from memory. Keep
	// ContentLength accurate; the proxy uses it to pick chunked
	// vs Content-Length on the wire to upstream, and a stale -1
	// here would force chunked re-encoding that some engines parse
	// less efficiently.
	req.Body = io.NopCloser(bytes.NewReader(buf))
	req.ContentLength = int64(len(buf))
	req.Header.Set(headerContentLength, itoa(int64(len(buf))))
	// Strip Transfer-Encoding now that we're framed by Content-Length;
	// otherwise httputil.ReverseProxy may forward both, which some
	// engines treat as malformed.
	req.TransferEncoding = nil
	req.Header.Del("Transfer-Encoding")
	return buf, false
}

// writeRewriteTooLarge emits the canonical 413 envelope. Extracted so
// the Content-Length and chunked rejection paths agree byte-for-byte
// on the response shape (clients write conformance tests against this).
func writeRewriteTooLarge(w http.ResponseWriter, req *http.Request, observed int64, cap int) {
	w.Header().Set(headerContentType, contentTypeJSON)
	w.Header().Set("X-Llm-Init-Max-Rewrite-Bytes", itoa(int64(cap)))
	w.WriteHeader(http.StatusRequestEntityTooLarge)
	capMiB := cap / (1 << 20)
	_ = json.NewEncoder(w).Encode(map[string]any{
		jsonKeyError: map[string]string{
			jsonKeyCode: errCodeRewriteBodyTooLarge,
			jsonKeyMessage: fmt.Sprintf(
				"request body exceeds the model-rewrite cap (%d MiB); reduce payload size or address the engine directly",
				capMiB),
		},
	})
	slog.Warn("proxy: rejected oversized rewrite body",
		slog.String("path", req.URL.Path),
		slog.Int64("observed_bytes", observed),
		slog.Int("cap", cap))
}

// rewriteJSONModel returns body with body["model"] set to modelName, or
// (body, false) if body is not a JSON object.
func rewriteJSONModel(body []byte, modelName string) ([]byte, bool) {
	trimmed := bytes.TrimLeft(body, " \t\r\n")
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return body, false
	}
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return body, false
	}
	obj["model"] = modelName
	out, err := json.Marshal(obj)
	if err != nil {
		return body, false
	}
	return out, true
}

// itoa avoids pulling strconv into the rewrite hot path. The values we
// emit are small non-negative content lengths so the simple loop is
// faster than strconv.Itoa.
func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for n > 0 {
		pos--
		buf[pos] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[pos:])
}

// stripAuthorization removes the Authorization header. The HF / control
// plane bearer (if any) must never reach the upstream engine because the
// engine is a sibling container with no concept of llm-init's auth.
func stripAuthorization(h http.Header) {
	h.Del(headerAuthorization)
	// Some clients send "Proxy-Authorization" too; strip it for the same
	// reason. The HTTP spec says it is hop-by-hop so technically Go's
	// reverse proxy already drops it, but being explicit costs nothing.
	h.Del("Proxy-Authorization")
}

// joinPath ensures we don't double-slash when stitching the upstream
// base URL onto an inbound path.
func joinPath(base, path string) string {
	base = strings.TrimRight(base, "/")
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return base + path
}
