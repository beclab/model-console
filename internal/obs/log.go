// Package obs supplies cross-cutting observability primitives: log
// redaction and the Prometheus metric registry.
//
// All log messages and structured attribute values pass through Redact so
// HF / OpenAI / Bearer tokens never leak into stdout. Metrics use a private
// Registry to avoid colliding with any embedding host that already collects
// /metrics for itself.
package obs

import (
	"context"
	"log/slog"
	"regexp"
	"strings"
)

// patterns drives Redact. Each entry is a regex describing a token shape
// found in the wild; the substitution is the placeholder.
//
// We deliberately keep the list short and conservative: each new pattern
// risks redacting innocent strings (e.g. log lines that mention "Bearer" in
// docs). Add patterns only when they are known-bad and high-signal.

// RedactedBearer is the canonical replacement string for matched Bearer
// tokens. Exported because log_test.go asserts on it; centralising the
// literal here means a future "use [REDACTED] instead" decision is a
// one-site edit.
const RedactedBearer = "Bearer ***"

var redactPatterns = []struct {
	re   *regexp.Regexp
	repl string
}{
	// HTTP Authorization: Bearer XXXX (rest of header). Stop at whitespace,
	// quote, or comma so we do not eat unrelated trailing text.
	{regexp.MustCompile(`(?i)bearer\s+[a-z0-9._\-]+`), RedactedBearer},
	// Hugging Face tokens: prefix `hf_` followed by 30+ chars.
	{regexp.MustCompile(`hf_[A-Za-z0-9]{20,}`), "hf_***"},
	// OpenAI / sk-style tokens: prefix `sk-` followed by 20+ chars.
	{regexp.MustCompile(`sk-[A-Za-z0-9_\-]{20,}`), "sk-***"},
	// query string ?token=... / &token=... up to next & or end.
	{regexp.MustCompile(`([?&]token)=[^&\s]*`), `$1=***`},
	// query string ?key=... / &api_key=... — same idea, common in
	// pre-signed URLs (e.g. S3, OSS).
	{regexp.MustCompile(`([?&](?:api_key|apikey|signature|sig|x-amz-signature))=[^&\s]*`), `$1=***`},
}

// Redact replaces sensitive substrings in s with placeholders.
//
// The function is best-effort: it covers the cases we know about and is
// safe against empty / very-large inputs. It returns s unchanged when no
// pattern matches (zero-allocation fast path).
func Redact(s string) string {
	if s == "" {
		return s
	}
	for _, p := range redactPatterns {
		s = p.re.ReplaceAllString(s, p.repl)
	}
	return s
}

// RedactingHandler wraps a slog.Handler and applies Redact to every string
// attribute value (including the message) before delegating.
//
// Non-string values pass through verbatim. Group nesting is preserved.
type RedactingHandler struct {
	inner slog.Handler
}

// NewRedactingHandler wraps inner. Use NewLogger as a convenience.
func NewRedactingHandler(inner slog.Handler) *RedactingHandler {
	return &RedactingHandler{inner: inner}
}

// Enabled implements slog.Handler by delegating to the wrapped inner
// handler -- redaction is a transformation, not a level filter.
func (h *RedactingHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.inner.Enabled(ctx, level)
}

// Handle implements slog.Handler. It redacts the record's message and
// every attribute (recursively for groups) before forwarding to the
// inner handler.
func (h *RedactingHandler) Handle(ctx context.Context, r slog.Record) error {
	r.Message = Redact(r.Message)
	clone := slog.NewRecord(r.Time, r.Level, r.Message, r.PC)
	r.Attrs(func(a slog.Attr) bool {
		clone.AddAttrs(redactAttr(a))
		return true
	})
	return h.inner.Handle(ctx, clone)
}

// WithAttrs implements slog.Handler. The supplied attrs are redacted
// before being attached to the returned handler so subsequent log
// calls inherit clean values.
func (h *RedactingHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	red := make([]slog.Attr, len(attrs))
	for i, a := range attrs {
		red[i] = redactAttr(a)
	}
	return &RedactingHandler{inner: h.inner.WithAttrs(red)}
}

// WithGroup implements slog.Handler. The group name itself is not
// redacted (it is operator-controlled, not user input); attrs added
// under the group are redacted as they pass through Handle.
func (h *RedactingHandler) WithGroup(name string) slog.Handler {
	return &RedactingHandler{inner: h.inner.WithGroup(name)}
}

// redactAttr returns a deep-copied attr with all string leaves redacted.
func redactAttr(a slog.Attr) slog.Attr {
	v := a.Value
	switch v.Kind() {
	case slog.KindString:
		return slog.String(a.Key, Redact(v.String()))
	case slog.KindGroup:
		grp := v.Group()
		out := make([]slog.Attr, len(grp))
		for i, g := range grp {
			out[i] = redactAttr(g)
		}
		return slog.Attr{Key: a.Key, Value: slog.GroupValue(out...)}
	case slog.KindAny:
		// Stringify only obvious string-bearing types; anything else is
		// considered structured data the producer chose to log explicitly.
		if s, ok := v.Any().(string); ok {
			return slog.String(a.Key, Redact(s))
		}
		// fmt.Stringer values may carry tokens too (e.g. URL.String()).
		if str, ok := v.Any().(interface{ String() string }); ok && !looksStructured(v.Any()) {
			return slog.String(a.Key, Redact(str.String()))
		}
		return a
	default:
		return a
	}
}

// looksStructured filters out a few common types whose String() method is
// non-redaction-relevant (e.g. time.Duration). Conservative: when in doubt
// we keep the value as-is.
func looksStructured(x any) bool {
	switch x.(type) {
	case interface{ Nanoseconds() int64 }, // time.Duration
		interface{ UnixNano() int64 }: // time.Time
		return true
	}
	return false
}

// HasSensitiveMarker is a tiny helper used by tests and CLI to avoid logging
// strings that obviously contain a token. Not a security boundary.
func HasSensitiveMarker(s string) bool {
	return strings.Contains(s, "Bearer ") ||
		strings.Contains(s, "hf_") ||
		strings.Contains(s, "sk-")
}
