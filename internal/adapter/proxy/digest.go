// Reporting which knobs a request carried, without reporting what it said.
//
// MIRROR OF: router/backend/internal/dataplane/paramx/digest.go. The two ends
// of one hop each summarize the request they saw, and the summaries are only
// worth comparing if they are spelled the same way -- a `reasoning_effort=low`
// on the gateway's line and a `reasoning-effort: low` on this one is a diff a
// reader has to do by hand. There is no shared package to hold this: the two
// repositories ship separately, and one JSON walk is not worth a dependency.
// Adding a key here means adding it there, and the reverse.
//
// The allowlist is the load-bearing half. This proxy sees the whole body,
// content and all, so a denylist would log the next vendor field carrying user
// text from the day the vendor ships it, by a rule nobody revisits.
//
// Why this exists at all: for a llama.cpp or vLLM sibling the adapter is a
// reverse proxy that rewrites one field, so a successful request used to leave
// nothing behind but a Prometheus counter. Whether a level a client swears it
// sent survived the two hops in front of this one was unanswerable after the
// fact, and the engines do not log what they were asked for either.

package proxy

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/llm-init/llm-init/internal/reasoning"
)

// digestScalarKeys are the parameters rendered by value, in this order. Each
// has a domain of numbers, bools or short enums, so the value is the useful
// part and cannot be a sentence. The fixed order keeps one request rendering
// the same way every time, which map iteration would not.
var digestScalarKeys = []string{
	reasoning.Param,
	"verbosity",
	"enable_thinking",
	"temperature",
	"top_p",
	"top_k",
	keyMaxTokens,
	"max_completion_tokens",
	"max_output_tokens",
	"n",
	"seed",
	keyStream,
	"parallel_tool_calls",
	"tool_choice",
}

// digestCountKeys are the content-bearing arrays, reported as a length. How
// many messages a request carried explains a token count; what they said
// explains nothing this log exists to answer.
var digestCountKeys = []string{
	keyMessages,
	"input",
	"tools",
}

// digestMaxValueLen caps a rendered scalar. Every allowlisted key has a small
// domain, so a longer value means the client sent something unexpected, and
// this log line is not the place to find out how long it was.
const digestMaxValueLen = 32

// paramDigest renders a one-line summary of the request parameters body
// carries, as space-separated `key=value` pairs in a fixed order. It returns
// "" for a body that is not a JSON object, or one that carries nothing on the
// allowlist, so the caller can leave the log field off.
func paramDigest(body []byte) string {
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(body, &doc); err != nil {
		return ""
	}
	var parts []string
	for _, key := range digestScalarKeys {
		if v, ok := digestScalar(doc[key]); ok {
			parts = append(parts, key+"="+v)
		}
	}
	parts = append(parts, digestNested(doc)...)
	for _, key := range digestCountKeys {
		if n, ok := digestLen(doc[key]); ok {
			parts = append(parts, key+"="+strconv.Itoa(n))
		}
	}
	return strings.Join(parts, " ")
}

// digestNested renders the two allowlisted keys whose value is an object.
//
// `response_format` is reduced to its `type` because the rest of it is a JSON
// schema, and a schema names fields the caller's own data is shaped like.
// `chat_template_kwargs` is rendered by key name only: what it does -- steer
// the engine's chat template -- is exactly what this log is for, but it is an
// open bag, and some templates accept a system prompt in it.
func digestNested(doc map[string]json.RawMessage) []string {
	var parts []string
	if raw, ok := doc["response_format"]; ok {
		var obj struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(raw, &obj); err == nil && obj.Type != "" {
			parts = append(parts, "response_format="+truncateDigestValue(obj.Type))
		}
	}
	if keys := digestObjectKeys(doc["chat_template_kwargs"]); len(keys) > 0 {
		parts = append(parts, "chat_template_kwargs={"+strings.Join(keys, ",")+"}")
	}
	return parts
}

// digestScalar renders a JSON string, number or bool, reporting false for
// anything else -- an object under a scalar key is a malformed request, and the
// engine's own error is a better account of it than a mangled log field.
func digestScalar(raw json.RawMessage) (string, bool) {
	var value any
	if len(raw) == 0 || json.Unmarshal(raw, &value) != nil {
		return "", false
	}
	switch v := value.(type) {
	case string:
		if v == "" {
			return "", false
		}
		return truncateDigestValue(v), true
	case bool, float64:
		out, err := json.Marshal(v)
		if err != nil {
			return "", false
		}
		return string(out), true
	}
	return "", false
}

// digestLen reports the element count of a JSON array, false for anything
// else. A Responses `input` may also be a bare string; it is skipped rather
// than counted as one item, which would read as a one-message conversation.
func digestLen(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var items []json.RawMessage
	if json.Unmarshal(raw, &items) != nil {
		return 0, false
	}
	return len(items), true
}

// digestObjectKeys returns the sorted key names of a JSON object, nil for
// anything else. Sorted so two requests with the same kwargs render the same
// line whatever order they were serialized in.
func digestObjectKeys(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, truncateDigestValue(k))
	}
	sort.Strings(keys)
	return keys
}

// truncateDigestValue makes a string safe to sit inside a space-separated
// line: whitespace becomes an underscore so one value cannot masquerade as two
// fields, and the result is capped.
func truncateDigestValue(s string) string {
	s = strings.Map(func(r rune) rune {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			return '_'
		}
		return r
	}, s)
	runes := []rune(s)
	if len(runes) > digestMaxValueLen {
		return string(runes[:digestMaxValueLen]) + "…"
	}
	return s
}

// digestModel reads back the top-level model field, which is the one thing the
// director rewrites. Logging both sides of that rewrite is what turns "the
// engine answered as X" into "the client asked for Y and got X".
func digestModel(body []byte) string {
	var doc struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return ""
	}
	return truncateDigestValue(doc.Model)
}
