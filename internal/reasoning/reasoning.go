// Package reasoning holds the one vocabulary for reasoning-intensity
// values on this side of the wire.
//
// MIRROR OF: router/backend/internal/core/reasoning. Router owns the
// value domain because it is what a client sends; this package exists so
// the two places here that read those values -- the card validator that
// decides which levels a deployment may declare, and the Ollama adapter
// that translates a level into that daemon's native `think` -- agree on
// the spelling without retyping it.
//
// EffortThinking is deliberately absent. It is Router's own literal for
// a model whose upstream metadata says it thinks but never says at what
// intensities, and it never reaches an engine: Router either translates
// it into a vendor's thinking object or strips it before forwarding. A
// local deployment always knows which levels its engine and chat
// template accept, so a card written here names them.
package reasoning

import "strings"

// Param is the request field carrying the intent, and the name a model
// card's parameter rule must have to be projected onto Router's
// /v1/models. `think`, `enable_thinking` and `chat_template_kwargs` are
// translation details inside an adapter; none of them reaches a client.
const Param = "reasoning_effort"

// The effort ladder, weakest first. These are OpenAI's values plus the
// two llama.cpp and Ollama added above `high`.
const (
	EffortNone    = "none"
	EffortMinimal = "minimal"
	EffortLow     = "low"
	EffortMedium  = "medium"
	EffortHigh    = "high"
	EffortXHigh   = "xhigh"
	EffortMax     = "max"
)

// Ladder is the full value domain, weakest effort first. The order is
// the point: an option list is rendered as a slider or a menu, and
// sorting these alphabetically would put high before low.
var Ladder = []string{
	EffortNone, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax,
}

// offSpellings are the three ways a client says "do not think". `off`
// and `disabled` predate OpenAI's `none` and still arrive.
var offSpellings = map[string]struct{}{
	EffortNone: {},
	"off":      {},
	"disabled": {},
}

// Normalize trims and lowercases an effort value, so a stray " High "
// is one level and not an unknown one.
func Normalize(effort string) string {
	return strings.ToLower(strings.TrimSpace(effort))
}

// IsOff reports whether an already-normalized value means "do not think".
func IsOff(effort string) bool {
	_, ok := offSpellings[effort]
	return ok
}

// IsLevel reports whether an already-normalized value is on the ladder.
func IsLevel(effort string) bool {
	for _, l := range Ladder {
		if l == effort {
			return true
		}
	}
	return false
}
