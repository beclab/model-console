package kvbudget

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
)

// This file answers "how much of the pool will this request take", which has
// two halves that fail differently.
//
// The prompt is countable, and counting it exactly means asking the engine --
// two local round trips, done in tokenize.go and only when the answer could
// change the decision. The completion is not countable at all: how many
// tokens a model will generate is known when it stops. So the reservation is
// a prompt measurement plus an output bound, and the bound is the part that
// makes this gate conservative on purpose.

// maxBodyBytes caps how much of a request body is read to size it. A body
// past this is charged by its length rather than parsed, which is the safe
// direction: the charge is an over-estimate and the request is still
// forwarded in full.
const maxBodyBytes = 8 << 20

// defaultOutputReserve is what a request that names no output ceiling is
// charged for its completion.
//
// Deliberately not `context_size - prompt`, which is what the engine will
// actually allow it to generate: charging that would let one request reserve
// the entire pool and hold every other caller at the door behind it, for a
// completion that in practice ends after a few hundred tokens. A fixed bound
// is wrong in both directions and only one of them is recoverable -- charge
// too little and the pool can still be oversubscribed by requests that all
// run long, charge the whole window and the gate becomes a single-file queue.
const defaultOutputReserve = 2048

// bodyOf reads and restores a request body so the handler downstream still
// sees it. The gate sits in front of a reverse proxy that forwards the body
// verbatim, so anything read here has to be put back.
func bodyOf(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	raw, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes+1))
	_ = r.Body.Close()
	if err != nil {
		// Restore what was read even on a failure: the proxy's own error
		// is a better report than one invented here.
		r.Body = io.NopCloser(bytes.NewReader(raw))
		return nil, err
	}
	r.Body = io.NopCloser(bytes.NewReader(raw))
	r.ContentLength = int64(len(raw))
	return raw, nil
}

// requestShape is the part of an OpenAI request body this gate reads. Every
// field is optional: an unparseable or unfamiliar body still gets a charge,
// from its length.
type requestShape struct {
	Prompt    json.RawMessage `json:"prompt"`
	Messages  json.RawMessage `json:"messages"`
	MaxTokens *int            `json:"max_tokens"`
	// MaxCompletionTokens is what the newer OpenAI clients send. Reading
	// only max_tokens would charge every one of them the default.
	MaxCompletionTokens *int `json:"max_completion_tokens"`
}

// generates reports whether the body asks the model to produce something,
// which is the same question as whether it will occupy KV cache.
//
// The two fields are the two ways an OpenAI-compatible request states a
// prompt, and between them they cover every route that puts tokens in the
// cache on a chat deployment. A body with neither is a metadata call, a
// malformed request, or a shape this package does not model -- and none of
// those should be charged for a completion.
func (s requestShape) generates() bool {
	return len(s.Messages) > 0 || len(s.Prompt) > 0
}

// outputReserve is what the body asks to be allowed to generate, bounded by
// what the card says the model will give back.
//
// A stated ceiling above the card's is not honoured, because the engine will
// not honour it either; charging for tokens that cannot be generated would
// reserve pool nobody can use.
func outputReserve(shape requestShape, cardMax int) int {
	stated := 0
	if shape.MaxCompletionTokens != nil && *shape.MaxCompletionTokens > 0 {
		stated = *shape.MaxCompletionTokens
	}
	if shape.MaxTokens != nil && *shape.MaxTokens > 0 {
		stated = *shape.MaxTokens
	}
	if stated <= 0 {
		stated = defaultOutputReserve
		if cardMax > 0 && cardMax < stated {
			stated = cardMax
		}
		return stated
	}
	if cardMax > 0 && stated > cardMax {
		return cardMax
	}
	return stated
}

// promptUpperBound is the most tokens a body of this length can contain.
//
// One byte per token, because that is the floor: a BPE vocabulary contains
// single-byte tokens, so nothing rules out a body that tokenizes one token
// per byte. It is a loose bound for ordinary text -- English averages around
// four bytes a token, CJK three bytes for often a single token, and the JSON
// envelope around the content is counted too -- and loose in the safe
// direction.
//
// Its job is not to be the charge. It is to answer "could tokenizing
// possibly change the decision", so the two round trips in tokenize.go are
// paid only by requests near the capacity line rather than by every short
// one.
func promptUpperBound(body []byte) int {
	return len(body)
}
