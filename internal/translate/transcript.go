package translate

import (
	"fmt"
	"strings"
)

// The transcript surface translates a stretch of dialogue as one unit. It is
// the only route in this package that is not MTranServer-compatible.
//
//	POST /translate/transcript
//	  {from,to,segments[{id,speaker?,text}],context?,background?,glossary?}
//	  → {segments:[{id,text,error?}]}
//
// The MTran routes translate each string on its own, which is what a page
// translator needs and what a transcript cannot use: a turn reading "对" is
// "For" alone and "Yes" after a question. Translating turns together fixes
// that, but only if something guarantees one line back per turn — and a
// caller reaching the model through /v1/chat/completions has to invent that
// guarantee, parse it back, and throw away the whole group when it does not
// hold. This route owns the guarantee instead.

const (
	// maxTranscriptSegments bounds one request, as maxBatchTexts does for
	// the batch route. The turns are translated together rather than one
	// after another, so this is a bound on the prompt rather than on the
	// number of calls — but a group this size is already past what the
	// models here hold a numbered reply together for.
	maxTranscriptSegments = 64

	// maxGlossaryEntries bounds the term table a caller may attach. It is
	// repeated in every request of a meeting, so an unbounded one is paid
	// for on every group.
	maxGlossaryEntries = 256

	// maxContextLines bounds the read-only lead-in. Enough for the question
	// a "yes" is answering, which is what it exists for.
	maxContextLines = 16
)

// Per-segment failure reasons. A segment always comes back — with a
// translation, or with one of these saying why it has none. Callers match on
// them, so they are wire vocabulary rather than prose.
const (
	// segmentErrUnaligned means the model's reply could not be matched to
	// this turn. Reported rather than guessed at: a translation applied to
	// the wrong turn reads as a plausible transcript of a conversation
	// nobody had, and is not recoverable by looking at it.
	segmentErrUnaligned = "unaligned"

	// segmentErrDeadline means the retry budget ran out before this turn
	// was translated. The caller can ask again for just these.
	segmentErrDeadline = "deadline"

	// segmentErrUpstream means the model call covering this turn failed
	// while other turns in the same request succeeded.
	segmentErrUpstream = "upstream"
)

// transcriptSegment is one turn to translate.
//
// Speaker travels as a field rather than as a prefix inside Text because it
// is not part of what is being translated. A model shown "[Alice] 你好"
// mirrors the format back however plainly it is told not to, and then every
// caller needs the same cleanup.
type transcriptSegment struct {
	// ID is the caller's own name for this turn. It comes back unchanged
	// and is what the caller matches on, so it must be unique within one
	// request.
	ID      string `json:"id"`
	Speaker string `json:"speaker,omitempty"`
	Text    string `json:"text"`
}

// transcriptLine is a turn the model may read but must not translate.
//
// A separate type from transcriptSegment, and deliberately without an ID: a
// line that cannot be named cannot be expected back. This is the whole reason
// context is its own field. Sent inside the same list as the turns to
// translate — which is what a hand-built prompt does — "please ignore this"
// and "please translate this" travel in one channel, and a model that
// translates the context anyway shifts every result onto the wrong turn.
// Here the extra lines have nowhere to land and are dropped by the parser.
type transcriptLine struct {
	Speaker string `json:"speaker,omitempty"`
	Text    string `json:"text"`
}

// transcriptContext is what the model reads for sense and does not translate.
type transcriptContext struct {
	// Before is the tail of the preceding conversation, in order.
	Before []transcriptLine `json:"before,omitempty"`
}

// glossaryEntry pins one term to one rendering.
//
// Structured rather than prose so the server can hand it to whatever the
// model's own terminology mechanism is, instead of every caller rendering its
// own paragraph and hoping.
type glossaryEntry struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

type transcriptRequest struct {
	// From may be empty or "auto"; the model is told the source language
	// when one is named, which on a two-word turn is most of what it has to
	// go on.
	From       string              `json:"from"`
	To         string              `json:"to"`
	Segments   []transcriptSegment `json:"segments"`
	Context    *transcriptContext  `json:"context,omitempty"`
	Background string              `json:"background,omitempty"`
	Glossary   []glossaryEntry     `json:"glossary,omitempty"`
}

// transcriptResult is one turn's answer. Exactly one of Text and Error is set.
type transcriptResult struct {
	ID    string `json:"id"`
	Text  string `json:"text"`
	Error string `json:"error,omitempty"`
}

// transcriptResponse holds one result per requested segment, in request
// order, under the same ids. Never shorter: a short array is how a caller
// matching by position silently shifts every translation onto the wrong turn,
// and this route exists to make that unrepresentable.
type transcriptResponse struct {
	Segments []transcriptResult `json:"segments"`
}

// transcriptPlan is a validated request: the languages resolved, the caller's
// input normalised, and nothing left to check.
type transcriptPlan struct {
	from     Language
	to       Language
	autoFrom bool
	segments []transcriptSegment
	context  []transcriptLine
	// background is free text describing what the meeting is about.
	background string
	glossary   []glossaryEntry
}

// validate resolves the languages and rejects a request that cannot be
// answered. Everything it lets through produces a full-length response.
func (h *Handler) validate(req transcriptRequest) (transcriptPlan, *clientError) {
	var plan transcriptPlan

	to := strings.TrimSpace(req.To)
	if to == "" {
		return plan, &clientError{code: codeInvalidRequest, msg: msgToRequired}
	}
	resolved, ok := h.catalog.Resolve(to)
	if !ok {
		return plan, &clientError{
			code: codeInvalidRequest,
			msg:  "unsupported target language for this model: " + to,
		}
	}
	plan.to = resolved

	from := strings.TrimSpace(req.From)
	plan.autoFrom = from == "" || strings.EqualFold(from, "auto")
	if !plan.autoFrom {
		resolved, ok := h.catalog.Resolve(from)
		if !ok {
			return plan, &clientError{
				code: codeInvalidRequest,
				msg:  "unsupported source language for this model: " + from,
			}
		}
		plan.from = resolved
	}

	if len(req.Segments) == 0 {
		return plan, &clientError{code: codeInvalidRequest, msg: "segments is required"}
	}
	if len(req.Segments) > maxTranscriptSegments {
		return plan, &clientError{
			code: codeInvalidRequest,
			msg:  fmt.Sprintf("segments exceeds max of %d", maxTranscriptSegments),
		}
	}

	seen := make(map[string]struct{}, len(req.Segments))
	plan.segments = make([]transcriptSegment, 0, len(req.Segments))
	for i, sg := range req.Segments {
		id := strings.TrimSpace(sg.ID)
		if id == "" {
			return plan, &clientError{
				code: codeInvalidRequest,
				msg:  fmt.Sprintf("segments[%d].id is required", i),
			}
		}
		// Duplicated rather than tolerated: the answer is keyed by id, so a
		// caller reading results back into a map would apply one turn's
		// translation to two and never see that it had.
		if _, dup := seen[id]; dup {
			return plan, &clientError{
				code: codeInvalidRequest,
				msg:  "duplicate segment id: " + id,
			}
		}
		seen[id] = struct{}{}

		text := collapseSpace(sg.Text)
		if text == "" {
			return plan, &clientError{
				code: codeInvalidRequest,
				msg:  fmt.Sprintf("segments[%d].text is required", i),
			}
		}
		plan.segments = append(plan.segments, transcriptSegment{
			ID:      id,
			Speaker: strings.TrimSpace(sg.Speaker),
			Text:    text,
		})
	}

	if req.Context != nil {
		lines := req.Context.Before
		// Trimmed from the front: what a group needs is what was said
		// immediately before it, not the start of a stretch a page away.
		if len(lines) > maxContextLines {
			lines = lines[len(lines)-maxContextLines:]
		}
		for _, line := range lines {
			text := collapseSpace(line.Text)
			if text == "" {
				continue
			}
			plan.context = append(plan.context, transcriptLine{
				Speaker: strings.TrimSpace(line.Speaker),
				Text:    text,
			})
		}
	}

	plan.background = collapseSpace(req.Background)

	// A blank row is skipped rather than rejected. The glossary is advisory,
	// usually machine-built, and failing a page of transcript over one empty
	// term is out of proportion to what it costs to ignore it.
	for _, entry := range req.Glossary {
		if len(plan.glossary) >= maxGlossaryEntries {
			break
		}
		source := collapseSpace(entry.Source)
		target := collapseSpace(entry.Target)
		if source == "" || target == "" {
			continue
		}
		plan.glossary = append(plan.glossary, glossaryEntry{Source: source, Target: target})
	}

	return plan, nil
}

// collapseSpace folds runs of whitespace into single spaces and trims.
//
// Unlike the MTran routes, which preserve caller whitespace because a page
// translator round-trips the result into the layout it came from. Here every
// turn becomes one line of a numbered prompt, so an embedded newline would
// look to the parser like the start of the next turn.
func collapseSpace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}
