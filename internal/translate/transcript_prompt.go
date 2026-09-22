package translate

import (
	"fmt"
	"strconv"
	"strings"
)

// transcriptSystem is the instruction one group is translated under.
//
// Written in English, like DefaultPrompt and for the same reason: this route
// serves whatever language pair the deployed model declares, and rules in the
// caller's language would be one more thing every caller has to agree on. The
// languages inside it are named rather than coded — telling a model to
// translate into "en" tells it less than telling it to translate into English.
func transcriptSystem(plan transcriptPlan) string {
	var b strings.Builder
	b.WriteString("You are translating a transcript of spoken conversation")
	if !plan.autoFrom {
		fmt.Fprintf(&b, " from %s", languageName(plan.from))
	}
	fmt.Fprintf(&b, " into %s.\n", languageName(plan.to))

	b.WriteString("Rules:\n" +
		"1. Output one line per numbered input line, keeping the numbers and their order. Never merge or split lines.\n" +
		"2. A name in square brackets is a speaker label. Do not translate it and do not repeat it in your output.\n" +
		"3. Everything above the numbered lines is reference. Read it, then translate only the numbered lines.\n" +
		"4. This is speech. Translate acknowledgements, filler and half-finished sentences as speech; do not write in what was not said, and do not turn it into prose.\n" +
		"5. For people, places and organisations, use the established rendering in the target language when there is one and keep the original when there is not. Keep product names, code identifiers, filenames and acronyms as they are. One name is written one way throughout.\n" +
		"6. Apart from what rule 5 keeps, leave no source-language words in the translation.\n" +
		"7. A line may be the second half of the sentence the line before it started. Read the whole sentence, then output only the part belonging to this line: do not repeat what the previous line already covered, and do not pull the next line's content forward.\n" +
		"8. Output nothing else: no explanations, no headings, no blank lines.\n")

	if plan.background != "" {
		fmt.Fprintf(&b, "\nBackground (reference only; do not translate or output this):\n%s\n", plan.background)
	}
	if len(plan.glossary) > 0 {
		b.WriteString("\nTerms (reference only; use these renderings, do not output this):\n")
		for _, entry := range plan.glossary {
			fmt.Fprintf(&b, "%s = %s\n", entry.Source, entry.Target)
		}
	}
	return b.String()
}

// transcriptLeadIn renders the turns preceding a group, to be appended to the
// system message rather than shown beside the turns to translate.
//
// Beside them is where it was, and a small translation model translated the
// lead-in as turn one and shifted every answer onto the wrong turn — four runs
// out of four against Hy-MT2-1.8B, and undetectable downstream because the
// reply still has one line per turn. The same text carried in the system
// message, where Background and Terms already sit, translated all turns
// correctly. A model that translates whatever is in front of it needs the
// material it must not translate somewhere else entirely.
func transcriptLeadIn(before []transcriptLine) string {
	if len(before) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\nWhat was said just before (reference only; do not translate or output this):\n")
	for _, line := range before {
		b.WriteString(labelled(line.Speaker, line.Text))
		b.WriteString("\n")
	}
	return b.String()
}

// transcriptUser renders the turns to translate, numbered from one.
//
// Numbered from one in every call, including the halves a failed group is
// split into, so what the model is asked for never depends on where the turns
// sat in the caller's request.
func transcriptUser(segments []transcriptSegment) string {
	var b strings.Builder
	b.WriteString("Lines to translate:\n")
	for i, sg := range segments {
		fmt.Fprintf(&b, "%d. %s\n", i+1, labelled(sg.Speaker, sg.Text))
	}
	return b.String()
}

func labelled(speaker, text string) string {
	if speaker == "" {
		return text
	}
	return "[" + speaker + "] " + text
}

// languageName names a language for the prompt, falling back to the code when
// the catalog carries no name for it — a bare code still says more than
// nothing.
func languageName(l Language) string {
	if l.NameEN != "" {
		return l.NameEN
	}
	return l.Code
}

// transcriptMaxTokens caps one call's reply, sized from the turns because a
// translation is about as long as what it translates.
//
// Left unset, a local model generates until it decides to stop: a thinking
// model spent 492 tokens on eight short lines and printed the answer twice.
// The headroom on top covers models that reason before answering out of the
// same budget, and it is deliberately generous — a reply cut off mid-line
// fails the alignment check, which costs the whole call rather than the line.
func transcriptMaxTokens(segments []transcriptSegment) int {
	chars := 0
	for _, sg := range segments {
		chars += len([]rune(sg.Text))
		chars += len([]rune(sg.Speaker))
	}
	return 2000 + 2*chars
}

// estimateTokens is a rough token count for a prompt, used only to decide
// whether a call fits in what the engine serves one conversation at a time.
//
// Deliberately coarse, and biased high: an overestimate costs one extra
// split, an underestimate costs a truncated reply, a retry truncated the same
// way, and the split anyway. One token per CJK character and one per three
// characters otherwise, which is close enough on either side of a bilingual
// transcript without a tokenizer this package has no other use for.
func estimateTokens(s string) int {
	cjk, other := 0, 0
	for _, r := range s {
		if ideograph(r) {
			cjk++
			continue
		}
		other++
	}
	return cjk + (other+2)/3
}

// parseNumbered reads a numbered reply back into one string per input line,
// returning nil when the reply does not line up.
//
// Every line has to be accounted for. A model that merges two turns or adds a
// preamble produces a reply that still parses, and applying it would shift
// every remaining translation onto the wrong turn — which reads as a plausible
// transcript of a conversation nobody had.
func parseNumbered(answer string, want int) []string {
	out := make([]string, want)
	filled := 0

	for _, raw := range strings.Split(answer, "\n") {
		n, text, ok := splitNumberPrefix(strings.TrimSpace(raw))
		if !ok || n < 1 || n > want || out[n-1] != "" || text == "" {
			continue
		}
		out[n-1] = text
		filled++
	}
	if filled != want {
		return nil
	}
	return out
}

// unnumberedSingle reads a reply to a one-turn call that did not number its
// answer, returning "" when the reply is not unambiguously that turn.
//
// Numbering is what makes a reply matchable, which is why a group without it
// is rejected whole. One turn is the exception: there is no other line the
// answer could land on, so the number carries no information the call does not
// already have. Rejecting it cost real turns — Hy-MT2-1.8B leaves the number
// off a lone line often enough that a meeting came back with holes where the
// model had in fact translated.
//
// More than one line of text is still rejected. Rule 3 says to translate only
// the numbered lines, and a model answering one turn with several lines is
// translating the reference material too — the failure that moved the lead-in
// out of the user message in the first place. Joining those lines would read
// as a plausible sentence of a conversation nobody had.
func unnumberedSingle(answer string) string {
	found := ""
	for _, raw := range strings.Split(answer, "\n") {
		line := strings.TrimSpace(raw)
		// A number parseNumbered could not use: out of range for a
		// one-turn call, or one this reply started counting from.
		if _, text, ok := splitNumberPrefix(line); ok {
			line = text
		}
		if line == "" {
			continue
		}
		if found != "" {
			return ""
		}
		found = line
	}
	return found
}

// splitNumberPrefix parses a leading list number, tolerating the several
// punctuation styles a model may pick: "1. x", "1、x", "1) x", "1：x".
func splitNumberPrefix(line string) (int, string, bool) {
	digits := 0
	for digits < len(line) && line[digits] >= '0' && line[digits] <= '9' {
		digits++
	}
	if digits == 0 {
		return 0, "", false
	}
	n, err := strconv.Atoi(line[:digits])
	if err != nil {
		return 0, "", false
	}
	rest := strings.TrimLeft(strings.TrimSpace(line[digits:]), ".、．)）:：- ")
	return n, strings.TrimSpace(rest), true
}

// stripEchoedLabels removes speaker labels a model mirrored back.
//
// The turns it is shown carry one, because who is speaking is what makes a
// dialogue readable and what a pronoun refers back to. A model mirrors the
// format it was shown however plainly rule 2 tells it not to, and this is the
// cleanup — applied only when every line carries a label, since a model
// echoing the format does it for the whole reply while a line that genuinely
// opens with a bracket is one line.
func stripEchoedLabels(lines []string) []string {
	for _, line := range lines {
		if !strings.HasPrefix(line, "[") || !strings.Contains(line, "]") {
			return lines
		}
	}
	out := make([]string, len(lines))
	for i, line := range lines {
		_, after, _ := strings.Cut(line, "]")
		out[i] = strings.TrimSpace(after)
		if out[i] == "" {
			// Whatever that was, it was not only a label.
			return lines
		}
	}
	return out
}
