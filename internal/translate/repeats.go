package translate

import (
	"strings"
	"unicode"
)

const (
	// repeatMaxBlock bounds the length of a repeating unit that is looked for,
	// in tokens. A looping model repeats a phrase or a sentence, not a
	// paragraph, and searching every block length up to the whole line is
	// quadratic for no return.
	repeatMaxBlock = 40
	// repeatMinShort is how many copies of a one- or two-token block it takes
	// to count as a loop. Higher than for longer blocks because short
	// repetition is sometimes what was said: 哈哈哈哈, "no no no", "很 很 很".
	repeatMinShort = 6
	// repeatMinLong is the same for a block of three tokens or more. Three
	// copies of a phrase that long is not something a person says.
	repeatMinLong = 3
)

// collapseRepeats cuts a repeated run down to one copy of what repeats.
//
// This is a guard against a decoder that gets stuck, which is a failure mode
// of the local models this runs on and not a hypothetical: a group of short
// turns came back with its answer printed twice. It survives the alignment
// check — the loop is inside one line — so nothing else catches it, and what
// reaches the reader is a translation that says the same thing five times.
//
// Blocks are tried longest first, so a repeated sentence is recognised as a
// sentence rather than reduced word by word.
func collapseRepeats(text string) string {
	toks := tokenize(text)
	if len(toks) < 2*repeatMinLong {
		return text
	}
	collapsed := collapseTokenRepeats(toks)
	if len(collapsed) == len(toks) {
		return text
	}
	return strings.TrimSpace(joinTokens(collapsed))
}

// token is a word or a single ideograph, with the whitespace that followed it.
// The separator travels with the token so dropping a repeat does not also drop
// the spacing around what is kept.
type token struct {
	text, sep string
}

// tokenize splits on whitespace, and additionally between ideographs, because
// Chinese, Japanese and Korean are written without spaces and a whole clause
// would otherwise be one token.
func tokenize(text string) []token {
	var (
		out  []token
		word strings.Builder
	)
	flush := func() {
		if word.Len() > 0 {
			out = append(out, token{text: word.String()})
			word.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
			if len(out) > 0 {
				out[len(out)-1].sep += string(r)
			}
		case ideograph(r):
			flush()
			out = append(out, token{text: string(r)})
		default:
			word.WriteRune(r)
		}
	}
	flush()
	return out
}

func ideograph(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r)
}

func joinTokens(toks []token) string {
	var b strings.Builder
	for _, t := range toks {
		b.WriteString(t.text)
		b.WriteString(t.sep)
	}
	return b.String()
}

func collapseTokenRepeats(toks []token) []token {
	for n := repeatMaxBlock; n >= 1; n-- {
		need := repeatMinLong
		if n < 3 {
			need = repeatMinShort
		}
		for i := 0; i+n*need <= len(toks); {
			copies := 1
			for i+(copies+1)*n <= len(toks) && sameBlock(toks, i, i+copies*n, n) {
				copies++
			}
			if copies < need {
				i++
				continue
			}
			toks = append(toks[:i+n], toks[i+copies*n:]...)
			// Not advancing: what follows the kept copy has moved to i+n and
			// may itself begin a run.
		}
	}
	return toks
}

func sameBlock(toks []token, a, b, n int) bool {
	for k := range n {
		if toks[a+k].text != toks[b+k].text {
			return false
		}
	}
	return true
}
