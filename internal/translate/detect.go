package translate

import (
	"context"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

const defaultDetectConfidence = 0.9

// DetectPrompt asks the model to emit a single MTran language code.
func DetectPrompt(text string, codes []string) string {
	return fmt.Sprintf(
		"Detect the language of the following text. Reply with ONLY one language code from this list, with no explanation:\n%s\n\nText:\n%s",
		strings.Join(codes, ", "), text)
}

type detectResult struct {
	Language   string
	Confidence float64

	// Usage is what the model answer cost, and is the zero value when a
	// heuristic answered instead. Detection is a model call like any other
	// on this handler, so it declares its tokens rather than being metered
	// as free.
	Usage chatUsage
}

// DetectLanguage identifies the language of text. Fast-path heuristics
// cover clear CJK (matching MTran's pure-CJK short-circuit); otherwise
// a single non-streaming chat completion is used. confidence is a
// synthetic score (1.0 for heuristics, defaultDetectConfidence for a
// successfully parsed model answer) for MTran minConfidence clients.
func (h *Handler) DetectLanguage(ctx context.Context, text string) (detectResult, error) {
	if text == "" {
		return detectResult{}, nil
	}

	if code, ok := detectPureCJK(text); ok {
		if h.catalog.HasLanguage(code) {
			return detectResult{Language: code, Confidence: 1.0}, nil
		}
		// Declared catalog excludes this CJK code — ask the model.
	}

	raw, usage, err := h.Completer.Complete(ctx, DetectPrompt(text, h.catalog.Codes()))
	if err != nil {
		return detectResult{}, err
	}
	// An answer this cannot parse still cost what it cost.
	code := h.parseDetectOutput(raw)
	if code == "" {
		return detectResult{Language: "", Confidence: 0, Usage: usage}, nil
	}
	return detectResult{Language: code, Confidence: defaultDetectConfidence, Usage: usage}, nil
}

func (h *Handler) parseDetectOutput(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	fields := strings.FieldsFunc(raw, func(r rune) bool {
		return unicode.IsSpace(r) || r == ',' || r == ';' || r == ':' ||
			r == '"' || r == '\'' || r == '`'
	})
	if len(fields) == 0 {
		return ""
	}
	if lang, ok := h.catalog.Resolve(fields[0]); ok {
		return lang.Code
	}
	// NameEN match against builtin table, then accept if in catalog.
	for _, l := range mtranLanguages {
		if strings.EqualFold(l.NameEN, fields[0]) && h.catalog.HasLanguage(l.Code) {
			return l.Code
		}
	}
	return ""
}

// detectPureCJK mirrors MTranServer's cheap Han/Hiragana/Katakana/
// Hangul short-circuit so common East-Asian text skips an LLM round-trip.
func detectPureCJK(text string) (string, bool) {
	start := 0
	for start < len(text) {
		r, size := utf8.DecodeRuneInString(text[start:])
		if r == utf8.RuneError && size == 1 {
			start++
			continue
		}
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			break
		}
		start += size
	}
	if start >= len(text) {
		return "", false
	}

	var hasHan, hasHira, hasKata, hasHangul, hasOther bool
	limit := start
	n := 0
	for limit < len(text) && n < 2000 {
		r, size := utf8.DecodeRuneInString(text[limit:])
		if r == utf8.RuneError && size == 1 {
			limit++
			n++
			continue
		}
		limit += size
		n++
		switch {
		case isHangul(r):
			hasHangul = true
		case isHiragana(r):
			hasHira = true
		case isKatakana(r):
			hasKata = true
		case unicode.Is(unicode.Han, r):
			hasHan = true
		case unicode.IsLetter(r) || unicode.IsNumber(r):
			hasOther = true
		}
	}
	if hasOther {
		return "", false
	}
	switch {
	case hasHangul && !hasHira && !hasKata:
		return "ko", true
	case hasHira || hasKata:
		return "ja", true
	case hasHan:
		return langZhHans, true
	default:
		return "", false
	}
}

func isHiragana(r rune) bool { return r >= 0x3040 && r <= 0x309f }
func isKatakana(r rune) bool { return r >= 0x30a0 && r <= 0x30ff }
func isHangul(r rune) bool {
	return (r >= 0xac00 && r <= 0xd7af) || (r >= 0x1100 && r <= 0x11ff) || (r >= 0x3130 && r <= 0x318f)
}
