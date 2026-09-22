package translate

import (
	"strings"
	"testing"
)

func testHandler() *Handler {
	return &Handler{prompt: DefaultPrompt, catalog: BuiltinCatalog()}
}

func seg(id, speaker, text string) transcriptSegment {
	return transcriptSegment{ID: id, Speaker: speaker, Text: text}
}

func TestValidate_Rejects(t *testing.T) {
	t.Parallel()
	h := testHandler()
	good := []transcriptSegment{seg("a", "Alice", "hello")}

	cases := []struct {
		name string
		req  transcriptRequest
		want string
	}{
		{
			name: "no target",
			req:  transcriptRequest{Segments: good},
			want: "to is required",
		},
		{
			name: "target outside the catalog",
			req:  transcriptRequest{To: "kli", Segments: good},
			want: "unsupported target language",
		},
		{
			name: "source outside the catalog",
			req:  transcriptRequest{From: "kli", To: "en", Segments: good},
			want: "unsupported source language",
		},
		{
			name: "no segments",
			req:  transcriptRequest{To: "en"},
			want: "segments is required",
		},
		{
			name: "too many segments",
			req:  transcriptRequest{To: "en", Segments: manySegments(maxTranscriptSegments + 1)},
			want: "segments exceeds max",
		},
		{
			name: "segment without an id",
			req:  transcriptRequest{To: "en", Segments: []transcriptSegment{seg(" ", "", "hi")}},
			want: "segments[0].id is required",
		},
		{
			// The answer is keyed by id, so two turns under one id is a
			// caller about to overwrite one translation with another.
			name: "duplicate ids",
			req: transcriptRequest{To: "en", Segments: []transcriptSegment{
				seg("a", "", "hi"), seg("a", "", "there"),
			}},
			want: "duplicate segment id: a",
		},
		{
			name: "segment with no text",
			req: transcriptRequest{To: "en", Segments: []transcriptSegment{
				seg("a", "", "hi"), seg("b", "", "   "),
			}},
			want: "segments[1].text is required",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := h.validate(tc.req)
			if err == nil {
				t.Fatalf("validate accepted %+v", tc.req)
			}
			if err.code != codeInvalidRequest {
				t.Errorf("code=%q want %q", err.code, codeInvalidRequest)
			}
			if !strings.Contains(err.msg, tc.want) {
				t.Errorf("msg=%q want it to contain %q", err.msg, tc.want)
			}
		})
	}
}

func TestValidate_NormalisesInput(t *testing.T) {
	t.Parallel()
	h := testHandler()

	plan, err := h.validate(transcriptRequest{
		From: " zh-CN ",
		To:   "en",
		Segments: []transcriptSegment{
			seg("  s1  ", "  Alice  ", "  你好\n世界  "),
		},
		Context: &transcriptContext{Before: []transcriptLine{
			{Speaker: "Bob", Text: "  earlier  "},
			{Speaker: "Bob", Text: "   "},
		}},
		Background: "  a\tmeeting  ",
		Glossary: []glossaryEntry{
			{Source: " Spinoza ", Target: " 斯宾诺莎 "},
			{Source: "", Target: "dropped"},
			{Source: "dropped", Target: ""},
		},
	})
	if err != nil {
		t.Fatalf("validate: %v", err)
	}

	if plan.autoFrom {
		t.Error("autoFrom set although the caller named a source")
	}
	if plan.from.Code != langZhHans {
		t.Errorf("from=%q want %q (alias should resolve)", plan.from.Code, langZhHans)
	}
	if plan.to.Code != "en" {
		t.Errorf("to=%q want en", plan.to.Code)
	}

	if got := plan.segments[0].ID; got != "s1" {
		t.Errorf("id=%q want s1", got)
	}
	if got := plan.segments[0].Speaker; got != "Alice" {
		t.Errorf("speaker=%q want Alice", got)
	}
	// One turn is one line of a numbered prompt, so an embedded newline would
	// read to the parser as the start of the next turn.
	if got := plan.segments[0].Text; got != "你好 世界" {
		t.Errorf("text=%q want %q", got, "你好 世界")
	}

	if len(plan.context) != 1 || plan.context[0].Text != "earlier" {
		t.Errorf("context=%+v want the one non-blank line", plan.context)
	}
	if plan.background != "a meeting" {
		t.Errorf("background=%q", plan.background)
	}
	// A blank term row is skipped rather than failing the request.
	if len(plan.glossary) != 1 || plan.glossary[0].Source != "Spinoza" {
		t.Errorf("glossary=%+v want only the complete entry", plan.glossary)
	}
}

func TestValidate_AutoSource(t *testing.T) {
	t.Parallel()
	h := testHandler()

	for _, from := range []string{"", "auto", "AUTO", "  "} {
		plan, err := h.validate(transcriptRequest{
			From: from, To: "en", Segments: []transcriptSegment{seg("a", "", "你好")},
		})
		if err != nil {
			t.Fatalf("from=%q: %v", from, err)
		}
		if !plan.autoFrom {
			t.Errorf("from=%q did not read as auto", from)
		}
	}
}

// The lead-in is trimmed from the front: what a group needs is what was said
// immediately before it, not the start of a stretch a page away.
func TestValidate_KeepsTheNearestContext(t *testing.T) {
	t.Parallel()
	h := testHandler()

	var before []transcriptLine
	for i := range maxContextLines + 5 {
		before = append(before, transcriptLine{Text: string(rune('a' + i))})
	}
	plan, err := h.validate(transcriptRequest{
		To:       "en",
		Segments: []transcriptSegment{seg("a", "", "hi")},
		Context:  &transcriptContext{Before: before},
	})
	if err != nil {
		t.Fatalf("validate: %v", err)
	}
	if len(plan.context) != maxContextLines {
		t.Fatalf("context len=%d want %d", len(plan.context), maxContextLines)
	}
	if want := before[len(before)-1].Text; plan.context[maxContextLines-1].Text != want {
		t.Errorf("last context line=%q want %q", plan.context[maxContextLines-1].Text, want)
	}
}

func manySegments(n int) []transcriptSegment {
	out := make([]transcriptSegment, 0, n)
	for i := range n {
		out = append(out, seg(strings.Repeat("x", i+1), "", "hi"))
	}
	return out
}
