package translate

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// engineCall is one request the route made, as the engine saw it.
type engineCall struct {
	System    string
	User      string
	MaxTokens int
	// Lines is how many numbered turns the user message asked for.
	Lines int
}

// fakeEngine stands in for the mounted chat handler. reply decides what each
// call answers, so a test can describe a model that merges lines, one that
// runs out of budget, or an engine that is not there.
type fakeEngine struct {
	calls []engineCall
	reply func(n int, call engineCall) (status int, content, finish string)
}

func (f *fakeEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Messages  []chatMessage `json:"messages"`
		MaxTokens int           `json:"max_tokens"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	call := engineCall{MaxTokens: req.MaxTokens}
	for _, m := range req.Messages {
		switch m.Role {
		case "system":
			call.System = m.Content
		case "user":
			call.User = m.Content
		}
	}
	_, body, _ := strings.Cut(call.User, "Lines to translate:\n")
	call.Lines = countNumbered(body)
	f.calls = append(f.calls, call)

	status, content, finish := f.reply(len(f.calls), call)
	if status != http.StatusOK {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"error":{"message":"engine down"}}`))
		return
	}
	if finish == "" {
		finish = "stop"
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"choices": []map[string]any{
			{"message": map[string]any{"content": content}, "finish_reason": finish},
		},
		"usage": map[string]any{
			"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
		},
	})
}

func countNumbered(body string) int {
	n := 0
	for _, line := range strings.Split(strings.TrimSpace(body), "\n") {
		if _, _, ok := splitNumberPrefix(strings.TrimSpace(line)); ok {
			n++
		}
	}
	return n
}

// echoNumbered answers every turn with its number, which is all most of these
// tests need: what is under test is the alignment, not the translation.
func echoNumbered(_ int, call engineCall) (int, string, string) {
	var b strings.Builder
	for i := 1; i <= call.Lines; i++ {
		fmt.Fprintf(&b, "%d. translated %d\n", i, i)
	}
	return http.StatusOK, b.String(), ""
}

func mountFake(t *testing.T, engine *fakeEngine) *http.ServeMux {
	t.Helper()
	mux := http.NewServeMux()
	Mount(mux, Options{Completer: Completer{Handler: engine, ModelName: "mt"}})
	return mux
}

func postTranscript(t *testing.T, mux *http.ServeMux, req transcriptRequest) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/translate/transcript", strings.NewReader(string(body)))
	r.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, r)
	return rec
}

func decodeTranscript(t *testing.T, rec *httptest.ResponseRecorder) transcriptResponse {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out transcriptResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func turns(n int) []transcriptSegment {
	out := make([]transcriptSegment, 0, n)
	for i := 1; i <= n; i++ {
		out = append(out, transcriptSegment{
			ID:      fmt.Sprintf("s%d", i),
			Speaker: "Alice",
			Text:    fmt.Sprintf("line %d", i),
		})
	}
	return out
}

// The invariant the route exists for: one answer per turn, under the caller's
// own ids, in the order they were sent.
func TestTranscript_AnswersEveryTurnInOrder(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: echoNumbered}
	rec := postTranscript(t, mountFake(t, engine), transcriptRequest{
		From: "en", To: "zh-Hans", Segments: turns(5),
	})
	out := decodeTranscript(t, rec)

	if len(out.Segments) != 5 {
		t.Fatalf("got %d results for 5 turns", len(out.Segments))
	}
	for i, res := range out.Segments {
		want := fmt.Sprintf("s%d", i+1)
		if res.ID != want {
			t.Errorf("result %d id=%q want %q", i, res.ID, want)
		}
		if res.Text == "" || res.Error != "" {
			t.Errorf("result %d = %+v, want a translation and no error", i, res)
		}
	}
	if len(engine.calls) != 1 {
		t.Errorf("calls=%d want one", len(engine.calls))
	}
	if got := rec.Header().Get("X-Model-Usage-Total-Tokens"); got != "15" {
		t.Errorf("usage header=%q want 15", got)
	}
}

// Context, background and glossary have to reach the model, and the context
// must not come back as a result.
func TestTranscript_CarriesContextAndTerms(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: echoNumbered}
	rec := postTranscript(t, mountFake(t, engine), transcriptRequest{
		From: "en", To: "zh-Hans", Segments: turns(2),
		Context:    &transcriptContext{Before: []transcriptLine{{Speaker: "Bob", Text: "what about Spinoza"}}},
		Background: "a talk about ethics",
		Glossary:   []glossaryEntry{{Source: "Spinoza", Target: "斯宾诺莎"}},
	})
	out := decodeTranscript(t, rec)

	if len(out.Segments) != 2 {
		t.Fatalf("got %d results, want 2 — the context line is not a turn", len(out.Segments))
	}
	call := engine.calls[0]
	if !strings.Contains(call.System, "斯宾诺莎") {
		t.Errorf("glossary missing from the system message:\n%s", call.System)
	}
	if !strings.Contains(call.System, "a talk about ethics") {
		t.Errorf("background missing from the system message:\n%s", call.System)
	}
	if !strings.Contains(call.System, "into Chinese") {
		t.Errorf("target language not named in the system message:\n%s", call.System)
	}
	// The lead-in rides with the other reference material, in the system
	// message. Shown beside the turns, a small model translates it as turn
	// one and shifts every answer onto the wrong turn.
	if !strings.Contains(call.System, "what about Spinoza") {
		t.Errorf("lead-in missing from the system message:\n%s", call.System)
	}
	if strings.Contains(call.User, "what about Spinoza") {
		t.Errorf("lead-in reached the message holding the turns to translate:\n%s", call.User)
	}
	if !strings.Contains(call.User, "1. [Alice] line 1") {
		t.Errorf("speaker label missing from the numbered turns:\n%s", call.User)
	}
}

// A model that mirrors the label format back is cleaned up rather than shown
// to the reader.
func TestTranscript_StripsEchoedSpeakerLabels(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(_ int, call engineCall) (int, string, string) {
		var b strings.Builder
		for i := 1; i <= call.Lines; i++ {
			fmt.Fprintf(&b, "%d. [Alice] 第%d句\n", i, i)
		}
		return http.StatusOK, b.String(), ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(3),
	}))
	for _, res := range out.Segments {
		if strings.HasPrefix(res.Text, "[") {
			t.Errorf("label survived: %q", res.Text)
		}
	}
}

// Labels are removed line by line, and only labels this request showed. A
// model that echoes some lines and not others still has its echoes cleaned, a
// neighbour's label is a label too, and a bracket that is part of what was
// said stays.
func TestTranscript_StripsOnlyTheLabelsItShowed(t *testing.T) {
	t.Parallel()
	segments := []transcriptSegment{
		{ID: "s1", Speaker: "SPEAKER_02", Text: "line 1"},
		{ID: "s2", Speaker: "SPEAKER_01", Text: "line 2"},
		{ID: "s3", Speaker: "SPEAKER_01", Text: "line 3"},
		{ID: "s4", Speaker: "SPEAKER_02", Text: "line 4"},
	}
	engine := &fakeEngine{reply: func(_ int, _ engineCall) (int, string, string) {
		return http.StatusOK, "1. [SPEAKER_02] 第一句\n" +
			"2. 第二句\n" +
			"3. [SPEAKER_02] 第三句\n" +
			"4. [笑] 第四句\n", ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: segments,
	}))
	want := []string{"第一句", "第二句", "第三句", "[笑] 第四句"}
	for i, res := range out.Segments {
		if res.Text != want[i] || res.Error != "" {
			t.Errorf("turn %d = %+v, want text %q", i+1, res, want[i])
		}
	}
	if len(engine.calls) != 1 {
		t.Errorf("calls=%d want one: the reply lined up", len(engine.calls))
	}
}

// A line that is only a label is a turn the model folded into its neighbour,
// as Hy-MT2-1.8B did with "Innovator." on a real meeting. The rest of the
// reply stands, and only that turn and the two either side of it are asked
// for again, reading what came before them as lead-in.
func TestTranscript_AsksAgainOnlyAroundALabelAlone(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(_ int, call engineCall) (int, string, string) {
		if call.Lines == 3 {
			return http.StatusOK, "1. [Alice] 补二\n2. 补三\n3. 补四\n", ""
		}
		var b strings.Builder
		for i := 1; i <= call.Lines; i++ {
			if i == 3 {
				fmt.Fprintf(&b, "%d. [Alice]\n", i)
				continue
			}
			fmt.Fprintf(&b, "%d. [Alice] 第%d句\n", i, i)
		}
		return http.StatusOK, b.String(), ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(6),
	}))
	want := []string{"第1句", "补二", "补三", "补四", "第5句", "第6句"}
	for i, res := range out.Segments {
		if res.Text != want[i] || res.Error != "" {
			t.Errorf("turn %d = %+v, want text %q", i+1, res, want[i])
		}
	}
	if len(engine.calls) != 2 {
		t.Fatalf("calls=%d want two: the group, then the three turns around the hole", len(engine.calls))
	}
	again := engine.calls[1]
	if !strings.Contains(again.User, "[Alice] line 2") || !strings.Contains(again.User, "[Alice] line 4") ||
		strings.Contains(again.User, "line 5") {
		t.Errorf("second call asked for:\n%s\nwant turns 2 to 4 only", again.User)
	}
	if !strings.Contains(again.System, "[Alice] line 1") {
		t.Errorf("second call lead-in lacks turn 1:\n%s", again.System)
	}
}

// When neither the neighbours nor the turn on its own get an answer, the
// turn comes back empty and reported, and its neighbours keep the first
// answer rather than being lost with it.
func TestTranscript_LeavesALabelAloneEmptyWhenAskingAgainDoesNotHelp(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(_ int, call engineCall) (int, string, string) {
		var b strings.Builder
		for i := 1; i <= call.Lines; i++ {
			if strings.Contains(call.User, fmt.Sprintf("%d. [Alice] line 3", i)) {
				fmt.Fprintf(&b, "%d. [Alice]\n", i)
				continue
			}
			fmt.Fprintf(&b, "%d. [Alice] 第%d句\n", i, i)
		}
		return http.StatusOK, b.String(), ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(4),
	}))
	want := []string{"第1句", "第2句", "", "第4句"}
	for i, res := range out.Segments {
		wantErr := ""
		if i == 2 {
			wantErr = segmentErrUnaligned
		}
		if res.Text != want[i] || res.Error != wantErr {
			t.Errorf("turn %d = %+v, want text %q error %q", i+1, res, want[i], wantErr)
		}
	}
	if len(engine.calls) != 3 {
		t.Errorf("calls=%d want three: the group, the window, the turn alone", len(engine.calls))
	}
}

// When asking with the neighbours folds the turn away again, the turn is
// asked for on its own, where it has nothing to fold into. Its answer is
// taken, and the neighbours keep the first one.
func TestTranscript_AsksForAStubbornHoleOnItsOwn(t *testing.T) {
	t.Parallel()
	folding := labelAlone("line 3")
	engine := &fakeEngine{reply: func(n int, call engineCall) (int, string, string) {
		if call.Lines == 1 {
			return http.StatusOK, "1. 单独一句", ""
		}
		return folding(n, call)
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(4),
	}))
	want := []string{"译1", "译2", "单独一句", "译4"}
	for i, res := range out.Segments {
		if res.Text != want[i] || res.Error != "" {
			t.Errorf("turn %d = %+v, want text %q", i+1, res, want[i])
		}
	}
	if len(engine.calls) != 3 {
		t.Fatalf("calls=%d want three: the group, the window, the turn alone", len(engine.calls))
	}
	if alone := engine.calls[2]; alone.Lines != 1 || !strings.Contains(alone.System, "[Alice] line 2") {
		t.Errorf("last call asked for %d turns with lead-in:\n%s\nwant turn 3 alone after turn 2", alone.Lines, alone.System)
	}
}

// labelAlone answers every numbered line, giving a label alone to the turns
// whose source text is in alone.
func labelAlone(alone ...string) func(int, engineCall) (int, string, string) {
	return func(_ int, call engineCall) (int, string, string) {
		_, body, _ := strings.Cut(call.User, "Lines to translate:\n")
		var b strings.Builder
		for i, line := range strings.Split(strings.TrimSpace(body), "\n") {
			folded := false
			for _, text := range alone {
				folded = folded || strings.HasSuffix(line, "] "+text)
			}
			if folded {
				fmt.Fprintf(&b, "%d. [Alice]\n", i+1)
				continue
			}
			fmt.Fprintf(&b, "%d. 译%d\n", i+1, i+1)
		}
		return http.StatusOK, b.String(), ""
	}
}

// A hole found in the second half of a split group is reported on the turn
// it belongs to, not on the turn at the same position in the first half.
func TestTranscript_ReportsAHoleOnItsOwnTurnAfterASplit(t *testing.T) {
	t.Parallel()
	answer := labelAlone("line 7")
	engine := &fakeEngine{reply: func(n int, call engineCall) (int, string, string) {
		if call.Lines == 8 {
			return http.StatusOK, "1. 只有一行", ""
		}
		return answer(n, call)
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(8),
	}))
	for i, res := range out.Segments {
		wantErr := ""
		if i == 6 {
			wantErr = segmentErrUnaligned
		}
		if res.Error != wantErr || (wantErr == "") != (res.Text != "") {
			t.Errorf("turn %d = %+v, want error %q", i+1, res, wantErr)
		}
	}
}

// Holes next to each other are asked for in one call, and a hole on the first
// or last line is asked for with the one neighbour it has.
func TestTranscript_AsksForAdjacentHolesTogether(t *testing.T) {
	t.Parallel()
	first := labelAlone("line 1", "line 2", "line 6")
	answer := labelAlone()
	engine := &fakeEngine{reply: func(n int, call engineCall) (int, string, string) {
		if n == 1 {
			return first(n, call)
		}
		return answer(n, call)
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(6),
	}))
	for i, res := range out.Segments {
		if res.Error != "" || res.Text == "" {
			t.Errorf("turn %d = %+v, want it translated", i+1, res)
		}
	}
	if len(engine.calls) != 3 {
		t.Fatalf("calls=%d want three: the group, turns 1-3, turns 5-6", len(engine.calls))
	}
	if got := []int{engine.calls[1].Lines, engine.calls[2].Lines}; got[0] != 3 || got[1] != 2 {
		t.Errorf("asked again for %v turns, want [3 2]", got)
	}
}

// A reply that answers no turn at all is not patched: it is a reply that does
// not line up, owed the retry and the halving.
func TestTranscript_RetriesAReplyThatIsAllLabels(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: labelAlone("line 1", "line 2", "line 3")}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(3),
	}))
	for i, res := range out.Segments {
		if res.Error != segmentErrUnaligned {
			t.Errorf("turn %d = %+v, want it unaligned", i+1, res)
		}
	}
	// The group and its retry, then each half and its retry: 1+2 split.
	if len(engine.calls) != 6 {
		t.Errorf("calls=%d want six: retry and halving, not a patch", len(engine.calls))
	}
}

// When asking again fails at the engine, the hole says so: a caller deciding
// whether to ask again reads the reason.
func TestTranscript_ReportsAnEngineFailureWhenAskingAgain(t *testing.T) {
	t.Parallel()
	first := labelAlone("line 2")
	engine := &fakeEngine{reply: func(n int, call engineCall) (int, string, string) {
		if n == 1 {
			return first(n, call)
		}
		return http.StatusInternalServerError, "", ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(3),
	}))
	want := []string{"", segmentErrUpstream, ""}
	for i, res := range out.Segments {
		if res.Error != want[i] {
			t.Errorf("turn %d = %+v, want error %q", i+1, res, want[i])
		}
	}
	if len(engine.calls) != 2 {
		t.Errorf("calls=%d want two: a failing engine is not asked again turn by turn", len(engine.calls))
	}
}

// A stuck decoder repeats inside one line, which the alignment check cannot
// see.
func TestTranscript_CollapsesARepeatingReply(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(_ int, _ engineCall) (int, string, string) {
		loop := strings.TrimSpace(strings.Repeat("这是一句话。", 5))
		return http.StatusOK, "1. " + loop, ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(1),
	}))
	if got := out.Segments[0].Text; got != "这是一句话。" {
		t.Errorf("text=%q want the loop collapsed to one copy", got)
	}
}

// A reply that does not line up is retried identically, then split, so one
// unanswerable turn costs one turn instead of the whole group.
func TestTranscript_SplitsAroundTheTurnItCannotAnswer(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(_ int, call engineCall) (int, string, string) {
		// The model refuses turn 4 whenever it is asked for, which it can
		// only express by returning one line short.
		var b strings.Builder
		n := 0
		for i := 1; i <= call.Lines; i++ {
			if strings.Contains(call.User, fmt.Sprintf("%d. [Alice] line 4", i)) {
				continue
			}
			n++
			fmt.Fprintf(&b, "%d. translated\n", i)
		}
		return http.StatusOK, b.String(), ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(4),
	}))

	for i, res := range out.Segments {
		switch i {
		case 3:
			if res.Error != segmentErrUnaligned {
				t.Errorf("turn 4 = %+v, want it reported as unaligned", res)
			}
			if res.Text != "" {
				t.Errorf("turn 4 carries both a translation and a failure: %+v", res)
			}
		default:
			if res.Text == "" || res.Error != "" {
				t.Errorf("turn %d = %+v, want it translated despite its neighbour", i+1, res)
			}
		}
	}
}

// An engine that is not there is a failure of the request, not of every turn
// in it: retrying and splitting would turn one outage into a call per turn,
// and record each as a turn that could not be translated.
func TestTranscript_FailsTheRequestWhenTheEngineIsGone(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(_ int, _ engineCall) (int, string, string) {
		return http.StatusServiceUnavailable, "", ""
	}}
	rec := postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(16),
	})
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status=%d want 500 body=%s", rec.Code, rec.Body.String())
	}
	if len(engine.calls) != 2 {
		t.Errorf("calls=%d want the attempt and its retry, and no tree below them", len(engine.calls))
	}
}

// One failing call among working ones is a per-segment failure: the engine is
// there, so the rest of the request is still worth having.
func TestTranscript_ReportsAPartialFailure(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(n int, call engineCall) (int, string, string) {
		// Fail the whole group twice, then fail only the half holding the
		// last two turns.
		if n <= 2 {
			return http.StatusOK, "1. only one line", ""
		}
		if strings.Contains(call.User, "line 4") {
			return http.StatusInternalServerError, "", ""
		}
		return echoNumbered(n, call)
	}}
	rec := postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(4),
	})
	out := decodeTranscript(t, rec)

	// The split narrows the failure onto the one turn that causes it; its
	// neighbour in the same half is still translated.
	for i, res := range out.Segments[:3] {
		if res.Text == "" || res.Error != "" {
			t.Errorf("turn %d = %+v, want it translated", i+1, res)
		}
	}
	if got := out.Segments[3].Error; got != segmentErrUpstream {
		t.Errorf("turn 4 error=%q want %q", got, segmentErrUpstream)
	}
}

// A reply cut off by the token budget is asked for again with more room. The
// model was not wrong, it ran out of space.
func TestTranscript_RetriesATruncatedReplyWithMoreRoom(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(n int, call engineCall) (int, string, string) {
		if n == 1 {
			return http.StatusOK, "1. translated 1\n2. transla", "length"
		}
		return echoNumbered(n, call)
	}}
	rec := postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(3),
	})
	out := decodeTranscript(t, rec)

	for i, res := range out.Segments {
		if res.Text == "" {
			t.Errorf("turn %d not translated: %+v", i+1, res)
		}
	}
	if len(engine.calls) != 2 {
		t.Fatalf("calls=%d want the attempt and one retry", len(engine.calls))
	}
	if engine.calls[1].MaxTokens != 2*engine.calls[0].MaxTokens {
		t.Errorf("retry budget=%d want double the first call's %d",
			engine.calls[1].MaxTokens, engine.calls[0].MaxTokens)
	}
}

// One turn is the single case where an unnumbered reply is not ambiguous:
// there is no other line it could belong to. Rejecting it discarded turns the
// model had in fact translated — Hy-MT2-1.8B drops the number on a lone line
// often enough that a meeting came back with holes in it.
func TestTranscript_TakesAnUnnumberedReplyToOneTurn(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct{ name, reply, want string }{
		{"no number at all", "这是译文", "这是译文"},
		{"a number this group does not have", "2. 这是译文", "这是译文"},
		{"a speaker label mirrored back", "[Alice] 这是译文", "这是译文"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := &fakeEngine{reply: func(_ int, _ engineCall) (int, string, string) {
				return http.StatusOK, tc.reply, ""
			}}
			out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
				To: "zh-Hans", Segments: turns(1),
			}))
			if got := out.Segments[0]; got.Text != tc.want || got.Error != "" {
				t.Errorf("segment = %+v, want text %q", got, tc.want)
			}
			if len(engine.calls) != 1 {
				t.Errorf("calls=%d want one: the reply was usable as it stood", len(engine.calls))
			}
		})
	}
}

// Without numbering, one turn can only be matched to one line of text. More
// than one means the model translated the reference material as well, which is
// what moving the lead-in into the system message exists to prevent; joining
// them would read as a plausible sentence of a conversation nobody had.
func TestTranscript_RejectsAReplyThatIsNotOneTurnsAnswer(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct{ name, reply string }{
		{"nothing at all", ""},
		{"a number and no text", "1."},
		{"more lines than turns", "这是译文\n这是背景的译文"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := &fakeEngine{reply: func(_ int, _ engineCall) (int, string, string) {
				return http.StatusOK, tc.reply, ""
			}}
			out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
				To: "zh-Hans", Segments: turns(1),
			}))
			if got := out.Segments[0]; got.Error != segmentErrUnaligned || got.Text != "" {
				t.Errorf("segment = %+v, want %q", got, segmentErrUnaligned)
			}
			if len(engine.calls) != 2 {
				t.Errorf("calls=%d want the attempt and its retry", len(engine.calls))
			}
		})
	}
}

// Half a sentence is the one unnumbered reply that is wrong even though only
// one turn could own it. It costs a call to ask again, and reads as a finished
// translation forever if it is taken.
func TestTranscript_DoesNotTakeATruncatedReplyToOneTurn(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(n int, _ engineCall) (int, string, string) {
		if n == 1 {
			return http.StatusOK, "这是译文的前半", "length"
		}
		return http.StatusOK, "这是完整的译文", ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(1),
	}))
	if got := out.Segments[0].Text; got != "这是完整的译文" {
		t.Errorf("text=%q want the retry's reply rather than the cut-off one", got)
	}
	if len(engine.calls) != 2 {
		t.Fatalf("calls=%d want the attempt and one retry", len(engine.calls))
	}
	if engine.calls[1].MaxTokens != 2*engine.calls[0].MaxTokens {
		t.Errorf("retry budget=%d want double the first call's %d",
			engine.calls[1].MaxTokens, engine.calls[0].MaxTokens)
	}
}

// The tolerance is one turn wide. A group answered with a single unnumbered
// line is still rejected and divided, because which turn that line answers is
// exactly what is missing.
func TestTranscript_DoesNotTakeAnUnnumberedReplyForAGroup(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: func(_ int, _ engineCall) (int, string, string) {
		return http.StatusOK, "一行答案", ""
	}}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(2),
	}))
	if len(engine.calls) != 4 {
		t.Fatalf("calls=%d want the group, its retry, and one call per turn", len(engine.calls))
	}
	for i, call := range engine.calls[2:] {
		if call.Lines != 1 {
			t.Errorf("call %d asked for %d turns, want the group divided", i+3, call.Lines)
		}
	}
	// Each half is one turn, so each half may take its own reply.
	for i, res := range out.Segments {
		if res.Text != "一行答案" || res.Error != "" {
			t.Errorf("turn %d = %+v, want the answer to its own call", i+1, res)
		}
	}
}

// A group too large for one conversation is divided before it is asked for.
// Being cut off costs a truncated reply, a retry truncated the same way, and
// then the split that was going to happen regardless.
func TestTranscript_SplitsAGroupTooBigForTheContextWindow(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: echoNumbered}
	mux := http.NewServeMux()
	// Small enough that eight turns cannot fit but two can.
	Mount(mux, Options{
		Completer:   Completer{Handler: engine, ModelName: "mt"},
		ContextSize: 2600,
	})

	out := decodeTranscript(t, postTranscript(t, mux, transcriptRequest{
		To: "zh-Hans", Segments: turns(8),
	}))
	for i, res := range out.Segments {
		if res.Text == "" || res.Error != "" {
			t.Errorf("turn %d = %+v, want it translated", i+1, res)
		}
	}
	if len(engine.calls) < 2 {
		t.Fatalf("calls=%d want the group divided", len(engine.calls))
	}
	for i, call := range engine.calls {
		if call.Lines == 8 {
			t.Errorf("call %d asked for all eight turns despite the window", i)
		}
	}
	// The second half is shown the first, so dividing costs no context.
	if !strings.Contains(engine.calls[1].System, "What was said just before") {
		t.Errorf("the half after the first has no lead-in:\n%s", engine.calls[1].System)
	}
}

// A card that pins no window down means the engine took it from the model's
// own metadata, which cannot be read from here. Guessing low would divide
// every group for nothing.
func TestTranscript_DoesNotSplitWhenTheWindowIsUnknown(t *testing.T) {
	t.Parallel()
	engine := &fakeEngine{reply: echoNumbered}
	out := decodeTranscript(t, postTranscript(t, mountFake(t, engine), transcriptRequest{
		To: "zh-Hans", Segments: turns(8),
	}))
	if len(out.Segments) != 8 || len(engine.calls) != 1 {
		t.Fatalf("results=%d calls=%d want eight turns in one call",
			len(out.Segments), len(engine.calls))
	}
}

// A caller that hangs up gets what was finished and a reason for the rest,
// rather than the outage the failed call would otherwise look like.
func TestTranscript_MarksWhatTheBudgetDidNotReach(t *testing.T) {
	t.Parallel()
	h := &Handler{prompt: DefaultPrompt, catalog: BuiltinCatalog()}
	plan, cerr := h.validate(transcriptRequest{To: "zh-Hans", Segments: turns(4)})
	if cerr != nil {
		t.Fatal(cerr)
	}
	run := &transcriptRun{
		h:        h,
		log:      slog.New(slog.NewTextHandler(io.Discard, nil)),
		deadline: time.Now().Add(transcriptBudget),
		out:      make([]transcriptResult, len(plan.segments)),
	}
	for i, sg := range plan.segments {
		run.out[i].ID = sg.ID
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := run.translate(ctx, transcriptTask{segments: plan.segments, retry: true}); err != nil {
		t.Fatalf("translate returned %v, want the partial answer", err)
	}
	for _, res := range run.out {
		if res.Error != segmentErrDeadline {
			t.Errorf("%+v: want the turn marked as out of time", res)
		}
	}
}
