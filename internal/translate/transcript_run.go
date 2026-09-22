package translate

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"time"
)

// transcriptBudget is the wall clock this route may spend starting calls.
//
// The gateway in front of this gives one upstream request 60 seconds, and the
// retry-then-halve tree below can turn one slow group into a dozen calls. Left
// unbounded it would spend that budget and hand back nothing at all, when what
// it has by then is most of the group translated. So no new call is started
// after the budget is gone: the turns already done are returned, and the rest
// come back marked so the caller can ask for them again.
//
// A call already in flight is not cut short — the remaining twenty seconds are
// for it — and a client that disconnects cancels the context, which stops the
// tree wherever it is.
const transcriptBudget = 40 * time.Second

// errMisaligned is the engine answering with something that cannot be matched
// back to the turns it was sent. Distinct from a failed call: the engine is
// there, so retrying costs a call rather than confirming an outage.
var errMisaligned = errors.New("reply does not line up with the turns sent")

// errTruncated is the engine stopping on the token budget. The same turns are
// worth asking for again with more room, which is not true of a mangled reply.
var errTruncated = errors.New("reply cut off by the token budget")

func (h *Handler) handleTranscript(w http.ResponseWriter, r *http.Request) {
	var req transcriptRequest
	if err := decodeJSONBody(r, &req); err != nil {
		writeRequestErr(w, err)
		return
	}
	plan, cerr := h.validate(req)
	if cerr != nil {
		writeRequestErr(w, cerr)
		return
	}

	run := &transcriptRun{
		h:        h,
		system:   transcriptSystem(plan),
		log:      slog.Default(),
		deadline: time.Now().Add(transcriptBudget),
		out:      make([]transcriptResult, len(plan.segments)),
	}
	run.systemTokens = estimateTokens(run.system)
	for i, sg := range plan.segments {
		run.out[i].ID = sg.ID
	}

	err := run.translate(r.Context(), transcriptTask{
		segments: plan.segments,
		context:  plan.context,
		retry:    true,
		budget:   transcriptMaxTokens(plan.segments),
	})
	if err != nil {
		writeTranslateErr(w, err)
		return
	}

	run.declareUsage(w)
	run.log.Info("transcript translated",
		"turns", len(plan.segments),
		"failed", run.failedSegments(),
		"calls", run.calls,
		"prompt_tokens", run.usage.PromptTokens,
		"completion_tokens", run.usage.CompletionTokens)
	writeJSON(w, http.StatusOK, transcriptResponse{Segments: run.out})
}

// transcriptRun is one request in progress. It owns the answer being filled
// in, so a call several levels into the halving writes its turns in place
// rather than returning a slice somebody has to splice back.
type transcriptRun struct {
	h      *Handler
	system string
	// systemTokens is the rules and the term table, which are identical in
	// every call of the request and are most of what a small group spends.
	systemTokens int
	log          *slog.Logger
	deadline     time.Time

	out   []transcriptResult
	usage chatUsage
	calls int
	// answered counts calls the engine returned something for, usable or
	// not. What separates a model that will not follow the numbering from
	// an engine that is not there.
	answered int
	// upstreamFails counts calls that produced no answer at all.
	upstreamFails int
}

// transcriptTask is one node of the retry tree: which turns, what they follow,
// and what is still owed to them.
type transcriptTask struct {
	// offset is where these turns sit in the answer.
	offset   int
	segments []transcriptSegment
	// context is what precedes these turns, shown and not translated. The
	// right half of a split reads the left half's own turns here, so
	// splitting a group does not cost the second half its lead-in.
	context []transcriptLine
	// retry is whether an identical second attempt is still owed. The
	// common failure is a model merging two lines, which it does not
	// reliably repeat.
	retry  bool
	budget int
}

// translate fills in one task's turns, retrying once and then splitting in
// half. It returns an error only when the whole request should fail.
//
// A group is rejected whole when the reply does not line up, and a group here
// is a page of transcript: one dropped line would otherwise discard thirty-two
// turns. Retrying identically first because the common failure does not
// reliably repeat; splitting after that because the failure that does repeat
// is one turn the model will not answer for, and a smaller group both isolates
// it and asks less of the model.
func (run *transcriptRun) translate(ctx context.Context, task transcriptTask) error {
	if len(task.segments) == 0 {
		return nil
	}
	if ctx.Err() != nil || !time.Now().Before(run.deadline) {
		run.mark(task, segmentErrDeadline)
		return nil
	}

	leadIn := transcriptLeadIn(task.context)
	user := transcriptUser(task.segments)

	// Split before asking rather than after being cut off. A group that does
	// not fit in one conversation's context comes back truncated, which the
	// alignment check turns into a rejected group, a retry that is truncated
	// in exactly the same way, and only then the split that was going to
	// happen regardless. Checking first skips two calls that cannot succeed.
	if len(task.segments) > 1 && !run.fits(leadIn, user, task.budget) {
		run.log.Info("transcript group split to fit the context window",
			"turns", len(task.segments), "context_size", run.h.contextSize)
		return run.split(ctx, task)
	}

	lines, err := run.call(ctx, task, leadIn, user)
	if err == nil {
		for i, line := range lines {
			run.out[task.offset+i].Text = collapseRepeats(line)
		}
		return nil
	}

	// A call that failed because the caller hung up is not a group to try
	// harder on, and checking it here rather than only on the next attempt
	// keeps a cancellation from being reported as an outage.
	if ctx.Err() != nil {
		run.mark(task, segmentErrDeadline)
		return nil
	}

	// Nothing below helps against an engine that is not answering, and the
	// tree would turn one outage into a call per turn. A restart during one
	// meeting once burned 197 of 231 turns in three seconds, each recorded
	// as a turn that could not be translated. Failing the request instead
	// leaves the caller's own scheduler to try again once the condition has
	// had time to clear.
	if run.answered == 0 && run.upstreamFails >= 2 {
		return err
	}

	run.log.Warn("transcript group failed",
		"turns", len(task.segments), "retry", task.retry, "err", err)

	if task.retry {
		next := task
		next.retry = false
		if errors.Is(err, errTruncated) {
			// The reply was not wrong, only cut off. Same turns, more room.
			next.budget = 2 * task.budget
		}
		return run.translate(ctx, next)
	}

	if len(task.segments) == 1 {
		reason := segmentErrUnaligned
		if !errors.Is(err, errMisaligned) && !errors.Is(err, errTruncated) {
			reason = segmentErrUpstream
		}
		run.mark(task, reason)
		return nil
	}

	return run.split(ctx, task)
}

// split translates the two halves of a task, the second reading the first as
// its lead-in so a group loses no context by being divided.
//
// Whatever the halves are still owed carries over: reached because the group
// did not fit, they have not been tried yet and keep their retry; reached
// after a failure, the retry has already been spent and task.retry is false.
func (run *transcriptRun) split(ctx context.Context, task transcriptTask) error {
	mid := len(task.segments) / 2

	left := task
	left.segments = task.segments[:mid]
	left.budget = transcriptMaxTokens(left.segments)
	if err := run.translate(ctx, left); err != nil {
		return err
	}

	right := task
	right.offset = task.offset + mid
	right.segments = task.segments[mid:]
	right.context = trailingContext(task.segments[:mid])
	right.budget = transcriptMaxTokens(right.segments)
	return run.translate(ctx, right)
}

// fits reports whether a call's prompt and the reply it is allowed both sit
// inside what the engine serves one conversation at a time.
//
// True when the card pins no window down: the engine took the window from the
// model's own metadata, which is not a number readable from here, and
// guessing low would split every group for nothing.
func (run *transcriptRun) fits(leadIn, user string, budget int) bool {
	if run.h.contextSize <= 0 {
		return true
	}
	return run.systemTokens+estimateTokens(leadIn)+estimateTokens(user)+budget <= run.h.contextSize
}

// call asks the engine for one task and returns one line per turn.
func (run *transcriptRun) call(ctx context.Context, task transcriptTask, leadIn, user string) ([]string, error) {
	run.calls++
	res, err := run.h.Completer.complete(ctx, chatCall{
		System:    run.system + leadIn,
		User:      user,
		MaxTokens: task.budget,
	})
	if err != nil {
		run.upstreamFails++
		return nil, err
	}
	run.answered++
	run.usage = run.usage.add(res.Usage)

	if lines := parseNumbered(res.Text, len(task.segments)); lines != nil {
		return stripEchoedLabels(lines), nil
	}
	if res.Truncated {
		return nil, errTruncated
	}
	if len(task.segments) == 1 {
		if line := unnumberedSingle(res.Text); line != "" {
			run.log.Info("transcript reply was not numbered, and one turn has nowhere else to land")
			return stripEchoedLabels([]string{line}), nil
		}
	}
	// With the reply, because nothing downstream can see it: the turn
	// reaches the caller as the word "unaligned", and what tells a model
	// that will not number its answer from one answering something else
	// entirely is the answer itself.
	run.log.Warn("transcript reply did not line up",
		"turns", len(task.segments), "reply", truncate(res.Text, 512))
	return nil, errMisaligned
}

// mark records why these turns have no translation. A turn always comes back;
// this is what it comes back as when it came back empty.
func (run *transcriptRun) mark(task transcriptTask, reason string) {
	for i := range task.segments {
		run.out[task.offset+i].Error = reason
	}
}

func (run *transcriptRun) failedSegments() int {
	n := 0
	for _, res := range run.out {
		if res.Error != "" {
			n++
		}
	}
	return n
}

// declareUsage reports what the whole request cost.
func (run *transcriptRun) declareUsage(w http.ResponseWriter) {
	declareUsage(w, run.usage)
}

// trailingContext turns the turns just translated into the lead-in for the
// ones that follow them. The source text, not the translation: what the next
// half needs is the conversation, and the translation of it is one more thing
// that could be wrong.
func trailingContext(segments []transcriptSegment) []transcriptLine {
	if len(segments) > maxContextLines {
		segments = segments[len(segments)-maxContextLines:]
	}
	out := make([]transcriptLine, 0, len(segments))
	for _, sg := range segments {
		out = append(out, transcriptLine{Speaker: sg.Speaker, Text: sg.Text})
	}
	return out
}
