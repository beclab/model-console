// Package kvbudget keeps concurrent requests from promising a llama.cpp
// unified KV pool more tokens than it holds.
//
// It exists for one configuration, and only that one. In split mode each
// slot owns a hard share of the cache, so an over-long prompt is refused at
// admission with a 400 naming the limit -- the caller learns what happened
// and nobody else is affected. In unified mode there is no per-slot ceiling
// to check against: every request is admitted against the size of the whole
// pool, and the pool running out is discovered mid-decode, where llama.cpp's
// recovery releases every sequence that is currently generating. One
// oversized request therefore kills its neighbours, including ones already
// streaming an answer to somebody.
//
// So the gate is an accounting of what has been promised, not a limit anyone
// configured. A request reserves what it may consume before it is forwarded,
// holds the reservation until it is done, and waits at the door when the pool
// is full rather than being let in to take the whole thing down. Waiting is
// the point: the engine is perfectly capable of serving these requests one
// after another, and refusing on arrival would turn a queue into an error.
//
// What it deliberately does not do is refuse on a number it does not have.
// With no pool size, no cost estimate, or a request larger than the pool
// itself, it forwards exactly as if it were not mounted. The last of those is
// the least obvious and the most important: a request that cannot be made to
// fit by waiting must reach the engine, because the engine's own refusal
// names the real limit and this gate's would not.
//
// Which of the two modes a deployment is in is not settled when the process
// starts. The flags are on an editable card and the engine is relaunched
// under the new ones, so the gate is asked per request whether it applies
// rather than being mounted or left out once and for all.
package kvbudget

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"
)

// Options configure a Guard.
type Options struct {
	// Active reports whether the deployment this guard fronts wants its
	// pool accounted for, read per request. Nil means it always does.
	//
	// A function because the answer is a property of the launch flags,
	// and those move: a card edit rewrites engine_args and the wrapper
	// relaunches the engine under them, so a deployment that booted with
	// one slot can be sharing a pool between eight of them a minute
	// later. Answering once, while the process was assembling itself,
	// left the gate absent for the rest of its life in exactly the
	// configuration it exists for.
	Active func() bool

	// Capacity reports the KV pool's size in tokens, and false when
	// nothing knows it. Read per request rather than once: at mount time
	// the engine has not started, so a measured pool size does not exist
	// yet, and a card edit that changes -c relaunches the engine with a
	// different one.
	Capacity func() (int, bool)

	// Tokenizer counts a prompt exactly by asking the engine. Nil leaves
	// every request charged by the upper bound its body length implies,
	// which is what the gate falls back to when a count fails anyway.
	Tokenizer *Tokenizer

	// MaxOutputTokens is the card's ceiling on a completion, used to cap
	// what a request may reserve for the answer it has not received yet.
	// Nil, or a zero result, means the card does not say. Read per
	// request, for the same reason as Active: the card is editable.
	MaxOutputTokens func() int

	// Wait is how long a request may queue for room. Zero selects
	// defaultWait.
	Wait time.Duration
}

// defaultWait matches the Router's own chat queue budget. The two queues sit
// on one call path, and giving up before the queue in front has finished
// counting turns a wait into a failure twice over.
const defaultWait = 10 * time.Second

// Guard is the middleware. Build it with New.
type Guard struct {
	opts Options

	// mu guards the lazily built semaphore and the capacity it was built
	// for. A semaphore.Weighted cannot be resized, and its size is not
	// known when the process starts.
	mu       sync.Mutex
	sem      *semaphore.Weighted
	capacity int
}

// New returns a Guard reading its pool size and per-request cost from opts.
func New(opts Options) *Guard {
	return &Guard{opts: opts}
}

// Wrap returns next fronted by the gate.
func (g *Guard) Wrap(next http.Handler) http.Handler {
	if g == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		release, ok := g.admit(r)
		if !ok {
			writeExhausted(w, g.wait())
			return
		}
		// Held until the handler returns, which for a streamed response is
		// after the last token: the proxy forwards with FlushInterval -1 and
		// returns only once the upstream body is drained. A client that
		// disconnects cancels the request context, the proxy returns, and
		// this runs then -- so a reservation cannot outlive its request.
		defer release()
		next.ServeHTTP(w, r)
	})
}

// noRelease is what an unmetered admission hands back.
func noRelease() {}

// admit reserves room for r, or reports that it waited and got none.
func (g *Guard) admit(r *http.Request) (release func(), ok bool) {
	if g.opts.Active != nil && !g.opts.Active() {
		return noRelease, true
	}
	sem, capacity := g.semaphore()
	if sem == nil {
		return noRelease, true
	}

	body, err := bodyOf(r)
	if err != nil {
		// The body could not be read here, so it cannot be sized -- and
		// the proxy downstream will fail on the same body and say so
		// better than an invented error would.
		return noRelease, true
	}
	var shape requestShape
	// A body that does not parse is still charged, from its length. These
	// are the same bytes an engine is about to reject, and the reservation
	// only has to outlive that rejection.
	_ = json.Unmarshal(body, &shape)
	if !shape.generates() {
		// Nothing to prompt with means nothing to keep in the cache. This
		// is the whole reason the gate reads the body rather than the
		// path: it is mounted on the `/v1/` catch-all, so `GET /v1/models`
		// passes through it, and a request that occupies no KV must not
		// reserve any. It also leaves anything shaped in a way this
		// package cannot size -- `/v1/responses` and its `input` field --
		// forwarded unmetered rather than charged a guess.
		return noRelease, true
	}
	maxOutput := 0
	if g.opts.MaxOutputTokens != nil {
		maxOutput = g.opts.MaxOutputTokens()
	}
	output := outputReserve(shape, maxOutput)

	bound := promptUpperBound(body) + output
	if bound <= 0 {
		return noRelease, true
	}

	// The fast path: small enough that measuring it exactly could not
	// change any decision. It is charged its loose bound, which
	// over-reserves -- capped at a fraction of the pool precisely so that
	// the over-reservation cannot accumulate into a queue. Anything larger
	// pays for two local round trips and gets charged what it really
	// costs, which is the trade this gate exists to make.
	if int64(bound) <= capacity/fastPathDivisor {
		if sem.TryAcquire(int64(bound)) {
			return releaseOnce(sem, int64(bound)), true
		}
	}

	cost := bound
	if measured, okCount := g.count(r.Context(), shape); okCount {
		cost = measured + output
	}
	if int64(cost) > capacity {
		// Waiting cannot make this fit. Forward it and let the engine
		// answer with the number that is actually true of it.
		slog.Warn("kv budget: a single request wants more of the KV pool than it holds; forwarding it for the engine to refuse",
			"request_tokens", cost, "kv_pool_tokens", capacity, "path", r.URL.Path)
		return noRelease, true
	}

	ctx, cancel := context.WithTimeout(r.Context(), g.wait())
	defer cancel()
	if err := sem.Acquire(ctx, int64(cost)); err != nil {
		return noRelease, false
	}
	return releaseOnce(sem, int64(cost)), true
}

// fastPathDivisor bounds how much of the pool may be held by requests
// charged their loose byte bound rather than a real count.
//
// It is the one tuned number here, and what it tunes is not accuracy but
// contamination. A byte bound over-reserves several-fold, so a large request
// admitted on one would make the pool look full while most of what it holds
// is slack -- and the requests queueing behind that slack would be waiting
// for nothing. Restricting the fast path to an eighth of the pool caps that
// slack at a size that cannot produce a queue on its own, and sends every
// request big enough to matter through the tokenizer.
const fastPathDivisor = 8

// releaseOnce returns a release that is safe to call more than once. It is
// called exactly once today, from a defer; the guard is here because a second
// release would hand the pool capacity that does not exist, which is the one
// failure this package cannot detect afterwards.
func releaseOnce(sem *semaphore.Weighted, n int64) func() {
	var once sync.Once
	return func() { once.Do(func() { sem.Release(n) }) }
}

// count asks the engine how many tokens the prompt really is. The bool is
// false when there is no tokenizer, nothing to count, or the engine could not
// answer -- and the last of those is why this returns a bool rather than an
// error: a tokenizer that is down must not stop the model serving.
func (g *Guard) count(ctx context.Context, shape requestShape) (int, bool) {
	if g.opts.Tokenizer == nil {
		return 0, false
	}
	n, err := g.opts.Tokenizer.CountPrompt(ctx, shape)
	if err != nil {
		slog.Warn("kv budget: could not count this prompt; charging the upper bound its length implies",
			"err", err.Error())
		return 0, false
	}
	return n, n > 0
}

// semaphore returns the live semaphore and its capacity, building it the
// first time a capacity is known and rebuilding it when the capacity has
// changed.
//
// The rebuild is for a relaunch. A card edit rewrites engine_args and
// restarts the engine, which comes back with a different pool; a guard
// holding the old number would enforce it until the process itself
// restarted. Requests still holding room on the discarded semaphore release
// against it and are forgotten, which is correct rather than merely
// tolerable: the engine they were talking to no longer exists.
func (g *Guard) semaphore() (*semaphore.Weighted, int64) {
	if g.opts.Capacity == nil {
		return nil, 0
	}
	capacity, ok := g.opts.Capacity()
	if !ok || capacity <= 0 {
		return nil, 0
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	if g.sem == nil || g.capacity != capacity {
		g.sem = semaphore.NewWeighted(int64(capacity))
		g.capacity = capacity
	}
	return g.sem, int64(capacity)
}

func (g *Guard) wait() time.Duration {
	if g.opts.Wait > 0 {
		return g.opts.Wait
	}
	return defaultWait
}

// CodeExhausted is the error code on the 503. Clients and the Router pin on
// the literal string, so it is a wire value.
const CodeExhausted = "kv_budget_exhausted"

// writeExhausted answers a request that could not get room in time.
//
// 503 rather than 429, and the distinction is the whole message: 429 is a
// limit somebody configured and 503 is the machine being full. The Router
// already treats a 503 from here as retryable, so it fails over to another
// backend or retries this one -- which is the reason this gate is worth
// having in front of the engine rather than in front of the gateway.
//
// The envelope matches the not-ready 503 from the same data plane, because a
// client should not need a second parser to learn it has to come back later.
func writeExhausted(w http.ResponseWriter, wait time.Duration) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", retryAfterSeconds(wait))
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"code": CodeExhausted,
			"message": "the model's KV cache is fully reserved by requests already in " +
				"flight; this request waited for room and did not get it",
		},
	})
}

// retryAfterSeconds renders the wait as whole seconds, rounded up and never
// below one. A `Retry-After: 0` reads as "immediately", which sends the
// client straight back into the same full queue.
func retryAfterSeconds(wait time.Duration) string {
	secs := int((wait + time.Second - 1) / time.Second)
	if secs < 1 {
		secs = 1
	}
	return strconv.Itoa(secs)
}
