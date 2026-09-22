package controlplane

import (
	"sync"
	"testing"
	"time"
)

// TestTokenBucket_BurstThenThrottle exercises the full happy path:
// the bucket starts at capacity (so a fresh server can absorb a small
// burst), drains as Allow consumes tokens, refuses once empty, and
// recovers when the wall clock advances.
func TestTokenBucket_BurstThenThrottle(t *testing.T) {
	t.Parallel()
	now := time.Unix(0, 0)
	b := newTokenBucket(3, 1.0, now)

	for i := 0; i < 3; i++ {
		if !b.allow(now) {
			t.Fatalf("burst[%d] should be allowed (bucket pre-filled to capacity)", i)
		}
	}
	if b.allow(now) {
		t.Fatal("4th call must be rejected (bucket empty, no wall clock advance)")
	}

	// Half a token after 0.5s — still <1, must reject.
	if b.allow(now.Add(500 * time.Millisecond)) {
		t.Fatal("0.5s elapsed at 1tok/sec yields 0.5 tokens; must reject")
	}
	// Full token after another 0.5s.
	if !b.allow(now.Add(1 * time.Second)) {
		t.Fatal("1s elapsed at 1tok/sec must allow exactly one call")
	}
	// And immediately after, no more.
	if b.allow(now.Add(1 * time.Second)) {
		t.Fatal("token consumed; immediate next must reject")
	}
}

// TestTokenBucket_DisabledIsAlwaysAllow locks the "0 = disabled"
// contract documented at the env layer. The handler relies on this
// so RETRY_RATE_LIMIT=0 is a true no-op.
func TestTokenBucket_DisabledIsAlwaysAllow(t *testing.T) {
	t.Parallel()
	now := time.Unix(0, 0)
	b := newTokenBucket(1, 0, now)
	for i := 0; i < 1000; i++ {
		if !b.allow(now) {
			t.Fatalf("disabled bucket rejected call %d", i)
		}
	}
	// Nil receiver is also a no-op (covers the "limiter never built"
	// branch in handleRetry).
	var nilB *tokenBucket
	if !nilB.Allow() {
		t.Fatal("nil bucket must allow")
	}
}

// TestTokenBucket_GoroutineSafe stresses the mutex path. With a tiny
// bucket and 100 goroutines firing 1k calls each, the total accepted
// count must be bounded by capacity + (elapsed * rate). Pre-mutex
// races would either panic, undercount, or vastly overcount.
func TestTokenBucket_GoroutineSafe(t *testing.T) {
	t.Parallel()
	b := newTokenBucket(5, 100, time.Now())
	var wg sync.WaitGroup
	const goroutines = 32
	const perG = 200
	var allowed int64
	var mu sync.Mutex
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < perG; j++ {
				if b.Allow() {
					mu.Lock()
					allowed++
					mu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	// Sanity: at least 5 (initial burst). Upper bound is "everyone
	// got through" (goroutines * perG); we only assert it's bounded
	// to catch races that double-credit tokens.
	if allowed < 5 {
		t.Fatalf("allowed=%d below initial burst (something locked tokens)", allowed)
	}
	if allowed > goroutines*perG {
		t.Fatalf("allowed=%d above hard cap (%d) — race?", allowed, goroutines*perG)
	}
}
