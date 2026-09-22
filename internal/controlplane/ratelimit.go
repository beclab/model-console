package controlplane

import (
	"sync"
	"time"
)

// tokenBucket is a small, lock-protected, monotonic-clock token bucket
// limiter sized for the operator-control-plane workload (POST
// /api/retry being the only known caller as of v1.0.5). It is
// deliberately not a public package: a new external dependency
// (golang.org/x/time/rate) is overkill for one handler that fires at
// most a few times per minute in normal operation, and the v1.0.x
// dependency policy is "only prometheus/client_golang".
//
// Behavior:
//
//   - capacity is the bucket size (== burst). Filled to capacity on
//     creation so a fresh server accepts a flurry up to that count
//     before throttling kicks in. This matters for restart scenarios
//     where an orchestrator may fire a few retries while bringing the
//     pod back into rotation.
//   - refillPerSec is the steady-state rate. We refill in fractional
//     tokens (float64) so non-integer rates like 0.5/s are supported
//     without rounding errors over long uptimes.
//   - Allow() returns true if at least one whole token is available;
//     it consumes the token and returns true. Otherwise returns false
//     without consuming anything.
//
// Disabled-by-config sentinel: the caller passes refillPerSec <= 0 to
// indicate "feature disabled, always Allow()". This matches the
// RETRY_RATE_LIMIT=0 env semantics documented in config.go.
type tokenBucket struct {
	capacity     float64
	refillPerSec float64

	mu     sync.Mutex
	tokens float64
	last   time.Time
}

// newTokenBucket returns a limiter pre-filled to capacity. nowFn is a
// monotonic clock; production callers pass time.Now and tests pass a
// controllable clock.
func newTokenBucket(capacity int, refillPerSec float64, now time.Time) *tokenBucket {
	return &tokenBucket{
		capacity:     float64(capacity),
		refillPerSec: refillPerSec,
		tokens:       float64(capacity),
		last:         now,
	}
}

// allow at the supplied wall-clock time. Hot path takes a single
// mutex acquisition. The "always allow" branch is kept above the
// mutex acquire so a fully-disabled limiter has no contention cost.
func (t *tokenBucket) allow(now time.Time) bool {
	if t == nil || t.refillPerSec <= 0 {
		return true
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	elapsed := now.Sub(t.last).Seconds()
	if elapsed > 0 {
		t.tokens += elapsed * t.refillPerSec
		if t.tokens > t.capacity {
			t.tokens = t.capacity
		}
		t.last = now
	}
	if t.tokens < 1 {
		return false
	}
	t.tokens--
	return true
}

// Allow is the production entry point.
func (t *tokenBucket) Allow() bool { return t.allow(time.Now()) }
