package fetch

import (
	"errors"
	"io"
	"math/rand/v2"
	"net"
	"net/http"
	"net/url"
	"syscall"
	"time"
)

// HTTP-level retry parameters. These are independent of the lifecycle outer
// retry (5s base / 5min cap) — they bound a single
// RangeDownloader.Download invocation. Production code uses the defaults
// below; tests configure per-instance overrides on defaultDownloader.backoffBase
// / backoffCap so parallel tests don't race on package state.
const (
	productionBackoffBase = 500 * time.Millisecond
	productionBackoffCap  = 60 * time.Second
	defaultMaxRetries     = 5
	jitterPct             = 0.20
)

// backoff returns the wait duration before retry attempt N (0-indexed).
//
// Formula: min(base * 2^attempt, cap) +/- jitterPct. We use the
// math/rand/v2 package globals (goroutine-safe by design — Go 1.22+)
// so concurrent fan-out callers from `lifecycle.downloadAll` cannot
// race on a shared `*rand.Rand`. Pre-fix backoff was called from the
// MaxConcurrentDownloads worker pool against a single `*rand.Rand`,
// which is documented as not goroutine-safe; under contention the
// jitter values were undefined and `-race` flagged it.
//
// The randomness is not security-sensitive — it spreads concurrent
// retries by ±20%. The math/rand/v2 globals are seeded automatically
// from a per-process random source, so seed determinism (the v1
// version's only reason for a package-private rng) is no longer
// useful here. Tests assert distributions, not exact values.
func backoff(attempt int, base, cap time.Duration) time.Duration {
	if attempt < 0 {
		attempt = 0
	}
	d := base
	for i := 0; i < attempt && d < cap; i++ {
		d *= 2
	}
	if d > cap {
		d = cap
	}
	jitter := time.Duration(float64(d) * jitterPct)
	if jitter <= 0 {
		return d
	}
	// Range: [d - jitter, d + jitter]. rand.Int64N panics on n<=0; the
	// jitter > 0 guard above is sufficient since 2*jitter > 0.
	delta := rand.Int64N(int64(2*jitter)) - int64(jitter)
	out := d + time.Duration(delta)
	if out < 0 {
		out = 0
	}
	return out
}

// isRetriableStatus reports whether an HTTP response code should be retried
// at the fetch layer. 4xx are mostly permanent (auth, not-found) except 408
// and 429 which the upstream is asking us to back off.
func isRetriableStatus(code int) bool {
	switch {
	case code == http.StatusRequestTimeout, code == http.StatusTooManyRequests:
		return true
	case code >= 500 && code <= 599:
		return true
	default:
		return false
	}
}

// isRetriableError reports whether a transport-layer error (returned by
// http.Client.Do or by reading the body) should be retried. We retry the
// usual transient classes: connection reset, EOF mid-stream, timeouts,
// and context-deadline cases that aren't rooted in the parent ctx.
func isRetriableError(err error) bool {
	if err == nil {
		return false
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	if errors.Is(err, syscall.ECONNRESET) ||
		errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.ECONNABORTED) {
		return true
	}
	// io.EOF / unexpected EOF mid-stream. Use errors.Is so wrapped
	// variants (fmt.Errorf("read x: %w", io.ErrUnexpectedEOF), etc.)
	// also match -- pre-fix the comparison was on err.Error() which
	// silently failed once any layer added a prefix.
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	// url.Error wraps the underlying net error; unwrap and retry on hops.
	var ue *url.Error
	if errors.As(err, &ue) {
		return isRetriableError(ue.Err)
	}
	return false
}
