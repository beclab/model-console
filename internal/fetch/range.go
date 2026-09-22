// Package fetch implements the HTTP download core for llm-init.
//
// The core type is RangeDownloader: HTTP-Range based, resumable, retryable
// downloads of a single file. Higher-level callers (fetch.URLClient and
// the per-Kind switch in internal/lifecycle/ensure.go) wire it up with the
// right URL, headers and progress sink. (HF downloads no longer flow
// through this package — they go through internal/adapter/hfwrap, which
// drives the `hf` CLI from Go. Range/Retry/ETag for HF is owned by that
// CLI.) The state machine for one Download() call:
//
//	HEAD URL → ETag, Content-Length
//	check existing <dest>.part size N:
//	  N == 0           → GET full
//	  0 < N < Total    → GET with Range bytes=N-, If-Range: ETag
//	    206            → resume append into .part
//	    200            → server ignored Range, truncate and write from 0
//	    416            → if N == Total accept; else truncate + retry
//	    other          → retry with backoff
//	all bytes written  → fsync → rename(.part → dest)
//
// Failure during streaming bumps the attempt counter and re-issues the
// request. attempt > MaxRetries returns ErrTooManyRetries wrapping the
// last underlying error.
package fetch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/progress"
)

// Sentinel errors callers can match with errors.Is.
var (
	// ErrTooManyRetries is returned when the per-request retry budget is
	// exhausted. The underlying cause is wrapped via fmt.Errorf("...%w").
	ErrTooManyRetries = errors.New("fetch: too many retries")

	// ErrSizeMismatch indicates the bytes downloaded did not match
	// ExpectedSize after the stream completed. Treated as a hard error;
	// retry semantics belong to the lifecycle caller.
	ErrSizeMismatch = errors.New("fetch: size mismatch")

	// ErrSHA256Mismatch is returned by Verify (and by URLClient when an
	// EXPECTED_SHA256 is supplied). The .part file is left in place so
	// callers can inspect it.
	ErrSHA256Mismatch = errors.New("fetch: sha256 mismatch")

	// ErrMaxBytesExceeded is returned when a download writes more than
	// RangeOptions.MaxBytes. A hard (non-retriable) error: it guards
	// against an endless / dishonestly-sized URL filling the cache
	// volume when the server reports no usable Content-Length.
	ErrMaxBytesExceeded = errors.New("fetch: max download bytes exceeded")
)

// HTTPStatusError reports a non-2xx response from an upstream HTTP
// call (HEAD/GET/tree). Callers use errors.As to match it instead of
// the pre-v1.0.5 `strings.Contains(err.Error(), "status 4")` pattern,
// which broke whenever a wrapping layer reformatted the message.
//
// Op is one of "HEAD", "GET", or "tree" so failure messages stay
// useful in logs; downstream classification (lifecycle's
// classifyDownloadErr) keys only on StatusCode and ignores Op/Body.
//
// Body is bounded to ~512 bytes by the call sites and passed through
// redactSecrets (credentials an upstream may have echoed are masked)
// before storage; do NOT log it as a metric label.
type HTTPStatusError struct {
	Op         string
	StatusCode int
	Body       string
}

func (e *HTTPStatusError) Error() string {
	if e.Body == "" {
		return fmt.Sprintf("fetch: %s status %d", e.Op, e.StatusCode)
	}
	return fmt.Sprintf("fetch: %s status %d: %s", e.Op, e.StatusCode, e.Body)
}

// RangeOptions configures one Download invocation.
type RangeOptions struct {
	URL          string
	DestPath     string
	ExpectedSize int64       // -1 means unknown (no Content-Length validation)
	ETag         string      // last-known ETag; used as If-Range when resuming
	Headers      http.Header // extra headers (Authorization, User-Agent, ...)
	MaxRetries   int         // 0 → defaultMaxRetries
	BufSize      int         // 0 → defaultBufSize
	// MaxBytes caps total bytes written for one Download. 0 means
	// unlimited. Exceeding it is a hard error (ErrMaxBytesExceeded);
	// it bounds disk usage when the upstream sends no / a dishonest
	// Content-Length so total stays <= 0.
	MaxBytes int64

	// OnRetry, when non-nil, is called once per retried attempt
	// (i.e. NOT for the very first try, only when the loop is about
	// to backoff and re-issue). attempt is 1-indexed: 1 means
	// "second try", 2 means "third try", etc. err is the underlying
	// retriable error that triggered the retry.
	//
	// The callback is fired BEFORE the backoff sleep so observers
	// can stamp metrics with the same wall-clock as the retry
	// decision. It must not block — fetch loops cannot afford to
	// stall on user code. Production wires this to the
	// `llm_init_download_retries_total` counter via a small
	// classify() helper in internal/lifecycle.
	OnRetry func(attempt int, err error)

	// OnMeta, when non-nil, is called once with what the upstream said
	// about the object: its length and ETag, either of which may be
	// absent (0 / ""). It fires whether or not any bytes end up being
	// transferred, so a caller recording download metadata still learns
	// the ETag of a file that turned out to be complete already.
	//
	// It must not block, for the same reason OnRetry must not.
	OnMeta func(size int64, etag string)
}

const defaultBufSize = 1 << 20 // 1 MiB

// RangeDownloader is the abstraction the rest of llm-init depends on.
type RangeDownloader interface {
	Download(ctx context.Context, opts RangeOptions, sink progress.Sink) error
	Verify(ctx context.Context, path string, expectedSize int64, expectedSHA string) error
}

// New returns the default RangeDownloader. It is safe to share across
// goroutines.
//
// The supplied client is reused for every request (HEAD + GET + retries);
// pass nil to build one with sane defaults (no body timeout — streaming
// downloads can take an hour for 7B safetensors).
func New(client *http.Client) RangeDownloader {
	if client == nil {
		client = newDefaultClient()
	}
	return &defaultDownloader{client: client}
}

// newDefaultClient builds an http.Client with no overall timeout (Range
// downloads stream for a long time) but reasonable header timeouts. The
// HF-specific redirect stripping previously lived in HFClient (removed
// in v1.1.0 with the move to the Python wrapper); URLClient uses this
// vanilla client.
func newDefaultClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			ResponseHeaderTimeout: 30 * time.Second,
			IdleConnTimeout:       90 * time.Second,
			ExpectContinueTimeout: time.Second,
		},
	}
}

type defaultDownloader struct {
	client *http.Client
	// backoffBase / backoffCap default to productionBackoffBase /
	// productionBackoffCap. Tests construct downloaders with much
	// smaller values via newDownloader(t, withTinyBackoff()) -- this
	// replaces the pre-v1.0.5 package-global mutable
	// defaultBackoffBase / defaultBackoffCap which raced under
	// `go test -race -count=1 ./...` once two backoff-sensitive
	// tests started running in parallel (caught on the first CI
	// run, lint-H aftermath).
	backoffBase time.Duration
	backoffCap  time.Duration
}

// Download executes the state machine described in the package doc.
//
// Progress is reported through sink. For resumable inputs sink.OnFileStart
// is called once with the *total* size (not "remaining bytes"); subsequent
// OnBytes deltas only count newly-fetched bytes — callers tracking absolute
// progress must seed BytesCompleted before invoking Download.
func (d *defaultDownloader) Download(ctx context.Context, opts RangeOptions, sink progress.Sink) error {
	if opts.URL == "" {
		return errors.New("fetch: URL is required")
	}
	if opts.DestPath == "" {
		return errors.New("fetch: DestPath is required")
	}
	if opts.MaxRetries <= 0 {
		opts.MaxRetries = defaultMaxRetries
	}
	if opts.BufSize <= 0 {
		opts.BufSize = defaultBufSize
	}
	if sink == nil {
		sink = progress.NopSink{}
	}

	if err := os.MkdirAll(parentDir(opts.DestPath), 0o755); err != nil {
		return fmt.Errorf("fetch: mkdir parent: %w", err)
	}
	partPath := opts.DestPath + ".part"

	// Probe the upstream to learn ETag + size. HEAD is cheap and avoids
	// downloading anything until we know whether resume is possible. If
	// HEAD is not allowed (some mirrors), fall through to GET-on-demand.
	total, etag, err := d.probe(ctx, opts)
	if err != nil {
		return err
	}
	if opts.ExpectedSize > 0 && total > 0 && opts.ExpectedSize != total {
		return fmt.Errorf("%w: head=%d expected=%d", ErrSizeMismatch, total, opts.ExpectedSize)
	}
	if total <= 0 {
		total = opts.ExpectedSize
	}
	if opts.ETag != "" && etag != "" && opts.ETag != etag {
		// Caller's recorded ETag is stale; force a clean download.
		_ = os.Remove(partPath)
	}
	if etag != "" {
		opts.ETag = etag
	}
	if opts.OnMeta != nil {
		opts.OnMeta(total, etag)
	}
	sink.OnFileStart(opts.DestPath, total)

	// If destination already exists at the right size we are done.
	if total > 0 {
		if info, err := os.Stat(opts.DestPath); err == nil && info.Size() == total {
			sink.OnFileDone(opts.DestPath)
			return nil
		}
	}

	var (
		lastErr error
		ok      bool
	)
	for attempt := 0; attempt <= opts.MaxRetries; attempt++ {
		if attempt > 0 {
			base := d.backoffBase
			if base <= 0 {
				base = productionBackoffBase
			}
			ceil := d.backoffCap
			if ceil <= 0 {
				ceil = productionBackoffCap
			}
			wait := backoff(attempt-1, base, ceil)
			if err := sleep(ctx, wait); err != nil {
				return err
			}
		}
		err := d.attempt(ctx, opts, partPath, total, sink)
		if err == nil {
			ok = true
			break
		}
		if !shouldRetry(err) {
			return err
		}
		lastErr = err
		sink.OnError(err)
		// OnRetry fires only on the retriable-error path so observers
		// don't double-count permanent failures (which already exit
		// via shouldRetry). attempt+1 is the 1-indexed retry number
		// the next iteration will execute.
		if opts.OnRetry != nil {
			opts.OnRetry(attempt+1, err)
		}
	}
	if !ok {
		return fmt.Errorf("%w: %w", ErrTooManyRetries, lastErr)
	}

	if err := os.Rename(partPath, opts.DestPath); err != nil {
		return fmt.Errorf("fetch: rename: %w", err)
	}
	sink.OnFileDone(opts.DestPath)
	return nil
}

// probe issues HEAD to discover size and ETag. Returning (0, "", nil) is
// allowed: some mirrors disallow HEAD or omit Content-Length, and we let
// the caller proceed without size validation in that case.
func (d *defaultDownloader) probe(ctx context.Context, opts RangeOptions) (int64, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, opts.URL, nil)
	if err != nil {
		return 0, "", fmt.Errorf("fetch: build HEAD: %w", err)
	}
	for k, vs := range opts.Headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	resp, err := d.client.Do(req)
	if err != nil {
		// Some servers (e.g. CDN) refuse HEAD outright; treat as "unknown
		// size, no etag" rather than a hard fault.
		if isRetriableError(err) {
			return 0, "", nil
		}
		return 0, "", fmt.Errorf("fetch: HEAD: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusMethodNotAllowed || resp.StatusCode == http.StatusForbidden {
		return 0, "", nil
	}
	if resp.StatusCode != http.StatusOK {
		return 0, "", &HTTPStatusError{Op: "HEAD", StatusCode: resp.StatusCode}
	}
	var size int64 = -1
	if cl := resp.Header.Get("Content-Length"); cl != "" {
		if v, err := strconv.ParseInt(cl, 10, 64); err == nil {
			size = v
		}
	}
	return size, resp.Header.Get("ETag"), nil
}

// attempt is one body-streaming pass. Returns nil if the file is fully on
// disk; returns a retriable error to ask the outer loop to back off, or a
// permanent error to abort.
func (d *defaultDownloader) attempt(ctx context.Context, opts RangeOptions, partPath string, total int64, sink progress.Sink) error {
	existing, err := partSize(partPath)
	if err != nil {
		return err
	}
	if total > 0 && existing > total {
		// Our local copy is bigger than the upstream — corrupt or wrong
		// file; truncate and retry from scratch.
		_ = os.Remove(partPath)
		existing = 0
	}
	if total > 0 && existing == total {
		// Local already complete; verify by attempting the rename.
		return nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, opts.URL, nil)
	if err != nil {
		return fmt.Errorf("fetch: build GET: %w", err)
	}
	for k, vs := range opts.Headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	if existing > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existing))
		if opts.ETag != "" {
			req.Header.Set("If-Range", opts.ETag)
		}
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignored Range (or we asked for the whole file). Truncate
		// any partial bytes and rewrite from zero.
		if existing > 0 {
			_ = os.Remove(partPath)
			existing = 0
		}
		return d.streamInto(ctx, resp, partPath, false, total, existing, opts.BufSize, opts.MaxBytes, sink)
	case http.StatusPartialContent:
		// Verify the server gave us the byte range we asked for.
		if cr := resp.Header.Get("Content-Range"); cr != "" {
			start, ok := parseContentRangeStart(cr)
			if ok && start != existing {
				// Range mismatch; reset and retry.
				_ = os.Remove(partPath)
				return retryable(fmt.Errorf("Content-Range start %d != existing %d", start, existing))
			}
		}
		return d.streamInto(ctx, resp, partPath, true, total, existing, opts.BufSize, opts.MaxBytes, sink)
	case http.StatusRequestedRangeNotSatisfiable:
		// existing >= total either means we already have the file (size
		// match) or the server changed the file (size shrank). Stat to
		// distinguish.
		if total > 0 && existing == total {
			return nil
		}
		_ = os.Remove(partPath)
		return retryable(errors.New("416 with size mismatch; truncated and retrying"))
	default:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		err := &HTTPStatusError{
			Op:         "GET",
			StatusCode: resp.StatusCode,
			// Scrub any credentials the upstream echoed (e.g. an S3 403
			// reproducing the signed request) before this snippet reaches
			// logs / last_error via HTTPStatusError.Error().
			Body: redactSecrets(strings.TrimSpace(string(body))),
		}
		if isRetriableStatus(resp.StatusCode) {
			return retryable(err)
		}
		return err
	}
}

func (d *defaultDownloader) streamInto(ctx context.Context, resp *http.Response, partPath string, append bool, total, existing int64, bufSize int, maxBytes int64, sink progress.Sink) error {
	flag := os.O_WRONLY | os.O_CREATE
	if append {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	f, err := os.OpenFile(partPath, flag, 0o644)
	if err != nil {
		return fmt.Errorf("fetch: open %s: %w", partPath, err)
	}
	defer f.Close()

	// Cancellation watchdog: pre-v1.0.5 the loop only checked ctx via
	// `select { case <-ctx.Done(): default: }` once per iteration and
	// then committed to a blocking resp.Body.Read. Go's http.Transport
	// does eventually close the body on ctx cancel, but slow servers
	// (TCP retransmits, syscall scheduling) could keep a single Read
	// blocked for the full ResponseHeaderTimeout window — meaning
	// SIGTERM during a download could stall shutdown by RTT * worker
	// count, occasionally past k8s preStop's 30s grace and into
	// SIGKILL territory.
	//
	// Defensive belt-and-braces: spawn one watchdog goroutine that
	// closes resp.Body the moment ctx is done, forcing the in-flight
	// Body.Read to return immediately. After every Body.Read error we
	// explicitly check ctx.Err() so the caller sees context.Canceled /
	// DeadlineExceeded rather than a generic "use of closed network
	// connection" — the latter would otherwise be classified as a
	// retriable network error and trigger pointless retry churn during
	// shutdown.
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			_ = resp.Body.Close()
		case <-done:
		}
	}()

	buf := make([]byte, bufSize)
	written := existing
	for {
		// Cheap pre-read check so a ctx already-cancelled before the
		// first read returns immediately without taking a syscall.
		if err := ctx.Err(); err != nil {
			return err
		}
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				return fmt.Errorf("fetch: write %s: %w", partPath, werr)
			}
			written += int64(n)
			sink.OnBytes(int64(n))
			if maxBytes > 0 && written > maxBytes {
				return fmt.Errorf("%w: wrote %d, cap %d", ErrMaxBytesExceeded, written, maxBytes)
			}
		}
		if errors.Is(rerr, io.EOF) {
			break
		}
		if rerr != nil {
			// If ctx was cancelled the watchdog closed Body and rerr is
			// almost certainly a "closed connection" wrapping; surface
			// the ctx error so callers can distinguish shutdown from
			// transient I/O.
			if cerr := ctx.Err(); cerr != nil {
				return cerr
			}
			if isRetriableError(rerr) {
				return retryable(rerr)
			}
			return fmt.Errorf("fetch: read body: %w", rerr)
		}
	}
	if total > 0 && written != total {
		return retryable(fmt.Errorf("short read: got %d want %d", written, total))
	}
	if err := f.Sync(); err != nil {
		return fmt.Errorf("fetch: fsync %s: %w", partPath, err)
	}
	return nil
}

// retryableErr is a thin wrapper used to flag errors that came from
// transport / streaming code so the outer loop knows to back off rather
// than abort. retryable() / shouldRetry() form a tiny private API.
type retryableErr struct{ err error }

func (e retryableErr) Error() string { return e.err.Error() }
func (e retryableErr) Unwrap() error { return e.err }

func retryable(err error) error { return retryableErr{err: err} }

func shouldRetry(err error) bool {
	if err == nil {
		return false
	}
	var re retryableErr
	if errors.As(err, &re) {
		return true
	}
	return isRetriableError(err)
}

// removeIfExists deletes path if it exists, returning nil for ENOENT.
// Used by URLClient on SHA mismatch and by tests.
func removeIfExists(path string) error {
	err := os.Remove(path)
	if err == nil || errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func partSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, nil
		}
		return 0, fmt.Errorf("fetch: stat %s: %w", path, err)
	}
	return info.Size(), nil
}

func parentDir(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' || p[i] == '\\' {
			return p[:i]
		}
	}
	return "."
}

// parseContentRangeStart returns the byte offset from a "bytes A-B/C"
// header. Returns (0, false) for malformed headers; callers fall through
// to permissive behavior.
func parseContentRangeStart(s string) (int64, bool) {
	const prefix = "bytes "
	if !strings.HasPrefix(s, prefix) {
		return 0, false
	}
	rest := s[len(prefix):]
	dash := strings.IndexByte(rest, '-')
	if dash < 0 {
		return 0, false
	}
	v, err := strconv.ParseInt(rest[:dash], 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// sleep blocks for d unless ctx is cancelled first.
func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
