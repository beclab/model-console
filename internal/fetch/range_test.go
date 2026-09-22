package fetch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/progress"
)

// payload is reused across tests; not too small (we want Range slicing
// to actually slice) and not too large (so tests stay fast).
var payload = func() []byte {
	b := make([]byte, 64*1024)
	for i := range b {
		b[i] = byte(i % 251)
	}
	return b
}()

func payloadSHA() string {
	h := sha256.Sum256(payload)
	return hex.EncodeToString(h[:])
}

// rangeServer is a minimal HF-like static server that supports HEAD, GET
// and the Range/If-Range headers. The harness lets each test inject
// behavior via the hooks.
type rangeServer struct {
	body []byte
	etag string

	// hooks
	onHEAD func(w http.ResponseWriter, r *http.Request) bool // return false → fall through
	onGET  func(w http.ResponseWriter, r *http.Request) bool

	requests atomic.Int32
}

func newRangeServer() *rangeServer {
	return &rangeServer{body: payload, etag: `"etag-v1"`}
}

func (s *rangeServer) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.requests.Add(1)
		switch r.Method {
		case http.MethodHead:
			if s.onHEAD != nil && s.onHEAD(w, r) {
				return
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(s.body)))
			w.Header().Set("ETag", s.etag)
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			if s.onGET != nil && s.onGET(w, r) {
				return
			}
			s.serveGET(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})
}

func (s *rangeServer) serveGET(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("ETag", s.etag)
	rng := r.Header.Get("Range")
	if rng == "" {
		w.Header().Set("Content-Length", strconv.Itoa(len(s.body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(s.body)
		return
	}
	// Parse "bytes=N-".
	if !strings.HasPrefix(rng, "bytes=") {
		http.Error(w, "bad range", http.StatusBadRequest)
		return
	}
	spec := strings.TrimPrefix(rng, "bytes=")
	dash := strings.IndexByte(spec, '-')
	if dash < 0 {
		http.Error(w, "bad range", http.StatusBadRequest)
		return
	}
	start, err := strconv.ParseInt(spec[:dash], 10, 64)
	if err != nil {
		http.Error(w, "bad range", http.StatusBadRequest)
		return
	}
	if start >= int64(len(s.body)) {
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
		return
	}
	// If-Range mismatch → return full 200 with current ETag (per spec).
	if cond := r.Header.Get("If-Range"); cond != "" && cond != s.etag {
		w.Header().Set("Content-Length", strconv.Itoa(len(s.body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(s.body)
		return
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(s.body)-int(start)))
	w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(s.body)-1, len(s.body)))
	w.WriteHeader(http.StatusPartialContent)
	_, _ = w.Write(s.body[start:])
}

// newDownloader returns a downloader for tests that don't drive the
// backoff path (single attempt, fast happy-path tests).
//
// Pre-v1.0.5 the backoff base / cap were package-level mutable globals
// (defaultBackoffBase / defaultBackoffCap) that retry-sensitive tests
// dialled down via overrideBase / overrideCap. Once two such tests both
// ran with t.Parallel() the race detector caught a real read/write
// conflict on those globals. Backoff knobs now live on the downloader
// itself (see range.go); retry-sensitive tests construct downloaders
// inline with `dl.backoffBase = time.Millisecond` etc., so each
// parallel test owns its own copy.
func newDownloader(t *testing.T) (*defaultDownloader, string) {
	t.Helper()
	dl := New(nil).(*defaultDownloader)
	return dl, filepath.Join(t.TempDir(), "out.bin")
}

func TestDownload_Full200(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	sink := &countSink{}
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
	}, sink); err != nil {
		t.Fatalf("Download: %v", err)
	}

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Errorf("payload mismatch: len=%d want=%d", len(got), len(payload))
	}
	if sink.bytes.Load() != int64(len(payload)) {
		t.Errorf("OnBytes total = %d, want %d", sink.bytes.Load(), len(payload))
	}
	if sink.starts.Load() != 1 || sink.dones.Load() != 1 {
		t.Errorf("starts=%d dones=%d, want 1/1", sink.starts.Load(), sink.dones.Load())
	}
}

func TestDownload_Resume206(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	// Pre-seed half the file in <dest>.part to simulate a previous
	// partial download.
	if err := os.WriteFile(dest+".part", payload[:len(payload)/2], 0o644); err != nil {
		t.Fatal(err)
	}

	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		ETag:         srv.etag, // matches → server returns 206
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(payload) {
		t.Errorf("resume mismatch: len=%d want=%d", len(got), len(payload))
	}
}

func TestDownload_Server200IgnoresRange(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	srv.onGET = func(w http.ResponseWriter, r *http.Request) bool {
		// Always reply with full body, ignoring Range. Client must
		// truncate any partial bytes and rewrite from zero.
		w.Header().Set("ETag", srv.etag)
		w.Header().Set("Content-Length", strconv.Itoa(len(srv.body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(srv.body)
		return true
	}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	if err := os.WriteFile(dest+".part", []byte("STALE"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		ETag:         srv.etag,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(payload) {
		t.Errorf("payload mismatch after 200 reset")
	}
}

func TestDownload_416AlreadyComplete(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	// Pre-seed the .part with the full body. Range start would be == size
	// → server returns 416; client must accept since size matches.
	if err := os.WriteFile(dest+".part", payload, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		ETag:         srv.etag,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if len(got) != len(payload) {
		t.Errorf("size = %d, want %d", len(got), len(payload))
	}
}

func TestDownload_416OnSizeShrink(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	srv.body = payload[:len(payload)/2] // upstream shrank
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	if err := os.WriteFile(dest+".part", payload, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(srv.body)),
		ETag:         srv.etag,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if len(got) != len(srv.body) {
		t.Errorf("size = %d, want %d", len(got), len(srv.body))
	}
}

func TestDownload_ETagMismatchTriggersReset(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	if err := os.WriteFile(dest+".part", payload[:1024], 0o644); err != nil {
		t.Fatal(err)
	}

	// Caller passes a stale ETag. Probe sees server's current ETag and
	// removes the .part before retrying. Final size must still match.
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		ETag:         `"old-etag"`,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(payload) {
		t.Errorf("payload mismatch after etag reset")
	}
}

func TestDownload_ConnectionResetThenResume(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	var dropped atomic.Bool
	var gets atomic.Int32
	srv.onGET = func(w http.ResponseWriter, r *http.Request) bool {
		n := gets.Add(1)
		if n == 1 {
			// First GET: return half the body and hijack/close the
			// connection mid-stream so the client sees an early EOF.
			// Connection: close tells net/http the conn is gone for
			// good so it must not try to reuse it on the retry.
			h, ok := w.(http.Hijacker)
			if !ok {
				t.Fatal("hijack unsupported")
			}
			conn, bw, err := h.Hijack()
			if err != nil {
				t.Fatal(err)
			}
			fmt.Fprintf(bw, "HTTP/1.1 200 OK\r\nContent-Length: %d\r\nETag: %s\r\nConnection: close\r\n\r\n",
				len(srv.body), srv.etag)
			bw.Write(srv.body[:len(srv.body)/2])
			bw.Flush()
			conn.Close()
			dropped.Store(true)
			return true
		}
		return false // fall through to default which honours Range
	}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		MaxRetries:   3,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(payload) {
		t.Errorf("payload mismatch after resume (got %d bytes)", len(got))
	}
	if !dropped.Load() {
		t.Error("connection-drop branch was never exercised")
	}
}

func TestDownload_5xxStormExhaustsRetries(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	srv.onGET = func(w http.ResponseWriter, r *http.Request) bool {
		http.Error(w, "boom", http.StatusBadGateway)
		return true
	}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		MaxRetries:   2,
	}, nil)
	if !errors.Is(err, ErrTooManyRetries) {
		t.Errorf("err = %v, want ErrTooManyRetries", err)
	}
}

func TestDownload_PermanentErrorAborts(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	srv.onGET = func(w http.ResponseWriter, r *http.Request) bool {
		http.Error(w, "no", http.StatusUnauthorized)
		return true
	}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		MaxRetries:   5,
	}, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if errors.Is(err, ErrTooManyRetries) {
		t.Errorf("permanent 401 should not trigger retry budget: %v", err)
	}
	if srv.requests.Load() > 2 { // HEAD + 1 GET
		t.Errorf("too many requests for permanent error: %d", srv.requests.Load())
	}
}

func TestDownload_HeadSizeMismatchAborts(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)) + 100, // wrong
	}, nil)
	if !errors.Is(err, ErrSizeMismatch) {
		t.Errorf("err = %v, want ErrSizeMismatch", err)
	}
}

func TestDownload_MaxBytesExceeded(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	err := dl.Download(context.Background(), RangeOptions{
		URL:      ts.URL + "/file",
		DestPath: dest,
		MaxBytes: int64(len(payload)) - 1, // one byte short of the body
	}, nil)
	if !errors.Is(err, ErrMaxBytesExceeded) {
		t.Errorf("err = %v, want ErrMaxBytesExceeded", err)
	}
	if _, statErr := os.Stat(dest); statErr == nil {
		t.Error("dest should not be promoted from .part when the cap trips")
	}
}

func TestDownload_DestAlreadyComplete(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	dl, dest := newDownloader(t)
	if err := os.WriteFile(dest, payload, 0o644); err != nil {
		t.Fatal(err)
	}

	if err := dl.Download(context.Background(), RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	// Only HEAD should have been issued.
	if got := srv.requests.Load(); got != 1 {
		t.Errorf("requests = %d, want 1 (HEAD only)", got)
	}
}

func TestDownload_ContextCancel(t *testing.T) {
	t.Parallel()
	srv := newRangeServer()
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	dl, dest := newDownloader(t)
	err := dl.Download(ctx, RangeOptions{
		URL:          ts.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
	}, nil)
	if err == nil {
		t.Error("expected ctx cancel error")
	}
}

// TestStreamInto_RespectsCtxCancelMidRead locks the v1.0.5 fix: pre-fix
// the streamInto loop only checked ctx via a `select { default }` then
// committed to a blocking resp.Body.Read. A slow / wedged upstream
// could keep one Read blocked for the full ResponseHeaderTimeout
// window, stalling SIGTERM by RTT * worker count. Post-fix a watchdog
// goroutine closes resp.Body on ctx cancellation, forcing the in-flight
// Read to return immediately, and the loop surfaces ctx.Err() rather
// than a generic "use of closed network connection".
//
// The test fixture: a server that writes a few bytes then sleeps for
// many seconds. Cancelling ctx mid-read must abort within ~250ms (we
// give a generous slack vs the plan's 50ms target to avoid CI flakes
// on slow runners; the pre-fix path stalled for the entire 30s+ test
// timeout, so any sub-second budget exposes the regression).
func TestStreamInto_RespectsCtxCancelMidRead(t *testing.T) {
	t.Parallel()

	// Server writes one chunk then blocks indefinitely without closing
	// the connection. The blocked read inside streamInto is what we
	// want to interrupt.
	//
	// firstChunkSent fires once the handler has flushed the prefix
	// onto the wire. The test waits on this signal (rather than a
	// best-guess time.Sleep) before firing cancel(); the previous
	// 150ms delay was a heuristic for "by now the downloader is
	// inside Body.Read" and was a documented flake source on slow
	// runners. With the channel we cancel exactly when the handler
	// has demonstrably yielded a body byte, which is the property
	// the test cares about.
	released := make(chan struct{})
	firstChunkSent := make(chan struct{})
	var firstChunkOnce sync.Once
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1048576") // 1 MiB advertised; we'll never deliver it
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("first-chunk"))
		if flusher != nil {
			flusher.Flush()
		}
		// sync.Once guards against retries: if the retry layer
		// reissues the request after the first connection died
		// early, the second handler invocation must NOT panic on a
		// re-close of firstChunkSent.
		firstChunkOnce.Do(func() { close(firstChunkSent) })
		select {
		case <-released:
		case <-r.Context().Done():
		}
	}))
	// Tear down in the right order: release blocked handlers first so
	// httptest.Server.Close() (which waits for all in-flight handlers)
	// does not deadlock against the still-blocked goroutine.
	t.Cleanup(func() {
		close(released)
		ts.Close()
	})

	ctx, cancel := context.WithCancel(context.Background())
	dl, dest := newDownloader(t)

	type res struct {
		err     error
		elapsed time.Duration
	}
	done := make(chan res, 1)
	go func() {
		start := time.Now()
		err := dl.Download(ctx, RangeOptions{
			URL:          ts.URL + "/slow",
			DestPath:     dest,
			ExpectedSize: 1 << 20,
		}, nil)
		done <- res{err: err, elapsed: time.Since(start)}
	}()

	// Wait for the handler to flush "first-chunk" onto the wire. Once
	// firstChunkSent fires we know the downloader has either already
	// returned from its first Read (and is back in the loop) or is
	// inside the next blocking Read -- either way ctx cancel exercises
	// the watchdog path. Pre-fix this was a `time.Sleep(150ms)` race.
	select {
	case <-firstChunkSent:
	case <-time.After(2 * time.Second):
		t.Fatal("handler never flushed first chunk; downloader probably never connected")
	}
	cancelTime := time.Now()
	cancel()

	select {
	case r := <-done:
		stallAfterCancel := r.elapsed - (cancelTime.Sub(cancelTime) /* zero */)
		_ = stallAfterCancel
		if r.err == nil {
			t.Fatal("Download returned nil error after ctx cancel")
		}
		if !errors.Is(r.err, context.Canceled) {
			// retry layer wraps with retryable() then surfaces; either
			// way the chain must include context.Canceled. Pre-fix the
			// chain typically did NOT contain context.Canceled because
			// Body.Close manifested as a generic net error.
			t.Fatalf("error chain missing context.Canceled: %v", r.err)
		}
		// 2 second budget: gives plenty of room for retry-layer backoff
		// to finish unwinding while still being far below the pre-fix
		// stall horizon (which was bounded only by ResponseHeaderTimeout
		// or test timeout).
		if r.elapsed > 2*time.Second {
			t.Errorf("Download took %v after cancel; expected sub-second", r.elapsed)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Download did not return within 5s after ctx.Cancel — streamInto wedged?")
	}
}

func TestDownload_ValidatesRequiredFields(t *testing.T) {
	t.Parallel()
	dl := New(nil)
	if err := dl.Download(context.Background(), RangeOptions{}, nil); err == nil {
		t.Error("missing URL+DestPath should error")
	}
	if err := dl.Download(context.Background(), RangeOptions{URL: "http://x"}, nil); err == nil {
		t.Error("missing DestPath should error")
	}
}

func TestVerify_OK(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	p := filepath.Join(dir, "f")
	if err := os.WriteFile(p, payload, 0o644); err != nil {
		t.Fatal(err)
	}
	dl := New(nil)
	if err := dl.Verify(context.Background(), p, int64(len(payload)), payloadSHA()); err != nil {
		t.Errorf("Verify: %v", err)
	}
	if err := dl.Verify(context.Background(), p, int64(len(payload)), "sha256:"+payloadSHA()); err != nil {
		t.Errorf("Verify with prefix: %v", err)
	}
}

func TestVerify_SizeMismatch(t *testing.T) {
	t.Parallel()
	p := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(p, []byte("short"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := New(nil).Verify(context.Background(), p, 999, "")
	if !errors.Is(err, ErrSizeMismatch) {
		t.Errorf("err = %v, want ErrSizeMismatch", err)
	}
}

func TestVerify_SHAMismatch(t *testing.T) {
	t.Parallel()
	p := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(p, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	err := New(nil).Verify(context.Background(), p, 4,
		"deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
	if !errors.Is(err, ErrSHA256Mismatch) {
		t.Errorf("err = %v, want ErrSHA256Mismatch", err)
	}
}

func TestVerify_SkipsWhenSHAEmpty(t *testing.T) {
	t.Parallel()
	p := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(p, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := New(nil).Verify(context.Background(), p, 4, ""); err != nil {
		t.Errorf("Verify with empty SHA: %v", err)
	}
}

func TestVerify_MissingFile(t *testing.T) {
	t.Parallel()
	err := New(nil).Verify(context.Background(), filepath.Join(t.TempDir(), "ghost"), 0, "")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestBackoff_GrowsAndCaps(t *testing.T) {
	t.Parallel()
	base := 100 * time.Millisecond
	cap := time.Second
	prev := time.Duration(0)
	for i := 0; i < 8; i++ {
		d := backoff(i, base, cap)
		if d < 0 {
			t.Errorf("attempt %d: negative %v", i, d)
		}
		if d > cap+time.Duration(float64(cap)*jitterPct)+time.Millisecond {
			t.Errorf("attempt %d: %v exceeds cap %v + jitter", i, d, cap)
		}
		_ = prev
	}
}

func TestBackoff_RespectsContextCancel(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := sleep(ctx, 10*time.Second); err == nil {
		t.Error("sleep should observe cancelled ctx")
	}
}

// TestRangeDownloader_OnRetryFiresOnRetriedAttempts asserts the new
// (v1.0.4) RangeOptions.OnRetry hook fires once per retried attempt
// — the producer side of `llm_init_download_retries_total`. The fake
// server returns 503 twice, then 200; the harness counts callback
// invocations and verifies they equal the number of retries actually
// taken (2). The first attempt is NOT a retry, so the callback fires
// 0 times when no retries are needed.
func TestRangeDownloader_OnRetryFiresOnRetriedAttempts(t *testing.T) {
	t.Parallel()

	s := newRangeServer()
	var failsLeft atomic.Int32
	failsLeft.Store(2)
	s.onGET = func(w http.ResponseWriter, _ *http.Request) bool {
		if failsLeft.Load() > 0 {
			failsLeft.Add(-1)
			http.Error(w, "upstream is grumpy", http.StatusServiceUnavailable)
			return true
		}
		return false // fall through to default 200 GET
	}
	srv := httptest.NewServer(s.handler())
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "blob.bin")
	var calls atomic.Int32
	var lastErr error
	var lastErrMu sync.Mutex
	d := New(nil).(*defaultDownloader)
	d.backoffBase = time.Millisecond
	d.backoffCap = 5 * time.Millisecond
	err := d.Download(context.Background(), RangeOptions{
		URL:          srv.URL + "/m",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		MaxRetries:   5,
		OnRetry: func(attempt int, err error) {
			calls.Add(1)
			lastErrMu.Lock()
			lastErr = err
			lastErrMu.Unlock()
			if attempt < 1 {
				t.Errorf("OnRetry attempt=%d, want >= 1 (1-indexed)", attempt)
			}
		},
	}, nil)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if got := calls.Load(); got != 2 {
		t.Errorf("OnRetry fired %d times, want 2 (one per 503)", got)
	}
	lastErrMu.Lock()
	defer lastErrMu.Unlock()
	if lastErr == nil || !strings.Contains(lastErr.Error(), "503") {
		t.Errorf("last OnRetry err = %v, want containing 503", lastErr)
	}
}

// TestRangeDownloader_OnRetryNotFiredOnSuccess asserts the hook stays
// silent when no retries happen, so a clean download doesn't pollute
// the retries counter.
func TestRangeDownloader_OnRetryNotFiredOnSuccess(t *testing.T) {
	t.Parallel()
	s := newRangeServer()
	srv := httptest.NewServer(s.handler())
	defer srv.Close()
	dest := filepath.Join(t.TempDir(), "blob.bin")
	var calls atomic.Int32
	d := New(nil)
	err := d.Download(context.Background(), RangeOptions{
		URL:          srv.URL + "/m",
		DestPath:     dest,
		ExpectedSize: int64(len(payload)),
		OnRetry:      func(_ int, _ error) { calls.Add(1) },
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("OnRetry fired %d times on a clean run; want 0", got)
	}
}

// TestBackoff_GoroutineSafe exercises the jitter RNG concurrently to
// match `lifecycle.downloadAll`'s fan-out under MaxConcurrentDownloads
// > 1. Pre-fix backoff used a shared `*rand.Rand` from `math/rand`,
// which is documented as not goroutine-safe; under contention `-race`
// flagged a write conflict on the LCG state.
//
// The test runs 100 workers, each calling backoff(2, 1ms, 100ms) 100
// times. With the math/rand/v2 globals the package functions are
// goroutine-safe by design, so the test must complete cleanly under
// `go test -race`. We additionally assert every value is non-negative
// to catch a future regression where jitter underflows.
func TestBackoff_GoroutineSafe(t *testing.T) {
	t.Parallel()
	const workers = 100
	const iters = 100
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				d := backoff(2, time.Millisecond, 100*time.Millisecond)
				if d < 0 {
					t.Errorf("backoff returned negative: %v", d)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestParseContentRangeStart(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in   string
		want int64
		ok   bool
	}{
		{"bytes 1024-2047/4096", 1024, true},
		{"bytes 0-99/100", 0, true},
		{"items 1-2/3", 0, false},
		{"bytes nope/100", 0, false},
		{"", 0, false},
	}
	for _, c := range cases {
		got, ok := parseContentRangeStart(c.in)
		if got != c.want || ok != c.ok {
			t.Errorf("parseContentRangeStart(%q) = (%d,%v), want (%d,%v)",
				c.in, got, ok, c.want, c.ok)
		}
	}
}

// countSink is a thread-safe progress.Sink used to assert OnFileStart /
// OnBytes / OnFileDone delivery.
type countSink struct {
	starts atomic.Int32
	dones  atomic.Int32
	bytes  atomic.Int64
	errs   atomic.Int32
	mu     sync.Mutex
	last   error
}

func (s *countSink) OnFileStart(string, int64) { s.starts.Add(1) }
func (s *countSink) OnBytes(n int64)           { s.bytes.Add(n) }
func (s *countSink) OnFileDone(string)         { s.dones.Add(1) }
func (s *countSink) OnError(err error) {
	s.errs.Add(1)
	s.mu.Lock()
	s.last = err
	s.mu.Unlock()
}
func (s *countSink) OnPhase(progress.Phase)  {}
func (s *countSink) OnResolvedCommit(string) {}
func (s *countSink) OnNote(string)           {}
func (s *countSink) OnFilesTotal(int)        {}

// silence unused import "io"
var _ = io.Discard
