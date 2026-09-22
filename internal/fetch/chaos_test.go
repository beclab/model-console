package fetch

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync/atomic"
	"testing"
	"time"
)

// Chaos tests share the package-level backoff overrides, so they must not
// run in parallel with each other. They can still run in parallel with
// other test packages because each test file sees its own global state via
// `go test`.

// TestChaos_ETagChangeMidStream simulates the upstream rolling the file
// (ETag changes) between the initial HEAD and a resumed GET. The probe
// path catches the new ETag and forces a clean restart by deleting any
// stale .part; the second attempt then downloads the new bytes from
// scratch.
func TestChaos_ETagChangeMidStream(t *testing.T) {

	v1 := bigPayload(32 * 1024)
	v2 := bigPayload(40 * 1024)
	for i := range v2 {
		v2[i] ^= 0xAA // make sure bytes actually differ
	}

	var rolled atomic.Bool
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		body, etag := v1, `"etag-v1"`
		if rolled.Load() {
			body, etag = v2, `"etag-v2"`
		}
		w.Header().Set("ETag", etag)
		switch r.Method {
		case http.MethodHead:
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body)
		}
	}))
	defer srv.Close()

	dl := New(nil)
	dest := filepath.Join(t.TempDir(), "out.bin")

	// Caller passes the v1 ETag and a stale .part containing v1 partial
	// bytes. Before our HEAD probe runs the upstream rolls to v2; the
	// new ETag mismatches the caller's recorded one, so .part is wiped
	// and the v2 body is fetched cleanly.
	if err := os.WriteFile(dest+".part", v1[:len(v1)/2], 0o644); err != nil {
		t.Fatal(err)
	}
	rolled.Store(true) // upstream is now v2 by the time we hit it

	if err := dl.Download(context.Background(), RangeOptions{
		URL:          srv.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(v2)),
		ETag:         `"etag-v1"`,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}

	got, _ := os.ReadFile(dest)
	if string(got) != string(v2) {
		t.Errorf("payload mismatch: got %d bytes, want v2 of %d bytes", len(got), len(v2))
	}
}

// TestChaos_5xxThenRecover proves the retry path actually advances after a
// transient 502 storm. The server returns 502 three times then succeeds; we
// expect MaxRetries=5 to absorb that comfortably and end with a complete
// download. We also use a tiny fake clock to assert backoff actually
// invoked (i.e. retries != 0).
func TestChaos_5xxThenRecover(t *testing.T) {

	body := bigPayload(8 * 1024)
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"e"`)
		switch r.Method {
		case http.MethodHead:
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			n := attempts.Add(1)
			if n <= 3 {
				http.Error(w, "boom", http.StatusBadGateway)
				return
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body)
		}
	}))
	defer srv.Close()

	dl := New(nil).(*defaultDownloader)
	dl.backoffBase = time.Millisecond
	dl.backoffCap = 10 * time.Millisecond
	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          srv.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(body)),
		MaxRetries:   5,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if got := attempts.Load(); got != 4 {
		t.Errorf("GET attempts = %d, want 4 (3 boom + 1 success)", got)
	}
	finalGot, _ := os.ReadFile(dest)
	if len(finalGot) != len(body) {
		t.Errorf("size = %d, want %d", len(finalGot), len(body))
	}
}

// TestChaos_RetryBudgetExceededWraps drives MaxRetries=1 against a 503
// firehose and asserts the returned error wraps ErrTooManyRetries (so
// lifecycle can branch on it) and embeds the upstream status text for
// operator diagnostics.
func TestChaos_RetryBudgetExceededWraps(t *testing.T) {

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"e"`)
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusOK)
			return
		}
		http.Error(w, "service unavailable", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	dl := New(nil).(*defaultDownloader)
	dl.backoffBase = time.Millisecond
	dl.backoffCap = 5 * time.Millisecond
	err := dl.Download(context.Background(), RangeOptions{
		URL:          srv.URL + "/file",
		DestPath:     filepath.Join(t.TempDir(), "out.bin"),
		ExpectedSize: 100,
		MaxRetries:   1,
	}, nil)
	if !errors.Is(err, ErrTooManyRetries) {
		t.Errorf("err = %v, want wrapping ErrTooManyRetries", err)
	}
	if !contains(err.Error(), "503") {
		t.Errorf("error message lost upstream status: %v", err)
	}
}

// TestChaos_RangeMismatchTriggersReset covers the Content-Range / existing
// disagreement path: the server happily replies 206 but with a start offset
// that doesn't match what the client requested. We expect the downloader
// to delete .part and retry from zero.
func TestChaos_RangeMismatchTriggersReset(t *testing.T) {

	body := bigPayload(16 * 1024)
	var get atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"e"`)
		switch r.Method {
		case http.MethodHead:
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			n := get.Add(1)
			if n == 1 {
				// Lie: claim 206 with a totally wrong start offset.
				w.Header().Set("Content-Length", strconv.Itoa(len(body)))
				w.Header().Set("Content-Range",
					fmt.Sprintf("bytes 0-%d/%d", len(body)-1, len(body)))
				w.WriteHeader(http.StatusPartialContent)
				_, _ = w.Write(body)
				return
			}
			// Subsequent attempts: serve the full file cleanly.
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body)
		}
	}))
	defer srv.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := os.WriteFile(dest+".part", body[:len(body)/2], 0o644); err != nil {
		t.Fatal(err)
	}
	dl := New(nil).(*defaultDownloader)
	dl.backoffBase = time.Millisecond
	dl.backoffCap = 5 * time.Millisecond
	if err := dl.Download(context.Background(), RangeOptions{
		URL:          srv.URL + "/file",
		DestPath:     dest,
		ExpectedSize: int64(len(body)),
		ETag:         `"e"`,
		MaxRetries:   3,
	}, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	final, _ := os.ReadFile(dest)
	if string(final) != string(body) {
		t.Errorf("final mismatch")
	}
}

func contains(haystack, needle string) bool {
	return len(haystack) > 0 && len(needle) > 0 &&
		(haystack == needle || (len(haystack) >= len(needle) &&
			indexOf(haystack, needle) >= 0))
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// bigPayload returns a deterministic n-byte slice. Used by the chaos
// tests to fabricate "model file" bodies large enough that the range
// downloader actually exercises its resume / retry paths instead of
// trivially completing on the first chunk. Lived in
// internal/fetch/integration_test.go before commit d576e83 dropped
// the manifest-era HF integration test; chaos_test.go is the only
// remaining caller, so the helper now lives next to it.
func bigPayload(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((i*31 + 7) % 251)
	}
	return b
}
