package fetch

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func newURLServer(body []byte) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
			return
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
}

func TestURLClient_FetchOK(t *testing.T) {
	t.Parallel()
	ts := newURLServer(payload)
	defer ts.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	c := NewURLClient()
	if err := c.Fetch(context.Background(), ts.URL+"/file", dest,
		int64(len(payload)), payloadSHA(), nil); err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != string(payload) {
		t.Errorf("payload mismatch")
	}
}

func TestURLClient_FetchSizeOnly(t *testing.T) {
	t.Parallel()
	ts := newURLServer(payload)
	defer ts.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	if err := NewURLClient().Fetch(context.Background(), ts.URL+"/file", dest,
		int64(len(payload)), "", nil); err != nil {
		t.Fatalf("Fetch: %v", err)
	}
}

func TestURLClient_FetchSHAMismatchRemovesFile(t *testing.T) {
	t.Parallel()
	ts := newURLServer(payload)
	defer ts.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	bogusSHA := "deadbeef" + payloadSHA()[8:]
	err := NewURLClient().Fetch(context.Background(), ts.URL+"/file", dest,
		int64(len(payload)), bogusSHA, nil)
	if !errors.Is(err, ErrSHA256Mismatch) {
		t.Errorf("err = %v, want ErrSHA256Mismatch", err)
	}
	if _, statErr := os.Stat(dest); !errors.Is(statErr, os.ErrNotExist) {
		t.Errorf("dest should have been removed, stat err = %v", statErr)
	}
}

func TestURLClient_FetchSizeMismatchKeepsError(t *testing.T) {
	t.Parallel()
	ts := newURLServer(payload)
	defer ts.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	err := NewURLClient().Fetch(context.Background(), ts.URL+"/file", dest,
		int64(len(payload))+100, "", nil)
	if !errors.Is(err, ErrSizeMismatch) {
		t.Errorf("err = %v, want ErrSizeMismatch", err)
	}
}

func TestURLClient_FetchValidatesArgs(t *testing.T) {
	t.Parallel()
	c := NewURLClient()
	if err := c.Fetch(context.Background(), "", "x", 0, "", nil); err == nil {
		t.Error("missing url should error")
	}
	if err := c.Fetch(context.Background(), "http://x", "", 0, "", nil); err == nil {
		t.Error("missing dest should error")
	}
}

func TestURLClient_FetchWithHeaders(t *testing.T) {
	t.Parallel()
	gotAuth := ""
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
			return
		}
		_, _ = w.Write(payload)
	}))
	defer ts.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	hdrs := http.Header{}
	hdrs.Set("Authorization", "Bearer custom-token")
	if err := NewURLClient().FetchWithHeaders(context.Background(), ts.URL+"/file",
		dest, int64(len(payload)), payloadSHA(), hdrs, nil); err != nil {
		t.Fatalf("FetchWithHeaders: %v", err)
	}
	if gotAuth != "Bearer custom-token" {
		t.Errorf("Authorization not propagated, got %q", gotAuth)
	}
}

func TestURLClient_FetchWithHeadersSHAMismatch(t *testing.T) {
	t.Parallel()
	ts := newURLServer(payload)
	defer ts.Close()

	dest := filepath.Join(t.TempDir(), "out.bin")
	err := NewURLClient().FetchWithHeaders(context.Background(), ts.URL+"/file",
		dest, int64(len(payload)),
		"0000000000000000000000000000000000000000000000000000000000000000",
		nil, nil)
	if !errors.Is(err, ErrSHA256Mismatch) {
		t.Errorf("err = %v, want ErrSHA256Mismatch", err)
	}
	if _, statErr := os.Stat(dest); !errors.Is(statErr, os.ErrNotExist) {
		t.Errorf("dest should be removed on SHA mismatch")
	}
}

func TestRemoveIfExists(t *testing.T) {
	t.Parallel()
	if err := removeIfExists(filepath.Join(t.TempDir(), "ghost")); err != nil {
		t.Errorf("ENOENT should be silent, got %v", err)
	}
	p := filepath.Join(t.TempDir(), "f")
	_ = os.WriteFile(p, []byte("x"), 0o644)
	if err := removeIfExists(p); err != nil {
		t.Errorf("remove: %v", err)
	}
}
