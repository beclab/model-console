package fetch

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSumProbeContentLength(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", "100")
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	u1 := srv.URL + "/a.gguf"
	u2 := srv.URL + "/b.gguf"
	total, ok, allKnown := SumProbeContentLength(context.Background(), nil, []string{u1, u2})
	if !ok || !allKnown || total != 200 {
		t.Fatalf("total=%d ok=%v allKnown=%v, want 200 true true", total, ok, allKnown)
	}
}

func TestSumProbeContentLength_PartialKnown(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead {
			http.NotFound(w, r)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/a.gguf") {
			w.Header().Set("Content-Length", "100")
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	total, ok, allKnown := SumProbeContentLength(context.Background(), nil, []string{
		srv.URL + "/a.gguf",
		srv.URL + "/b.gguf",
	})
	if !ok || allKnown || total != 100 {
		t.Fatalf("total=%d ok=%v allKnown=%v, want 100 true false", total, ok, allKnown)
	}
}
