// Command mockollama runs the download suite's mock Ollama daemon as a
// standalone process for the manual runbook. Point llm-init at it with
// LLM_INIT_TEST_ENGINE_URL=http://127.0.0.1:11434 and ENGINE_KIND=ollama.
//
// Example (simulate an unknown library tag -> 404):
//
//	go run ./tests/download/cmd/mockollama -addr 127.0.0.1:11434 -pull-status 404
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/llm-init/llm-init/tests/download/mockollama"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:11434", "listen address")
	pullStatus := flag.Int("pull-status", 200, "POST /api/pull status (404 = unknown tag, 403 = denied)")
	pullOmitSuccess := flag.Bool("pull-omit-success", false, "end /api/pull without a success frame")
	createOmitSuccess := flag.Bool("create-omit-success", false, "end /api/create without a success frame")
	blobStatus := flag.Int("blob-status", 201, "POST /api/blobs status (403 = denied)")
	existing := flag.String("existing", "", "comma-separated models to pre-populate /api/tags")
	flag.Parse()

	cfg := mockollama.Config{
		PullStatus:        *pullStatus,
		PullOmitSuccess:   *pullOmitSuccess,
		CreateOmitSuccess: *createOmitSuccess,
		BlobPushStatus:    *blobStatus,
	}
	if *existing != "" {
		cfg.ExistingModels = strings.Split(*existing, ",")
	}

	srv := mockollama.New(cfg)
	defer srv.Close()

	fmt.Printf("mockollama backing server at %s\n", srv.URL)
	fmt.Printf("point llm-init at: LLM_INIT_TEST_ENGINE_URL=http://%s ENGINE_KIND=ollama\n", *addr)

	// Reverse-proxy the fixed address onto the httptest backend so the
	// runbook URL is stable.
	proxy := func(w http.ResponseWriter, r *http.Request) {
		req, err := http.NewRequestWithContext(r.Context(), r.Method, srv.URL+r.URL.Path, r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		req.Header = r.Header.Clone()
		resp, err := srv.Client().Do(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		for k, vs := range resp.Header {
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(resp.StatusCode)
		buf := make([]byte, 32*1024)
		for {
			n, rerr := resp.Body.Read(buf)
			if n > 0 {
				_, _ = w.Write(buf[:n])
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			}
			if rerr != nil {
				return
			}
		}
	}

	server := &http.Server{
		Addr:              *addr,
		Handler:           http.HandlerFunc(proxy),
		ReadHeaderTimeout: 10 * time.Second,
	}
	// Return (not log.Fatal) on error so the deferred srv.Close() runs.
	if err := server.ListenAndServe(); err != nil {
		log.Print(err)
	}
}
