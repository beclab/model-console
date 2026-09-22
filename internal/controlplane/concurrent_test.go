package controlplane

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// TestConcurrentCardEditsAndReads is the acceptance test for the runtime
// config having one owner. Editing the model card and reading it back are
// ordinary concurrent operations — Router PATCHes while the dashboard
// polls, and an operator's /api/config is a support artefact that must be
// collectable at any moment — and until the store existed the write was
// two unlocked assignments into a struct holding maps that these three
// handlers were reading from other goroutines.
//
// The assertion is the race detector's, so this test is only meaningful
// under `go test -race`. Without it the old code passed too.
func TestConcurrentCardEditsAndReads(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	const rounds = 25
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := range rounds {
			// Each round moves the launch flags, so every iteration
			// takes the full write path: normalise, persist, publish
			// engine_args, signal a restart.
			body := `{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len ` +
				strconv.Itoa(2048+i*8) + `"}`
			req, err := http.NewRequest(http.MethodPut, srv.URL+"/api/model-spec",
				strings.NewReader(body))
			if err != nil {
				t.Error(err)
				return
			}
			req.Header.Set("Content-Type", contentTypeJSON)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Error(err)
				return
			}
			if resp.StatusCode != http.StatusOK {
				t.Errorf("PUT status = %d", resp.StatusCode)
			}
			resp.Body.Close()
		}
	}()

	for _, path := range []string{"/api/config", "/api/endpoints", "/api/model-spec", "/healthz"} {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range rounds {
				resp, err := http.Get(srv.URL + path)
				if err != nil {
					t.Error(err)
					return
				}
				if resp.StatusCode != http.StatusOK {
					t.Errorf("GET %s status = %d", path, resp.StatusCode)
				}
				resp.Body.Close()
			}
		}()
	}

	wg.Wait()
}
