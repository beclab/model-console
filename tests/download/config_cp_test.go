//go:build download

package download

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/controlplane"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/runtimecfg"
)

// --- E. config negative-input parsing (clear prompts for bad input) --

func TestConfig_InvalidInput_GivesClearErrors(t *testing.T) {
	cases := []struct {
		name    string
		env     map[string]string
		wantSub string
	}{
		{"model_source_num_removed",
			map[string]string{"MODEL_SOURCE": "hf://a/b", "MODEL_SOURCE_NUM": "1"},
			"comma-separated MODEL_SOURCE"},
		{"missing_source",
			map[string]string{},
			"required"},
		{"bad_scheme",
			map[string]string{"MODEL_SOURCE": "ftp://host/x"},
			"unsupported scheme"},
		{"hf_bad_repo",
			map[string]string{"MODEL_SOURCE": "hf://noslash"},
			"owner/repo"},
		{"url_missing_local",
			map[string]string{"MODEL_SOURCE": "https://host/model.bin"},
			"MODEL_SOURCE_LOCAL"},
		{"url_inline_flags",
			map[string]string{"MODEL_SOURCE": "https://host/model.bin --include x", "MODEL_SOURCE_LOCAL": "/tmp/m"},
			"does not accept inline flags"},
		{"ollama_inline_flags",
			map[string]string{"MODEL_SOURCE": "ollama://qwen2.5:7b --revision x"},
			"does not accept inline flags"},
		{"hf_endpoint_inline",
			map[string]string{"MODEL_SOURCE": "hf://a/b --endpoint https://m.co"},
			"not allowed inline"},
		{"bad_sha_fragment",
			map[string]string{"MODEL_SOURCE": "https://host/m.bin#sha256=xyz", "MODEL_SOURCE_LOCAL": "/tmp/m"},
			"sha256"},
		{"comma_local_mismatch",
			map[string]string{
				"MODEL_SOURCE":       "ollama://a,ollama://b",
				"MODEL_SOURCE_LOCAL": "/x",
			},
			"MODEL_SOURCE_LOCAL"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			g := func(k string) string { return tc.env[k] }
			_, err := config.LoadModelSources(g)
			if err == nil {
				t.Fatalf("expected an error for %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error = %q, want substring %q", err.Error(), tc.wantSub)
			}
		})
	}
}

// --- F. control plane: /api/progress + /api/retry --------------------

func newControlPlane(t *testing.T, pm progress.Manager, retry controlplane.RetryFunc, rate float64) *httptest.Server {
	t.Helper()
	srv := controlplane.NewServer(controlplane.Options{
		Manager:        pm,
		Config:         runtimecfg.New(config.Config{Model: config.Model{Name: "m"}, Engine: config.Engine{Kind: config.EngineLlamaCpp}}, nil),
		Retry:          retry,
		RetryRateLimit: rate,
	})
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

func TestControlPlane_ProgressExposesLastError(t *testing.T) {
	pm := progress.New(time.Now())
	pm.Update(func(s *progress.State) {
		s.Phase = progress.PhaseFailed
		s.LastError = "HuggingFace repository is gated: request access on the model page and set HF_TOKEN"
	})
	ts := newControlPlane(t, pm, func(context.Context, controlplane.RetryOptions) error { return nil }, 0)

	resp, err := http.Get(ts.URL + "/api/progress")
	if err != nil {
		t.Fatalf("GET /api/progress: %v", err)
	}
	defer resp.Body.Close()
	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body["phase"] != "failed" {
		t.Errorf("phase = %v want failed", body["phase"])
	}
	le, _ := body["last_error"].(string)
	if !strings.Contains(le, "gated") {
		t.Errorf("last_error = %q, want the operator-facing gated message", le)
	}
}

func TestControlPlane_RetryForwardsForce(t *testing.T) {
	pm := progress.New(time.Now())
	var gotForce atomic.Bool
	var called atomic.Int32
	retry := func(_ context.Context, opts controlplane.RetryOptions) error {
		called.Add(1)
		gotForce.Store(opts.Force)
		return nil
	}
	ts := newControlPlane(t, pm, retry, 0)

	resp, err := http.Post(ts.URL+"/api/retry?force=true", "", nil)
	if err != nil {
		t.Fatalf("POST /api/retry: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Errorf("status = %d want 202", resp.StatusCode)
	}
	if called.Load() != 1 || !gotForce.Load() {
		t.Errorf("retry called=%d force=%v, want 1/true", called.Load(), gotForce.Load())
	}
}

func TestControlPlane_RetryRateLimited(t *testing.T) {
	pm := progress.New(time.Now())
	retry := func(context.Context, controlplane.RetryOptions) error { return nil }
	ts := newControlPlane(t, pm, retry, 1.0) // burst 5, 1/s refill

	var got429 bool
	for i := 0; i < 12; i++ {
		resp, err := http.Post(ts.URL+"/api/retry", "", nil)
		if err != nil {
			t.Fatalf("POST: %v", err)
		}
		if resp.StatusCode == http.StatusTooManyRequests {
			got429 = true
		}
		resp.Body.Close()
	}
	if !got429 {
		t.Error("expected at least one 429 after exhausting the retry burst")
	}
}
