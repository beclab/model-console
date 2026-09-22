//go:build integration

// Package negative_integration exercises the real-upstream download
// failure paths that cannot be faked hermetically: a HuggingFace repo
// that does not exist, a bad HF revision, and an unknown Ollama library
// tag. They run the production deploy/compose/<engine>.yml files directly
// (those seed the model-spec from MODEL_MODE, default chat) against the
// live huggingface.co / ollama.com upstreams, so they are gated behind
// LLM_INIT_INTEGRATION_NEGATIVE and never run in CI.
//
//	LLM_INIT_INTEGRATION_NEGATIVE=1 \
//	    go test -tags integration -timeout 20m -v ./tests/integration/negative/...
//
// Each case asserts the operator-facing message on GET /api/progress
// (the same last_error the dashboard renders), proving the lifecycle
// turns a raw upstream failure into an actionable prompt.
package negative_integration

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

const waitPhase = 8 * time.Minute

func TestNegative_RealUpstream(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_NEGATIVE") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_NEGATIVE=1 to run the real-upstream negative suite")
	}

	llmImage := shared.EnsureImage(t, "LLM_INIT_IMAGE", "llm-init:integration", "")
	llamacpp := shared.RepoPath(t, "deploy/compose/llamacpp.yml")
	ollama := shared.RepoPath(t, "deploy/compose/ollama.yml")

	cases := []struct {
		name      string
		compose   string
		env       map[string]string
		wantPhase string
		wantErr   string
	}{
		{
			name:    "hf_repo_not_found",
			compose: llamacpp,
			env: map[string]string{
				"MODEL_NAME":   "missing-repo",
				"MODEL_SOURCE": "hf://llm-init-test/this-repo-does-not-exist-zzz --include model.gguf",
				"ENGINE_ARGS":  "-c 512",
			},
			wantPhase: "failed", wantErr: "repository not found",
		},
		{
			name:    "hf_bad_revision",
			compose: llamacpp,
			env: map[string]string{
				"MODEL_NAME": "bad-revision",
				"MODEL_SOURCE": "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF " +
					"--include tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf " +
					"--revision 0000000000000000000000000000000000000000",
				"ENGINE_ARGS": "-c 512",
			},
			wantPhase: "failed", wantErr: "revision not found",
		},
		{
			name:    "ollama_invalid_tag",
			compose: ollama,
			env: map[string]string{
				"MODEL_NAME":   "missing-tag",
				"MODEL_SOURCE": "ollama://this-model-definitely-does-not-exist-zzz",
			},
			wantPhase: "degraded", wantErr: "not found in the library",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("LLM_INIT_IMAGE", llmImage)
			for k, v := range tc.env {
				t.Setenv(k, v)
			}

			port := os.Getenv("LLM_INIT_PORT")
			if port == "" {
				port = "8080"
			}
			shared.AssertPortFree(t, port)

			stack := shared.Up(t, tc.compose)
			got := stack.WaitProgress(t, func(p shared.Progress) bool {
				return p.Phase == tc.wantPhase
			}, waitPhase)
			t.Logf("phase=%s retry_count=%d last_error=%q", got.Phase, got.RetryCount, got.LastError)

			if !strings.Contains(strings.ToLower(got.LastError), tc.wantErr) {
				t.Fatalf("last_error=%q does not contain %q (phase=%s)", got.LastError, tc.wantErr, got.Phase)
			}
		})
	}
}
