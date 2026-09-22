//go:build integration

// Package ollamaurl_integration drives the ollama-url-channel
// download-fault matrix: llm-init (ENGINE_KIND=ollama) downloads the
// model URL from faultsrv before registering it into a real ollama
// daemon (tests/integration/compose/faultsrv-ollama-url.yml). Every
// case here fails during download/verify, before register, so the
// daemon only needs to be alive (ENGINE_KIND=ollama is AliveBeforeBoot,
// so Run blocks on it before the download step runs).
//
// Gated behind LLM_INIT_INTEGRATION_DOWNLOAD (same switch as the URL
// matrix); not part of the default CI integration job. Run locally:
//
//	LLM_INIT_INTEGRATION_DOWNLOAD=1 \
//	    go test -tags integration -timeout 30m -v ./tests/integration/ollamaurl/...
package ollamaurl_integration

import (
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

const (
	payloadSize = 1 << 20
	wrongSHA    = "0000000000000000000000000000000000000000000000000000000000000000"
	// ollama:// URL overload: body contains "://" so the parser routes
	// it to the KindOllamaURL channel.
	modelURL  = "ollama://http://faultsrv:9000/model.bin"
	waitPhase = 6 * time.Minute
)

func TestDownloadFaults_OllamaURL(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_DOWNLOAD") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_DOWNLOAD=1 to run the ollama-url download-fault matrix")
	}

	llmImage := shared.EnsureImage(t, "LLM_INIT_IMAGE", "llm-init:integration", "")
	faultImage := shared.EnsureImage(t, "FAULTSRV_IMAGE", "llm-init-faultsrv:integration",
		"tests/integration/faultsrv/Dockerfile")

	cases := []struct {
		name        string
		modelSource string
		forceStatus string
		wantPhase   string
		wantErr     string
	}{
		{
			name:        "download_retriable_stuck",
			modelSource: modelURL,
			forceStatus: "503",
			wantPhase:   "degraded", wantErr: "retry",
		},
		{
			name:        "download_non_retriable_404",
			modelSource: modelURL,
			forceStatus: "404",
			wantPhase:   "failed", wantErr: "not found",
		},
		{
			name:        "sha_mismatch",
			modelSource: modelURL + "#sha256=" + wrongSHA,
			forceStatus: "0",
			wantPhase:   "degraded", wantErr: "verify",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("LLM_INIT_IMAGE", llmImage)
			t.Setenv("FAULTSRV_IMAGE", faultImage)
			t.Setenv("FAULTSRV_SIZE", strconv.Itoa(payloadSize))
			t.Setenv("FAULTSRV_FORCE_STATUS", tc.forceStatus)
			t.Setenv("FAULTSRV_FAIL_FIRST", "0")
			t.Setenv("MODEL_SOURCE", tc.modelSource)

			port := os.Getenv("LLM_INIT_PORT")
			if port == "" {
				port = "8080"
			}
			shared.AssertPortFree(t, port)

			compose := shared.RepoPath(t, "tests/integration/compose/faultsrv-ollama-url.yml")
			stack := shared.Up(t, compose)

			got := stack.WaitProgress(t, func(p shared.Progress) bool {
				return p.Phase == tc.wantPhase
			}, waitPhase)
			t.Logf("phase=%s retry_count=%d last_error=%q", got.Phase, got.RetryCount, got.LastError)

			if tc.wantErr != "" && !strings.Contains(strings.ToLower(got.LastError), tc.wantErr) {
				t.Fatalf("last_error=%q does not contain %q (phase=%s)", got.LastError, tc.wantErr, got.Phase)
			}
		})
	}
}
