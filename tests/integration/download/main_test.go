//go:build integration

// Package download_integration drives the URL-channel download-fault
// matrix against a real llm-init container fronting a faultsrv (see
// tests/integration/compose/faultsrv-url.yml). It is gated behind
// LLM_INIT_INTEGRATION_DOWNLOAD because it builds two images and runs a
// compose stack per case; it is NOT part of the default CI integration
// job (which only runs the llamacpp happy path) — run it locally:
//
//	LLM_INIT_INTEGRATION_DOWNLOAD=1 \
//	    go test -tags integration -timeout 30m -v ./tests/integration/download/...
//
// Each case asserts on GET /api/progress, not /readyz: the stack starts
// no real engine, so the lifecycle reaches degraded/failed (or flips
// ready after writing the sentinel) independent of WaitAlive.
package download_integration

import (
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/download/faultsrv"
	"github.com/llm-init/llm-init/tests/integration/shared"
)

const (
	// payloadSize is the faultsrv body size; the SHA-256 of a body of
	// this size is what the happy-path MODEL_SOURCE pins via #sha256=.
	payloadSize = 1 << 20
	// wrongSHA is a syntactically valid but incorrect digest used to
	// drive the verify-failure case.
	wrongSHA  = "0000000000000000000000000000000000000000000000000000000000000000"
	modelURL  = "http://faultsrv:9000/model.bin"
	waitPhase = 6 * time.Minute
)

func TestDownloadFaults_URL(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_DOWNLOAD") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_DOWNLOAD=1 to run the URL download-fault matrix")
	}

	goodSHA := faultsrv.SHA256Hex(faultsrv.GenerateBody(payloadSize))

	llmImage := shared.EnsureImage(t, "LLM_INIT_IMAGE", "llm-init:integration", "")
	faultImage := shared.EnsureImage(t, "FAULTSRV_IMAGE", "llm-init-faultsrv:integration",
		"tests/integration/faultsrv/Dockerfile")

	cases := []struct {
		name        string
		modelSource string
		forceStatus string
		failFirst   string
		wantPhase   string
		wantErr     string // case-insensitive substring of last_error; "" skips the check
	}{
		{
			name:        "normal",
			modelSource: modelURL + "#sha256=" + goodSHA,
			forceStatus: "0", failFirst: "0",
			wantPhase: "ready",
		},
		{
			name:        "retriable_recover",
			modelSource: modelURL + "#sha256=" + goodSHA,
			forceStatus: "0", failFirst: "2",
			wantPhase: "ready",
		},
		{
			name:        "retriable_stuck",
			modelSource: modelURL,
			forceStatus: "503", failFirst: "0",
			wantPhase: "degraded", wantErr: "retry",
		},
		{
			name:        "non_retriable_404",
			modelSource: modelURL,
			forceStatus: "404", failFirst: "0",
			wantPhase: "failed", wantErr: "not found",
		},
		{
			name:        "sha_mismatch",
			modelSource: modelURL + "#sha256=" + wrongSHA,
			forceStatus: "0", failFirst: "0",
			wantPhase: "degraded", wantErr: "verify",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("LLM_INIT_IMAGE", llmImage)
			t.Setenv("FAULTSRV_IMAGE", faultImage)
			t.Setenv("FAULTSRV_SIZE", strconv.Itoa(payloadSize))
			t.Setenv("FAULTSRV_FORCE_STATUS", tc.forceStatus)
			t.Setenv("FAULTSRV_FAIL_FIRST", tc.failFirst)
			t.Setenv("MODEL_SOURCE", tc.modelSource)

			port := os.Getenv("LLM_INIT_PORT")
			if port == "" {
				port = "8080"
			}
			shared.AssertPortFree(t, port)

			compose := shared.RepoPath(t, "tests/integration/compose/faultsrv-url.yml")
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
