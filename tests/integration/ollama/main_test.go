//go:build integration

// Package ollama_integration exercises deploy/compose/ollama.yml. It
// is env-gated: the test skips unless LLM_INIT_INTEGRATION_OLLAMA=1 is
// set, since CI does not run Ollama (model pulls are heavy and the
// daemon image is GPU-friendly only).
//
// Local invocation:
//
//	LLM_INIT_INTEGRATION_OLLAMA=1 go test -tags integration -timeout 15m \
//	    ./tests/integration/ollama/...
package ollama_integration

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

func TestOllama_FullStack(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_OLLAMA") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_OLLAMA=1 to enable; CI never runs this test")
	}

	env := map[string]string{
		"MODEL_NAME":    "qwen2.5:0.5b",
		"MODEL_SOURCE":  "ollama://qwen2.5:0.5b",
		"LLM_INIT_PORT": "8080",
		"LOG_LEVEL":     "info",
		"LOG_FORMAT":    "json",
	}
	for k, v := range env {
		if os.Getenv(k) == "" {
			t.Setenv(k, v)
		}
	}
	shared.AssertPortFree(t, env["LLM_INIT_PORT"])

	composePath := repoRelative(t, "deploy/compose/ollama.yml")
	stack := shared.Up(t, composePath)

	// Pull budget: 0.5B model is ~400 MB, plus daemon startup.
	ready := stack.WaitReady(t, 10*time.Minute)
	t.Logf("ready in %s", ready)

	ttft, content := shared.ProbeChat(t, stack.BaseURL, env["MODEL_NAME"])
	if content == "" {
		t.Fatalf("empty completion")
	}
	t.Logf("TTFT=%s content=%q", ttft, content)

	shared.RecordBaseline(t, "ollama", ready, ttft)
}

func repoRelative(t *testing.T, rel string) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for dir := wd; ; {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Join(dir, rel)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("go.mod not found above %s", wd)
		}
		dir = parent
	}
}
