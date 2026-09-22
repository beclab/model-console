//go:build integration

// Package sglang_integration exercises deploy/compose/sglang.yml.
// Env-gated: skip unless LLM_INIT_INTEGRATION_SGLANG=1 is set. SGLang
// also needs an NVIDIA GPU.
//
// Local invocation (chat, default):
//
//	LLM_INIT_INTEGRATION_SGLANG=1 go test -tags integration -timeout 30m \
//	    ./tests/integration/sglang/...
//
// Local invocation (embedding; see llminit/integration_test_sglang_embedding.sh):
//
//	LLM_INIT_INTEGRATION_SGLANG=1 MODEL_MODE=embedding MODEL_SOURCE='hf://...' \
//	    go test -tags integration -timeout 30m ./tests/integration/sglang/...
package sglang_integration

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

// modelEnv defaults to a small chat instruct model. Override via env when
// running embedding (BAAI/bge-small-en-v1.5 or intfloat/e5-small-v2 + --is-embedding).
var modelEnv = map[string]string{
	"MODEL_NAME": "qwen2.5-0.5b",
	"MODEL_MODE": "chat",
	// Commit SHA pinned at the time of writing; refresh when HF retags.
	"MODEL_SOURCE":  "hf://Qwen/Qwen2.5-0.5B-Instruct --revision a8b602d9dafd3a75d382e62757d83d89fca3be54",
	"ENGINE_ARGS":   "--context-length 4096",
	"LLM_INIT_PORT": "8080",
	"SGLANG_PORT":   "30000",
	"LOG_LEVEL":     "info",
	"LOG_FORMAT":    "json",
}

func TestSGLang_FullStack(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_SGLANG") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_SGLANG=1 (and provide an NVIDIA GPU) to enable")
	}
	if os.Getenv("LLM_INIT_INTEGRATION_DISABLED") != "" {
		t.Skip("LLM_INIT_INTEGRATION_DISABLED set")
	}

	for k, v := range modelEnv {
		if os.Getenv(k) == "" {
			t.Setenv(k, v)
		}
	}
	port := envOr("LLM_INIT_PORT", modelEnv["LLM_INIT_PORT"])
	shared.AssertPortFree(t, port)

	composePath := repoRelative(t, "deploy/compose/sglang.yml")
	stack := shared.Up(t, composePath)

	// Wait for phase=ready; log /api/progress on each change (run go test -v).
	ready := stack.WaitReadyWithProgressLog(t, 20*time.Minute)
	t.Logf("ready in %s", ready)

	modelName := envOr("MODEL_NAME", modelEnv["MODEL_NAME"])
	mode := envOr("MODEL_MODE", modelEnv["MODEL_MODE"])

	var dataLatency time.Duration
	switch mode {
	case "embedding":
		latency, dim := shared.ProbeEmbeddings(t, stack.BaseURL, modelName, "hello")
		t.Logf("embeddings latency=%s dim=%d", latency, dim)
		dataLatency = latency
		shared.AssertEmbeddingSimilarityCases(t, stack.BaseURL, modelName, 0.3,
			shared.EmbeddingSimilarityCases)
	default:
		ttft, content := shared.ProbeChat(t, stack.BaseURL, modelName)
		t.Logf("TTFT=%s content=%q", ttft, content)
		if content == "" {
			t.Fatalf("empty completion")
		}
		dataLatency = ttft
	}

	shared.RecordBaseline(t, "sglang", ready, dataLatency)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
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
