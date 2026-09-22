//go:build integration

// Package llamacpp_integration exercises deploy/compose/llamacpp.yml
// against a real llm-init + llama.cpp-server pair. CI runs this test
// (commit M5.7's `integration` job) using a TinyLlama Q4 GGUF; the
// total budget is 8 minutes including model download (was 5 min
// pre-v1.0.7; bumped after PR #12 hit two consecutive 5m0s timeouts
// on a clean rebase due to HF CDN slowdown during a Dependabot
// burst -- see WaitReady call site for the data).
//
// Trigger locally (chat, default):
//
//	go test -tags integration -timeout 14m \
//	    ./tests/integration/llamacpp/...
//
// Trigger locally (embedding; override model source to an embedding GGUF):
//
//	MODEL_MODE=embedding MODEL_NAME=my-embed MODEL_SOURCE='hf://...' \
//	    go test -tags integration -timeout 14m ./tests/integration/llamacpp/...
package llamacpp_integration

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

// modelEnv defines a TinyLlama Q4 download. It is small (~600 MB) and
// works on CPU-only runners. The CI workflow can override the values
// to point at a cached mirror without code changes.
var modelEnv = map[string]string{
	"MODEL_NAME": "tinyllama-q4",
	"MODEL_MODE": "chat",
	// Commit SHA pinned at the time of writing; CI overrides via env
	// when HF retags the repo.
	"MODEL_SOURCE":  "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --include tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --revision 52e7645ba7c309695bec7ac98f4f005b139cf465",
	"ENGINE_ARGS":   "-c 2048",
	"LLM_INIT_PORT": "8080",
	"LLAMACPP_PORT": "8081",
	"LOG_LEVEL":     "info",
	"LOG_FORMAT":    "json",
}

func TestLlamacpp_FullStack(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_DISABLED") != "" {
		t.Skip("LLM_INIT_INTEGRATION_DISABLED set")
	}

	// docker compose --env-file is the supported way to point at a
	// custom env file, but we want the values to come from t.Setenv so
	// CI can override per-job. Export them now.
	for k, v := range modelEnv {
		if os.Getenv(k) == "" {
			t.Setenv(k, v)
		}
	}
	port := envOr("LLM_INIT_PORT", modelEnv["LLM_INIT_PORT"])
	shared.AssertPortFree(t, port)

	composePath := repoRelative(t, "deploy/compose/llamacpp.yml")
	stack := shared.Up(t, composePath)

	// Cold-deploy budget: 10 min download + 1 min llama-server warmup.
	// The harness only times the readiness phase; model download is
	// included since /readyz waits for phase=ready.
	//
	// 8 min was the v1.0.7 bump but GHA still times out when HF returns
	// sustained 429s to anonymous egress (lifecycle retries correctly,
	// the budget just runs out). CI passes HF_TOKEN for a higher quota;
	// 10 min covers the remaining anonymous/fallback variance.
	ready := stack.WaitReady(t, 10*time.Minute)
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
			t.Fatalf("empty completion content")
		}
		dataLatency = ttft
	}

	// GPU residency endpoint. The compose stack leaves n_gpu_layers
	// at the shipped default ("all"), so the placement contract holds
	// and source must be "llamacpp_fit_off_alive" on both the GPU rig
	// and the CPU-only runner (the source is an env-contract verdict,
	// not a CUDA measurement). "unavailable" means someone diverged
	// the compose ENGINE_ARGS from the wrapper flags; failing here is
	// the whole point of the gate.
	//
	// The numeric fields stay pointers: llama.cpp exposes no layer
	// counts, and kv_cache_usage_perc only appears when the engine
	// serves /metrics, so both are logged rather than required.
	gpuResp, err := http.Get(stack.BaseURL + "/api/diag/gpu")
	if err != nil {
		t.Fatalf("GET /api/diag/gpu: %v", err)
	}
	gpuBody, _ := io.ReadAll(gpuResp.Body)
	gpuResp.Body.Close()
	if gpuResp.StatusCode != http.StatusOK {
		t.Fatalf("/api/diag/gpu status=%d body=%s", gpuResp.StatusCode, gpuBody)
	}
	var gpuReport struct {
		SchemaVersion int    `json:"schema_version"`
		EngineKind    string `json:"engine_kind"`
		GPU           struct {
			Source           string   `json:"source"`
			GPULayers        *int     `json:"gpu_layers"`
			KVCacheUsagePerc *float64 `json:"kv_cache_usage_perc"`
		} `json:"gpu"`
		Warnings []string `json:"warnings"`
	}
	if err := json.Unmarshal(gpuBody, &gpuReport); err != nil {
		t.Fatalf("/api/diag/gpu decode: %v body=%s", err, gpuBody)
	}
	if gpuReport.SchemaVersion != 1 {
		t.Errorf("schema_version=%d, want 1 (body=%s)", gpuReport.SchemaVersion, gpuBody)
	}
	if gpuReport.EngineKind != "llamacpp" {
		t.Errorf("engine_kind=%q, want llamacpp (body=%s)", gpuReport.EngineKind, gpuBody)
	}
	if gpuReport.GPU.Source != "llamacpp_fit_off_alive" {
		t.Fatalf("/api/diag/gpu reports gpu.source=%q on a contract-compliant compose stack, "+
			"want llamacpp_fit_off_alive; this means someone diverged ENGINE_ARGS n_gpu_layers "+
			"from the wrapper flags. warnings=%v body=%s", gpuReport.GPU.Source, gpuReport.Warnings, gpuBody)
	}
	t.Logf("/api/diag/gpu source=%s gpu_layers=%v kv_cache_usage_perc=%v warnings=%v",
		gpuReport.GPU.Source, gpuReport.GPU.GPULayers, gpuReport.GPU.KVCacheUsagePerc, gpuReport.Warnings)

	shared.RecordBaseline(t, "llamacpp", ready, dataLatency)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// repoRelative resolves a path against the repository root (located by
// walking up to find go.mod). Failing the lookup fails the test.
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
