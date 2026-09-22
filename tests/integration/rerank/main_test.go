//go:build integration

// Package rerank_integration exercises deploy/compose/rerank.yml against
// a real llm-init + rerank-server pair. Env-gated: skip unless
// LLM_INIT_INTEGRATION_RERANK=1 (needs rerank-server image + HF model).
//
// Local invocation (see llminit/integration_test_rerank_unified.sh):
//
//	LLM_INIT_INTEGRATION_RERANK=1 \
//	    ACCELERATOR=cpu \
//	    RERANK_IMAGE=rerank-server:onnx-cpu-amd64 \
//	    MODEL_NAME=bge-reranker-v2-m3 \
//	    MODEL_ID=bge-reranker-v2-m3 \
//	    MODEL_SOURCE='hf://beclab/bge-reranker-v2-m3 --revision main --exclude "openvino/**" --subdir onnx' \
//	    go test -tags integration -timeout 45m -v ./tests/integration/rerank/...
package rerank_integration

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

var modelEnv = map[string]string{
	"MODEL_NAME": "bge-reranker-v2-m3",
	"MODEL_ID":   "bge-reranker-v2-m3",
	"MODEL_MODE": "rerank",
	"LOG_LEVEL":  "info",
	"LOG_FORMAT": "json",
}

// TestRerank_FullStack is the end-to-end contract for ENGINE_KIND=rerank.
func TestRerank_FullStack(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_RERANK") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_RERANK=1 to enable (needs rerank-server image + model)")
	}
	if os.Getenv("LLM_INIT_INTEGRATION_DISABLED") != "" {
		t.Skip("LLM_INIT_INTEGRATION_DISABLED set")
	}

	for k, v := range modelEnv {
		if os.Getenv(k) == "" {
			t.Setenv(k, v)
		}
	}
	port := strings.TrimSpace(os.Getenv("LLM_INIT_PORT"))
	if port != "" {
		shared.AssertPortFree(t, port)
	}

	if os.Getenv("RERANK_USE_HOST_NETWORK") == "1" {
		if os.Getenv("RERANK_ENGINE_PORT") == "" {
			t.Setenv("RERANK_ENGINE_PORT", shared.FreePort(t))
		}
		t.Logf("host-network overlay: RERANK_ENGINE_PORT=%s", os.Getenv("RERANK_ENGINE_PORT"))
	}

	t.Logf("ACCELERATOR=%s RERANK_RUNTIME=%s RERANK_IMAGE=%s",
		envOr("ACCELERATOR", ""), envOr("RERANK_RUNTIME", ""), envOr("RERANK_IMAGE", ""))
	t.Logf("MODEL_SOURCE=%s", envOr("MODEL_SOURCE", ""))
	t.Logf("HF_ENDPOINT=%s HF_CACHE_SOURCE=%s",
		envOr("HF_ENDPOINT", ""), envOr("HF_CACHE_SOURCE", ""))

	composeFiles := []string{shared.RepoPath(t, "deploy/compose/rerank.yml")}
	if os.Getenv("RERANK_USE_HOST_NETWORK") == "1" {
		composeFiles = append(composeFiles, shared.RepoPath(t, "deploy/compose/rerank.integration.yml"))
	}
	if os.Getenv("RERANK_COMPOSE_GPU") != "" {
		composeFiles = append(composeFiles, shared.RepoPath(t, "deploy/compose/rerank.gpu.yml"))
		if os.Getenv("RERANK_HOST_PORT") == "" {
			t.Setenv("RERANK_HOST_PORT", shared.FreePort(t))
		}
		t.Logf("merging GPU overlay rerank.gpu.yml (RERANK_GPU_COUNT=%s RERANK_HOST_PORT=%s)",
			envOr("RERANK_GPU_COUNT", "all"), os.Getenv("RERANK_HOST_PORT"))
	}
	stack := shared.UpFiles(t, composeFiles...)

	ready := stack.WaitReadyWithProgressLog(t, 45*time.Minute)
	t.Logf("ready in %s", ready)

	logRerankModels(t, stack.BaseURL)

	model := envOr("MODEL_NAME", modelEnv["MODEL_NAME"])
	docs := []string{
		"hello world",
		"the giant panda is a bear species endemic to China",
		"quantum computing uses qubits",
	}
	query := "what is a panda?"
	t.Logf("probe rerank model=%q MODEL_ID=%q", model, envOr("MODEL_ID", modelEnv["MODEL_ID"]))
	latency, topScore := shared.ProbeRerank(t, stack.BaseURL, model, query, docs)
	t.Logf("rerank latency=%s top_score=%.4f", latency, topScore)

	// BGE cross-encoder emits raw logits (often negative); only check ordering.
	pandaScore := rerankOne(t, stack.BaseURL, model, query, "the giant panda is a bear species endemic to China")
	noiseScore := rerankOne(t, stack.BaseURL, model, query, "quantum computing uses qubits")
	t.Logf("ordering smoke: panda=%.4f quantum=%.4f", pandaScore, noiseScore)
	if pandaScore <= noiseScore {
		t.Errorf("expected panda doc score > quantum doc score; got %.4f vs %.4f", pandaScore, noiseScore)
	}

	assertRerankModeGuard(t, stack.BaseURL)

	shared.RecordBaseline(t, "rerank-"+envOr("ACCELERATOR", "default"), ready, latency)
}

func rerankOne(t *testing.T, baseURL, model, query, doc string) float64 {
	t.Helper()
	_, score := shared.ProbeRerank(t, baseURL, model, query, []string{doc})
	return score
}

func assertRerankModeGuard(t *testing.T, baseURL string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		baseURL+"/v1/chat/completions",
		strings.NewReader(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("build chat request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("chat request: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("rerank mode should refuse chat with 404, got %d body=%s", resp.StatusCode, raw)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("chat 404 body not JSON: %s", raw)
	}
	errObj, _ := body["error"].(map[string]any)
	if errObj == nil || errObj["code"] != "endpoint_not_served" {
		t.Fatalf("chat 404 envelope = %v", body)
	}
}

func logRerankModels(t *testing.T, baseURL string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/models", nil)
	if err != nil {
		t.Logf("GET /v1/models: build request: %v", err)
		return
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Logf("GET /v1/models: %v", err)
		return
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var pretty json.RawMessage
	if err := json.Unmarshal(raw, &pretty); err != nil {
		t.Logf("GET /v1/models status=%d body=%s", resp.StatusCode, raw)
		return
	}
	formatted, _ := json.MarshalIndent(pretty, "", "  ")
	t.Logf("GET /v1/models status=%d:\n%s", resp.StatusCode, formatted)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
