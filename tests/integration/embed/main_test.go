//go:build integration

// Package embed_integration exercises deploy/compose/embed.yml against
// a real llm-init + IREmbeddingServer pair. Env-gated: skip unless
// LLM_INIT_INTEGRATION_EMBED=1 (needs IREmbeddingServer image + model).
//
// Local invocation:
//
//	LLM_INIT_INTEGRATION_EMBED=1 \
//	    EMBED_IMAGE=embed-server:ov-intel-amd64 \
//	    MODEL_ID=embeddinggemma-300m-ov \
//	    MODEL_NAME=embeddinggemma-300m \
//	    MODEL_SOURCE="hf://wangzhong/embeddinggemma-300m-ov,hf://wangzhong/embeddinggemma-300m-onnx-export" \
//	    go test -tags integration -timeout 20m ./tests/integration/embed/...
//
// Metrics / GPU residency (optional; enabled by integration_test_pureembed_unified.sh):
//
//	LLM_INIT_INTEGRATION_EMBED_METRICS=1  — assert /api/diag/gpu + embed GET /metrics
//	EMBED_COMPOSE_GPU=1                  — merge deploy/compose/embed.gpu.yml (/dev/dri + NVIDIA)
package embed_integration

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

// modelEnv defaults for a local OpenVINO IR stack. Override via env when
// running ONNX/CUDA images (swap MODEL_ID and EMBED_IMAGE). MODEL_NAME
// must equal ModelSpec.model_name for MODEL_ID (see IREmbedding registry).
var modelEnv = map[string]string{
	"MODEL_NAME":    "embeddinggemma-300m",
	"MODEL_ID":      "embeddinggemma-300m-ov",
	"MODEL_MODE":    "embedding",
	"MODEL_SOURCE":  "hf://google/embeddinggemma-300m",
	"LLM_INIT_PORT": "8080",
	"EMBED_PORT":    "8080",
	"EMBED_DEVICE":  "auto",
	"LOG_LEVEL":     "info",
	"LOG_FORMAT":    "json",
}

// TestEmbed_FullStack is the end-to-end contract test for ENGINE_KIND=embed.
// It mirrors tests/integration/{ollama,vllm,llamacpp}/main_test.go but probes
// POST /v1/embeddings instead of chat completions.
//
// Flow:
//  1. Gate on LLM_INIT_INTEGRATION_EMBED (skipped in default go test ./...).
//  2. Seed compose env (MODEL_ID on embed, MODEL_NAME on llm-init, etc.).
//  3. docker compose up deploy/compose/embed.yml (+ optional embed.gpu.yml).
//  4. Poll GET /readyz until 200, logging GET /api/progress when phase or
//     byte counters change (use go test -v to see download/loading lines).
//  5. POST /v1/embeddings through llm-init; assert non-empty vector.
//  6. Record ready + embedding latency into tests/integration/baseline.json.
//  7. Embed two similar sentences and assert cosine similarity is high enough.
//  8. When LLM_INIT_INTEGRATION_EMBED_METRICS=1: assert embed /metrics gpu_*
//     and llm-init /api/diag/gpu (source=engine_metrics, mode≠unknown).
func TestEmbed_FullStack(t *testing.T) {
	// Opt-in only: needs IREmbeddingServer image, model bytes, and often GPU/NPU.
	if os.Getenv("LLM_INIT_INTEGRATION_EMBED") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_EMBED=1 to enable (needs IREmbeddingServer image + model)")
	}
	if os.Getenv("LLM_INIT_INTEGRATION_DISABLED") != "" {
		t.Skip("LLM_INIT_INTEGRATION_DISABLED set")
	}

	// Export defaults into the process env so docker compose inherits them.
	for k, v := range modelEnv {
		if os.Getenv(k) == "" {
			t.Setenv(k, v)
		}
	}
	port := envOr("LLM_INIT_PORT", modelEnv["LLM_INIT_PORT"])
	shared.AssertPortFree(t, port)

	composeFiles := []string{shared.RepoPath(t, "deploy/compose/embed.yml")}
	if os.Getenv("EMBED_COMPOSE_GPU") != "" {
		composeFiles = append(composeFiles, shared.RepoPath(t, "deploy/compose/embed.gpu.yml"))
		if os.Getenv("EMBED_HOST_PORT") == "" {
			t.Setenv("EMBED_HOST_PORT", shared.FreePort(t))
		}
		t.Logf("merging GPU overlay embed.gpu.yml (EMBED_GPU_COUNT=%s EMBED_HOST_PORT=%s)",
			envOr("EMBED_GPU_COUNT", "all"), os.Getenv("EMBED_HOST_PORT"))
	}
	stack := shared.UpFiles(t, composeFiles...)

	// Wait for phase=ready; log /api/progress on each change (run go test -v).
	ready := stack.WaitReadyWithProgressLog(t, 15*time.Minute)
	t.Logf("ready in %s", ready)

	logEmbedModels(t, stack.BaseURL)

	// Data-plane smoke: llm-init proxy → embed POST /v1/embeddings.
	model := envOr("MODEL_NAME", modelEnv["MODEL_NAME"])
	t.Logf("probe embeddings with model=%q (MODEL_ID=%q)", model, envOr("MODEL_ID", modelEnv["MODEL_ID"]))
	latency, dim := shared.ProbeEmbeddings(t, stack.BaseURL, model, "hello")
	t.Logf("embeddings latency=%s dim=%d", latency, dim)

	shared.RecordBaseline(t, "embed", ready, latency)

	shared.AssertEmbeddingSimilarityCases(t, stack.BaseURL, model, 0.3,
		shared.EmbeddingSimilarityCases)

	if os.Getenv("LLM_INIT_INTEGRATION_EMBED_METRICS") != "" {
		present, used := assertEmbedMetrics(t, stack)
		assertDiagGPU(t, stack.BaseURL, present, used)
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// assertEmbedMetrics GETs embed /metrics (host-published port or container IP)
// and checks gpu_* samples match CapabilityReport mapping rules.
// Returns (gpu_present, gpu_mem_used_bytes) for cross-checks against /api/diag/gpu.
func assertEmbedMetrics(t *testing.T, stack *shared.Stack) (present, used float64) {
	t.Helper()
	body := fetchEmbedMetrics(t, stack)
	t.Logf("embed GET /metrics:\n%s", body)

	var okPresent, okUsed bool
	present, okPresent = parsePromGauge(body, "gpu_present")
	used, okUsed = parsePromGauge(body, "gpu_mem_used_bytes")
	if !okPresent || !okUsed {
		t.Fatalf("embed /metrics missing gpu_present or gpu_mem_used_bytes; body=%s", body)
	}

	// CUDA: present=1 and used>0 (NVML or 1-byte placeholder).
	// Intel iGPU/dGPU: present=1 and used=0 means VRAM unknown (by design).
	if present == 0 && used != 0 {
		t.Fatalf("gpu_present=0 but gpu_mem_used_bytes=%v (want 0)", used)
	}
	if present != 0 && present != 1 {
		t.Fatalf("gpu_present=%v, want 0 or 1", present)
	}
	t.Logf("embed /metrics gauges: gpu_present=%v gpu_mem_used_bytes=%v", present, used)
	return present, used
}

// fetchEmbedMetrics returns the Prometheus text from the embed service.
// Prefer EMBED_HOST_PORT (published by embed.gpu.yml); fall back to the
// container's Docker network IP (image has no curl).
func fetchEmbedMetrics(t *testing.T, stack *shared.Stack) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	var urls []string
	if p := strings.TrimSpace(os.Getenv("EMBED_HOST_PORT")); p != "" {
		urls = append(urls, "http://127.0.0.1:"+p+"/metrics")
	}
	if ip := embedContainerIP(t, stack); ip != "" {
		urls = append(urls, "http://"+ip+":8080/metrics")
	}
	if len(urls) == 0 {
		t.Fatal("cannot reach embed /metrics: set EMBED_COMPOSE_GPU=1 (publishes EMBED_HOST_PORT) or ensure compose network IP is inspectable")
	}

	var lastErr error
	for _, u := range urls {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		if err != nil {
			lastErr = err
			continue
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			lastErr = err
			t.Logf("GET %s: %v", u, err)
			continue
		}
		raw, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("status=%d body=%s", resp.StatusCode, raw)
			t.Logf("GET %s: %v", u, lastErr)
			continue
		}
		return string(raw)
	}
	t.Fatalf("embed /metrics unreachable via %v: %v", urls, lastErr)
	return ""
}

func embedContainerIP(t *testing.T, stack *shared.Stack) string {
	t.Helper()
	args := []string{"compose", "-f", stack.Compose}
	if os.Getenv("EMBED_COMPOSE_GPU") != "" {
		args = append(args, "-f", shared.RepoPath(t, "deploy/compose/embed.gpu.yml"))
	}
	args = append(args, "ps", "-q", "embed")
	out, err := exec.Command("docker", args...).Output()
	if err != nil {
		t.Logf("compose ps -q embed: %v", err)
		return ""
	}
	id := strings.TrimSpace(string(out))
	if id == "" {
		return ""
	}
	ipOut, err := exec.Command("docker", "inspect", "-f",
		"{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", id).Output()
	if err != nil {
		t.Logf("docker inspect embed IP: %v", err)
		return ""
	}
	return strings.TrimSpace(string(ipOut))
}

// assertDiagGPU checks llm-init GET /api/diag/gpu reflects embed /metrics.
//
// Note: since llm-init v1.1.0 the wire no longer carries gpu.mode; dashboards
// use source + vram_bytes. We assert source=engine_metrics and that
// vram_bytes matches the embed gauges (present=1 → vram_bytes>0).
func assertDiagGPU(t *testing.T, baseURL string, present, used float64) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/diag/gpu", nil)
	if err != nil {
		t.Fatalf("GET /api/diag/gpu: build request: %v", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET /api/diag/gpu: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/api/diag/gpu status=%d body=%s", resp.StatusCode, raw)
	}

	var report struct {
		EngineKind string `json:"engine_kind"`
		GPU        struct {
			Source    string `json:"source"`
			VRAMBytes *int64 `json:"vram_bytes"`
		} `json:"gpu"`
		Warnings []string `json:"warnings"`
	}
	if err := json.Unmarshal(raw, &report); err != nil {
		t.Fatalf("/api/diag/gpu decode: %v body=%s", err, raw)
	}
	vram := int64(0)
	if report.GPU.VRAMBytes != nil {
		vram = *report.GPU.VRAMBytes
	}
	t.Logf("/api/diag/gpu engine_kind=%s source=%s vram_bytes=%v warnings=%v body=%s",
		report.EngineKind, report.GPU.Source, report.GPU.VRAMBytes, report.Warnings, raw)

	if report.EngineKind != "embed" {
		t.Errorf("engine_kind=%q, want embed", report.EngineKind)
	}
	if report.GPU.Source != "engine_metrics" {
		t.Fatalf("/api/diag/gpu source=%q, want engine_metrics (embed /metrics gpu_*); warnings=%v body=%s",
			report.GPU.Source, report.Warnings, raw)
	}
	// CUDA (used>0): diag should surface vram_bytes. Intel iGPU (used=0):
	// VRAM unknown — vram_bytes may be absent; that is expected.
	if present == 1 && used > 0 {
		if report.GPU.VRAMBytes == nil || vram <= 0 {
			t.Fatalf("gpu_present=1 used>0 but diag vram_bytes=%v (want >0); body=%s",
				report.GPU.VRAMBytes, raw)
		}
	}
}

// parsePromGauge returns the first sample value for an unlabelled gauge name.
func parsePromGauge(body, name string) (float64, bool) {
	prefix := name + " "
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, prefix) {
			v, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(line, prefix)), 64)
			if err != nil {
				return 0, false
			}
			return v, true
		}
	}
	return 0, false
}

// logEmbedModels dumps GET /v1/models (llm-init passthrough to embed_server)
// so failures like model_not_found show what the upstream actually advertises.
func logEmbedModels(t *testing.T, baseURL string) {
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
		t.Logf("GET /v1/models: request failed: %v", err)
		return
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Logf("GET /v1/models status=%d: read body: %v", resp.StatusCode, err)
		return
	}

	var pretty json.RawMessage
	if err := json.Unmarshal(raw, &pretty); err != nil {
		t.Logf("GET /v1/models status=%d body=%s", resp.StatusCode, raw)
		return
	}
	formatted, err := json.MarshalIndent(pretty, "", "  ")
	if err != nil {
		t.Logf("GET /v1/models status=%d body=%s", resp.StatusCode, raw)
		return
	}
	t.Logf("GET /v1/models status=%d:\n%s", resp.StatusCode, formatted)

	modelName := envOr("MODEL_NAME", modelEnv["MODEL_NAME"])
	modelID := envOr("MODEL_ID", modelEnv["MODEL_ID"])
	req2, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("%s/v1/models/%s", baseURL, modelName), nil)
	if err == nil {
		if resp2, err := http.DefaultClient.Do(req2); err == nil {
			body2, _ := io.ReadAll(resp2.Body)
			resp2.Body.Close()
			t.Logf("GET /v1/models/%s status=%d body=%s", modelName, resp2.StatusCode, body2)
		}
	}
	req3, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("%s/v1/models/%s", baseURL, modelID), nil)
	if err == nil {
		if resp3, err := http.DefaultClient.Do(req3); err == nil {
			body3, _ := io.ReadAll(resp3.Body)
			resp3.Body.Close()
			t.Logf("GET /v1/models/%s status=%d body=%s", modelID, resp3.StatusCode, body3)
		}
	}
}
