package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// TestRun_StartsControlPlane brings up the full process, hits each control
// plane endpoint, and asserts every route is wired correctly. Shutdown
// semantics (SIGINT -> exit 0) live in TestRun_GracefulShutdown_Linux so
// this test passes cleanly on every platform without conditional skips.
//
// Uses a unique TCP port so it can run in parallel with other test binaries.
func TestRun_StartsControlPlane(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}

	// MODEL_DIR / RUN_DIR must be writable by the test process; using
	// the production "/models" / "/run/llm-init" defaults makes ensure()
	// fail with EACCES on Linux CI runners (non-root cannot mkdir /models)
	// fast enough that PhaseDegraded shows up in the SSE stream BEFORE
	// the subscriber connects, racing the phase=init assertion below.
	// Per-test temp dirs are the contract-correct fix.
	dir := t.TempDir()
	env := mapEnv(map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
		"RUN_DIR":      dir + "/run-state",
		"PORT":         port,
	})

	stdout, _ := os.Create(dir + "/out")
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(dir + "/err")
	t.Cleanup(func() { stderr.Close() })

	// run() blocks until SIGINT/SIGTERM. We start it in a goroutine and
	// leave it running for the rest of the test process — this is a
	// bounded, intentional leak: `go test` exits quickly after the
	// suite completes, so the goroutine costs ~one server per test
	// invocation. The shutdown contract is exercised separately in
	// TestRun_GracefulShutdown_Linux.
	go func() { _ = run(nil, stdout, stderr, env) }()

	// Wait until the server starts accepting connections.
	if err := waitFor("http://127.0.0.1:"+port+"/livez", 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("server never came up: %v", err)
	}

	checks := []struct {
		path string
		want int
	}{
		{"/livez", 200},
		{"/readyz", 503},
		{"/healthz", 200},
		{"/api/progress", 200},
		{"/api/config", 200},
		// /api/model-spec must answer 200 before the lifecycle is
		// ready so Router can stage the upsert during boot.
		{"/api/model-spec", 200},
		{"/metrics", 200},
		{"/", 200},
	}
	for _, c := range checks {
		resp, err := http.Get("http://127.0.0.1:" + port + c.path)
		if err != nil {
			t.Errorf("GET %s: %v", c.path, err)
			continue
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != c.want {
			t.Errorf("%s status = %d, want %d (body=%s)", c.path, resp.StatusCode, c.want, body)
		}
	}

	// /api/progress sanity: with the SSE stream gone in v1.1.0 the
	// dashboard polls this every 1 s instead. The phase set is closed
	// (internal/progress/progress.go); an unrecognised value is a
	// regression worth catching.
	resp, err := http.Get("http://127.0.0.1:" + port + "/api/progress")
	if err != nil {
		t.Errorf("GET /api/progress: %v", err)
		return
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	validPhases := []string{
		`"phase":"init"`,
		`"phase":"download"`,
		`"phase":"loading"`,
		`"phase":"ready"`,
		`"phase":"degraded"`,
		`"phase":"failed"`,
	}
	ok := false
	for _, p := range validPhases {
		if strings.Contains(string(body), p) {
			ok = true
			break
		}
	}
	if !ok {
		t.Errorf("/api/progress missing a known phase value: %q", body)
	}
}

func freePort() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer l.Close()
	_, p, err := net.SplitHostPort(l.Addr().String())
	return p, err
}

// waitFor polls url with exponential backoff (50 ms → 1 s ceiling) until
// it gets a 2xx/3xx/4xx (anything < 500), the budget runs out, or the
// HTTP layer fails. Returns the last error on timeout. Total budget is
// callable; smoke tests use 30 s to ride out CI cold-start jitter.
func waitFor(url string, total time.Duration) error {
	deadline := time.Now().Add(total)
	var last error
	delay := 50 * time.Millisecond
	const maxDelay = 1 * time.Second
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode < 500 {
				return nil
			}
			last = errors.New(resp.Status)
		} else {
			last = err
		}
		time.Sleep(delay)
		if delay < maxDelay {
			delay *= 2
			if delay > maxDelay {
				delay = maxDelay
			}
		}
	}
	if last == nil {
		last = errors.New("waitFor: deadline exceeded without any probe")
	}
	return last
}

// dumpServerLogs reads the server's stderr file and emits it to t.Log.
// Best-effort; a missing or unreadable file is silently ignored. This is
// only a diagnostic aid for the rare CI case where the smoke server fails
// to bind — the operator gets the actual server logs instead of a generic
// "never came up" message.
func dumpServerLogs(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	if len(data) == 0 {
		t.Logf("server stderr empty (%s)", path)
		return
	}
	t.Logf("--- server stderr (%s) ---\n%s\n--- end ---", path, data)
}

// TestRun_M4ReachesReadyAndDataPlane proves the M4 wiring works end to
// end for an ollama-name source: a fake Ollama daemon answers /api/tags
// + /api/pull, and the embedded lifecycle drives the manager to
// PhaseReady, after which /v1/chat/completions is proxied to the same
// fake daemon's /api/chat (translated to SSE).
func TestRun_M4ReachesReadyAndDataPlane(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}

	// Fake Ollama daemon: tags lists the model, pull emits a single
	// success frame, chat returns a NDJSON frame with content.
	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen-smoke"}]}`))
		case "/api/ps":
			_, _ = w.Write([]byte(`{"models":[]}`))
		case "/api/pull":
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "success"})
		case "/api/chat":
			w.Header().Set("Content-Type", "application/x-ndjson")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"message": map[string]any{"role": "assistant", "content": "smoke"},
				"done":    true,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer fake.Close()

	env := mapEnv(map[string]string{
		"ENGINE_KIND":              "ollama",
		"LLM_INIT_TEST_ENGINE_URL": fake.URL,
		"MODEL_NAME":               "qwen-smoke",
		"MODEL_MODE":               "chat",
		"MODEL_SOURCE":             "ollama://qwen-smoke",
		"PORT":                     port,
		"RUN_DIR":                  t.TempDir(),
	})

	dir := t.TempDir()
	stdout, _ := os.Create(dir + "/out")
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(dir + "/err")
	t.Cleanup(func() { stderr.Close() })

	// Same intentional-leak pattern as TestRun_StartsControlPlane: run()
	// is left running until process exit so the shutdown contract can be
	// exercised in isolation in TestRun_GracefulShutdown_Linux. This
	// keeps the metric-contract assertions below platform-independent.
	go func() { _ = run(nil, stdout, stderr, env) }()

	// Wait for control plane and then for /readyz to flip to 200.
	if err := waitFor("http://127.0.0.1:"+port+"/livez", 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("server never came up: %v", err)
	}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get("http://127.0.0.1:" + port + "/readyz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	resp, err := http.Get("http://127.0.0.1:" + port + "/readyz")
	if err != nil {
		t.Fatalf("readyz: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("readyz status=%d body=%s (lifecycle did not reach Ready)", resp.StatusCode, body)
	}
	resp.Body.Close()

	// /v1/chat/completions should now reach the adapter and produce SSE.
	resp, err = http.Post("http://127.0.0.1:"+port+"/v1/chat/completions",
		"application/json",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"stream":true}`))
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat status=%d body=%s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "smoke") || !strings.Contains(string(body), "[DONE]") {
		t.Errorf("SSE body lacks expected content: %q", body)
	}

	// v1.0.3 contract: the four metric families that pre-train were
	// stuck at zero must now be non-trivially populated. Pre-fix all
	// three of these grep targets came back empty (or with `0` values
	// that don't match the substring); locking them in here is the
	// integration-level observable behind the obs subscriber and the
	// dataplane / healthloop wiring.
	mResp, err := http.Get("http://127.0.0.1:" + port + "/metrics")
	if err != nil {
		t.Fatalf("/metrics: %v", err)
	}
	mBody, _ := io.ReadAll(mResp.Body)
	mResp.Body.Close()
	mText := string(mBody)
	expectations := []string{
		`llm_init_phase{phase="ready"} 1`,
		`llm_init_engine_ready{engine_kind="ollama"} 1`,
		`llm_init_dataplane_requests_total{route="chat_completions",status="2xx"}`,
	}
	for _, want := range expectations {
		if !strings.Contains(mText, want) {
			t.Errorf("/metrics missing %q\n--- /metrics ---\n%s\n--- end ---",
				want, mText)
		}
	}
	t.Logf("v1.0.3 metric contract OK: phase=ready, engine_ready=1, dataplane requests counted")

	// Diag surface — happy path. /api/diag/gpu must respond 200 with a
	// populated GPUReport. The fake Ollama daemon above wires /api/ps to
	// "models:[]" (model not resident), so the residency numbers are
	// absent by design; we only assert the envelope fields, not a
	// particular residency outcome.
	gpuResp, err := http.Get("http://127.0.0.1:" + port + "/api/diag/gpu")
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
			Source string `json:"source"`
		} `json:"gpu"`
	}
	if err := json.Unmarshal(gpuBody, &gpuReport); err != nil {
		t.Fatalf("/api/diag/gpu decode: %v body=%s", err, gpuBody)
	}
	if gpuReport.SchemaVersion != 1 {
		t.Errorf("/api/diag/gpu schema_version=%d, want 1", gpuReport.SchemaVersion)
	}
	if gpuReport.EngineKind != "ollama" {
		t.Errorf("/api/diag/gpu engine_kind=%q, want ollama", gpuReport.EngineKind)
	}
	if gpuReport.GPU.Source == "" {
		t.Errorf("/api/diag/gpu source empty (body=%s)", gpuBody)
	}

	// The perf endpoints were removed in v1.3.0; the diag surface is
	// GPU-only and the old paths must not be served by anything else.
	perfResp, err := http.Post("http://127.0.0.1:"+port+"/api/diag/perf",
		"application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatalf("POST /api/diag/perf: %v", err)
	}
	perfBody, _ := io.ReadAll(perfResp.Body)
	perfResp.Body.Close()
	if perfResp.StatusCode != http.StatusNotFound {
		t.Errorf("/api/diag/perf status=%d, want 404 body=%s", perfResp.StatusCode, perfBody)
	}

	lastResp, err := http.Get("http://127.0.0.1:" + port + "/api/diag/perf/last")
	if err != nil {
		t.Fatalf("GET /api/diag/perf/last: %v", err)
	}
	lastBody, _ := io.ReadAll(lastResp.Body)
	lastResp.Body.Close()
	if lastResp.StatusCode != http.StatusNotFound {
		t.Errorf("/api/diag/perf/last status=%d, want 404 body=%s", lastResp.StatusCode, lastBody)
	}

	// /api/diag/gpu metrics counter must have ticked.
	mResp2, err := http.Get("http://127.0.0.1:" + port + "/metrics")
	if err != nil {
		t.Fatalf("/metrics #2: %v", err)
	}
	mBody2, _ := io.ReadAll(mResp2.Body)
	mResp2.Body.Close()
	if !strings.Contains(string(mBody2), `llm_init_diag_gpu_calls_total{result="ok"}`) {
		t.Errorf("/metrics missing diag_gpu_calls_total{result=ok} after diag call\n%s",
			string(mBody2))
	}
}

// TestRun_Embed_ReachesReadyAndEmbeddings drives the proxy path for
// ENGINE_KIND=embed: a fake upstream serves /v1/models + /v1/embeddings
// and doubles as the https?:// MODEL_SOURCE download target so ensure()
// completes without hitting the public internet. MODEL_NAME must match the
// fake /v1/models id (IREmbeddingServer ModelSpec.model_name semantics).
//
// Process-level ready+dataplane smoke exists today only for ollama
// (TestRun_M4ReachesReadyAndDataPlane) and Embed (this test).
// vllm/llamacpp/sglang rely on unit/integration tests under
// internal/adapter/proxy and tests/integration/.
func TestRun_Embed_ReachesReadyAndEmbeddings(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}

	const modelName = "embeddinggemma-300m"
	modelBytes := []byte("embed-smoke-model-bytes")
	shaHex := hex.EncodeToString(sha256Sum(modelBytes))

	runDir := t.TempDir()
	modelPath := filepath.Join(runDir, "model.bin")

	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/model.bin" && r.Method == http.MethodHead:
			w.Header().Set("Content-Length", strconv.Itoa(len(modelBytes)))
			w.WriteHeader(http.StatusOK)
		case r.URL.Path == "/model.bin":
			_, _ = w.Write(modelBytes)
		case r.URL.Path == "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data":   []map[string]string{{"id": modelName, "object": "model"}},
			})
		case r.URL.Path == "/v1/embeddings":
			var req struct {
				Model string `json:"model"`
				Input string `json:"input"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model != "" && req.Model != modelName {
				http.Error(w, "model mismatch", http.StatusNotFound)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"model":  modelName,
				"data": []map[string]any{{
					"object":    "embedding",
					"index":     0,
					"embedding": []float64{0.1, 0.2},
				}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer fake.Close()

	env := mapEnv(map[string]string{
		"ENGINE_KIND":              "embed",
		"LLM_INIT_TEST_ENGINE_URL": fake.URL,
		"MODEL_NAME":               modelName,
		"MODEL_MODE":               "embedding",
		"MODEL_SOURCE":             fake.URL + "/model.bin#sha256=" + shaHex,
		"MODEL_SOURCE_LOCAL":       modelPath,
		"PORT":                     port,
		"RUN_DIR":                  runDir,
	})

	dir := t.TempDir()
	stdout, _ := os.Create(dir + "/out")
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(dir + "/err")
	t.Cleanup(func() { stderr.Close() })

	go func() { _ = run(nil, stdout, stderr, env) }()

	if err := waitFor("http://127.0.0.1:"+port+"/livez", 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("server never came up: %v", err)
	}
	if err := waitForReady(port, 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("readyz never flipped: %v", err)
	}

	resp, err := http.Post("http://127.0.0.1:"+port+"/v1/embeddings",
		"application/json",
		strings.NewReader(`{"input":"hello"}`))
	if err != nil {
		t.Fatalf("embeddings: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("embeddings status=%d body=%s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "0.1") {
		t.Errorf("embeddings body lacks vector: %q", body)
	}

	mResp, err := http.Get("http://127.0.0.1:" + port + "/metrics")
	if err != nil {
		t.Fatalf("/metrics: %v", err)
	}
	mBody, _ := io.ReadAll(mResp.Body)
	mResp.Body.Close()
	mText := string(mBody)
	for _, want := range []string{
		`llm_init_phase{phase="ready"} 1`,
		`llm_init_engine_ready{engine_kind="embed"} 1`,
		`llm_init_dataplane_requests_total{route="embeddings",status="2xx"}`,
	} {
		if !strings.Contains(mText, want) {
			t.Errorf("/metrics missing %q\n--- /metrics ---\n%s\n--- end ---", want, mText)
		}
	}
}

// TestRun_Rerank_ReachesReadyAndRerank drives the proxy path for
// ENGINE_KIND=rerank: a fake upstream serves /v1/models + /v1/rerank
// and doubles as the MODEL_SOURCE download target.
func TestRun_Rerank_ReachesReadyAndRerank(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}

	const modelName = "bge-reranker-v2-m3"
	modelBytes := []byte("rerank-smoke-model-bytes")
	shaHex := hex.EncodeToString(sha256Sum(modelBytes))

	runDir := t.TempDir()
	modelPath := filepath.Join(runDir, "model.bin")

	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/model.bin" && r.Method == http.MethodHead:
			w.Header().Set("Content-Length", strconv.Itoa(len(modelBytes)))
			w.WriteHeader(http.StatusOK)
		case r.URL.Path == "/model.bin":
			_, _ = w.Write(modelBytes)
		case r.URL.Path == "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data":   []map[string]string{{"id": modelName, "object": "model"}},
			})
		case r.URL.Path == "/v1/rerank":
			var req struct {
				Model string `json:"model"`
				Query string `json:"query"`
			}
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req.Model != "" && req.Model != modelName {
				http.Error(w, "model mismatch", http.StatusNotFound)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":    "rerank-smoke",
				"model": modelName,
				"usage": map[string]any{"total_tokens": 12},
				"results": []map[string]any{{
					"index":           0,
					"relevance_score": 0.9,
					"document":        map[string]string{"text": "panda is a mammal"},
				}},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer fake.Close()

	env := mapEnv(map[string]string{
		"ENGINE_KIND":              "rerank",
		"LLM_INIT_TEST_ENGINE_URL": fake.URL,
		"MODEL_NAME":               modelName,
		"MODEL_MODE":               "rerank",
		"MODEL_SOURCE":             fake.URL + "/model.bin#sha256=" + shaHex,
		"MODEL_SOURCE_LOCAL":       modelPath,
		"PORT":                     port,
		"RUN_DIR":                  runDir,
	})

	dir := t.TempDir()
	stdout, _ := os.Create(dir + "/out")
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(dir + "/err")
	t.Cleanup(func() { stderr.Close() })

	go func() { _ = run(nil, stdout, stderr, env) }()

	if err := waitFor("http://127.0.0.1:"+port+"/livez", 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("server never came up: %v", err)
	}
	if err := waitForReady(port, 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("readyz never flipped: %v", err)
	}

	resp, err := http.Post("http://127.0.0.1:"+port+"/v1/rerank",
		"application/json",
		strings.NewReader(`{"query":"what is panda?","documents":["hi","panda is a mammal"]}`))
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("rerank status=%d body=%s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "relevance_score") {
		t.Errorf("rerank body lacks score: %q", body)
	}

	mResp, err := http.Get("http://127.0.0.1:" + port + "/metrics")
	if err != nil {
		t.Fatalf("/metrics: %v", err)
	}
	mBody, _ := io.ReadAll(mResp.Body)
	mResp.Body.Close()
	mText := string(mBody)
	for _, want := range []string{
		`llm_init_phase{phase="ready"} 1`,
		`llm_init_engine_ready{engine_kind="rerank"} 1`,
		`llm_init_dataplane_requests_total{route="rerank",status="2xx"}`,
	} {
		if !strings.Contains(mText, want) {
			t.Errorf("/metrics missing %q\n--- /metrics ---\n%s\n--- end ---", want, mText)
		}
	}
}

// TestRun_Audio_RelaysTheV2SpecAndTheAsyncChain covers ENGINE_KIND=audio,
// the one proxied mode whose data plane is not a single request/response.
// An audio job is submitted, polled and collected over three round trips,
// and everything llm-init contributes to that is pass-through: the 202 and
// its task id, the status document the poll returns, the result body, and
// the engine's own operation list on /api/endpoints, which is what tells
// Router the operation exists at all.
//
// Pass-through is exactly what tends to break silently. A relay that
// buffers the 202 into a 200, or drops input_duration_seconds out of the
// status document, leaves a caller polling an id that no longer means
// anything, or a job nobody can bill. The fake engine below answers the
// audio-engines contract: schema v2 on /api/engine-spec, 202 + task doc on
// submit, /v1/audio/tasks/{id} and .../result for the rest.
func TestRun_Audio_RelaysTheV2SpecAndTheAsyncChain(t *testing.T) {
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}

	const modelName = "Qwen/Qwen3-ASR-1.7B"
	const taskID = "task_smoke_1"
	modelBytes := []byte("audio-smoke-model-bytes")
	shaHex := hex.EncodeToString(sha256Sum(modelBytes))

	runDir := t.TempDir()
	modelPath := filepath.Join(runDir, "model.bin")

	// Two polls: the first says the job is still running, the second that
	// it finished. A relay that answers the second poll from something it
	// cached during the first would pass a single-poll test.
	var polls atomic.Int32

	fake := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/model.bin" && r.Method == http.MethodHead:
			w.Header().Set("Content-Length", strconv.Itoa(len(modelBytes)))
			w.WriteHeader(http.StatusOK)
		case r.URL.Path == "/model.bin":
			_, _ = w.Write(modelBytes)
		case r.URL.Path == "/v1/models":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{{
					"id": modelName, "object": "model",
					"mode": "audio", "supports": []string{"stt"},
				}},
			})
		case r.URL.Path == "/api/engine-spec":
			_ = json.NewEncoder(w).Encode(audioEngineSpec(modelName))
		case r.URL.Path == "/v1/audio/transcriptions":
			// audio-engines answers async submissions with 202 and the
			// task document; the id in it is the client's only handle.
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"task": audioTaskDoc(taskID, modelName, "queued"),
			})
		case r.URL.Path == "/v1/audio/tasks/"+taskID:
			status := "running"
			if polls.Add(1) > 1 {
				status = "succeeded"
			}
			_ = json.NewEncoder(w).Encode(audioTaskDoc(taskID, modelName, status))
		case r.URL.Path == "/v1/audio/tasks/"+taskID+"/result":
			_ = json.NewEncoder(w).Encode(map[string]any{"text": "a smoke transcript"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer fake.Close()

	env := mapEnv(map[string]string{
		"ENGINE_KIND":              "audio",
		"LLM_INIT_TEST_ENGINE_URL": fake.URL,
		"MODEL_NAME":               modelName,
		"MODEL_MODE":               "audio",
		"MODEL_SUPPORTS":           "supports_stt",
		"MODEL_SOURCE":             fake.URL + "/model.bin#sha256=" + shaHex,
		"MODEL_SOURCE_LOCAL":       modelPath,
		"PORT":                     port,
		"RUN_DIR":                  runDir,
	})

	dir := t.TempDir()
	stdout, _ := os.Create(dir + "/out")
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(dir + "/err")
	t.Cleanup(func() { stderr.Close() })

	go func() { _ = run(nil, stdout, stderr, env) }()

	if err := waitFor("http://127.0.0.1:"+port+"/livez", 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("server never came up: %v", err)
	}
	if err := waitForReady(port, 30*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("readyz never flipped: %v", err)
	}

	base := "http://127.0.0.1:" + port

	// Submit. The status is the contract: 202 means "come back for it",
	// and a 200 here would tell the caller the transcript is in the body.
	submit, err := http.Post(base+"/v1/audio/transcriptions?async=1",
		"application/json", strings.NewReader(`{"model":"`+modelName+`"}`))
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	submitBody, _ := io.ReadAll(submit.Body)
	submit.Body.Close()
	if submit.StatusCode != http.StatusAccepted {
		t.Fatalf("submit status=%d, want 202 body=%s", submit.StatusCode, submitBody)
	}
	var accepted struct {
		Task struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		} `json:"task"`
	}
	if err := json.Unmarshal(submitBody, &accepted); err != nil {
		t.Fatalf("submit body not a task document: %v body=%s", err, submitBody)
	}
	if accepted.Task.ID != taskID {
		t.Fatalf("submit returned task id %q, want %q", accepted.Task.ID, taskID)
	}

	// Poll until it finishes, then read what the last poll actually said.
	var status struct {
		ID       string   `json:"id"`
		Status   string   `json:"status"`
		Seconds  *float64 `json:"input_duration_seconds"`
		Progress any      `json:"progress"`
	}
	deadline := time.Now().Add(10 * time.Second)
	for {
		poll, err := http.Get(base + "/v1/audio/tasks/" + accepted.Task.ID)
		if err != nil {
			t.Fatalf("poll: %v", err)
		}
		pollBody, _ := io.ReadAll(poll.Body)
		poll.Body.Close()
		if poll.StatusCode != http.StatusOK {
			t.Fatalf("poll status=%d body=%s", poll.StatusCode, pollBody)
		}
		if err := json.Unmarshal(pollBody, &status); err != nil {
			t.Fatalf("poll body not a task document: %v body=%s", err, pollBody)
		}
		if status.Status == "succeeded" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("task never left %q: %s", status.Status, pollBody)
		}
		time.Sleep(20 * time.Millisecond)
	}
	// Duration is what a caller downstream bills on, and it exists only
	// on the finished document. Losing it in the relay is invisible until
	// someone reconciles usage against an invoice.
	if status.Seconds == nil || *status.Seconds != 12.5 {
		t.Errorf("input_duration_seconds did not survive the relay: %v", status.Seconds)
	}

	result, err := http.Get(base + "/v1/audio/tasks/" + accepted.Task.ID + "/result")
	if err != nil {
		t.Fatalf("result: %v", err)
	}
	resultBody, _ := io.ReadAll(result.Body)
	result.Body.Close()
	if result.StatusCode != http.StatusOK {
		t.Fatalf("result status=%d body=%s", result.StatusCode, resultBody)
	}
	if !strings.Contains(string(resultBody), "a smoke transcript") {
		t.Errorf("result body lacks the transcript: %s", resultBody)
	}

	// The v2 relay: /api/endpoints is where Router learns the operation
	// exists, and the fields it keys on are operation_id, transport and
	// the two support flags. A row that arrives without them is dropped
	// downstream and the operation is refused as unsupported.
	endpointsResp, err := http.Get(base + "/api/endpoints")
	if err != nil {
		t.Fatalf("GET /api/endpoints: %v", err)
	}
	endpointsBody, _ := io.ReadAll(endpointsResp.Body)
	endpointsResp.Body.Close()
	var endpoints struct {
		Endpoints []struct {
			Method         string `json:"method"`
			Path           string `json:"path"`
			Available      bool   `json:"available"`
			OperationID    string `json:"operation_id"`
			Transport      string `json:"transport"`
			Protocol       string `json:"protocol"`
			SyncSupported  *bool  `json:"sync_supported"`
			AsyncSupported *bool  `json:"async_supported"`
		} `json:"endpoints"`
	}
	if err := json.Unmarshal(endpointsBody, &endpoints); err != nil {
		t.Fatalf("/api/endpoints decode: %v body=%s", err, endpointsBody)
	}
	wantOperations := map[string]string{
		"POST /v1/audio/transcriptions": "audio.transcribe",
		"GET /v1/audio/tasks/{id}":      "task.read",
	}
	for _, e := range endpoints.Endpoints {
		key := e.Method + " " + e.Path
		want, ours := wantOperations[key]
		if !ours {
			continue
		}
		delete(wantOperations, key)
		if e.OperationID != want {
			t.Errorf("%s operation_id=%q, want %q", key, e.OperationID, want)
		}
		if e.Transport != "http" || e.Protocol == "" || e.SyncSupported == nil {
			t.Errorf("%s lost its v2 fields: transport=%q protocol=%q sync_supported=%v",
				key, e.Transport, e.Protocol, e.SyncSupported)
		}
		if !e.Available {
			t.Errorf("%s came back unavailable", key)
		}
	}
	if len(wantOperations) > 0 {
		t.Errorf("engine-declared operations missing from /api/endpoints: %v\nbody=%s",
			wantOperations, endpointsBody)
	}

	// Chat and embeddings are not this model's, and an audio instance
	// forwards every /v1 path it is given, so the catalog is the only
	// thing that says so.
	for _, e := range endpoints.Endpoints {
		if (e.Path == "/v1/chat/completions" || e.Path == "/v1/embeddings") && e.Available {
			t.Errorf("%s advertised as available on an audio model", e.Path)
		}
	}

	mResp, err := http.Get(base + "/metrics")
	if err != nil {
		t.Fatalf("/metrics: %v", err)
	}
	mBody, _ := io.ReadAll(mResp.Body)
	mResp.Body.Close()
	mText := string(mBody)
	for _, want := range []string{
		`llm_init_phase{phase="ready"} 1`,
		`llm_init_engine_ready{engine_kind="audio"} 1`,
		// Per capability, not one bucket for the whole audio surface:
		// submit and poll have to be separable to mean anything.
		`llm_init_dataplane_requests_total{route="audio_transcriptions",status="2xx"}`,
		`llm_init_dataplane_requests_total{route="tasks",status="2xx"}`,
	} {
		if !strings.Contains(mText, want) {
			t.Errorf("/metrics missing %q\n--- /metrics ---\n%s\n--- end ---", want, mText)
		}
	}
}

// audioEngineSpec is the subset of an audio-engines schema v2 report this
// test depends on: the envelope fields llm-init validates, one capability
// operation and the task API that operation's 202 sends the caller to.
func audioEngineSpec(modelName string) map[string]any {
	row := func(method, path, operationID, protocol string, async bool) map[string]any {
		return map[string]any{
			"method": method, "path": path, "available": true,
			"operation_id": operationID, "protocol": protocol, "transport": "http",
			"sync_supported": true, "async_supported": async,
		}
	}
	return map[string]any{
		"schema_version": 2,
		"base":           "qwen",
		"model":          modelName,
		"implements":     []string{"stt"},
		"declares":       []string{"stt"},
		"serves":         []string{"stt"},
		"endpoints": []map[string]any{
			row("POST", "/v1/audio/transcriptions", "audio.transcribe", "openai.audio.v1", true),
			row("GET", "/v1/audio/tasks/{id}", "task.read", "olares.audio.v1", false),
			row("GET", "/v1/audio/tasks/{id}/result", "task.result", "olares.audio.v1", false),
			row("DELETE", "/v1/audio/tasks/{id}", "task.cancel", "olares.audio.v1", false),
		},
	}
}

// audioTaskDoc is the task document shape audio-engines answers with, cut
// down to the fields that have to survive a relay.
func audioTaskDoc(id, modelName, status string) map[string]any {
	doc := map[string]any{
		"object": "task", "id": id, "kind": "audio",
		"cap": "stt", "model": modelName, "status": status,
		"result_kind": "json",
	}
	if status == "succeeded" {
		doc["input_duration_seconds"] = 12.5
	}
	return doc
}

func sha256Sum(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}

func waitForReady(port string, budget time.Duration) error {
	deadline := time.Now().Add(budget)
	for time.Now().Before(deadline) {
		resp, err := http.Get("http://127.0.0.1:" + port + "/readyz")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	return errors.New("readyz did not return 200 before deadline")
}
