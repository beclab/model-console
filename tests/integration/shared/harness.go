// Package shared holds the docker-compose-driven harness used by every
// integration test under tests/integration/<engine>/. The shape is the
// same across engines: bring the compose stack up, poll /readyz until
// it flips to 200, fire one streaming chat request, record wall-clock
// timings, tear down.
//
// Every export here is a free function rather than a struct method so
// that engine test files stay tiny (one main_test.go each) and read
// like a checklist.
package shared

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// composeSubcmd is the docker subcommand used for all stack operations.
const composeSubcmd = "compose"

// Stack is the running compose stack with a base URL pointing at
// llm-init's published port. Cleanup must be deferred or registered via
// t.Cleanup; calling it twice is safe.
type Stack struct {
	BaseURL string
	// Compose is the primary compose file (the first one passed to
	// UpFiles); kept for callers/logs that want a single identifier.
	Compose string
	// files holds every -f compose file in order so Down and log dumps
	// target the exact same stack UpFiles started.
	files []string
	once  sync.Once
}

// composeArgs expands the stack's compose files into the
// `-f a.yml -f b.yml` argument list shared by up / down / logs.
func (s *Stack) composeArgs() []string {
	args := make([]string, 0, len(s.files)*2)
	for _, f := range s.files {
		args = append(args, "-f", f)
	}
	return args
}

// Up starts the compose stack defined by composePath. The function
// blocks until `docker compose up -d` returns (i.e. the `create` and
// `start` phases complete; the API server is detached and may still
// be doing its own bootstrap). Readiness is then asserted by the
// caller via WaitReady against /readyz, which is the contract-correct
// signal — llm-init flips that to 200 only after sentinel write +
// adapter Ready, which is "the engine can serve traffic" not just
// "the container is running". Pass a path relative to the repo root.
//
// We deliberately do NOT pass `--wait` here. v1.0.5 used `--wait` and
// hit two failure modes against newer docker-compose:
//  1. The engine container inherits a /health HEALTHCHECK from the
//     upstream image (vllm, llama.cpp, sglang) that 503s for the
//     entire cold-deploy download window; --wait declares the
//     container "unhealthy" within ~90s and aborts before download
//     can complete (deploy/compose/llamacpp.yml documents the
//     mitigation: explicit `healthcheck: disable: true`).
//  2. After (1) is mitigated, `--wait` against a service with no
//     healthcheck exits 1 with "has no healthcheck configured" on
//     compose v2.36+. Removing --wait closes the second gap; the
//     readiness contract is unchanged because the test path already
//     gates on /readyz.
//
// Tests must register Down via t.Cleanup so containers are reaped even
// when the test panics. We do not call t.Helper() — the caller owns
// the meaning of "this step failed", so the failing line is the most
// useful frame.
func Up(t *testing.T, composePath string) *Stack {
	t.Helper()
	return UpFiles(t, composePath)
}

// UpFiles is Up for a multi-file stack: every path is passed as its own
// `-f` flag, so a production compose file can be overlaid with a test
// override (e.g. a GPU device reservation or an alternate engine image).
// Paths are resolved to absolute so docker compose's project directory is
// stable regardless of the test's working directory.
func UpFiles(t *testing.T, composePaths ...string) *Stack {
	t.Helper()
	if len(composePaths) == 0 {
		t.Fatalf("UpFiles: at least one compose file is required")
	}

	files := make([]string, 0, len(composePaths))
	for _, p := range composePaths {
		abs, err := filepath.Abs(p)
		if err != nil {
			t.Fatalf("compose path %s: %v", p, err)
		}
		if _, err := os.Stat(abs); err != nil {
			t.Fatalf("compose file %s missing: %v", abs, err)
		}
		files = append(files, abs)
	}

	// Resolve the host port llm-init publishes. An explicit LLM_INIT_PORT
	// (set by a caller or CI) wins for backward compatibility; otherwise
	// grab a free ephemeral port and export it so the compose subprocess
	// (which inherits os.Environ and reads ${LLM_INIT_PORT}) and BaseURL
	// agree. The dynamic default lets repeated / overlapping stacks run
	// without colliding on a hard-coded 8080.
	port := strings.TrimSpace(os.Getenv("LLM_INIT_PORT"))
	if port == "" {
		port = FreePort(t)
		t.Setenv("LLM_INIT_PORT", port)
	}

	stack := &Stack{
		BaseURL: "http://127.0.0.1:" + port,
		Compose: files[0],
		files:   files,
	}

	args := append(stack.composeArgs(), "up", "-d")
	cmd := exec.Command("docker", append([]string{composeSubcmd}, args...)...)
	cmd.Stdout = &prefixedWriter{prefix: "[compose-up] ", out: testWriter{t}}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Run(); err != nil {
		// Best-effort log dump so CI surfaces the cause.
		stack.dumpLogs(t)
		t.Fatalf("docker compose up: %v", err)
	}

	t.Cleanup(stack.Down)
	return stack
}

// Down tears the stack down with `docker compose down -v`. Idempotent.
func (s *Stack) Down() {
	s.once.Do(func() {
		args := append(s.composeArgs(), "down", "-v", "--remove-orphans")
		cmd := exec.Command("docker", append([]string{composeSubcmd}, args...)...)
		cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
		_ = cmd.Run()
	})
}

// WaitReady blocks until GET /readyz returns 200 or max elapses.
// Returns the wall-clock duration measured from the first poll. The
// first 503 with phase=init is normal; downstream baseline numbers
// should be interpreted relative to compose-start time, which the
// caller times outside this helper.
//
// On timeout it dumps the last ~200 lines of `docker compose logs`
// (both llm-init and engine containers) to t.Log BEFORE calling
// t.Fatalf, so CI surfaces the actual container state at expiry
// rather than just the wall-clock budget. Through v1.0.5/v1.0.6/v1.0.7
// every tag-CI Integration job timed out at this exact line with no
// container visibility — operators got "readiness timed out after 8m0s"
// and nothing else, blocking every release's GitHub Release page on a
// black-box failure. The dump turns the next failure into an
// actionable signal (HF download stalled mid-stream? lifecycle stuck
// pre-sentinel? llama-server crashlooping? volume EACCES post Track
// P.7?). Track P.9.A.
func (s *Stack) WaitReady(t *testing.T, max time.Duration) time.Duration {
	t.Helper()
	start := time.Now()
	deadline := start.Add(max)
	client := &http.Client{Timeout: 5 * time.Second}

	for time.Now().Before(deadline) {
		resp, err := client.Get(s.BaseURL + "/readyz")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return time.Since(start)
			}
		}
		time.Sleep(2 * time.Second)
	}
	// Best-effort log dump; the t.Fatalf below is what actually
	// fails the test. Keep the dump first so the operator-visible
	// summary at the bottom of `go test` output is the budget value,
	// while the container logs are scrollable above it.
	s.dumpLogs(t)
	t.Fatalf("readiness timed out after %s", max)
	return 0
}

// FormatProgress renders a one-line summary of GET /api/progress for test logs.
func FormatProgress(p Progress) string {
	var pct string
	switch {
	case p.BytesTotal > 0:
		pct = fmt.Sprintf("%.1f%%", float64(p.BytesCompleted)/float64(p.BytesTotal)*100)
	case p.BytesCompleted > 0:
		pct = "streaming"
	default:
		pct = "—"
	}
	line := fmt.Sprintf("phase=%s download=%d/%d (%s)", p.Phase, p.BytesCompleted, p.BytesTotal, pct)
	if p.RetryCount > 0 || p.TransportRetries > 0 {
		line += fmt.Sprintf(" retries=%d transport_retries=%d", p.RetryCount, p.TransportRetries)
	}
	if p.LastError != "" {
		line += " last_error=" + p.LastError
	}
	return line
}

// WaitReadyWithProgressLog is WaitReady plus t.Log lines whenever /api/progress
// changes (phase, byte counters, or last_error). Use -v when running go test
// to see download / loading progress during long cold starts.
func (s *Stack) WaitReadyWithProgressLog(t *testing.T, max time.Duration) time.Duration {
	t.Helper()
	start := time.Now()
	deadline := start.Add(max)
	client := &http.Client{Timeout: 5 * time.Second}
	var lastProgressLog string

	logProgress := func() {
		p, err := GetProgress(s.BaseURL)
		if err != nil {
			return
		}
		line := FormatProgress(p)
		if line != lastProgressLog {
			lastProgressLog = line
			t.Logf("[progress +%s] %s", time.Since(start).Round(time.Second), line)
		}
	}

	for time.Now().Before(deadline) {
		logProgress()
		resp, err := client.Get(s.BaseURL + "/readyz")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				logProgress()
				return time.Since(start)
			}
		}
		time.Sleep(2 * time.Second)
	}
	s.dumpLogs(t)
	t.Fatalf("readiness timed out after %s (last progress: %s)", max, lastProgressLog)
	return 0
}

// ProbeChat fires a streaming POST /v1/chat/completions and returns
// the wall-clock TTFT (time to first SSE `data:` frame) plus the full
// concatenated content. Failure to receive at least one frame fails
// the test.
func ProbeChat(t *testing.T, baseURL, model string) (ttft time.Duration, content string) {
	t.Helper()
	body := map[string]any{
		"model":      model,
		"stream":     true,
		"max_tokens": 32,
		"messages": []map[string]string{
			{"role": "user", "content": "Reply with the single word: hi"},
		},
	}
	buf, _ := json.Marshal(body)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST",
		baseURL+"/v1/chat/completions", bytes.NewReader(buf))
	if err != nil {
		t.Fatalf("build chat request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("chat request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("chat status=%d body=%s", resp.StatusCode, string(raw))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1<<20)
	var sb strings.Builder
	gotFirst := false
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}
		if !gotFirst {
			ttft = time.Since(start)
			gotFirst = true
		}
		var frame struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &frame); err == nil {
			for _, ch := range frame.Choices {
				sb.WriteString(ch.Delta.Content)
			}
		}
	}
	if !gotFirst {
		t.Fatalf("no chat frames received before EOF; scanner err=%v", scanner.Err())
	}
	return ttft, sb.String()
}

// FetchEmbeddingVector POST /v1/embeddings and returns the first vector.
// model may be empty to omit the field (some engines accept omitted model).
func FetchEmbeddingVector(t *testing.T, baseURL, model, input string) []float64 {
	t.Helper()
	body := map[string]any{"input": input}
	if model != "" {
		body["model"] = model
	}
	return postEmbeddings(t, baseURL, body)
}

// FetchImageEmbeddingVector POST /v1/embeddings with a CLIP-style image input
// object (IREmbeddingServer input polymorphism: type=image + data URL).
func FetchImageEmbeddingVector(t *testing.T, baseURL, model, imagePath string) []float64 {
	t.Helper()
	raw, err := os.ReadFile(imagePath)
	if err != nil {
		t.Fatalf("read image %s: %v", imagePath, err)
	}
	mime := imageMIME(imagePath)
	dataURL := fmt.Sprintf("data:%s;base64,%s", mime, base64.StdEncoding.EncodeToString(raw))

	body := map[string]any{
		"input": map[string]any{
			"type":      "image",
			"image_url": dataURL,
		},
	}
	if model != "" {
		body["model"] = model
	}
	return postEmbeddings(t, baseURL, body)
}

func imageMIME(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".webp":
		return "image/webp"
	default:
		return "image/png"
	}
}

func postEmbeddings(t *testing.T, baseURL string, body map[string]any) []float64 {
	t.Helper()
	buf, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal embeddings body: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		baseURL+"/v1/embeddings", bytes.NewReader(buf))
	if err != nil {
		t.Fatalf("build embeddings request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("embeddings request: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("embeddings status=%d body=%s", resp.StatusCode, raw)
	}
	var out struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode embeddings: %v", err)
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		t.Fatal("empty embedding")
	}
	return out.Data[0].Embedding
}

// ProbeEmbeddings POST /v1/embeddings and returns wall-clock latency plus
// the embedding vector dimension.
func ProbeEmbeddings(t *testing.T, baseURL, model, input string) (latency time.Duration, dim int) {
	t.Helper()
	start := time.Now()
	vec := FetchEmbeddingVector(t, baseURL, model, input)
	return time.Since(start), len(vec)
}

// ProbeRerank POST /v1/rerank and returns wall-clock latency plus the top
// result's relevance_score. model may be empty to omit the field.
func ProbeRerank(t *testing.T, baseURL, model, query string, docs []string) (latency time.Duration, topScore float64) {
	t.Helper()
	body := map[string]any{
		"query":     query,
		"documents": docs,
	}
	if model != "" {
		body["model"] = model
	}
	buf, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal rerank body: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		baseURL+"/v1/rerank", bytes.NewReader(buf))
	if err != nil {
		t.Fatalf("build rerank request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("rerank request: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("rerank status=%d body=%s", resp.StatusCode, raw)
	}
	var out struct {
		Results []struct {
			RelevanceScore float64 `json:"relevance_score"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("decode rerank: %v body=%s", err, raw)
	}
	if len(out.Results) == 0 {
		t.Fatalf("empty rerank results: %s", raw)
	}
	return time.Since(start), out.Results[0].RelevanceScore
}

// CosineSimilarity returns the cosine similarity of two equal-length vectors.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// EmbeddingSimilarityCase is a paraphrase pair shared across all engine
// embedding integration tests so logged cosine scores are comparable.
type EmbeddingSimilarityCase struct {
	A, B string
}

// EmbeddingSimilarityCases are smoke-test paraphrases. minSim in callers is
// a low floor (engine/model variance); absolute scores are for logs only.
var EmbeddingSimilarityCases = []EmbeddingSimilarityCase{
	{"The cat sits on the mat.", "A kitten is resting on the rug."},
	{"Python is a popular programming language.", "Python is widely used for software development."},
	{"The weather is sunny today.", "It's a bright and clear day outside."},
	{"She bought fresh vegetables at the market.", "He picked up produce from the grocery store."},
	{"Machine learning models learn from data.", "AI systems find patterns in large datasets."},
}

// AssertEmbeddingSimilarity embeds two texts and checks cosine similarity.
func AssertEmbeddingSimilarity(t *testing.T, baseURL, model, sentA, sentB string, minSim float64) float64 {
	t.Helper()
	embA := FetchEmbeddingVector(t, baseURL, model, sentA)
	embB := FetchEmbeddingVector(t, baseURL, model, sentB)
	sim := CosineSimilarity(embA, embB)
	t.Logf("cosine similarity (%q, %q) = %.4f", sentA, sentB, sim)
	if sim < minSim {
		t.Errorf("expected cosine similarity >= %.2f for similar sentences, got %.4f", minSim, sim)
	}
	return sim
}

// AssertEmbeddingSimilarityCases runs AssertEmbeddingSimilarity for each case.
func AssertEmbeddingSimilarityCases(t *testing.T, baseURL, model string, minSim float64, cases []EmbeddingSimilarityCase) {
	t.Helper()
	for i, c := range cases {
		t.Logf("similarity case %d/%d", i+1, len(cases))
		AssertEmbeddingSimilarity(t, baseURL, model, c.A, c.B, minSim)
	}
}

// Progress is the subset of GET /api/progress the fault-injection
// suites assert on. The download-fault matrices in
// tests/integration/download and tests/integration/ollamaurl never
// reach a live engine, so /readyz cannot be the signal — the lifecycle
// flips PhaseReady (and exposes degraded/failed plus a LastError) on
// /api/progress independent of WaitAlive. Field names mirror
// internal/progress.State's JSON tags.
type Progress struct {
	Phase            string `json:"phase"`
	LastError        string `json:"last_error"`
	BytesTotal       int64  `json:"bytes_total"`
	BytesCompleted   int64  `json:"bytes_completed"`
	RetryCount       int    `json:"retry_count"`
	TransportRetries int    `json:"transport_retries"`
}

// GetProgress fetches and decodes GET /api/progress once. A transport
// error or non-200 is returned to the caller so WaitProgress can keep
// polling while the container is still booting (the API server binds
// the port a beat after `docker compose up -d` returns).
func GetProgress(baseURL string) (Progress, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(baseURL + "/api/progress")
	if err != nil {
		return Progress{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Progress{}, fmt.Errorf("/api/progress status=%d", resp.StatusCode)
	}
	var p Progress
	if err := json.NewDecoder(resp.Body).Decode(&p); err != nil {
		return Progress{}, fmt.Errorf("decode /api/progress: %w", err)
	}
	return p, nil
}

// WaitProgress polls GET /api/progress until pred returns true or max
// elapses, then returns the matching snapshot. On timeout it dumps the
// last ~200 lines of compose logs (mirroring WaitReady) before failing,
// so a stuck download surfaces the container state rather than a bare
// budget value. Transient fetch errors (pre-bind, mid-restart) are
// swallowed and retried; only the final timeout fails the test.
func (s *Stack) WaitProgress(t *testing.T, pred func(Progress) bool, max time.Duration) Progress {
	t.Helper()
	deadline := time.Now().Add(max)
	var last Progress
	for time.Now().Before(deadline) {
		p, err := GetProgress(s.BaseURL)
		if err == nil {
			last = p
			if pred(p) {
				return p
			}
		}
		time.Sleep(2 * time.Second)
	}
	s.dumpLogs(t)
	t.Fatalf("progress predicate not satisfied within %s; last seen phase=%q last_error=%q",
		max, last.Phase, last.LastError)
	return Progress{}
}

// Baseline is one row of tests/integration/baseline.json.
type Baseline struct {
	ReadyMs    int64  `json:"ready_ms"`
	TTFTMs     int64  `json:"ttft_ms"`
	RecordedAt string `json:"recorded_at"`
}

// RecordBaseline writes <repo-root>/tests/integration/baseline.json
// atomically (write to temp + rename). Existing entries are preserved
// so a single test run can update a single engine without erasing the
// others. The file is committed to the repo as a reference; CI does
// not assert on its contents to avoid flake.
func RecordBaseline(t *testing.T, kind string, ready, ttft time.Duration) {
	t.Helper()

	root, err := repoRoot()
	if err != nil {
		t.Logf("baseline skipped: cannot find repo root: %v", err)
		return
	}
	path := filepath.Join(root, "tests", "integration", "baseline.json")

	all := map[string]Baseline{}
	if data, err := os.ReadFile(path); err == nil {
		_ = json.Unmarshal(data, &all)
	}
	all[kind] = Baseline{
		ReadyMs:    ready.Milliseconds(),
		TTFTMs:     ttft.Milliseconds(),
		RecordedAt: time.Now().UTC().Format(time.RFC3339),
	}

	data, err := json.MarshalIndent(all, "", "  ")
	if err != nil {
		t.Fatalf("marshal baseline: %v", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("write baseline tmp: %v", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		t.Fatalf("rename baseline: %v", err)
	}
}

// RepoPath resolves rel against the repository root (located by walking
// up to find go.mod). Failing the lookup fails the test.
func RepoPath(t *testing.T, rel string) string {
	t.Helper()
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	return filepath.Join(root, filepath.FromSlash(rel))
}

// EnsureImage returns the image tag for a fault-suite container, building
// it on demand. If the env var named by envKey is set, its value is used
// verbatim (CI / power users supply a pre-built image). Otherwise the
// image is built from dockerfileRel with the repo root as context and
// tagged defaultTag. dockerfileRel == "" means the root Dockerfile.
// The chosen tag is exported back into envKey so the compose files
// (which read the same var) resolve to it.
func EnsureImage(t *testing.T, envKey, defaultTag, dockerfileRel string) string {
	t.Helper()
	if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
		return v
	}
	root, err := repoRoot()
	if err != nil {
		t.Fatalf("repo root: %v", err)
	}
	args := []string{"build", "-t", defaultTag}
	if dockerfileRel != "" {
		args = append(args, "-f", filepath.Join(root, filepath.FromSlash(dockerfileRel)))
	}
	args = append(args, root)

	cmd := exec.Command("docker", args...)
	cmd.Stdout = &prefixedWriter{prefix: "[docker-build] ", out: testWriter{t}}
	cmd.Stderr = cmd.Stdout
	if err := cmd.Run(); err != nil {
		t.Fatalf("docker build %s: %v", defaultTag, err)
	}
	t.Setenv(envKey, defaultTag)
	return defaultTag
}

// AssertPortFree refuses to start the stack if the published llm-init
// port already has a listener — otherwise the harness will silently
// route /readyz to whatever is squatting on 127.0.0.1:8080. An empty
// port is treated as "harness will allocate a free one in UpFiles" and
// skipped.
func AssertPortFree(t *testing.T, port string) {
	t.Helper()
	if strings.TrimSpace(port) == "" {
		return
	}
	conn, err := net.DialTimeout("tcp", "127.0.0.1:"+port, 200*time.Millisecond)
	if err == nil {
		_ = conn.Close()
		t.Fatalf("port %s already bound; refusing to overwrite an existing service", port)
	}
}

// FreePort asks the OS for an unused TCP port on the loopback interface
// and returns it as a string. There is an inherent TOCTOU window
// between closing the probe listener and docker binding the port, but
// it is far smaller than the collision risk of a hard-coded 8080 when
// stacks run back-to-back or in parallel.
func FreePort(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("FreePort: %v", err)
	}
	defer l.Close()
	_, p, err := net.SplitHostPort(l.Addr().String())
	if err != nil {
		t.Fatalf("FreePort split %q: %v", l.Addr(), err)
	}
	return p
}

// repoRoot walks parent directories looking for go.mod, which marks
// the repository root.
func repoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found above %s", dir)
		}
		dir = parent
	}
}

// dumpLogs streams `docker compose logs` for every file in the stack to
// t.Log on failure.
func (s *Stack) dumpLogs(t *testing.T) {
	t.Helper()
	args := append(s.composeArgs(), "logs", "--no-color", "--tail=200")
	cmd := exec.Command("docker", append([]string{composeSubcmd}, args...)...)
	out, _ := cmd.CombinedOutput()
	for _, line := range strings.Split(string(out), "\n") {
		if line == "" {
			continue
		}
		t.Log(line)
	}
}

// prefixedWriter prepends a fixed string to every line it forwards.
// Used so compose-up output is greppable in CI logs.
type prefixedWriter struct {
	prefix string
	out    io.Writer
}

func (p *prefixedWriter) Write(b []byte) (int, error) {
	for _, line := range strings.SplitAfter(string(b), "\n") {
		if line == "" {
			continue
		}
		fmt.Fprint(p.out, p.prefix, line)
	}
	return len(b), nil
}

// testWriter is an io.Writer that pipes lines into t.Log.
type testWriter struct{ t *testing.T }

func (w testWriter) Write(b []byte) (int, error) {
	w.t.Log(strings.TrimRight(string(b), "\r\n"))
	return len(b), nil
}
