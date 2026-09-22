package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/obs"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/runtimecfg"
	"github.com/llm-init/llm-init/internal/version"
)

// fixture returns a server configured against a vLLM + HF source so
// token redaction is exercised, alongside the manager so tests can drive
// state transitions.
func fixture(t *testing.T) (*Server, progress.Manager) {
	t.Helper()
	env := map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
		"HF_TOKEN":     "hf_abcdefghijklmnopqrstuvwxyz1234",
	}
	cfg, err := config.Load(func(k string) string { return env[k] })
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	mgr := progress.New(time.Now())
	return NewServer(Options{
		Manager: mgr,
		Metrics: obs.NewMetrics(),
		Config:  runtimecfg.New(cfg, nil),
		Version: version.Info{Version: "v1.2.3", Commit: "abc"},
		// Mirror cmd/llm-init/main.go's wiring so the fixture
		// exercises the same code path operators run. Pre-fix the
		// fixture omitted these and tests therefore did not catch
		// the BLOCKER where RetryRateLimit=0 was silently remapped
		// to the default 1.0/s.
		RetryRateLimit: cfg.Runtime.RetryRateLimit,
	}), mgr
}

// httpResp is the slice of an http.Response that server_test.go's
// get() helper actually exposes to callers. Returning this instead
// of *http.Response (which the pre-v1.0.5 helper did) keeps the
// body-drained-and-closed contract in one place AND prevents
// bodyclose from flagging every caller -- the linter was right
// that it could not prove the close, because the helper's return
// type leaked the *http.Response shape. See lint-D commit
// message.
type httpResp struct {
	Status int
	Header http.Header
	Body   []byte
}

func get(t *testing.T, s *Server, path string) *httpResp {
	t.Helper()
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	resp, err := http.Get(srv.URL + path)
	if err != nil {
		t.Fatalf("GET %s: %v", path, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return &httpResp{
		Status: resp.StatusCode,
		Header: resp.Header.Clone(),
		Body:   body,
	}
}

func TestLivez_Always200(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	r := get(t, s, "/livez")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	if !strings.Contains(string(r.Body), `"status":"ok"`) {
		t.Errorf("body = %s", r.Body)
	}
}

func TestReadyz_503WhenInit(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	r := get(t, s, "/readyz")
	if r.Status != 503 {
		t.Fatalf("status = %d, want 503", r.Status)
	}
	if !strings.Contains(string(r.Body), `"status":"not_ready"`) {
		t.Errorf("body missing not_ready: %s", r.Body)
	}
	if !strings.Contains(string(r.Body), `"phase":"init"`) {
		t.Errorf("body missing phase=init: %s", r.Body)
	}
}

func TestReadyz_200WhenAllGreen(t *testing.T) {
	t.Parallel()
	s, mgr := fixture(t)
	mgr.Update(func(st *progress.State) { st.Phase = progress.PhaseReady })
	s.opts.Readiness = func() Readiness {
		return Readiness{
			Phase:       progress.PhaseReady,
			EngineAlive: true,
			ModelExists: true,
			Ready:       true,
		}
	}
	r := get(t, s, "/readyz")
	if r.Status != 200 {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	var parsed map[string]string
	if err := json.Unmarshal(r.Body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if parsed["status"] != "ready" || parsed["model"] != "qwen2.5-7b" {
		t.Errorf("body = %v", parsed)
	}
}

func TestHealthz_FieldsPresent(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	r := get(t, s, "/healthz")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	var parsed map[string]any
	if err := json.Unmarshal(r.Body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, k := range []string{"status", "ready", "engine_alive", "model_exists",
		"phase", "last_verify", "last_verify_ok"} {
		if _, ok := parsed[k]; !ok {
			t.Errorf("healthz missing key %q", k)
		}
	}
	if parsed["phase"] != "init" {
		t.Errorf("phase = %v, want init", parsed["phase"])
	}
	// On a fresh manager LastVerifyAt is zero and LastVerifyOK is nil,
	// both of which must serialise as JSON null (not 0001-01-01T...).
	if parsed["last_verify"] != nil {
		t.Errorf("last_verify = %v, want null when no verify has run", parsed["last_verify"])
	}
	if parsed["last_verify_ok"] != nil {
		t.Errorf("last_verify_ok = %v, want null when no verify has run", parsed["last_verify_ok"])
	}
}

// An engine that is up without the model it was given is the case the
// two handlers used to answer differently: `status` only asked whether
// the engine was alive, so it read "ok" beside `"ready": false` while
// every /v1 request was being refused with 503 — for as long as the
// health loop's grace window, which is where a dashboard watching
// `status` would have shown a green light through an outage.
func TestReadinessIsOneVerdict_AliveEngineMissingModel(t *testing.T) {
	t.Parallel()
	s, mgr := fixture(t)
	mgr.Update(func(st *progress.State) { st.Phase = progress.PhaseReady })
	s.opts.Readiness = func() Readiness {
		return Readiness{
			Phase:       progress.PhaseReady,
			EngineAlive: true,
			ModelExists: false,
			Ready:       false,
			Reason:      "model_missing",
		}
	}

	if r := get(t, s, "/readyz"); r.Status != 503 {
		t.Errorf("/readyz = %d, want 503; body=%s", r.Status, r.Body)
	}

	r := get(t, s, "/healthz")
	var parsed map[string]any
	if err := json.Unmarshal(r.Body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if parsed["ready"] != false {
		t.Errorf("ready = %v, want false", parsed["ready"])
	}
	if parsed["status"] != "degraded" {
		t.Errorf("status = %v, want degraded beside ready=false", parsed["status"])
	}
}

// TestHealthz_PopulatesLastVerify asserts that once a verify outcome
// has been stamped onto progress.State, /healthz surfaces both
// `last_verify` (RFC3339 string) and `last_verify_ok` (bool).
// Pre-v1.0.3 the handler hard-coded nil for both, so a populated state
// still came back as null and the dashboard could not display verify
// health.
func TestHealthz_PopulatesLastVerify(t *testing.T) {
	t.Parallel()
	s, mgr := fixture(t)
	when := time.Date(2026, 5, 15, 10, 30, 0, 0, time.UTC)
	ok := true
	mgr.Update(func(st *progress.State) {
		st.LastVerifyAt = when
		st.LastVerifyOK = &ok
	})
	r := get(t, s, "/healthz")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	var parsed map[string]any
	if err := json.Unmarshal(r.Body, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := parsed["last_verify"]; got != "2026-05-15T10:30:00Z" {
		t.Errorf("last_verify = %v, want RFC3339 of 2026-05-15T10:30:00Z", got)
	}
	if got := parsed["last_verify_ok"]; got != true {
		t.Errorf("last_verify_ok = %v, want true", got)
	}
}

func TestProgress_ReturnsSnapshot(t *testing.T) {
	t.Parallel()
	s, mgr := fixture(t)
	mgr.Update(func(st *progress.State) {
		st.Phase = progress.PhaseDownload
		st.BytesTotal = 10
		st.BytesCompleted = 3
	})
	r := get(t, s, "/api/progress")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	var st progress.State
	if err := json.Unmarshal(r.Body, &st); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if st.Phase != progress.PhaseDownload || st.BytesCompleted != 3 {
		t.Errorf("snapshot = %+v", st)
	}
}

func TestConfig_RedactsSecrets(t *testing.T) {
	s, _ := fixture(t)
	r := get(t, s, "/api/config")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	got := string(r.Body)
	if strings.Contains(got, "hf_abcdefghijklmnopqrstuvwxyz1234") {
		t.Errorf("HF token leaked into /api/config: %s", got)
	}
	// v1.1 contract: HF_TOKEN never round-trips; only the boolean
	// hf_token_set: true reflects whether it was wired.
	if !strings.Contains(got, `"hf_token_set":true`) {
		t.Errorf("hf_token_set=true missing from /api/config: %s", got)
	}
}

// TestConfig_ExposesPublicURL pins the contract the dashboard's
// Overview "API URL" card reads: when APP_URL is set on the llm-init
// process, GET /api/config must surface it under Runtime.PublicURL
// (case-sensitive — encoding/json maps the Go field name verbatim).
// Olares charts populate APP_URL from .Values.domain.<entrance>; an
// empty APP_URL is also tested separately at the config layer (see
// TestLoad_PublicURL).
func TestConfig_ExposesPublicURL(t *testing.T) {
	env := map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
		"APP_URL":      "https://5f99418c.pptest01.olares.com",
	}
	cfg, err := config.Load(func(k string) string { return env[k] })
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	mgr := progress.New(time.Now())
	s := NewServer(Options{
		Manager:        mgr,
		Metrics:        obs.NewMetrics(),
		Config:         runtimecfg.New(cfg, nil),
		Version:        version.Info{Version: "v0.0.0"},
		RetryRateLimit: cfg.Runtime.RetryRateLimit,
	})
	r := get(t, s, "/api/config")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	// Decode rather than substring-match so we catch shape regressions
	// (e.g. a future refactor that nests public_url under a different
	// parent would silently keep passing a strings.Contains check).
	// Go's decoder falls back to case-insensitive field matching, so the
	// keys below must be spelled exactly as the dashboard reads them —
	// this test passed for a release against `Runtime.PublicURL` while
	// the response already said `runtime`, and the front-end lookup that
	// had no such fallback returned undefined the whole time.
	var got struct {
		Runtime struct {
			PublicURL string `json:"public_url"`
		} `json:"runtime"`
	}
	if err := json.Unmarshal(r.Body, &got); err != nil {
		t.Fatalf("unmarshal: %v\nbody=%s", err, r.Body)
	}
	if got.Runtime.PublicURL != "https://5f99418c.pptest01.olares.com" {
		t.Errorf("runtime.public_url = %q, want the APP_URL value\nbody=%s",
			got.Runtime.PublicURL, r.Body)
	}
	if !strings.Contains(string(r.Body), `"runtime":{"port":`) {
		t.Errorf("runtime object should marshal snake_case field names\nbody=%s", r.Body)
	}
}

func TestRetry_NotWiredReturns503(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/api/retry", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "lifecycle_not_wired") {
		t.Errorf("body = %s", body)
	}
}

func TestRetry_DispatchesAndReturns202(t *testing.T) {
	t.Parallel()
	s, mgr := fixture(t)
	mgr.Update(func(st *progress.State) { st.Phase = progress.PhaseFailed })
	called := make(chan RetryOptions, 1)
	s.opts.Retry = func(_ context.Context, opts RetryOptions) error {
		called <- opts
		return nil
	}

	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	resp, err := http.Post(srv.URL+"/api/retry?force=true&level=sha256", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 202 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	select {
	case opts := <-called:
		if !opts.Force || opts.Level != "sha256" {
			t.Errorf("opts = %+v", opts)
		}
	case <-time.After(time.Second):
		t.Fatal("retry not invoked")
	}

	body, _ := io.ReadAll(resp.Body)
	var parsed map[string]any
	_ = json.Unmarshal(body, &parsed)
	if parsed["previous_phase"] != "failed" {
		t.Errorf("previous_phase = %v", parsed["previous_phase"])
	}
}

// TestRetry_LevelForwardedAsIs locks the wire contract: the handler has
// no opinion on ?level=, it hands the string to the lifecycle, which
// falls back to VERIFY_LEVEL on anything it does not recognise. Keeping
// the rejection out of the HTTP layer is what lets a dashboard sending
// a value from an older release still trigger a retry.
func TestRetry_LevelForwardedAsIs(t *testing.T) {
	t.Parallel()
	for _, level := range []string{"size", "sha256", "remote", "md5", "sha-256", ""} {
		level := level
		name := level
		if name == "" {
			name = "empty"
		}
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			s, _ := fixture(t)
			got := make(chan string, 1)
			s.opts.Retry = func(_ context.Context, opts RetryOptions) error {
				got <- opts.Level
				return nil
			}
			srv := httptest.NewServer(s)
			t.Cleanup(srv.Close)
			resp, err := http.Post(srv.URL+"/api/retry?level="+level,
				"application/json", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusAccepted {
				t.Errorf("status=%d want 202", resp.StatusCode)
			}
			select {
			case g := <-got:
				if g != level {
					t.Errorf("forwarded level=%q want %q", g, level)
				}
			case <-time.After(time.Second):
				t.Fatal("Retry not invoked")
			}
		})
	}
}

// TestRetry_RateLimitedAfterBurst locks the v1.0.5 token-bucket guard.
// We construct a server with rate=1/s burst=5 (the production default),
// fire 6 requests in tight succession, and expect the first 5 to be
// accepted (202) and the 6th to be rejected (429 + Retry-After +
// rate_limited). The lifecycle Retry stub MUST NOT be dispatched for
// the rejected call. We also verify Retry-After is present and the
// llm_init_retry_throttled_total counter has incremented.
func TestRetry_RateLimitedAfterBurst(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	calls := 0
	s.opts.Retry = func(_ context.Context, _ RetryOptions) error {
		calls++
		return nil
	}
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	for i := 0; i < 5; i++ {
		resp, err := http.Post(srv.URL+"/api/retry", "application/json", nil)
		if err != nil {
			t.Fatalf("burst[%d]: %v", i, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusAccepted {
			t.Fatalf("burst[%d] status=%d, want 202", i, resp.StatusCode)
		}
	}
	if calls != 5 {
		t.Fatalf("Retry dispatched %d times, want 5", calls)
	}

	resp, err := http.Post(srv.URL+"/api/retry", "application/json", nil)
	if err != nil {
		t.Fatalf("over-burst: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("over-burst status=%d, want 429", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got == "" {
		t.Errorf("Retry-After missing on 429")
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"code":"rate_limited"`) {
		t.Errorf("body missing rate_limited code: %s", body)
	}
	if calls != 5 {
		t.Errorf("Retry dispatched on rate-limited call: calls=%d", calls)
	}
}

// TestRetry_RateLimitDisabled verifies that RetryRateLimit=0 makes
// the limiter a no-op so all calls succeed. 0 is the SAME value
// config.Load accepts on RETRY_RATE_LIMIT=0 (see config.go:431-432
// where the parser rejects negatives but allows 0); pre-fix this
// test used -1 which production config never produces, so the
// production disable path was uncovered and the BLOCKER bug
// (server.NewServer remapping 0 -> 1.0) shipped to v1.0.5
// undetected.
func TestRetry_RateLimitDisabled(t *testing.T) {
	t.Parallel()
	mgr := progress.New(time.Now())
	s := NewServer(Options{
		Manager:        mgr,
		Metrics:        obs.NewMetrics(),
		Version:        version.Info{Version: "v1.0.5"},
		RetryRateLimit: 0, // documented disable value (matches RETRY_RATE_LIMIT=0)
		Retry: func(_ context.Context, _ RetryOptions) error {
			return nil
		},
	})
	if s.retryLimiter != nil {
		t.Fatal("retryLimiter must be nil when RetryRateLimit=0; pre-fix v1.0.5 silently remapped 0 -> 1.0/s")
	}
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	// 50 calls is comfortably above any sane burst, so a smuggled-in
	// limiter at the production default (rate=1/s burst=5) would
	// trip 429 well before this loop completes.
	for i := 0; i < 50; i++ {
		resp, err := http.Post(srv.URL+"/api/retry", "application/json", nil)
		if err != nil {
			t.Fatalf("call[%d]: %v", i, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusAccepted {
			t.Fatalf("call[%d] status=%d, want 202 (limiter disabled)", i, resp.StatusCode)
		}
	}
}

func TestRetry_PropagatesError(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	s.opts.Retry = func(_ context.Context, _ RetryOptions) error {
		return errors.New("disk full")
	}
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	resp, err := http.Post(srv.URL+"/api/retry", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 500 {
		t.Errorf("status = %d, want 500", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "disk full") {
		t.Errorf("body = %s", body)
	}
}

func TestMetrics_Mounted(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	r := get(t, s, "/metrics")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	if !strings.Contains(string(r.Body), "# TYPE llm_init_phase gauge") {
		t.Errorf("metrics not exposed: %s", r.Body)
	}
}

func TestDataPlane_503BeforeReady(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got != "5" {
		t.Errorf("Retry-After = %q, want 5", got)
	}
	// v1.0.5 contract: the boot-time stub MUST emit the same machine
	// code as dataplane.writeNotReady so a client written against the
	// post-registration code does not regress when scraped during
	// boot. Pre-fix this stub said "model_not_ready" while
	// writeNotReady said "not_ready"; clients pinning either lost half
	// the timeline.
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"code":"not_ready"`) {
		t.Errorf("body missing canonical not_ready code: %s", body)
	}
	if strings.Contains(string(body), `"model_not_ready"`) {
		t.Errorf("body still emits legacy model_not_ready code: %s", body)
	}
}

func TestDataPlane_501AfterReady(t *testing.T) {
	t.Parallel()
	s, mgr := fixture(t)
	mgr.Update(func(st *progress.State) { st.Phase = progress.PhaseReady })

	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 501 {
		t.Errorf("status = %d, want 501 (legacy stub still active when DataPlane registrar absent)", resp.StatusCode)
	}
}

func TestDataPlane_RegistrarSuppressesStub(t *testing.T) {
	t.Parallel()
	mgr := progress.New(time.Now())
	called := false
	s := NewServer(Options{
		Manager: mgr,
		Metrics: obs.NewMetrics(),
		Version: version.Info{Version: "v1.2.3"},
		DataPlane: func(m *http.ServeMux) {
			m.HandleFunc("/v1/", func(w http.ResponseWriter, _ *http.Request) {
				called = true
				w.WriteHeader(http.StatusTeapot)
			})
		},
	})
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	// When the DataPlane registrar is supplied, the legacy 503 stub must
	// step out of the way so the M4 dataplane handler is reached even
	// before phase=ready.
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTeapot {
		t.Fatalf("status=%d (registrar handler should be reached)", resp.StatusCode)
	}
	if !called {
		t.Error("registrar handler not invoked")
	}
}

func TestUI_Root200(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	r := get(t, s, "/")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	if !strings.HasPrefix(r.Header.Get("Content-Type"), "text/html") {
		t.Errorf("content-type = %q", r.Header.Get("Content-Type"))
	}
	if !strings.Contains(string(r.Body), "llm-init") {
		t.Errorf("body should mention llm-init: %s", r.Body)
	}
}

func TestUI_StaticServed(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	r := get(t, s, "/static/index.html")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}
	if !strings.Contains(string(r.Body), "llm-init") {
		t.Errorf("body = %s", r.Body)
	}
}

// TestUI_DashboardAssetsEmbedded asserts that every static asset the
// v1.0.9 dashboard depends on is bundled into the embed.FS. A missing
// file would 404 the browser silently — easy to miss in manual smoke
// tests, deadly when the operator opens the production binary and
// sees a blank page.
func TestUI_DashboardAssetsEmbedded(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	assets := []string{
		"/static/dashboard.css",
		"/static/dashboard.js",
		"/static/charts.js",
		"/static/config.js",
		"/static/gpu.js",
		"/static/vendor/uPlot.iife.min.js",
		"/static/vendor/uPlot.min.css",
		"/static/vendor/LICENSE.uPlot",
	}
	for _, p := range assets {
		p := p
		t.Run(p, func(t *testing.T) {
			t.Parallel()
			r := get(t, s, p)
			if r.Status != http.StatusOK {
				t.Errorf("%s: status = %d, want 200", p, r.Status)
			}
			if len(r.Body) == 0 {
				t.Errorf("%s: empty body", p)
			}
		})
	}
}

func TestUI_UnknownReturns404(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	r := get(t, s, "/does-not-exist")
	if r.Status != 404 {
		t.Errorf("status = %d, want 404", r.Status)
	}
}

// TestRoot_AcceptJSON exercises the content-negotiation branch in
// attachUI: a request that asks specifically for application/json gets
// a tiny status payload, while a normal browser (Accept: text/html,...)
// still receives the dashboard HTML.
func TestRoot_AcceptJSON(t *testing.T) {
	t.Parallel()
	s, _ := fixture(t)
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	cases := []struct {
		name       string
		accept     string
		wantCT     string
		wantInBody string
	}{
		{"json_only", "application/json", "application/json", `"status"`},
		{"json_first_no_html", "application/json, */*", "application/json", `"status"`},
		{"browser_default", "text/html,application/xhtml+xml,application/xml", "text/html", "llm-init"},
		{"empty", "", "text/html", "llm-init"},
		{"both", "text/html,application/json", "text/html", "llm-init"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req, _ := http.NewRequest(http.MethodGet, srv.URL+"/", nil)
			if tc.accept != "" {
				req.Header.Set("Accept", tc.accept)
			} else {
				// Override Go's default Accept header.
				req.Header["Accept"] = nil
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode != 200 {
				t.Fatalf("status=%d", resp.StatusCode)
			}
			if !strings.HasPrefix(resp.Header.Get("Content-Type"), tc.wantCT) {
				t.Errorf("Content-Type=%q want prefix %q", resp.Header.Get("Content-Type"), tc.wantCT)
			}
			if !strings.Contains(string(body), tc.wantInBody) {
				t.Errorf("body lacks %q: %s", tc.wantInBody, body)
			}
		})
	}
}
