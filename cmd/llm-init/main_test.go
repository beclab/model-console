package main

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

// mapEnv builds a config.Getenv backed by the supplied map. Used everywhere
// in this test binary so tests are hermetic — the parent shell's env cannot
// leak into config.Load.
func mapEnv(m map[string]string) config.Getenv {
	return func(k string) string { return m[k] }
}

// setupSpecFile returns a writable temp path for the model-spec.json that
// ReconcileModelSpecFile seeds on first boot. Tests pass it via
// MODEL_SPEC_PATH so the reconcile step does not try to write the
// production default under /run (which is not writable in CI).
func setupSpecFile(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "model-spec.json")
}

// setupRunDir returns a writable RUN_DIR. engine_args handoff writes
// ${RUN_DIR}/engine_args during reconcile; the production default
// /run/llm-init is not writable on CI runners.
func setupRunDir(t *testing.T) string {
	t.Helper()
	return t.TempDir()
}

// captureOutput runs fn with stdout/stderr replaced by temp files and returns
// the captured contents. It avoids touching the parent process's real stdio.
func captureOutput(t *testing.T, fn func(stdout, stderr *os.File)) (stdout, stderr string) {
	t.Helper()
	dir := t.TempDir()
	out, err := os.Create(filepath.Join(dir, "out"))
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	errF, err := os.Create(filepath.Join(dir, "err"))
	if err != nil {
		t.Fatal(err)
	}
	defer errF.Close()
	fn(out, errF)
	_ = out.Sync()
	_ = errF.Sync()
	stdoutB, _ := os.ReadFile(out.Name())
	stderrB, _ := os.ReadFile(errF.Name())
	return string(stdoutB), string(stderrB)
}

func TestRun_VersionFlag(t *testing.T) {
	t.Parallel()
	var code int
	stdout, _ := captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--version"}, out, errF, mapEnv(nil))
	})
	if code != exitOK {
		t.Errorf("exit code = %d, want %d", code, exitOK)
	}
	if !strings.Contains(stdout, "llm-init") {
		t.Errorf("stdout missing version banner: %q", stdout)
	}
}

func TestRun_HelpFlag(t *testing.T) {
	t.Parallel()
	var code int
	captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"-h"}, out, errF, mapEnv(nil))
	})
	if code != exitOK {
		t.Errorf("help should exit 0; got %d", code)
	}
}

func TestRun_BadFlag(t *testing.T) {
	t.Parallel()
	var code int
	captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--no-such-flag"}, out, errF, mapEnv(nil))
	})
	if code != exitConfig {
		t.Errorf("bad flag exit = %d, want %d", code, exitConfig)
	}
}

func TestRun_PrintConfig(t *testing.T) {
	specPath := setupSpecFile(t)
	env := mapEnv(map[string]string{
		"ENGINE_KIND":     "ollama",
		"MODEL_NAME":      "qwen2.5:7b",
		"MODEL_MODE":      "chat",
		"MODEL_SOURCE":    "ollama://qwen2.5:7b",
		"MODEL_SPEC_PATH": specPath,
		"RUN_DIR":         setupRunDir(t),
	})

	var code int
	stdout, _ := captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--print-config"}, out, errF, env)
	})
	if code != exitOK {
		t.Errorf("exit code = %d", code)
	}
	if !strings.Contains(stdout, "ollama") {
		t.Errorf("redacted config should include engine kind: %q", stdout)
	}
}

func TestRun_ConfigError(t *testing.T) {
	setupSpecFile(t)
	env := mapEnv(map[string]string{
		"ENGINE_KIND": "not-a-real-engine",
		"MODEL_NAME":  "n",
	})

	var code int
	captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--print-config"}, out, errF, env)
	})
	if code != exitConfig {
		t.Errorf("config error exit = %d, want %d", code, exitConfig)
	}
}

// TestRun_NoEnvLeak proves the smoke-test bug from M5 verification cannot
// regress: an injected map-getenv must beat anything os.Setenv does, even
// when those os env vars would normally cause a fail-fast (HF_REPO is on
// the v1.1 22-removed list).
func TestRun_NoEnvLeak(t *testing.T) {
	specPath := setupSpecFile(t)

	// Set a hostile os env that would crash config validation if read.
	os.Setenv("HF_REPO", "leaked/from-parent")
	t.Cleanup(func() { os.Unsetenv("HF_REPO") })

	env := mapEnv(map[string]string{
		"ENGINE_KIND":     "ollama",
		"MODEL_NAME":      "qwen2.5:7b",
		"MODEL_MODE":      "chat",
		"MODEL_SOURCE":    "ollama://qwen2.5:7b",
		"MODEL_SPEC_PATH": specPath,
		"RUN_DIR":         setupRunDir(t),
		// No HF_REPO in the map — leaked os env must NOT be read.
	})
	var code int
	captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--print-config"}, out, errF, env)
	})
	if code != exitOK {
		t.Errorf("exit code = %d (env leak regression?)", code)
	}
}

// TestRun_Healthcheck_OK starts a tiny test server that answers /livez 200
// and verifies --healthcheck against its port returns 0.
func TestRun_Healthcheck_OK(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/livez" {
			w.WriteHeader(200)
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(srv.Close)

	port := srv.Listener.Addr().(*net.TCPAddr).Port
	env := mapEnv(map[string]string{"PORT": fmt.Sprintf("%d", port)})

	var code int
	stdout, _ := captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--healthcheck"}, out, errF, env)
	})
	if code != exitOK {
		t.Errorf("healthcheck OK: code = %d, want %d", code, exitOK)
	}
	if !strings.Contains(stdout, "ok") {
		t.Errorf("healthcheck OK: stdout = %q", stdout)
	}
}

// TestRun_Healthcheck_NoServer points at a closed port and verifies the
// probe exits with the documented failure code.
func TestRun_Healthcheck_NoServer(t *testing.T) {
	t.Parallel()
	// Bind a listener and close it to grab a port nothing's serving on.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()

	env := mapEnv(map[string]string{"PORT": fmt.Sprintf("%d", port)})
	var code int
	captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--healthcheck"}, out, errF, env)
	})
	if code != exitHealthcheckFail {
		t.Errorf("healthcheck no-server: code = %d, want %d", code, exitHealthcheckFail)
	}
}

// TestRun_Healthcheck_Non2xx asserts a server answering /livez 500 fails
// the probe.
func TestRun_Healthcheck_Non2xx(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
	}))
	t.Cleanup(srv.Close)

	port := srv.Listener.Addr().(*net.TCPAddr).Port
	env := mapEnv(map[string]string{"PORT": fmt.Sprintf("%d", port)})

	var code int
	captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--healthcheck"}, out, errF, env)
	})
	if code != exitHealthcheckFail {
		t.Errorf("healthcheck 500: code = %d, want %d", code, exitHealthcheckFail)
	}
}

func TestNewLogger(t *testing.T) {
	t.Parallel()
	cases := []struct{ level, format string }{
		{"debug", "json"},
		{"info", "text"},
		{"warn", "json"},
		{"error", "text"},
		{"unknown", "json"}, // falls back to info
	}
	for _, tc := range cases {
		l := newLogger(tc.level, tc.format, os.Stderr)
		if l == nil {
			t.Errorf("newLogger(%s, %s) returned nil", tc.level, tc.format)
		}
	}
}

// TestRun_ProgressStateLoadCorrupt asserts that a corrupt progress
// state file does NOT crash the bootstrap. Pre-fix a bad state file
// would percolate through mgr.LoadFrom and short-circuit run() before
// the control plane ever bound; today the path warns + continues. The
// test injects a non-JSON file at PROGRESS_STATE_PATH and verifies
// run() still reaches PhaseInit (the control plane comes up). The
// LoadFrom error branch (line ~129 in main.go) is otherwise
// uncovered on cross-platform runs; smoke_linux_test exercises only
// the happy path.
func TestRun_ProgressStateLoadCorrupt(t *testing.T) {
	t.Parallel()
	port, err := freePort()
	if err != nil {
		t.Fatalf("freePort: %v", err)
	}
	dir := t.TempDir()
	statePath := filepath.Join(dir, "progress.json")
	if err := os.WriteFile(statePath, []byte("{this is not valid json"), 0o600); err != nil {
		t.Fatalf("seed bad state: %v", err)
	}

	specPath := setupSpecFile(t)
	env := mapEnv(map[string]string{
		"ENGINE_KIND":         "vllm",
		"MODEL_NAME":          "qwen2.5-7b",
		"MODEL_MODE":          "chat",
		"MODEL_SOURCE":        "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
		"PORT":                port,
		"PROGRESS_STATE_PATH": statePath,
		"MODEL_SPEC_PATH":     specPath,
		"RUN_DIR":             setupRunDir(t),
	})

	stdout, _ := os.Create(filepath.Join(dir, "out"))
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(filepath.Join(dir, "err"))
	t.Cleanup(func() { stderr.Close() })

	go func() { _ = run(nil, stdout, stderr, env) }()

	if err := waitFor("http://127.0.0.1:"+port+"/livez", 15*time.Second); err != nil {
		dumpServerLogs(t, stderr.Name())
		t.Fatalf("server never came up despite corrupt progress state: %v", err)
	}

	// stderr should contain the warn the LoadFrom error branch emits.
	// We don't pin the exact wording (slog.Warn message text is
	// allowed to evolve); just confirm the server got past LoadFrom
	// rather than crashing on it.
	stderrBytes, _ := os.ReadFile(stderr.Name())
	if !strings.Contains(string(stderrBytes), "progress state load failed") {
		t.Errorf("expected LoadFrom warn in stderr, got: %s", stderrBytes)
	}
}

// TestRun_ControlPlanePortInUse drives the errCh != nil branch in
// run(). Pre-fix the path was uncovered on every platform: smoke
// tests only hit the happy ListenAndServe → ErrServerClosed branch
// via SIGINT shutdown, never the "port already bound" path that
// returns a real net error from the goroutine. The branch matters
// because it is exactly how operators on a misconfigured Pod
// (PORT collision with a sidecar) discover the conflict; pre-fix
// the only signal was the slog.Error log and exitControlPlane.
func TestRun_ControlPlanePortInUse(t *testing.T) {
	t.Parallel()
	// Hold a port hostage so ListenAndServe gets EADDRINUSE. The
	// control plane binds the wildcard ":PORT" (see buildHTTPServer
	// in main.go), so we do the same here — on Windows a loopback
	// bind ("127.0.0.1:PORT") does NOT collide with a subsequent
	// wildcard bind, so this test would silently succeed without
	// actually exercising the errCh != nil branch we want to cover.
	hold, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("hold listen: %v", err)
	}
	t.Cleanup(func() { hold.Close() })
	port := hold.Addr().(*net.TCPAddr).Port

	specPath := setupSpecFile(t)
	env := mapEnv(map[string]string{
		"ENGINE_KIND":     "vllm",
		"MODEL_NAME":      "qwen2.5-7b",
		"MODEL_MODE":      "chat",
		"MODEL_SOURCE":    "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
		"PORT":            fmt.Sprintf("%d", port),
		"MODEL_SPEC_PATH": specPath,
		"RUN_DIR":         setupRunDir(t),
	})

	dir := t.TempDir()
	stdout, _ := os.Create(filepath.Join(dir, "out"))
	t.Cleanup(func() { stdout.Close() })
	stderr, _ := os.Create(filepath.Join(dir, "err"))
	t.Cleanup(func() { stderr.Close() })

	done := make(chan int, 1)
	go func() { done <- run(nil, stdout, stderr, env) }()

	select {
	case code := <-done:
		if code != exitControlPlane {
			t.Errorf("exit code = %d, want %d (exitControlPlane)", code, exitControlPlane)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("run() did not exit within 15s on EADDRINUSE; errCh wiring broken?")
	}
}

// TestRun_NilGetenv_FallsBackToOSGetenv covers the `getenv == nil`
// branch at the top of run() (line ~65 in main.go). Pre-fix the
// branch was technically dead in tests because every test wired a
// non-nil mapEnv, but main() *does* call run(..., nil, ...) in
// production — the branch is the only way os.Getenv ever gets
// consulted. We verify by invoking --version, which short-circuits
// before any config touching, so no real env vars are read; only
// the nil-fallback is exercised.
func TestRun_NilGetenv_FallsBackToOSGetenv(t *testing.T) {
	t.Parallel()
	var code int
	stdout, _ := captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--version"}, out, errF, nil)
	})
	if code != exitOK {
		t.Errorf("exit code = %d, want %d (nil getenv must default to os.Getenv)", code, exitOK)
	}
	if !strings.Contains(stdout, "llm-init") {
		t.Errorf("stdout missing version banner: %q", stdout)
	}
}

// TestRun_Healthcheck_DefaultPort exercises the `port == ""` branch
// in doHealthcheck (line ~318 in main.go) which falls back to the
// documented default port 8080. The branch was uncovered because the
// rest of the healthcheck tests always set PORT explicitly. We use a
// real listener on 8080 with t.Setenv("PORT", "") to force the
// default-port code path; the listener only has to answer /livez to
// satisfy the probe.
//
// NOT t.Parallel(): t.Setenv conflicts with parallel tests, and
// binding 8080 is a process-wide resource we don't want to race on.
func TestRun_Healthcheck_DefaultPort(t *testing.T) {
	// Bind 8080 directly. If something else has it on the dev host,
	// skip — this isn't a regression in run().
	l, err := net.Listen("tcp", "127.0.0.1:8080")
	if err != nil {
		t.Skipf("port 8080 in use; skipping default-port healthcheck test: %v", err)
	}
	srv := &http.Server{Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/livez" {
			w.WriteHeader(200)
			return
		}
		http.NotFound(w, r)
	})}
	go func() { _ = srv.Serve(l) }()
	t.Cleanup(func() { _ = srv.Close() })

	t.Setenv("PORT", "")
	var code int
	captureOutput(t, func(out, errF *os.File) {
		code = run([]string{"--healthcheck"}, out, errF, nil)
	})
	if code != exitOK {
		t.Errorf("default-port healthcheck: code = %d, want %d", code, exitOK)
	}
}

// TestBuildHTTPServer pins the timeout policy of the control-plane
// http.Server. We assert IdleTimeout > 0 (slowloris / idle-keep-alive
// hardening) and that ReadTimeout / WriteTimeout stay zero so SSE
// streaming on /v1/* and large /v1/embeddings batches are never
// truncated by a server-side deadline.
func TestBuildHTTPServer(t *testing.T) {
	t.Parallel()
	srv := buildHTTPServer(":0", http.NewServeMux())
	if srv.Addr != ":0" {
		t.Errorf("Addr=%q", srv.Addr)
	}
	if srv.Handler == nil {
		t.Error("Handler nil")
	}
	if srv.ReadHeaderTimeout <= 0 {
		t.Errorf("ReadHeaderTimeout=%v, want > 0", srv.ReadHeaderTimeout)
	}
	if srv.IdleTimeout <= 0 {
		t.Errorf("IdleTimeout=%v, want > 0", srv.IdleTimeout)
	}
	if srv.ReadTimeout != 0 {
		t.Errorf("ReadTimeout=%v, want 0 (large request bodies)", srv.ReadTimeout)
	}
	if srv.WriteTimeout != 0 {
		t.Errorf("WriteTimeout=%v, want 0 (SSE streaming)", srv.WriteTimeout)
	}
}
