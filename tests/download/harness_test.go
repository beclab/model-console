//go:build download

// Shared harness for the offline download end-to-end suite. Everything
// here is gated behind the `download` build tag; see doc.go.
package download

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/lifecycle"
	"github.com/llm-init/llm-init/internal/progress"
)

// fakeHFBin is the path to the compiled fakehf binary, built once in
// TestMain and used to drive the real hfwrap.Run subprocess path.
var fakeHFBin string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "fakehf-bin")
	if err != nil {
		fmt.Fprintln(os.Stderr, "mktemp:", err)
		os.Exit(1)
	}
	bin := filepath.Join(dir, "hf")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	build := exec.Command("go", "build", "-o", bin,
		"github.com/llm-init/llm-init/tests/download/cmd/fakehf")
	build.Stdout, build.Stderr = os.Stdout, os.Stderr
	if err := build.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "build fakehf:", err)
		os.Exit(1)
	}
	fakeHFBin = bin
	code := m.Run()
	_ = os.RemoveAll(dir)
	os.Exit(code)
}

// fakeAdapter is the engine stand-in for the url/hf channels (vLLM /
// llama.cpp / SGLang), which never pull and treat Register as a no-op.
// It records Register calls so tests can assert on the engine-facing
// file paths.
type fakeAdapter struct {
	kind     config.EngineKind
	regMu    sync.Mutex
	regFiles [][]string
	regKnown map[string]string
	regErr   error
	regCalls atomic.Int32
}

func (f *fakeAdapter) Kind() config.EngineKind         { return f.kind }
func (f *fakeAdapter) WaitAlive(context.Context) error { return nil }
func (f *fakeAdapter) AliveBeforeBoot() bool           { return false }
func (f *fakeAdapter) Pull(context.Context, string, progress.Sink) error {
	return nil
}
func (f *fakeAdapter) Ready(context.Context) (adapter.ReadyState, error) {
	return adapter.ReadyState{Alive: true, ModelExists: true}, nil
}

// Register must keep adapter.ModelInstaller's signature exactly: an
// adapter that misses it is silently handed the no-op installer, and the
// assertions below then read an engine that was never told anything.
func (f *fakeAdapter) Register(_ context.Context, files []string, known map[string]string, _ progress.Sink) error {
	f.regCalls.Add(1)
	f.regMu.Lock()
	f.regFiles = append(f.regFiles, append([]string(nil), files...))
	f.regKnown = known
	f.regMu.Unlock()
	return f.regErr
}
func (f *fakeAdapter) OpenAIHandler(config.Config) http.Handler    { return nil }
func (f *fakeAdapter) AnthropicHandler(config.Config) http.Handler { return nil }
func (f *fakeAdapter) EngineNativeStats(context.Context) (adapter.NativeStats, error) {
	return adapter.NativeStats{Source: "test", GPU: adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown}}, nil
}

func (f *fakeAdapter) lastRegister() []string {
	f.regMu.Lock()
	defer f.regMu.Unlock()
	if len(f.regFiles) == 0 {
		return nil
	}
	return f.regFiles[len(f.regFiles)-1]
}

// rig owns a running lifecycle.Manager driven through its real Run loop
// (a single background goroutine), plus the progress.Manager the
// frontend reads via /api/progress.
type rig struct {
	t      *testing.T
	pm     progress.Manager
	mgr    *lifecycle.Manager
	cancel context.CancelFunc
	done   chan error
}

// startManager wires defaults, constructs the Manager, and starts Run in
// the background. The fast RetriableBackoff keeps degraded<->download
// oscillation tight so waitPhase observes it quickly.
func startManager(t *testing.T, opts lifecycle.Options) *rig {
	t.Helper()
	if opts.Manager == nil {
		opts.Manager = progress.New(time.Now())
	}
	if opts.Downloader == nil {
		opts.Downloader = fetch.New(nil)
	}
	if opts.RetriableBackoff == 0 {
		opts.RetriableBackoff = 30 * time.Millisecond
	}
	if opts.HealthInterval == 0 {
		opts.HealthInterval = time.Hour
	}
	mgr, err := lifecycle.New(opts)
	if err != nil {
		t.Fatalf("lifecycle.New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	r := &rig{t: t, pm: opts.Manager, mgr: mgr, cancel: cancel, done: make(chan error, 1)}
	go func() { r.done <- mgr.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-r.done:
		case <-time.After(5 * time.Second):
			t.Log("warning: lifecycle Run did not return within 5s of cancel")
		}
	})
	return r
}

// waitPhase polls the progress snapshot until phase == want or timeout.
func (r *rig) waitPhase(want progress.Phase, timeout time.Duration) progress.State {
	r.t.Helper()
	deadline := time.Now().Add(timeout)
	var last progress.State
	for time.Now().Before(deadline) {
		last = r.pm.Snapshot()
		if last.Phase == want {
			return last
		}
		time.Sleep(5 * time.Millisecond)
	}
	r.t.Fatalf("phase %q not reached within %s (last phase=%q, last_error=%q)",
		want, timeout, last.Phase, last.LastError)
	return last
}

// waitUntil polls pred until it returns true or timeout.
func (r *rig) waitUntil(timeout time.Duration, pred func(progress.State) bool) progress.State {
	r.t.Helper()
	deadline := time.Now().Add(timeout)
	var last progress.State
	for time.Now().Before(deadline) {
		last = r.pm.Snapshot()
		if pred(last) {
			return last
		}
		time.Sleep(5 * time.Millisecond)
	}
	r.t.Fatalf("predicate not satisfied within %s (last phase=%q, last_error=%q)",
		timeout, last.Phase, last.LastError)
	return last
}

// --- config builders -------------------------------------------------

// baseConfig returns a Config skeleton with a writable model/run dir.
func baseConfig(t *testing.T, kind config.EngineKind, engineURL string) config.Config {
	t.Helper()
	return config.Config{
		Engine:    config.Engine{Kind: kind, URL: engineURL},
		Model:     config.Model{Name: "test-model", Type: config.ModelChat, Dir: t.TempDir()},
		ModelName: "test-model",
		Runtime: config.Runtime{
			Port:                   8080,
			RunDir:                 t.TempDir(),
			MaxConcurrentDownloads: 2,
		},
	}
}

func urlSource(url, sha, local string) config.ModelSource {
	return config.ModelSource{
		Index: 1, Kind: config.KindURL, Role: config.RoleMain,
		URL: url, URLSha256: sha, LocalPath: local,
	}
}

func ollamaURLSource(url, sha string) config.ModelSource {
	return config.ModelSource{
		Index: 1, Kind: config.KindOllamaURL, Role: config.RoleMain,
		URL: url, URLSha256: sha,
	}
}

func ollamaNameSource(tag string) config.ModelSource {
	return config.ModelSource{
		Index: 1, Kind: config.KindOllama, Role: config.RoleMain, OllamaTag: tag,
	}
}
