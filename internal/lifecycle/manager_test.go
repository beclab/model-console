package lifecycle

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/sentinel"
)

// fakeAdapter implements adapter.Adapter and adapter.ModelInstaller for
// lifecycle tests. Dropping either method makes it an engine that
// installs nothing, and every assertion on pullCalls / regCalls below
// reads zero.
type fakeAdapter struct {
	kind            config.EngineKind
	waitErr         error
	waitFn          func(context.Context) error
	readyState      adapter.ReadyState
	readyErr        error
	readyFn         func(context.Context) (adapter.ReadyState, error)
	pullCalls       atomic.Int32
	pullErr         error
	regCalls        atomic.Int32
	regErr          error
	regFiles        [][]string
	regKnown        map[string]string
	regMu           sync.Mutex
	aliveBeforeBoot *bool
}

func (f *fakeAdapter) Kind() config.EngineKind { return f.kind }
func (f *fakeAdapter) WaitAlive(ctx context.Context) error {
	if f.waitFn != nil {
		return f.waitFn(ctx)
	}
	return f.waitErr
}
func (f *fakeAdapter) AliveBeforeBoot() bool {
	if f.aliveBeforeBoot == nil {
		return true
	}
	return *f.aliveBeforeBoot
}
func (f *fakeAdapter) Ready(ctx context.Context) (adapter.ReadyState, error) {
	if f.readyFn != nil {
		return f.readyFn(ctx)
	}
	return f.readyState, f.readyErr
}
func (f *fakeAdapter) Pull(_ context.Context, _ string, _ progress.Sink) error {
	f.pullCalls.Add(1)
	return f.pullErr
}
func (f *fakeAdapter) Register(_ context.Context, files []string, known map[string]string, _ progress.Sink) error {
	f.regCalls.Add(1)
	f.regMu.Lock()
	f.regFiles = append(f.regFiles, files)
	f.regKnown = known
	f.regMu.Unlock()
	return f.regErr
}

// registered reports the files and digest hints of the last Register.
func (f *fakeAdapter) registered() ([][]string, map[string]string) {
	f.regMu.Lock()
	defer f.regMu.Unlock()
	return f.regFiles, f.regKnown
}
func (f *fakeAdapter) OpenAIHandler(config.Config) http.Handler    { return nil }
func (f *fakeAdapter) AnthropicHandler(config.Config) http.Handler { return nil }
func (f *fakeAdapter) EngineNativeStats(context.Context) (adapter.NativeStats, error) {
	return adapter.NativeStats{Source: "test", GPU: adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown}}, nil
}

// fakeDownloader writes a fixed payload to dest and reports success
// unless downloadErr is set.
type fakeDownloader struct {
	files       map[string][]byte
	downloadErr error
	verifyErr   error
	etag        string
	hits        atomic.Int32
}

func (d *fakeDownloader) Download(_ context.Context, opts fetch.RangeOptions, _ progress.Sink) error {
	d.hits.Add(1)
	if d.downloadErr != nil {
		if opts.OnRetry != nil {
			opts.OnRetry(1, d.downloadErr)
		}
		return d.downloadErr
	}
	body, ok := d.files[opts.URL]
	if !ok {
		body = []byte("test")
	}
	if err := os.MkdirAll(filepath.Dir(opts.DestPath), 0o755); err != nil {
		return err
	}
	if opts.OnMeta != nil {
		opts.OnMeta(int64(len(body)), d.etag)
	}
	return os.WriteFile(opts.DestPath, body, 0o644)
}

// Verify defers to the production implementation unless a test asked for
// a specific failure. The sidecar hit path decides whether to skip a
// download based on what this returns, so a stub that always agreed
// would make every reuse test pass without checking anything.
func (d *fakeDownloader) Verify(ctx context.Context, path string, size int64, sha string) error {
	if d.verifyErr != nil {
		return d.verifyErr
	}
	return fetch.New(nil).Verify(ctx, path, size, sha)
}

// hfRunRecorder captures every hfwrap.Run invocation so tests can
// assert on Repo / Revision / ForceReload / AllowPatterns.
type hfRunRecorder struct {
	mu    sync.Mutex
	calls []hfwrap.Config
	res   hfwrap.Result
	err   error
}

func (r *hfRunRecorder) run(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
	r.mu.Lock()
	r.calls = append(r.calls, c)
	r.mu.Unlock()
	return r.res, r.err
}

func (r *hfRunRecorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.calls)
}

func (r *hfRunRecorder) lastCall() hfwrap.Config {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) == 0 {
		return hfwrap.Config{}
	}
	return r.calls[len(r.calls)-1]
}

// hfSource builds a single ModelSource with Kind=KindHF.
func hfSource(repo, revision string, include ...string) config.ModelSource {
	return config.ModelSource{
		Index:      1,
		Kind:       config.KindHF,
		Role:       config.RoleMain,
		HFRepo:     repo,
		HFRevision: revision,
		HFInclude:  include,
	}
}

// urlSource builds a single ModelSource with Kind=KindURL.
func urlSource(url, sha, local string) config.ModelSource {
	return config.ModelSource{
		Index:     1,
		Kind:      config.KindURL,
		Role:      config.RoleMain,
		URL:       url,
		URLSha256: sha,
		LocalPath: local,
	}
}

// ollamaNameSource builds a single ModelSource with Kind=KindOllama.
func ollamaNameSource(tag string) config.ModelSource {
	return config.ModelSource{
		Index:     1,
		Kind:      config.KindOllama,
		Role:      config.RoleMain,
		OllamaTag: tag,
	}
}

func ollamaExtraSource(index int, tag string) config.ModelSource {
	return config.ModelSource{
		Index:     index,
		Kind:      config.KindOllama,
		Role:      config.RoleExtra,
		OllamaTag: tag,
	}
}

// ollamaURLSource builds a single ModelSource with Kind=KindOllamaURL.
func ollamaURLSource(url, sha string) config.ModelSource {
	return config.ModelSource{
		Index:     1,
		Kind:      config.KindOllamaURL,
		Role:      config.RoleMain,
		URL:       url,
		URLSha256: sha,
	}
}

// baseOptions sets up a minimal Options that ensure() can drive without
// blowing up on nil deps.
func baseOptions(t *testing.T, sources []config.ModelSource) Options {
	t.Helper()
	dir := t.TempDir()
	cfg := config.Config{
		Engine:  config.Engine{Kind: config.EngineLlamaCpp, URL: "http://127.0.0.1:0"},
		Model:   config.Model{Name: "m", Type: config.ModelChat, Dir: dir},
		Sources: sources,
		Runtime: config.Runtime{
			Port:                   8080,
			RunDir:                 t.TempDir(),
			MaxConcurrentDownloads: 2,
		},
	}
	return Options{
		Config:                 cfg,
		Adapter:                &fakeAdapter{kind: cfg.Engine.Kind},
		Manager:                progress.New(time.Now()),
		Downloader:             &fakeDownloader{},
		MaxConcurrentDownloads: 2,
		RetriableBackoff:       10 * time.Millisecond,
		NowFunc:                time.Now,
	}
}

// notAnInstaller is an engine with nothing to install, which is every
// engine but Ollama. Embedding the interface rather than the fake is
// what makes that true: only Adapter's methods are promoted, so
// Pull and Register are out of reach.
type notAnInstaller struct{ adapter.Adapter }

// Three of the four engines read the path they were launched with, and
// lifecycle drives them through the same ensure as Ollama. It resolves
// an installer once and calls it unconditionally, so an engine that
// implements none of it has to arrive at a no-op rather than a nil.
func TestEnsure_AnEngineThatInstallsNothing(t *testing.T) {
	t.Parallel()
	fake := &fakeAdapter{kind: config.EngineLlamaCpp}
	o := baseOptions(t, []config.ModelSource{hfSource("owner/repo", "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567")})
	o.Adapter = notAnInstaller{Adapter: fake}
	rec := &hfRunRecorder{res: hfwrap.Result{Status: "ok", Path: o.Config.Model.Dir, Commit: "abc"}}
	o.HFRun = rec.run
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if got := fake.regCalls.Load(); got != 0 {
		t.Errorf("Register calls = %d, want 0: the fake is unreachable through this adapter", got)
	}
}

// TestSetPhase_DownloadResetsCounters pins the PhaseDownload entry
// contract: per-download gauges and LastError are zeroed so a recovered
// retry does not keep the Status card red; durable counters
// (RetryCount / TransportRetries) stay for operators.
func TestSetPhase_DownloadResetsCounters(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{hfSource("a/b", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")})
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	o.Manager.Update(func(s *progress.State) {
		s.Phase = progress.PhaseFailed
		s.BytesTotal = 1_000_000
		s.BytesCompleted = 500_000
		s.SpeedBytesPerSec = 1024
		s.ETASeconds = 60
		s.LastError = "prior error"
		s.RetryCount = 3
		s.TransportRetries = 7
	})
	mgr.setPhase(progress.PhaseDownload)
	st := o.Manager.Snapshot()
	if st.Phase != progress.PhaseDownload {
		t.Errorf("Phase = %v, want download", st.Phase)
	}
	if st.BytesTotal != 0 {
		t.Errorf("BytesTotal = %d, want 0", st.BytesTotal)
	}
	if st.BytesCompleted != 0 {
		t.Errorf("BytesCompleted = %d, want 0", st.BytesCompleted)
	}
	if st.SpeedBytesPerSec != 0 {
		t.Errorf("SpeedBytesPerSec = %v, want 0", st.SpeedBytesPerSec)
	}
	if st.ETASeconds != 0 {
		t.Errorf("ETASeconds = %d, want 0", st.ETASeconds)
	}
	if st.LastError != "" {
		t.Errorf("LastError = %q, want empty (cleared on download re-entry)", st.LastError)
	}
	if st.RetryCount != 3 {
		t.Errorf("RetryCount = %d, want 3 (preserved)", st.RetryCount)
	}
	if st.TransportRetries != 7 {
		t.Errorf("TransportRetries = %d, want 7 (preserved)", st.TransportRetries)
	}
}

// TestEnsureOllamaName_ExtraPreload pulls ROLE=extra tags before the main
// tag and does not flip PhaseReady until the main pull completes.
func TestEnsureOllamaName_ExtraPreload(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		ollamaNameSource("qwen3:0.6b"),
		ollamaExtraSource(2, "llama3:8b"),
	})
	o.Config.Engine.Kind = config.EngineOllama
	ad := &fakeAdapter{kind: config.EngineOllama}
	o.Adapter = ad
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if ad.pullCalls.Load() != 2 {
		t.Errorf("Pull calls = %d, want 2 (extra + main)", ad.pullCalls.Load())
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseReady {
		t.Errorf("phase = %v, want ready", got)
	}
}

// TestEnsureOllamaName drives the ollama:// Library tag branch. The
// daemon owns the model bytes; lifecycle just calls Adapter.Pull and
// flips PhaseReady.
func TestEnsureOllamaName(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{ollamaNameSource("qwen2.5:7b")})
	o.Config.Engine.Kind = config.EngineOllama
	ad := &fakeAdapter{kind: config.EngineOllama}
	o.Adapter = ad
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if ad.pullCalls.Load() != 1 {
		t.Errorf("Pull calls = %d, want 1", ad.pullCalls.Load())
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseReady {
		t.Errorf("phase = %v, want ready", got)
	}
}

// TestEnsureHF_HappyPath asserts the new HF dispatch: ms.Kind=KindHF,
// lifecycle invokes hfwrap.Run, then Register, then stops at PhaseLoading
// (proxy engines reach PhaseReady only after manager.Run's WaitAlive).
func TestEnsureHF_HappyPath(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		hfSource("Qwen/Qwen3-27B", "0123456789abcdef0123456789abcdef01234567"),
	})
	rec := &hfRunRecorder{res: hfwrap.Result{Status: "ok", Path: o.Config.Model.Dir, Commit: "abc"}}
	o.HFRun = rec.run

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if len(rec.calls) != 1 {
		t.Fatalf("expected 1 hfwrap call, got %d", len(rec.calls))
	}
	c := rec.lastCall()
	if c.Repo != "Qwen/Qwen3-27B" {
		t.Errorf("Repo = %q", c.Repo)
	}
	if c.Revision != "0123456789abcdef0123456789abcdef01234567" {
		t.Errorf("Revision = %q", c.Revision)
	}
	if c.CacheDir == "" {
		t.Errorf("CacheDir should be set, got empty")
	}
	if c.ForceReload {
		t.Error("ForceReload should default to false")
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseLoading {
		t.Errorf("phase = %v want loading", got)
	}
	if ad, ok := o.Adapter.(*fakeAdapter); ok {
		if ad.regCalls.Load() != 1 {
			t.Errorf("Register calls = %d want 1", ad.regCalls.Load())
		}
	}
}

func TestEnsureHF_PreservesAggregateProgressForExtras(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/owner/main/tree/main":
			_, _ = io.WriteString(w, `[{"path":"main.bin","size":100,"type":"file"}]`)
		case "/api/models/owner/extra/tree/main":
			_, _ = io.WriteString(w, `[{"path":"extra.bin","size":200,"type":"file"}]`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	main := hfSource("owner/main", "main", "main.bin")
	extra := hfSource("owner/extra", "main", "extra.bin")
	extra.Index = 2
	extra.Role = config.RoleExtra
	o := baseOptions(t, []config.ModelSource{main, extra})
	o.Config.HFEndpoint = server.URL

	mainPath := filepath.Join(t.TempDir(), "main.bin")
	extraPath := filepath.Join(t.TempDir(), "extra.bin")
	if err := os.WriteFile(mainPath, make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(extraPath, make([]byte, 200), 0o644); err != nil {
		t.Fatal(err)
	}
	o.HFRun = func(_ context.Context, cfg hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		switch cfg.Repo {
		case "owner/main":
			return hfwrap.Result{Status: "ok", Path: mainPath}, nil
		case "owner/extra":
			return hfwrap.Result{Status: "ok", Path: extraPath}, nil
		default:
			return hfwrap.Result{}, errors.New("unexpected HF repo " + cfg.Repo)
		}
	}
	ad := o.Adapter.(*fakeAdapter)

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	st := o.Manager.Snapshot()
	if st.Phase != progress.PhaseLoading {
		t.Fatalf("Phase = %v, want loading", st.Phase)
	}
	if st.BytesTotal != 300 || st.BytesCompleted != 300 {
		t.Fatalf("progress = %d/%d, want 300/300", st.BytesCompleted, st.BytesTotal)
	}
	ad.regMu.Lock()
	defer ad.regMu.Unlock()
	if len(ad.regFiles) != 1 || len(ad.regFiles[0]) != 1 || ad.regFiles[0][0] != mainPath {
		t.Fatalf("Register files = %v, want only main path %q", ad.regFiles, mainPath)
	}
}

func TestEnsureHF_ReconcilesUnestimableCachedMainAndExtra(t *testing.T) {
	t.Parallel()
	main := hfSource("owner/main", "main")
	extra := hfSource("owner/extra", "main")
	extra.Index = 2
	extra.Role = config.RoleExtra
	o := baseOptions(t, []config.ModelSource{main, extra})

	mainDir := filepath.Join(t.TempDir(), "main")
	extraDir := filepath.Join(t.TempDir(), "extra")
	if err := os.MkdirAll(mainDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(extraDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(mainDir, defaultURLModelFilename), make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(extraDir, "adapter.bin"), make([]byte, 200), 0o644); err != nil {
		t.Fatal(err)
	}
	o.HFRun = func(_ context.Context, cfg hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		switch cfg.Repo {
		case "owner/main":
			return hfwrap.Result{Status: "ok", Path: mainDir}, nil
		case "owner/extra":
			return hfwrap.Result{Status: "ok", Path: extraDir}, nil
		default:
			return hfwrap.Result{}, errors.New("unexpected HF repo " + cfg.Repo)
		}
	}
	ad := o.Adapter.(*fakeAdapter)

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	st := o.Manager.Snapshot()
	if st.BytesTotal != 300 || st.BytesCompleted != 300 {
		t.Errorf("progress = %d/%d, want 300/300", st.BytesCompleted, st.BytesTotal)
	}
	ad.regMu.Lock()
	if len(ad.regFiles) != 1 || len(ad.regFiles[0]) != 1 || ad.regFiles[0][0] != mainDir {
		t.Errorf("Register files = %v, want only main path %q", ad.regFiles, mainDir)
	}
	ad.regMu.Unlock()
	sentinelPath, err := os.ReadFile(filepath.Join(o.Config.Runtime.RunDir, "model_path"))
	if err != nil {
		t.Fatalf("read sentinel model_path: %v", err)
	}
	if string(sentinelPath) != mainDir+"\n" {
		t.Errorf("sentinel model_path = %q, want %q", sentinelPath, mainDir)
	}
}

// TestEnsureHF_ForceForwardsToWrapper proves that POST /api/retry?force=true
// flips the wrapper's --force-download flag.
func TestEnsureHF_ForceForwardsToWrapper(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		hfSource("x/y", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
	})
	rec := &hfRunRecorder{res: hfwrap.Result{Status: "ok"}}
	o.HFRun = rec.run

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mgr.RetryWith(RetryOptions{Force: true})

	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if !rec.lastCall().ForceReload {
		t.Error("ForceReload should propagate from RetryOptions.Force")
	}
}

// TestEnsureHF_RetriableErrorYieldsDegraded checks that hfwrap.Error
// with Retriable=true (CodeNetwork etc.) lands the manager in
// PhaseDegraded with retriableErr wrapping.
func TestEnsureHF_RetriableErrorYieldsDegraded(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		hfSource("x/y", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
	})
	o.HFRun = func(context.Context, hfwrap.Config, progress.Sink) (hfwrap.Result, error) {
		return hfwrap.Result{}, &hfwrap.Error{Code: hfwrap.CodeNetwork, Retriable: true,
			Stage: "progress", Message: "503 upstream"}
	}
	mgr, _ := New(o)
	err := mgr.ensure(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.As(err, new(retriableErr)) {
		t.Errorf("expected retriableErr, got %T", err)
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseDegraded {
		t.Errorf("phase = %v want degraded", got)
	}
}

// TestEnsureHF_PermanentErrorYieldsFailed asserts that a non-retriable
// hfwrap.Error surfaces as PhaseFailed without wrapping retriableErr.
func TestEnsureHF_PermanentErrorYieldsFailed(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		hfSource("x/y", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
	})
	o.HFRun = func(context.Context, hfwrap.Config, progress.Sink) (hfwrap.Result, error) {
		return hfwrap.Result{}, &hfwrap.Error{Code: hfwrap.CodeRepoNotFound,
			Retriable: false, Stage: "error", Message: "repo not found"}
	}
	mgr, _ := New(o)
	err := mgr.ensure(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if errors.As(err, new(retriableErr)) {
		t.Error("CodeRepoNotFound should be permanent, not retriableErr")
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseFailed {
		t.Errorf("phase = %v want failed", got)
	}
}

// TestEnsureHF_RegisterErrorYieldsFailed asserts the all_fail policy for
// the loading phase: an Adapter.Register failure is terminal (PhaseFailed)
// and is not wrapped as retriableErr, so the manager waits for /api/retry
// instead of auto-retrying via PhaseDegraded.
func TestEnsureHF_RegisterErrorYieldsFailed(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		hfSource("Qwen/Qwen3-27B", "0123456789abcdef0123456789abcdef01234567"),
	})
	rec := &hfRunRecorder{res: hfwrap.Result{Status: "ok", Path: o.Config.Model.Dir, Commit: "abc"}}
	o.HFRun = rec.run
	ad := &fakeAdapter{kind: config.EngineVLLM, regErr: errors.New("register boom")}
	o.Adapter = ad

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	err = mgr.ensure(context.Background())
	if err == nil {
		t.Fatal("expected error from Register failure")
	}
	if errors.As(err, new(retriableErr)) {
		t.Error("Register failure in loading should be permanent, not retriableErr")
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseFailed {
		t.Errorf("phase = %v want failed", got)
	}
}

// TestEnsureURL_HappyPath drives a single-file URL source and checks
// Adapter.Register receives the absolute file path.
func TestEnsureURL_HappyPath(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dest := filepath.Join(dir, "m.bin")
	o := baseOptions(t, []config.ModelSource{urlSource("https://example.com/m.bin", "", dest)})
	body := []byte("model bytes")
	dl := &fakeDownloader{files: map[string][]byte{"https://example.com/m.bin": body}}
	o.Downloader = dl
	ad := &fakeAdapter{kind: config.EngineLlamaCpp}
	o.Adapter = ad

	mgr, _ := New(o)
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if ad.regCalls.Load() != 1 {
		t.Errorf("Register calls = %d want 1", ad.regCalls.Load())
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseLoading {
		t.Errorf("phase = %v want loading", got)
	}
	if data, err := os.ReadFile(dest); err != nil {
		t.Errorf("download missing: %v", err)
	} else if string(data) != string(body) {
		t.Errorf("download body = %q want %q", data, body)
	}
}

// TestEnsureURL_ForceWipesAndRedownloads asserts that Force=true deletes
// the existing file before invoking the downloader.
func TestEnsureURL_ForceWipesAndRedownloads(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dest := filepath.Join(dir, "m.bin")
	if err := os.WriteFile(dest, []byte("STALE"), 0o644); err != nil {
		t.Fatal(err)
	}
	o := baseOptions(t, []config.ModelSource{urlSource("https://example.com/m.bin", "", dest)})
	dl := &fakeDownloader{files: map[string][]byte{"https://example.com/m.bin": []byte("FRESH")}}
	o.Downloader = dl
	mgr, _ := New(o)
	mgr.RetryWith(RetryOptions{Force: true})
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	got, _ := os.ReadFile(dest)
	if string(got) != "FRESH" {
		t.Errorf("body = %q want FRESH", got)
	}
}

// probeDuringDownload records what the run dir looked like at the one
// moment that matters: after the force pass deleted the old bytes and
// before it wrote the new ones.
type probeDuringDownload struct {
	*fakeDownloader
	runDir      string
	sawSentinel bool
}

func (d *probeDuringDownload) Download(ctx context.Context, opts fetch.RangeOptions, sink progress.Sink) error {
	if _, err := os.Stat(filepath.Join(d.runDir, sentinel.FinishFile)); err == nil {
		d.sawSentinel = true
	}
	return d.fakeDownloader.Download(ctx, opts, sink)
}

// A force pass deletes the exact file model_path names, and the download
// that replaces it takes minutes for a real model. A wrapper that starts
// in that window — a kubelet restart, a CrashLoop, a supervise loop on
// its first launch — reads the sentinel, is told the model is ready, and
// execs a path that is not there. Clearing the sentinel first makes it
// wait for the pass instead.
func TestEnsureURL_ForceClearsTheSentinelBeforeDeletingTheBytes(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dest := filepath.Join(dir, "m.bin")
	if err := os.WriteFile(dest, []byte("STALE"), 0o644); err != nil {
		t.Fatal(err)
	}
	o := baseOptions(t, []config.ModelSource{urlSource("https://example.com/m.bin", "", dest)})
	runDir := o.Config.Runtime.RunDir
	if err := sentinel.Write(runDir, dest); err != nil {
		t.Fatalf("seed sentinel: %v", err)
	}
	dl := &probeDuringDownload{
		fakeDownloader: &fakeDownloader{files: map[string][]byte{
			"https://example.com/m.bin": []byte("FRESH"),
		}},
		runDir: runDir,
	}
	o.Downloader = dl

	mgr, _ := New(o)
	mgr.RetryWith(RetryOptions{Force: true})
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	if dl.sawSentinel {
		t.Error("the sentinel still said 'ready, load this' while the file it named was gone")
	}
	// And the pass puts it back, so nothing is left waiting.
	if _, err := os.Stat(filepath.Join(runDir, sentinel.FinishFile)); err != nil {
		t.Errorf("sentinel not rewritten after a successful force pass: %v", err)
	}
}

// A payload and its real digest. The #sha256= fragment is compared
// against the bytes for real, so the two must agree.
const (
	ollamaBlobBody  = "ollama blob bytes"
	shaOfOllamaBlob = "19b725c02fec8bd69a9c720dbb6dc3db1b7a02f14ff8f05868551a2591f15f9d"
)

// TestEnsureOllamaURL_DriveBlobUpload asserts that ollama-URL sources
// download into RUN_DIR/url-fetched and feed Adapter.Register with the
// resulting file path.
func TestEnsureOllamaURL_DriveBlobUpload(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		ollamaURLSource("https://example.com/x.gguf", shaOfOllamaBlob),
	})
	o.Config.Engine.Kind = config.EngineOllama
	ad := &fakeAdapter{kind: config.EngineOllama}
	o.Adapter = ad
	dl := &fakeDownloader{files: map[string][]byte{
		"https://example.com/x.gguf": []byte(ollamaBlobBody),
	}}
	o.Downloader = dl
	mgr, _ := New(o)
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if ad.regCalls.Load() != 1 {
		t.Errorf("Register calls = %d want 1", ad.regCalls.Load())
	}
	ad.regMu.Lock()
	calls := ad.regFiles
	ad.regMu.Unlock()
	if len(calls) != 1 || len(calls[0]) != 1 {
		t.Fatalf("Register: want 1 call w/ 1 file, got %v", calls)
	}
	wantDir := filepath.Join(o.Config.Runtime.RunDir, "url-fetched")
	if got := filepath.Dir(calls[0][0]); got != wantDir {
		t.Errorf("blob dir = %q want %q", got, wantDir)
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseReady {
		t.Errorf("phase = %v want ready", got)
	}
}

// TestClassifyDownloadErr exercises the metric-label taxonomy.
func TestClassifyDownloadErr(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, errCodeOther},
		{"429", &fetch.HTTPStatusError{StatusCode: 429}, errCode429},
		{"503", &fetch.HTTPStatusError{StatusCode: 503}, errCode5xx},
		{"timeout", &net.OpError{Op: "read", Err: &timeoutErr{}}, errCodeNetTimeout},
		{"eof", io.EOF, errCodeEOF},
		{"unexpected_eof", io.ErrUnexpectedEOF, errCodeEOF},
		{"econnreset", syscall.ECONNRESET, errCodeEOF},
		{"other", errors.New("boom"), errCodeOther},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := classifyDownloadErr(tc.err); got != tc.want {
				t.Errorf("classifyDownloadErr(%v) = %q want %q", tc.err, got, tc.want)
			}
		})
	}
}

// TestClassifyHFWrapErr asserts the HF-wrapper-specific labels.
func TestClassifyHFWrapErr(t *testing.T) {
	t.Parallel()
	cases := []struct {
		code int
		want string
	}{
		{hfwrap.CodeRepoNotFound, errCodeHFRepoNotFound},
		{hfwrap.CodeGated, errCodeHFGated},
		{hfwrap.CodeRevision, errCodeHFRevision},
		{hfwrap.CodeTokenMissing, errCodeHFTokenMissing},
		{hfwrap.CodeNetwork, errCodeHFNetwork},
		{hfwrap.CodeProcessKilled, errCodeHFNetwork},
		{hfwrap.CodeDiskFull, errCodeHFDiskFull},
		{hfwrap.CodeInternal, errCodeHFInternal},
	}
	for _, tc := range cases {
		tc := tc
		got := classifyHFWrapErr(&hfwrap.Error{Code: tc.code})
		if got != tc.want {
			t.Errorf("code %d -> %q want %q", tc.code, got, tc.want)
		}
	}
	if got := classifyHFWrapErr(io.EOF); got != errCodeEOF {
		t.Errorf("EOF should fall through to classifyDownloadErr, got %q", got)
	}
}

// TestRun_ProxyReEnsureRestoresReady asserts that proxy engines return to
// PhaseReady after a periodic or manual re-ensure. ensureHF sets
// PhaseLoading on every pass; Run must WaitAlive and flip ready each time.
func TestRun_ProxyReEnsureRestoresReady(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{
		hfSource("Qwen/Qwen3-27B", "0123456789abcdef0123456789abcdef01234567"),
	})
	o.Config.Engine.Kind = config.EngineVLLM
	aliveBeforeBoot := false
	waitCalls := atomic.Int32{}
	ad := &fakeAdapter{
		kind:            config.EngineVLLM,
		aliveBeforeBoot: &aliveBeforeBoot,
		readyState:      adapter.ReadyState{Alive: true, ModelExists: true},
		waitFn: func(context.Context) error {
			waitCalls.Add(1)
			return nil
		},
	}
	o.Adapter = ad
	rec := &hfRunRecorder{res: hfwrap.Result{Status: "ok", Path: o.Config.Model.Dir, Commit: "abc"}}
	o.HFRun = rec.run

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() { done <- mgr.Run(ctx) }()

	waitPhase := func(want progress.Phase, timeout time.Duration) bool {
		deadline := time.Now().Add(timeout)
		for time.Now().Before(deadline) {
			if o.Manager.Snapshot().Phase == want {
				return true
			}
			time.Sleep(5 * time.Millisecond)
		}
		return false
	}

	if !waitPhase(progress.PhaseReady, 2*time.Second) {
		t.Fatalf("first pass: phase = %v, want ready", o.Manager.Snapshot().Phase)
	}
	if waitCalls.Load() != 1 {
		t.Fatalf("WaitAlive after first pass = %d, want 1", waitCalls.Load())
	}

	// Retry() closes a wake channel; if we signal before Run enters
	// waitForRetryOrCtx the close is lost. Loop Retry until the second
	// ensure pass completes (same idempotency prod relies on).
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		rec.mu.Lock()
		ensureRuns := len(rec.calls)
		rec.mu.Unlock()
		if waitCalls.Load() >= 2 && ensureRuns >= 2 &&
			o.Manager.Snapshot().Phase == progress.PhaseReady {
			break
		}
		mgr.Retry()
		time.Sleep(10 * time.Millisecond)
	}
	if waitCalls.Load() != 2 {
		t.Errorf("WaitAlive after re-ensure = %d, want 2", waitCalls.Load())
	}
	rec.mu.Lock()
	ensureRuns := len(rec.calls)
	rec.mu.Unlock()
	if ensureRuns != 2 {
		t.Errorf("ensure/HF runs = %d, want 2", ensureRuns)
	}
	if o.Manager.Snapshot().Phase != progress.PhaseReady {
		t.Errorf("after re-ensure: phase = %v, want ready", o.Manager.Snapshot().Phase)
	}

	cancel()
	if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
		t.Errorf("Run returned %v", err)
	}
}

// A ready model must not re-enter ensure on a timer nobody asked for.
// The pass opens with setPhase(download), which bounces every /v1/*
// request with 503, and it calls the upstream registry — so an hourly
// default took a model that was complete on disk offline once an hour
// and degraded it whenever the registry was unreachable.
func TestRun_ReadyModelStopsAfterOnePass(t *testing.T) {
	t.Parallel()
	mgr, o, rec, cancel := runUntilReady(t)
	defer cancel()

	time.Sleep(300 * time.Millisecond)
	if got := rec.count(); got != 1 {
		t.Errorf("ensure passes = %d, want 1 — something re-ran ensure on its own", got)
	}
	if got := o.Manager.Snapshot().Phase; got != progress.PhaseReady {
		t.Errorf("phase = %v, want ready", got)
	}
	mgr.Stop()
}

// runUntilReady starts Run on a proxy-shaped adapter and returns once the
// first pass has reached ready.
func runUntilReady(t *testing.T) (*Manager, Options, *hfRunRecorder, context.CancelFunc) {
	t.Helper()
	o := baseOptions(t, []config.ModelSource{
		hfSource("Qwen/Qwen3-27B", "0123456789abcdef0123456789abcdef01234567"),
	})
	o.Config.Engine.Kind = config.EngineVLLM
	aliveBeforeBoot := false
	o.Adapter = &fakeAdapter{
		kind:            config.EngineVLLM,
		aliveBeforeBoot: &aliveBeforeBoot,
		readyState:      adapter.ReadyState{Alive: true, ModelExists: true},
	}
	rec := &hfRunRecorder{res: hfwrap.Result{Status: "ok", Path: o.Config.Model.Dir, Commit: "abc"}}
	o.HFRun = rec.run

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = mgr.Run(ctx) }()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if o.Manager.Snapshot().Phase == progress.PhaseReady {
			return mgr, o, rec, cancel
		}
		time.Sleep(5 * time.Millisecond)
	}
	cancel()
	t.Fatalf("first pass: phase = %v, want ready", o.Manager.Snapshot().Phase)
	return nil, o, nil, cancel
}

// TestRunRejectsRunAfterStop verifies the post-Stop guard.
func TestRunRejectsRunAfterStop(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{ollamaNameSource("x")})
	o.Config.Engine.Kind = config.EngineOllama
	mgr, _ := New(o)
	mgr.Stop()
	if err := mgr.Run(context.Background()); err == nil {
		t.Error("Run after Stop should error")
	}
}

// TestRetryNonBlocking verifies Retry / RetryWith never block even when
// no Run is in flight (channel-swap idempotency).
func TestRetryNonBlocking(t *testing.T) {
	t.Parallel()
	o := baseOptions(t, []config.ModelSource{ollamaNameSource("x")})
	mgr, _ := New(o)
	mgr.Retry()
	mgr.Retry()
	mgr.RetryWith(RetryOptions{Force: true, Level: "ignored"})
	got := mgr.consumePendingOpts()
	if !got.Force {
		t.Errorf("consumed opts should reflect last write: %+v", got)
	}
	if again := mgr.consumePendingOpts(); again != (RetryOptions{}) {
		t.Errorf("second consume should be zero, got %+v", again)
	}
}

// timeoutErr satisfies net.Error.Timeout() for classifyDownloadErr.
type timeoutErr struct{}

func (timeoutErr) Error() string { return "timeout" }
func (timeoutErr) Timeout() bool { return true }

func TestOnDiskSize(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.bin"), make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(dir, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sub, "b.bin"), make([]byte, 50), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := onDiskSize(dir); got != 150 {
		t.Errorf("onDiskSize(dir) = %d, want 150", got)
	}
	if got := onDiskSize(filepath.Join(dir, "a.bin")); got != 100 {
		t.Errorf("onDiskSize(file) = %d, want 100", got)
	}

	// HF snapshot dirs symlink into blobs/: the size must follow the
	// link to the real blob, not count the ~0-byte link inode.
	blob := filepath.Join(dir, "blob.bin")
	if err := os.WriteFile(blob, make([]byte, 200), 0o644); err != nil {
		t.Fatal(err)
	}
	snap := filepath.Join(dir, "snapshot")
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(blob, filepath.Join(snap, defaultURLModelFilename)); err != nil {
		t.Skipf("symlinks unsupported on this platform: %v", err)
	}
	if got := onDiskSize(snap); got != 200 {
		t.Errorf("onDiskSize(symlinked snapshot) = %d, want 200 (must follow symlink)", got)
	}
}

// TestEnsureHF_ExtraWritesExtraModelPath asserts ROLE=extra HF paths land in
// extra_model_path (first line = first extra), while model_path stays on main
// and ROLE=mmproj is never written there.
func TestEnsureHF_ExtraWritesExtraModelPath(t *testing.T) {
	t.Parallel()
	runDir := t.TempDir()
	mainPath := filepath.Join(runDir, "snapshots", "sha", "main.gguf")
	extraPath := filepath.Join(runDir, "snapshots", "sha", "onnx")
	if err := os.MkdirAll(filepath.Dir(mainPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mainPath, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(extraPath, 0o755); err != nil {
		t.Fatal(err)
	}

	main := hfSource("owner/main", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "main.gguf")
	// Extra is a layout ONNX tree: whole-repo (or pattern) download + --subdir.
	extra := config.ModelSource{
		Index:      2,
		Kind:       config.KindHF,
		Role:       config.RoleExtra,
		HFRepo:     "owner/layout",
		HFRevision: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
		HFSubdir:   "onnx",
	}

	o := baseOptions(t, []config.ModelSource{main, extra})
	o.Config.Runtime.RunDir = runDir
	o.HFRun = func(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		if c.Repo == "owner/main" {
			return hfwrap.Result{Status: "ok", Path: mainPath}, nil
		}
		// snapshot root (no single HFFile)
		return hfwrap.Result{Status: "ok", Path: filepath.Dir(extraPath)}, nil
	}

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	gotMain, err := sentinel.ReadModelPath(runDir)
	if err != nil {
		t.Fatalf("ReadModelPath: %v", err)
	}
	if gotMain != mainPath {
		t.Errorf("model_path = %q, want %q", gotMain, mainPath)
	}
	gotExtra, err := sentinel.ReadExtraPaths(runDir)
	if err != nil {
		t.Fatalf("ReadExtraPaths: %v", err)
	}
	if len(gotExtra) != 1 || gotExtra[0] != extraPath {
		t.Fatalf("extra_model_path = %v, want [%q]", gotExtra, extraPath)
	}
}

// TestEnsureHF_NoExtraOmitsExtraModelPath: single-source HF must not leave
// an extra_model_path file (and clears a stale one from a prior run).
func TestEnsureHF_NoExtraOmitsExtraModelPath(t *testing.T) {
	t.Parallel()
	runDir := t.TempDir()
	mainPath := filepath.Join(runDir, "model.gguf")
	if err := os.WriteFile(mainPath, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := sentinel.WriteExtra(runDir, []string{"/stale/onnx"}); err != nil {
		t.Fatal(err)
	}

	o := baseOptions(t, []config.ModelSource{
		hfSource("a/b", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "model.gguf"),
	})
	o.Config.Runtime.RunDir = runDir
	o.HFRun = func(context.Context, hfwrap.Config, progress.Sink) (hfwrap.Result, error) {
		return hfwrap.Result{Status: "ok", Path: mainPath}, nil
	}
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if _, err := sentinel.ReadExtraPaths(runDir); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("extra_model_path should be absent, got %v", err)
	}
}

// TestEnsureHF_MmprojNotInExtraModelPath: ROLE=mmproj is downloaded inline
// and must not appear in extra_model_path.
func TestEnsureHF_MmprojNotInExtraModelPath(t *testing.T) {
	t.Parallel()
	runDir := t.TempDir()
	snap := filepath.Join(runDir, "snap")
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	mainPath := filepath.Join(snap, "main.gguf")
	mmprojPath := filepath.Join(snap, "mmproj.gguf")
	for _, p := range []string{mainPath, mmprojPath} {
		if err := os.WriteFile(p, []byte("g"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	main := hfSource("o/r", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "main.gguf")
	mmproj := config.ModelSource{
		Index:      2,
		Kind:       config.KindHF,
		Role:       config.RoleMmproj,
		HFRepo:     "o/r",
		HFRevision: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
		HFInclude:  []string{"mmproj.gguf"},
	}
	o := baseOptions(t, []config.ModelSource{main, mmproj})
	o.Config.Runtime.RunDir = runDir
	o.HFRun = func(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		if c.HFFile == "mmproj.gguf" {
			return hfwrap.Result{Status: "ok", Path: mmprojPath}, nil
		}
		return hfwrap.Result{Status: "ok", Path: mainPath}, nil
	}
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	if _, err := sentinel.ReadExtraPaths(runDir); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("mmproj must not create extra_model_path, got %v", err)
	}
	gotMain, err := sentinel.ReadModelPath(runDir)
	if err != nil || gotMain != mainPath {
		t.Errorf("model_path = %q err=%v", gotMain, err)
	}
}

// TestEnsureHF_MmprojFromAnotherRepoReachesRegister: the projector path comes
// from its own hf pass, so a projector published in a different repo than the
// main weights still reaches the engine and counts towards progress.
func TestEnsureHF_MmprojFromAnotherRepoReachesRegister(t *testing.T) {
	t.Parallel()
	mainSnap := filepath.Join(t.TempDir(), "main-snap")
	projSnap := filepath.Join(t.TempDir(), "proj-snap")
	for _, d := range []string{mainSnap, projSnap} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	mainPath := filepath.Join(mainSnap, "main.gguf")
	mmprojPath := filepath.Join(projSnap, "mmproj.gguf")
	if err := os.WriteFile(mainPath, make([]byte, 100), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mmprojPath, make([]byte, 200), 0o644); err != nil {
		t.Fatal(err)
	}

	main := hfSource("owner/main", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "main.gguf")
	mmproj := config.ModelSource{
		Index:      2,
		Kind:       config.KindHF,
		Role:       config.RoleMmproj,
		HFRepo:     "owner/proj",
		HFRevision: "cafecafecafecafecafecafecafecafecafecafe",
		HFInclude:  []string{"mmproj.gguf"},
	}
	o := baseOptions(t, []config.ModelSource{main, mmproj})
	o.HFRun = func(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
		switch c.Repo {
		case "owner/main":
			return hfwrap.Result{Status: "ok", Path: mainPath}, nil
		case "owner/proj":
			return hfwrap.Result{Status: "ok", Path: mmprojPath}, nil
		default:
			return hfwrap.Result{}, errors.New("unexpected HF repo " + c.Repo)
		}
	}
	ad := o.Adapter.(*fakeAdapter)

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	ad.regMu.Lock()
	defer ad.regMu.Unlock()
	if len(ad.regFiles) != 1 || len(ad.regFiles[0]) != 2 ||
		ad.regFiles[0][0] != mainPath || ad.regFiles[0][1] != mmprojPath {
		t.Fatalf("Register files = %v, want [%q %q]", ad.regFiles, mainPath, mmprojPath)
	}
	if st := o.Manager.Snapshot(); st.BytesTotal != 300 || st.BytesCompleted != 300 {
		t.Errorf("progress = %d/%d, want 300/300", st.BytesCompleted, st.BytesTotal)
	}
}
