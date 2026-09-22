package hfwrap

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/progress"
)

// TestBuildEnv_ForcesTqdmProgress locks the fix that makes huggingface_hub
// emit tqdm byte bars on our non-TTY stderr pipe (otherwise the dashboard
// shows 0/0 for the whole download). An ExtraEnv override must still win.
func TestBuildEnv_ForcesTqdmProgress(t *testing.T) {
	env := buildEnv(Config{})
	if !envHas(env, "TQDM_POSITION=-1") {
		t.Errorf("buildEnv missing TQDM_POSITION=-1; got %v", env)
	}

	override := buildEnv(Config{ExtraEnv: []string{"TQDM_POSITION=2"}})
	// The operator override is appended last, so it wins under exec's
	// last-key-wins semantics.
	last := ""
	for _, e := range override {
		if strings.HasPrefix(e, "TQDM_POSITION=") {
			last = e
		}
	}
	if last != "TQDM_POSITION=2" {
		t.Errorf("ExtraEnv override should win, last TQDM_POSITION = %q", last)
	}
}

func TestBuildEnv_DisablesXetByDefault(t *testing.T) {
	// Zero-value Config => stable LFS mode (hf_xet disabled).
	env := buildEnv(Config{})
	if !envHas(env, "HF_HUB_DISABLE_XET=1") {
		t.Errorf("buildEnv missing HF_HUB_DISABLE_XET=1 (LFS mode); got %v", env)
	}

	// EnableXet opts back into the hf_xet backend: no disable flag emitted.
	if on := buildEnv(Config{EnableXet: true}); envHas(on, "HF_HUB_DISABLE_XET=1") {
		t.Errorf("EnableXet=true should not disable xet; got %v", on)
	}

	// ExtraEnv is still the last-resort escape hatch (last-key-wins).
	override := buildEnv(Config{ExtraEnv: []string{"HF_HUB_DISABLE_XET=0"}})
	last := ""
	for _, e := range override {
		if strings.HasPrefix(e, "HF_HUB_DISABLE_XET=") {
			last = e
		}
	}
	if last != "HF_HUB_DISABLE_XET=0" {
		t.Errorf("ExtraEnv override should win, last HF_HUB_DISABLE_XET = %q", last)
	}
}

func envHas(env []string, kv string) bool {
	for _, e := range env {
		if e == kv {
			return true
		}
	}
	return false
}

// recordingSink captures every Sink event so tests can assert on the
// stderr→Sink translation without standing up the full progress.Manager.
type recordingSink struct {
	mu             sync.Mutex
	starts         []startEvent
	bytes          int64
	dones          []string
	errs           []string
	phase          progress.Phase
	resolvedCommit string
	note           string
}

type startEvent struct {
	Path  string
	Total int64
}

func (s *recordingSink) OnFileStart(path string, total int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.starts = append(s.starts, startEvent{Path: path, Total: total})
}
func (s *recordingSink) OnBytes(d int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.bytes += d
}
func (s *recordingSink) OnFileDone(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.dones = append(s.dones, path)
}
func (s *recordingSink) OnError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.errs = append(s.errs, err.Error())
}
func (s *recordingSink) OnPhase(p progress.Phase) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.phase = p
}
func (s *recordingSink) OnResolvedCommit(sha string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resolvedCommit = sha
}
func (s *recordingSink) OnNote(text string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.note = text
}
func (s *recordingSink) OnFilesTotal(int) {}

var _ progress.Sink = (*recordingSink)(nil)

// buildHFStub writes a tiny sh script that pretends to be the `hf`
// binary. The stub:
//   - emits the supplied stderr bytes line-by-line
//   - optionally writes a refs/<ref> file in the cache dir so the
//     runner's post-success ReadRefCommit picks up a synthetic SHA
//   - exits with the requested code
//
// Used to exercise the runner pipeline (argv parsing, stderr scan,
// classify, refs reader) without depending on a real `hf` CLI.
type stubArgs struct {
	exitCode int
	stderr   string
	cacheDir string
	repo     string
	ref      string
	sha      string
}

func buildHFStub(t *testing.T, a stubArgs) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "hf.sh")
	body := "#!/bin/sh\n"
	if a.stderr != "" {
		// printf with %s avoids \r interpretation; use heredoc to
		// pass arbitrary content unchanged.
		body += "cat 1>&2 <<'__EOF__'\n" + a.stderr + "\n__EOF__\n"
	}
	if a.cacheDir != "" && a.repo != "" && a.sha != "" {
		ref := a.ref
		if ref == "" {
			ref = "main"
		}
		body += "mkdir -p " + filepath.Join(a.cacheDir, "models--"+strings.ReplaceAll(a.repo, "/", "--"), "refs") + "\n"
		body += "echo " + a.sha + " > " + filepath.Join(a.cacheDir, "models--"+strings.ReplaceAll(a.repo, "/", "--"), "refs", ref) + "\n"
	}
	body += "exit " + itoa(a.exitCode) + "\n"
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write stub: %v", err)
	}
	return path
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	digits := make([]byte, 0, 4)
	for n > 0 {
		digits = append(digits, byte('0'+n%10))
		n /= 10
	}
	if neg {
		digits = append(digits, '-')
	}
	for i, j := 0, len(digits)-1; i < j; i, j = i+1, j-1 {
		digits[i], digits[j] = digits[j], digits[i]
	}
	return string(digits)
}

func TestRun_HappyPath(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stub uses sh; happy-path test runs on linux/mac CI")
	}
	cacheDir := t.TempDir()
	sha := "abcdef0123456789abcdef0123456789abcdef01"
	exe := buildHFStub(t, stubArgs{
		exitCode: 0,
		stderr: "Fetching 2 files\n" +
			"config.json: 100%|##########| 2KiB/2KiB [00:01<00:00, 1MB/s]\n" +
			"model.safetensors: 50%|#####     | 1.0G/2.0G [00:21<00:21, 50MB/s]\n",
		cacheDir: cacheDir,
		repo:     "Qwen/Qwen3-27B",
		ref:      "main",
		sha:      sha,
	})

	sink := &recordingSink{}
	res, err := Run(context.Background(), Config{
		HFExe:    exe,
		CacheDir: cacheDir,
		Repo:     "Qwen/Qwen3-27B",
	}, sink)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Commit != sha {
		t.Errorf("Commit = %q want %q", res.Commit, sha)
	}
	if !strings.Contains(res.Path, sha) {
		t.Errorf("Path should contain sha; got %q", res.Path)
	}
	if sink.phase != progress.PhaseDownload {
		t.Errorf("phase = %v want download", sink.phase)
	}
	if sink.resolvedCommit != sha {
		t.Errorf("resolvedCommit = %q want %q", sink.resolvedCommit, sha)
	}
}

func TestRun_HappyPath_HexSHARevision(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stub uses sh")
	}
	sha := "0123456789abcdef0123456789abcdef01234567"
	cacheDir := t.TempDir()
	exe := buildHFStub(t, stubArgs{
		exitCode: 0,
		stderr:   "Fetching 1 files\nconfig.json: 100%|##########| 2KiB/2KiB\n",
		cacheDir: cacheDir,
		repo:     "x/y",
		// Note: ref name matches input revision (hex SHA), not "main".
		// hf CLI does not write refs/<sha> files; ReadRefCommit
		// short-circuits on hex input, so we don't need the file.
		sha: sha,
	})

	sink := &recordingSink{}
	res, err := Run(context.Background(), Config{
		HFExe:    exe,
		CacheDir: cacheDir,
		Repo:     "x/y",
		Revision: sha,
	}, sink)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Commit != sha {
		t.Errorf("Commit = %q want %q", res.Commit, sha)
	}
	// OnResolvedCommit should fire eagerly at spawn time when
	// Revision is hex, before download even starts.
	if sink.resolvedCommit != sha {
		t.Errorf("resolvedCommit = %q want %q", sink.resolvedCommit, sha)
	}
}

func TestRun_RetriableExitCode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stub uses sh")
	}
	exe := buildHFStub(t, stubArgs{
		exitCode: 1,
		stderr:   "ConnectionError: 503 upstream busy\n",
	})

	_, err := Run(context.Background(), Config{
		HFExe:    exe,
		Repo:     "x/y",
		Revision: "main",
	}, &recordingSink{})

	if err == nil {
		t.Fatal("expected error")
	}
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("expected *Error, got %T", err)
	}
	if e.Code != CodeNetwork {
		t.Errorf("expected CodeNetwork, got %d", e.Code)
	}
	if !e.Retriable {
		t.Error("expected Retriable=true")
	}
}

func TestRun_PermanentExitCode_RepoNotFound(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stub uses sh")
	}
	exe := buildHFStub(t, stubArgs{
		exitCode: 1,
		stderr:   "huggingface_hub.errors.RepositoryNotFoundError: 404 not found\n",
	})

	_, err := Run(context.Background(), Config{
		HFExe: exe,
		Repo:  "x/y",
	}, &recordingSink{})
	var e *Error
	if !errors.As(err, &e) {
		t.Fatalf("expected *Error, got %T", err)
	}
	if e.Code != CodeRepoNotFound {
		t.Errorf("got %d, want CodeRepoNotFound", e.Code)
	}
	if e.Retriable {
		t.Error("repo_not_found must be permanent")
	}
}

func TestRun_RequiredArgs(t *testing.T) {
	_, err := Run(context.Background(), Config{}, nil)
	if err == nil {
		t.Fatal("expected error for missing Repo")
	}
	var e *Error
	if !errors.As(err, &e) || e.Code != CodeBadArgs {
		t.Fatalf("expected CodeBadArgs, got %v", err)
	}
}

func TestBuildArgs_AllOptionsForwarded(t *testing.T) {
	args := buildArgs(Config{
		Repo:            "Qwen/Qwen3-27B",
		Revision:        "abc123",
		CacheDir:        "/cache/hf/hub",
		AllowPatterns:   []string{"*.safetensors", "config.json"},
		ExcludePatterns: []string{"openvino/**"},
		MaxWorkers:      16,
		ForceReload:     true,
	})
	want := []string{
		"download", "Qwen/Qwen3-27B",
		"--cache-dir", "/cache/hf/hub",
		"--revision", "abc123",
		"--include", "*.safetensors",
		"--include", "config.json",
		"--exclude", "openvino/**",
		"--max-workers", "16",
		"--force-download",
	}
	if len(args) != len(want) {
		t.Fatalf("len mismatch: got %v want %v", args, want)
	}
	for i := range want {
		if args[i] != want[i] {
			t.Fatalf("arg[%d] got %q want %q", i, args[i], want[i])
		}
	}
}

func TestBuildArgs_ExcludeOnly(t *testing.T) {
	args := buildArgs(Config{
		Repo:            "beclab/embeddinggemma-300m",
		CacheDir:        "/cache/hf/hub",
		ExcludePatterns: []string{"onnx/**", "openvino/**"},
	})
	want := []string{
		"download", "beclab/embeddinggemma-300m",
		"--cache-dir", "/cache/hf/hub",
		"--exclude", "onnx/**",
		"--exclude", "openvino/**",
	}
	if len(args) != len(want) {
		t.Fatalf("len mismatch: got %v want %v", args, want)
	}
	for i := range want {
		if args[i] != want[i] {
			t.Fatalf("arg[%d] got %q want %q", i, args[i], want[i])
		}
	}
}

func TestBuildArgs_SingleFile(t *testing.T) {
	args := buildArgs(Config{
		Repo:     "x/y",
		HFFile:   "model.gguf",
		CacheDir: "/c",
	})
	want := []string{"download", "x/y", "model.gguf", "--cache-dir", "/c"}
	if strings.Join(args, " ") != strings.Join(want, " ") {
		t.Fatalf("got %v want %v", args, want)
	}
}

func TestBuildArgs_OmitsMainAndEmptyRevision(t *testing.T) {
	for _, rev := range []string{"", "main"} {
		args := buildArgs(Config{Repo: "x/y", Revision: rev, CacheDir: "/c"})
		for _, a := range args {
			if a == "--revision" {
				t.Errorf("revision %q should not produce --revision flag; got %v", rev, args)
			}
		}
	}
}

func TestRun_ContextCancel(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stub uses sh")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "sleeper.sh")
	body := "#!/bin/sh\nsleep 5\nexit 0\n"
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write stub: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	_, err := Run(ctx, Config{
		HFExe: path,
		Repo:  "x/y",
	}, &recordingSink{})
	if err == nil {
		t.Fatal("expected error from cancelled subprocess")
	}
}

func TestRun_HFTransferEnvSetsNote(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("stub uses sh")
	}
	cacheDir := t.TempDir()
	sha := "abcdef0123456789abcdef0123456789abcdef01"
	exe := buildHFStub(t, stubArgs{
		exitCode: 0,
		cacheDir: cacheDir,
		repo:     "x/y",
		sha:      sha,
	})

	sink := &recordingSink{}
	_, err := Run(context.Background(), Config{
		HFExe:    exe,
		CacheDir: cacheDir,
		Repo:     "x/y",
		ExtraEnv: []string{"HF_HUB_ENABLE_HF_TRANSFER=1"},
	}, sink)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(sink.note, "xet") {
		t.Errorf("note should mention xet, got %q", sink.note)
	}
}
