//go:build download

package download

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/lifecycle"
	"github.com/llm-init/llm-init/internal/progress"
)

// hfRecorder captures hfwrap.Config per call so tests can assert on what
// lifecycle passed to the wrapper (token, endpoint, force, repo).
type hfRecorder struct {
	mu    sync.Mutex
	calls []hfwrap.Config
	res   hfwrap.Result
	// pathByFile answers per single-file download the way hfwrap does:
	// snapshot path plus the requested file.
	pathByFile map[string]string
	err        error
}

func (r *hfRecorder) run(_ context.Context, c hfwrap.Config, _ progress.Sink) (hfwrap.Result, error) {
	r.mu.Lock()
	r.calls = append(r.calls, c)
	r.mu.Unlock()
	res := r.res
	if p, ok := r.pathByFile[c.HFFile]; ok {
		res.Path = p
	}
	return res, r.err
}

func (r *hfRecorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.calls)
}

func (r *hfRecorder) last() hfwrap.Config {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) == 0 {
		return hfwrap.Config{}
	}
	return r.calls[len(r.calls)-1]
}

func (r *hfRecorder) snapshot() []hfwrap.Config {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]hfwrap.Config(nil), r.calls...)
}

func hfConfig(t *testing.T, sources ...config.ModelSource) config.Config {
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = sources
	return cfg
}

func hfSource(repo, revision string, include ...string) config.ModelSource {
	return config.ModelSource{
		Index: 1, Kind: config.KindHF, Role: config.RoleMain,
		HFRepo: repo, HFRevision: revision, HFInclude: include,
	}
}

// --- lifecycle dispatch + frontend-message mapping -------------------

func TestHF_HappyPath(t *testing.T) {
	cfg := hfConfig(t, hfSource("Qwen/Qwen3-7B", "0123456789abcdef0123456789abcdef01234567"))
	snap := t.TempDir()
	rec := &hfRecorder{res: hfwrap.Result{Status: "ok", Path: snap, Commit: "abc"}}
	ad := &fakeAdapter{kind: config.EngineLlamaCpp}
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: ad, HFRun: rec.run})
	r.waitPhase(progress.PhaseReady, 10*time.Second)

	if rec.count() != 1 {
		t.Fatalf("hfwrap calls = %d want 1", rec.count())
	}
	if reg := ad.lastRegister(); len(reg) != 1 || reg[0] != snap {
		t.Errorf("Register files = %v want [%s]", reg, snap)
	}
}

// hfErrorCase asserts that a given hfwrap error code lands the manager in
// the expected phase with a frontend message containing wantSubstr.
type hfErrorCase struct {
	name      string
	code      int
	retriable bool
	stderr    string
	phase     progress.Phase
	wantSub   string
}

func TestHF_ErrorMessages_ForFrontend(t *testing.T) {
	cases := []hfErrorCase{
		{"repo_not_found", hfwrap.CodeRepoNotFound, false, "RepositoryNotFoundError", progress.PhaseFailed, "repository not found"},
		{"gated", hfwrap.CodeGated, false, "Cannot access gated repo", progress.PhaseFailed, "gated"},
		{"token_missing", hfwrap.CodeTokenMissing, false, "401 Client Error", progress.PhaseFailed, "authentication failed"},
		{"revision", hfwrap.CodeRevision, false, "RevisionNotFoundError", progress.PhaseFailed, "revision not found"},
		{"disk_full", hfwrap.CodeDiskFull, false, "No space left on device", progress.PhaseFailed, "disk full"},
		{"permission", hfwrap.CodePermission, false, "Permission denied", progress.PhaseFailed, "permission denied"},
		{"network", hfwrap.CodeNetwork, true, "ConnectionError", progress.PhaseDegraded, "network error"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			cfg := hfConfig(t, hfSource("x/y", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"))
			rec := &hfRecorder{err: &hfwrap.Error{
				Code: tc.code, Retriable: tc.retriable, Stage: "stderr", Message: tc.stderr,
			}}
			r := startManager(t, lifecycle.Options{
				Config:  cfg,
				Adapter: &fakeAdapter{kind: config.EngineLlamaCpp},
				HFRun:   rec.run,
			})
			var st progress.State
			if tc.phase == progress.PhaseDegraded {
				st = r.waitUntil(10*time.Second, func(s progress.State) bool {
					return s.Phase == progress.PhaseDegraded && s.LastError != ""
				})
			} else {
				st = r.waitPhase(tc.phase, 10*time.Second)
			}
			if !strings.Contains(strings.ToLower(st.LastError), tc.wantSub) {
				t.Errorf("last_error = %q, want substring %q", st.LastError, tc.wantSub)
			}
		})
	}
}

func TestHF_TokenAndEndpointPlumbedToWrapper(t *testing.T) {
	cfg := hfConfig(t, hfSource("acme/private", "main"))
	cfg.HFToken = "hf_secrettoken"
	cfg.HFEndpoint = "https://hf-mirror.example.com"
	rec := &hfRecorder{res: hfwrap.Result{Status: "ok", Path: t.TempDir()}}
	r := startManager(t, lifecycle.Options{
		Config:  cfg,
		Adapter: &fakeAdapter{kind: config.EngineLlamaCpp},
		HFRun:   rec.run,
	})
	r.waitPhase(progress.PhaseReady, 10*time.Second)

	last := rec.last()
	if last.Token != "hf_secrettoken" {
		t.Errorf("wrapper Token = %q, want it plumbed from HF_TOKEN", last.Token)
	}
	if last.Endpoint != "https://hf-mirror.example.com" {
		t.Errorf("wrapper Endpoint = %q, want it plumbed from HF_ENDPOINT", last.Endpoint)
	}
}

func TestHF_CommaSeparatedMainPlusExtra(t *testing.T) {
	main := config.ModelSource{
		Index: 1, Kind: config.KindHF, Role: config.RoleMain,
		HFRepo: "vendor/main", HFRevision: "main", HFInclude: []string{"model.gguf"},
	}
	extra := config.ModelSource{
		Index: 2, Kind: config.KindHF, Role: config.RoleExtra,
		HFRepo: "vendor/extra", HFRevision: "main", HFInclude: []string{"aux.gguf"},
	}
	cfg := hfConfig(t, main, extra)
	rec := &hfRecorder{res: hfwrap.Result{Status: "ok", Path: filepath.Join(t.TempDir(), "model.gguf")}}
	ad := &fakeAdapter{kind: config.EngineLlamaCpp}
	r := startManager(t, lifecycle.Options{
		Config:  cfg,
		Adapter: ad,
		HFRun:   rec.run,
	})
	r.waitPhase(progress.PhaseReady, 10*time.Second)
	calls := rec.snapshot()
	if len(calls) != 2 {
		t.Fatalf("hfwrap calls = %d, want 2 (extra + main)", len(calls))
	}
	if calls[0].Repo != "vendor/extra" || calls[1].Repo != "vendor/main" {
		t.Errorf("hfwrap repos = %q, %q; want extra then main", calls[0].Repo, calls[1].Repo)
	}
	if reg := ad.lastRegister(); len(reg) != 1 {
		t.Errorf("Register files = %v, want only the main source", reg)
	}
}

// A MODEL_SOURCE segment carrying --role mmproj is fetched by a second hf
// invocation and joins the main source in Register, which is what keeps a
// llama.cpp projector reachable without it becoming an extra.
func TestHF_MainPlusMmprojSibling(t *testing.T) {
	snap := t.TempDir()
	mainPath := filepath.Join(snap, "model.gguf")
	mmprojPath := filepath.Join(snap, "mmproj.gguf")
	for _, p := range []string{mainPath, mmprojPath} {
		if err := os.WriteFile(p, []byte("g"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	main := config.ModelSource{
		Index: 1, Kind: config.KindHF, Role: config.RoleMain,
		HFRepo: "vendor/vlm", HFRevision: "main", HFInclude: []string{"model.gguf"},
	}
	mmproj := config.ModelSource{
		Index: 2, Kind: config.KindHF, Role: config.RoleMmproj,
		HFRepo: "vendor/vlm", HFRevision: "main", HFInclude: []string{"mmproj.gguf"},
	}
	rec := &hfRecorder{
		res: hfwrap.Result{Status: "ok", Path: mainPath},
		pathByFile: map[string]string{
			"model.gguf":  mainPath,
			"mmproj.gguf": mmprojPath,
		},
	}
	ad := &fakeAdapter{kind: config.EngineLlamaCpp}
	r := startManager(t, lifecycle.Options{
		Config:  hfConfig(t, main, mmproj),
		Adapter: ad,
		HFRun:   rec.run,
	})
	r.waitPhase(progress.PhaseReady, 10*time.Second)
	calls := rec.snapshot()
	if len(calls) != 2 {
		t.Fatalf("hfwrap calls = %d, want 2 (main + mmproj)", len(calls))
	}
	if calls[0].HFFile != "model.gguf" || calls[1].HFFile != "mmproj.gguf" {
		t.Errorf("hfwrap files = %q, %q; want main then mmproj", calls[0].HFFile, calls[1].HFFile)
	}
	if reg := ad.lastRegister(); len(reg) != 2 || reg[1] != mmprojPath {
		t.Errorf("Register files = %v, want main plus %s", reg, mmprojPath)
	}
}

// --- real hfwrap.Run subprocess path via the fake hf binary ----------

func TestHF_Subprocess_HappyPath(t *testing.T) {
	cache := t.TempDir()
	commit := "1111111111111111111111111111111111111111"
	res, err := hfwrap.Run(context.Background(), hfwrap.Config{
		HFExe:    fakeHFBin,
		CacheDir: cache,
		Repo:     "owner/repo",
		ExtraEnv: []string{"FAKEHF_COMMIT=" + commit, "FAKEHF_FILES=model.safetensors,config.json"},
	}, nil)
	if err != nil {
		t.Fatalf("hfwrap.Run: %v", err)
	}
	wantPath := hfwrap.SnapshotPath(cache, "owner/repo", commit)
	if res.Path != wantPath {
		t.Errorf("Path = %q want %q", res.Path, wantPath)
	}
	if res.Commit != commit {
		t.Errorf("Commit = %q want %q", res.Commit, commit)
	}
	for _, f := range []string{"model.safetensors", "config.json"} {
		if _, err := os.Stat(filepath.Join(wantPath, f)); err != nil {
			t.Errorf("expected snapshot file %s: %v", f, err)
		}
	}
}

func TestHF_Subprocess_SingleFile(t *testing.T) {
	cache := t.TempDir()
	commit := "2222222222222222222222222222222222222222"
	res, err := hfwrap.Run(context.Background(), hfwrap.Config{
		HFExe:    fakeHFBin,
		CacheDir: cache,
		Repo:     "owner/repo",
		HFFile:   "model.gguf",
		ExtraEnv: []string{"FAKEHF_COMMIT=" + commit},
	}, nil)
	if err != nil {
		t.Fatalf("hfwrap.Run: %v", err)
	}
	want := hfwrap.SnapshotPath(cache, "owner/repo", commit) + "/model.gguf"
	if res.Path != want {
		t.Errorf("Path = %q want %q", res.Path, want)
	}
	if _, err := os.Stat(want); err != nil {
		t.Errorf("single file not written: %v", err)
	}
}

// TestHF_Subprocess_DuplicateBasenameProgress plays the FireRedTTS3
// shape: one repository, two directories, one file name. huggingface_hub
// labels both bars `model.safetensors`, so before the tree named them the
// second file landed on the first one's state, every tick it sent was
// dropped as a rewind, and its row sat at `downloading` for good.
func TestHF_Subprocess_DuplicateBasenameProgress(t *testing.T) {
	cache := t.TempDir()
	commit := "3333333333333333333333333333333333333333"

	// The sizes the Hub reports, which tqdm prints as 8.48G and 3.78G.
	const (
		instructSize = int64(8475259708)
		redaeSize    = int64(3775160672)
		instructPath = "fireredtts3_instruct/model.safetensors"
		redaePath    = "redae/model.safetensors"
	)
	tree := []hfwrap.TreeEntry{
		{Path: instructPath, Size: instructSize},
		{Path: redaePath, Size: redaeSize},
	}

	mgr := progress.New(time.Now())
	mgr.SeedDownloadBudget(instructSize+redaeSize, 0)
	mgr.SeedDownloadFileList([]progress.FileProgress{
		{Path: instructPath, BytesTotal: instructSize, Status: progress.FileWaiting},
		{Path: redaePath, BytesTotal: redaeSize, Status: progress.FileWaiting},
	})

	_, err := hfwrap.Run(context.Background(), hfwrap.Config{
		HFExe:    fakeHFBin,
		CacheDir: cache,
		Repo:     "FireRedTeam/FireRedTTS3",
		Tree:     tree,
		ExtraEnv: []string{
			"FAKEHF_COMMIT=" + commit,
			"FAKEHF_FILES=" + instructPath + "," + redaePath,
			fmt.Sprintf("FAKEHF_PROGRESS_SIZES=%d,%d", instructSize, redaeSize),
		},
	}, mgr.Sink())
	if err != nil {
		t.Fatalf("hfwrap.Run: %v", err)
	}

	st := mgr.Snapshot()
	// tqdm's three significant digits round both totals up; what matters
	// is that the second file contributed at all. It used to contribute
	// nothing, leaving this at 8480000000.
	if want := int64(8480000000 + 3780000000); st.BytesCompleted != want {
		t.Errorf("bytes_completed = %d, want %d", st.BytesCompleted, want)
	}
	if st.FilesCompleted != 2 {
		t.Errorf("files_completed = %d, want 2", st.FilesCompleted)
	}
	if len(st.Files) != 2 {
		t.Fatalf("rows = %d, want the two seeded ones: %+v", len(st.Files), st.Files)
	}
	for _, f := range st.Files {
		if f.Status != progress.FileDone {
			t.Errorf("row %s status = %q, want done", f.Path, f.Status)
		}
		if f.BytesCompleted != f.BytesTotal {
			t.Errorf("row %s = %d/%d, want a full row", f.Path, f.BytesCompleted, f.BytesTotal)
		}
	}
}

func TestHF_Subprocess_ErrorClassification(t *testing.T) {
	cases := []struct {
		name      string
		stderr    string
		wantCode  int
		retriable bool
	}{
		{"repo_not_found", "huggingface_hub.errors.RepositoryNotFoundError: 404", hfwrap.CodeRepoNotFound, false},
		{"gated", "GatedRepoError: Cannot access gated repo for url ...", hfwrap.CodeGated, false},
		{"token_missing", "401 Client Error: Unauthorized for url ...", hfwrap.CodeTokenMissing, false},
		{"revision", "RevisionNotFoundError: 404 for revision deadbeef", hfwrap.CodeRevision, false},
		{"disk_full", "OSError: [Errno 28] No space left on device", hfwrap.CodeDiskFull, false},
		{"network", "requests.exceptions.ConnectionError: HTTPSConnectionPool", hfwrap.CodeNetwork, true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			cache := t.TempDir()
			_, err := hfwrap.Run(context.Background(), hfwrap.Config{
				HFExe:    fakeHFBin,
				CacheDir: cache,
				Repo:     "owner/repo",
				ExtraEnv: []string{"FAKEHF_EXIT=1", "FAKEHF_STDERR=" + tc.stderr},
			}, nil)
			if err == nil {
				t.Fatal("expected error")
			}
			var e *hfwrap.Error
			if !errors.As(err, &e) {
				t.Fatalf("error is %T, want *hfwrap.Error", err)
			}
			if e.Code != tc.wantCode {
				t.Errorf("Code = %d want %d (msg=%q)", e.Code, tc.wantCode, e.Message)
			}
			if hfwrap.IsRetriable(err) != tc.retriable {
				t.Errorf("IsRetriable = %v want %v", hfwrap.IsRetriable(err), tc.retriable)
			}
		})
	}
}
