package lifecycle

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

func TestEnterDownloadPhase_SeedsWholeRepo(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/full/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[
			{"path":"a.bin","size":100,"type":"file"},
			{"path":"b.bin","size":200,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, "")
	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/full"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.Phase != progress.PhaseDownload {
		t.Errorf("phase=%q, want download", st.Phase)
	}
	if st.BytesTotal != 300 || st.FilesTotal != 2 {
		t.Errorf("seed bytes=%d files=%d, want 300/2", st.BytesTotal, st.FilesTotal)
	}
	if !m.pass.entered {
		t.Error("pass.entered must be true")
	}

	m.enterDownloadPhase(context.Background())
	if pm.Snapshot().BytesTotal != 300 {
		t.Error("second enter must be a no-op")
	}
}

func TestEnterDownloadPhase_TreeFailureLeavesDynamic(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusBadGateway)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, "")
	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/full"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 0 || st.FilesTotal != 0 {
		t.Errorf("failed tree must not pin: bytes=%d files=%d", st.BytesTotal, st.FilesTotal)
	}
}

func TestEnterDownloadPhase_ResumeSeedsCachedSnapshot(t *testing.T) {
	t.Parallel()
	cache, repo := writeHFSnapshot(t, "owner/full", "main", map[string]int{
		"a.bin": 100,
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/full/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[
			{"path":"a.bin","size":100,"type":"file"},
			{"path":"b.bin","size":200,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Kind: config.KindHF, HFRepo: repo, HFRevision: "main"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 300 || st.BytesCompleted != 100 {
		t.Errorf("bytes %d/%d, want 100/300 cached", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesTotal != 2 || st.FilesCompleted != 1 {
		t.Errorf("files %d/%d, want 1/2 cached", st.FilesCompleted, st.FilesTotal)
	}
	if len(st.Files) != 2 {
		t.Fatalf("file list len=%d, want 2", len(st.Files))
	}
	if st.Files[0].Path != "a.bin" || st.Files[0].Status != progress.FileDone {
		t.Errorf("files[0]=%+v, want a.bin done", st.Files[0])
	}
	if st.Files[1].Path != "b.bin" || st.Files[1].Status != progress.FileWaiting {
		t.Errorf("files[1]=%+v, want b.bin waiting", st.Files[1])
	}
}

func TestEnterDownloadPhase_SeedsWaitingFileList(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/full/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[
			{"path":"mmproj-F16.gguf","size":50,"type":"file"},
			{"path":"model.gguf","size":250,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, "")
	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/full"}
	m.pass = &passState{
		cfg:  config.Config{HFEndpoint: server.URL},
		main: &main,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if len(st.Files) != 2 {
		t.Fatalf("file list len=%d, want 2", len(st.Files))
	}
	if st.Files[0].Path != "mmproj-F16.gguf" || st.Files[0].Status != progress.FileWaiting {
		t.Errorf("files[0]=%+v, want mmproj waiting", st.Files[0])
	}
	if st.Files[1].Path != "model.gguf" || st.Files[1].BytesTotal != 250 {
		t.Errorf("files[1]=%+v, want model.gguf 250", st.Files[1])
	}
}

func TestEnterDownloadPhase_ForceDropsCachedSeed(t *testing.T) {
	t.Parallel()
	cache, repo := writeHFSnapshot(t, "owner/full", "main", map[string]int{
		"a.bin": 100,
	})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[{"path":"a.bin","size":100,"type":"file"}]`)
	}))
	t.Cleanup(server.Close)

	pm := progress.New(time.Now())
	m := newBudgetManager(t, pm, server.URL, cache)
	main := config.ModelSource{Kind: config.KindHF, HFRepo: repo, HFRevision: "main"}
	m.pass = &passState{
		cfg:   config.Config{HFEndpoint: server.URL},
		main:  &main,
		force: true,
	}
	m.enterDownloadPhase(context.Background())

	st := pm.Snapshot()
	if st.BytesTotal != 100 || st.BytesCompleted != 0 {
		t.Errorf("force: bytes %d/%d, want 0/100", st.BytesCompleted, st.BytesTotal)
	}
	if st.FilesCompleted != 0 {
		t.Errorf("force: FilesCompleted=%d, want 0", st.FilesCompleted)
	}
}

func TestEstimateDownloadBudget_ExcludeAndMissingExact(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[
			{"path":"README.md","size":10,"type":"file"},
			{"path":"model.gguf","size":90,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	excl := config.ModelSource{
		Kind: config.KindHF, HFRepo: "owner/r", HFExclude: []string{"*.md"},
	}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&excl, nil, nil,
	)
	if !b.canPin || b.bytes != 90 || b.files != 1 {
		t.Fatalf("exclude: %+v, want bytes=90 files=1", b)
	}

	missing := config.ModelSource{
		Kind: config.KindHF, HFRepo: "owner/r",
		HFInclude: []string{"model.gguf", "absent.gguf"},
	}
	b = (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&missing, nil, nil,
	)
	if b.canPin {
		t.Fatalf("missing exact include must not pin: %+v", b)
	}
}

func TestEstimateDownloadBudget_Subdir(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, `[
			{"path":"tokenizer.json","size":7,"type":"file"},
			{"path":"onnx/model.onnx","size":40,"type":"file"},
			{"path":"onnx/config.json","size":5,"type":"file"}
		]`)
	}))
	t.Cleanup(server.Close)

	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/emb", HFSubdir: "onnx"}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&main, nil, nil,
	)
	if !b.canPin || b.bytes != 45 || b.files != 2 {
		t.Fatalf("subdir: %+v, want bytes=45 files=2", b)
	}
}

func TestCachedHFSnapshot_ExcludeAndMissingRef(t *testing.T) {
	t.Parallel()
	cache, repo := writeHFSnapshot(t, "owner/r", "main", map[string]int{
		"README.md":  10,
		"model.gguf": 90,
	})
	ms := config.ModelSource{
		Kind: config.KindHF, HFRepo: repo, HFRevision: "main",
		HFExclude: []string{"*.md"},
	}
	got, n := cachedHFSnapshot(cache, ms)
	if got != 90 || n != 1 {
		t.Errorf("exclude: %d/%d, want 90/1", got, n)
	}

	ms.HFRevision = "does-not-exist"
	if got, n = cachedHFSnapshot(cache, ms); got != 0 || n != 0 {
		t.Errorf("missing ref: %d/%d, want 0/0", got, n)
	}
	if got, n = cachedHFSnapshot("", ms); got != 0 || n != 0 {
		t.Errorf("empty cache: %d/%d, want 0/0", got, n)
	}
}

func newBudgetManager(t *testing.T, pm progress.Manager, endpoint, cache string) *Manager {
	t.Helper()
	return &Manager{
		opts: Options{
			Manager:     pm,
			NowFunc:     time.Now,
			HFCacheRoot: cache,
			Config:      config.Config{HFEndpoint: endpoint},
		},
	}
}

func writeHFSnapshot(t *testing.T, repo, ref string, files map[string]int) (cache, repoID string) {
	t.Helper()
	cache = t.TempDir()
	sha := "cccccccccccccccccccccccccccccccccccccccc"
	refs := filepath.Join(hfwrap.HFRepoCacheDir(cache, repo), "refs")
	if err := os.MkdirAll(refs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(refs, ref), []byte(sha+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snap := hfwrap.SnapshotPath(cache, repo, sha)
	for rel, size := range files {
		p := filepath.Join(snap, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return cache, repo
}
