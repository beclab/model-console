package lifecycle

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
)

func TestEstimateDownloadBudget_AggregatesHFMainAndExtras(t *testing.T) {
	t.Parallel()
	sizes := map[string]int64{
		"/api/models/owner/main/tree/main":  100,
		"/api/models/owner/extra/tree/main": 200,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		size, ok := sizes[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		fmt.Fprintf(w, `[{"path":"model.bin","size":%d,"type":"file"}]`, size)
	}))
	t.Cleanup(server.Close)

	main := config.ModelSource{
		Kind: config.KindHF, HFRepo: "owner/main", HFInclude: []string{defaultURLModelFilename},
	}
	extras := []config.ModelSource{{
		Kind: config.KindHF, HFRepo: "owner/extra", HFInclude: []string{defaultURLModelFilename},
	}}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&main,
		nil,
		extras,
	)
	if !b.canPin || b.bytes != 300 || b.bytesCached != 0 || b.files != 2 {
		t.Fatalf("budget = %+v, want bytes=300 files=2 canPin", b)
	}
}

// An mmproj sibling is a download this pass performs, so its bytes belong
// in the pinned total alongside the main source.
func TestEstimateDownloadBudget_IncludesMmproj(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/vlm/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[{"path":"model.bin","size":100,"type":"file"},{"path":"mmproj.gguf","size":40,"type":"file"}]`)
	}))
	t.Cleanup(server.Close)

	main := config.ModelSource{
		Kind: config.KindHF, Role: config.RoleMain,
		HFRepo: "owner/vlm", HFInclude: []string{defaultURLModelFilename},
	}
	mmproj := config.ModelSource{
		Kind: config.KindHF, Role: config.RoleMmproj,
		HFRepo: "owner/vlm", HFInclude: []string{"mmproj.gguf"},
	}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&main,
		&mmproj,
		nil,
	)
	if !b.canPin || b.bytes != 140 || b.files != 2 {
		t.Fatalf("budget = %+v, want bytes=140 files=2 canPin", b)
	}
}

func TestEstimateDownloadBudget_StaysDynamicWithOllamaSource(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/owner/model/tree/main":
			fmt.Fprint(w, `[{"path":"model.bin","size":100,"type":"file"}]`)
		case "/model.bin":
			w.Header().Set("Content-Length", "100")
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)

	ollama := config.ModelSource{Kind: config.KindOllama, OllamaTag: "qwen3:0.6b"}
	for _, tc := range []struct {
		name   string
		main   config.ModelSource
		extras []config.ModelSource
	}{
		{
			name: "Ollama main with HF extra",
			main: ollama,
			extras: []config.ModelSource{{
				Kind: config.KindHF, HFRepo: "owner/model", HFInclude: []string{defaultURLModelFilename},
			}},
		},
		{
			name: "URL main with Ollama extra",
			main: config.ModelSource{
				Kind: config.KindURL, URL: server.URL + "/model.bin", LocalPath: "/tmp/model.bin",
			},
			extras: []config.ModelSource{ollama},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := (&Manager{}).estimateDownloadBudget(
				context.Background(),
				config.Config{HFEndpoint: server.URL},
				&tc.main,
				nil,
				tc.extras,
			)
			if b.bytes != 0 || b.canPin {
				t.Fatalf("budget = %+v, want dynamic 0/false", b)
			}
		})
	}
}

func TestResumeAwareFileBytes_PrefersFinishedThenPart(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	dest := filepath.Join(dir, defaultURLModelFilename)

	if got := resumeAwareFileBytes(dest); got != 0 {
		t.Fatalf("missing file: got %d want 0", got)
	}

	part := dest + ".part"
	if err := os.WriteFile(part, make([]byte, 1500), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := resumeAwareFileBytes(dest); got != 1500 {
		t.Fatalf(".part only: got %d want 1500", got)
	}

	if err := os.WriteFile(dest, make([]byte, 2000), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := resumeAwareFileBytes(dest); got != 2000 {
		t.Fatalf("finished wins over .part: got %d want 2000", got)
	}
}

func TestCachedURLResumeBytes_URLAndOllamaURL(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	local := filepath.Join(dir, "a.bin")
	if err := os.WriteFile(local+".part", make([]byte, 400), 0o644); err != nil {
		t.Fatal(err)
	}
	got := cachedURLResumeBytes(config.Config{}, config.ModelSource{
		Kind:      config.KindURL,
		LocalPath: local,
	})
	if got != 400 {
		t.Errorf("KindURL .part: got %d want 400", got)
	}

	runDir := t.TempDir()
	destDir := filepath.Join(runDir, "url-fetched")
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		t.Fatal(err)
	}
	ms := config.ModelSource{
		Kind: config.KindOllamaURL,
		URL:  "https://example.com/weights.gguf",
	}
	dest := filepath.Join(destDir, urlDestName(ms))
	if err := os.WriteFile(dest+".part", make([]byte, 700), 0o644); err != nil {
		t.Fatal(err)
	}
	got = cachedURLResumeBytes(config.Config{
		Runtime: config.Runtime{RunDir: runDir},
	}, ms)
	if got != 700 {
		t.Errorf("KindOllamaURL .part: got %d want 700", got)
	}
}

func TestCachedHFBytes_FinishedSnapshotOnly(t *testing.T) {
	t.Parallel()
	cache := t.TempDir()
	repo := "owner/repo"
	sha := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	refs := filepath.Join(hfwrap.HFRepoCacheDir(cache, repo), "refs")
	if err := os.MkdirAll(refs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(refs, "main"), []byte(sha+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	snap := hfwrap.SnapshotPath(cache, repo, sha)
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(snap, "model.gguf")
	if err := os.WriteFile(file, make([]byte, 2500), 0o644); err != nil {
		t.Fatal(err)
	}
	// Orphan incomplete must not inflate the seed.
	blobs := filepath.Join(hfwrap.HFRepoCacheDir(cache, repo), "blobs")
	if err := os.MkdirAll(blobs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blobs, "deadbeef.uuid.incomplete"), make([]byte, 9999), 0o644); err != nil {
		t.Fatal(err)
	}

	ms := config.ModelSource{
		Kind:       config.KindHF,
		HFRepo:     repo,
		HFRevision: "main",
		HFInclude:  []string{"model.gguf"},
	}
	if got, n := cachedHFSnapshot(cache, ms); got != 2500 || n != 1 {
		t.Errorf("finished snapshot: got %d/%d want 2500/1", got, n)
	}

	ms.HFInclude = []string{"*.gguf"}
	if got, n := cachedHFSnapshot(cache, ms); got != 2500 || n != 1 {
		t.Errorf("glob include: got %d/%d want 2500/1", got, n)
	}

	ms.HFInclude = nil
	if got, n := cachedHFSnapshot(cache, ms); got != 2500 || n != 1 {
		t.Errorf("whole repo: got %d/%d want 2500/1", got, n)
	}

	ms.HFInclude = []string{"missing.gguf"}
	if got, n := cachedHFSnapshot(cache, ms); got != 0 || n != 0 {
		t.Errorf("missing file: got %d/%d want 0/0", got, n)
	}
}

func TestSeedCompletedForPass_ForceDropsResumeBytes(t *testing.T) {
	t.Parallel()
	if got := seedCompletedForPass(false, 800); got != 800 {
		t.Errorf("force=false: got %d want 800", got)
	}
	if got := seedCompletedForPass(true, 800); got != 0 {
		t.Errorf("force=true: got %d want 0", got)
	}
}

func TestCachedHFBytes_Subdir(t *testing.T) {
	t.Parallel()
	cache := t.TempDir()
	repo := "org/embed"
	sha := "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	refs := filepath.Join(hfwrap.HFRepoCacheDir(cache, repo), "refs")
	if err := os.MkdirAll(refs, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(refs, "main"), []byte(sha), 0o644); err != nil {
		t.Fatal(err)
	}
	snap := filepath.Join(hfwrap.SnapshotPath(cache, repo, sha), "onnx")
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(snap, "model.onnx"), make([]byte, 120), 0o644); err != nil {
		t.Fatal(err)
	}
	ms := config.ModelSource{
		Kind:      config.KindHF,
		HFRepo:    repo,
		HFInclude: []string{"model.onnx"},
		HFSubdir:  "onnx",
	}
	if got, n := cachedHFSnapshot(cache, ms); got != 120 || n != 1 {
		t.Errorf("subdir snapshot: got %d/%d want 120/1", got, n)
	}
}

func TestEstimateDownloadBudget_WholeHFRepo(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/full/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[{"path":"config.json","size":10,"type":"file"},{"path":"model.safetensors","size":90,"type":"file"}]`)
	}))
	t.Cleanup(server.Close)

	main := config.ModelSource{Kind: config.KindHF, HFRepo: "owner/full"}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&main,
		nil,
		nil,
	)
	if !b.canPin || b.bytes != 100 || b.files != 2 {
		t.Fatalf("whole repo budget = %+v, want bytes=100 files=2 canPin", b)
	}
}

func TestEstimateDownloadBudget_GlobInclude(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/repo/tree/main" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `[{"path":"a.gguf","size":40,"type":"file"},{"path":"b.gguf","size":60,"type":"file"},{"path":"README.md","size":3,"type":"file"}]`)
	}))
	t.Cleanup(server.Close)

	main := config.ModelSource{
		Kind: config.KindHF, HFRepo: "owner/repo", HFInclude: []string{"*.gguf"},
	}
	b := (&Manager{}).estimateDownloadBudget(
		context.Background(),
		config.Config{HFEndpoint: server.URL},
		&main,
		nil,
		nil,
	)
	if !b.canPin || b.bytes != 100 || b.files != 2 {
		t.Fatalf("glob budget = %+v, want bytes=100 files=2 canPin", b)
	}
}

func TestSeedCompletedFiles_ForceDrops(t *testing.T) {
	t.Parallel()
	if got := seedCompletedFiles(false, 4); got != 4 {
		t.Errorf("force=false: got %d want 4", got)
	}
	if got := seedCompletedFiles(true, 4); got != 0 {
		t.Errorf("force=true: got %d want 0", got)
	}
}
