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

func TestHasHFBlob(t *testing.T) {
	t.Parallel()
	repo := "owner/repo"
	cache, _ := writeHFBlobs(t, repo, map[string]int{
		blobA:                 2500,
		gitOID1:               10,
		blobB + ".incomplete": 9999,
	})
	cases := []struct {
		name string
		e    hfwrap.TreeEntry
		want bool
	}{
		{"lfs blob present", hfwrap.TreeEntry{Path: "model.gguf", Size: 2500, BlobID: blobA}, true},
		{"git blob present", hfwrap.TreeEntry{Path: "config.json", Size: 10, BlobID: gitOID1}, true},
		{"size differs", hfwrap.TreeEntry{Path: "model.gguf", Size: 2600, BlobID: blobA}, false},
		{"only a partial", hfwrap.TreeEntry{Path: "model.gguf", Size: 9999, BlobID: blobB}, false},
		{"absent", hfwrap.TreeEntry{Path: "model.gguf", Size: 2500, BlobID: blobC}, false},
		{"no id", hfwrap.TreeEntry{Path: "model.gguf", Size: 2500}, false},
		{"id names a path", hfwrap.TreeEntry{Path: "x", Size: 0, BlobID: "../refs"}, false},
	}
	for _, tc := range cases {
		if got := hasHFBlob(cache, repo, tc.e); got != tc.want {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
	if hasHFBlob("", repo, cases[0].e) {
		t.Error("empty cache root must report nothing cached")
	}
}

func TestHasHFFile_TargetSnapshot(t *testing.T) {
	t.Parallel()
	const commit = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
	repo := "owner/repo"
	cache := t.TempDir()
	writeSnapshotAt(t, cache, repo, commit, map[string]int{"model.gguf": 2500, "sub/a.json": 7})
	writeBlobsInto(t, cache, repo, map[string]int{"refs-escape": 1})
	cases := []struct {
		name   string
		commit string
		e      hfwrap.TreeEntry
		want   bool
	}{
		{"in target snapshot", commit, hfwrap.TreeEntry{Path: "model.gguf", Size: 2500, BlobID: blobC}, true},
		{"nested path", commit, hfwrap.TreeEntry{Path: "sub/a.json", Size: 7}, true},
		{"size differs", commit, hfwrap.TreeEntry{Path: "model.gguf", Size: 2600}, false},
		{"other commit", "1111111111111111111111111111111111111111", hfwrap.TreeEntry{Path: "model.gguf", Size: 2500}, false},
		{"no commit", "", hfwrap.TreeEntry{Path: "model.gguf", Size: 2500}, false},
		{"branch name", "main", hfwrap.TreeEntry{Path: "model.gguf", Size: 2500}, false},
		{"path escapes", commit, hfwrap.TreeEntry{Path: "../../blobs/refs-escape", Size: 1}, false},
	}
	for _, tc := range cases {
		if got := hasHFFile(cache, repo, tc.commit, tc.e); got != tc.want {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
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
