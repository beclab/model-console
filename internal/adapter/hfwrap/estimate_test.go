package hfwrap

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestEstimateRepoFiles_SumsMatchingFiles(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
			{"type":"file","path":"Qwen3.5-9B-UD-IQ2_XXS.gguf","size":100},
			{"type":"file","path":"mmproj-BF16.gguf","size":50},
			{"type":"file","path":"README.md","size":1}
		]`))
	}))
	defer srv.Close()

	total, allMatched, err := EstimateRepoFiles(context.Background(), Config{
		Repo:     "unsloth/Qwen3.5-9B-GGUF",
		Revision: "main",
		Endpoint: srv.URL,
	}, []string{"Qwen3.5-9B-UD-IQ2_XXS.gguf", "mmproj-BF16.gguf"})
	if err != nil {
		t.Fatal(err)
	}
	if !allMatched || total != 150 {
		t.Fatalf("total=%d allMatched=%v, want 150 true", total, allMatched)
	}
}

func TestRepoTree_BlobIDFollowsCacheNaming(t *testing.T) {
	t.Parallel()
	const lfs = "4f2a8b0e3c1d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
			{"type":"file","path":"model.gguf","size":100,"oid":"1111111111111111111111111111111111111111",
			 "lfs":{"oid":"` + lfs + `","size":100}},
			{"type":"file","path":"config.json","size":10,"oid":"2222222222222222222222222222222222222222"}
		]`))
	}))
	defer srv.Close()

	entries, _, err := RepoTree(context.Background(), Config{Repo: "o/r", Endpoint: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("entries=%+v", entries)
	}
	if entries[0].BlobID != lfs || entries[0].SHA256 != lfs {
		t.Errorf("lfs entry=%+v, want BlobID and SHA256 = lfs oid", entries[0])
	}
	if entries[1].BlobID != "2222222222222222222222222222222222222222" || entries[1].SHA256 != "" {
		t.Errorf("plain entry=%+v, want BlobID = git oid and no SHA256", entries[1])
	}
}

func TestEstimateRepo_WholeRepoAndFilters(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
			{"type":"file","path":"README.md","size":10},
			{"type":"file","path":"onnx/model.onnx","size":100},
			{"type":"file","path":"onnx/config.json","size":5},
			{"type":"file","path":"weights/q4.gguf","size":200}
		]`))
	}))
	defer srv.Close()
	cfg := Config{Repo: "owner/repo", Revision: "main", Endpoint: srv.URL}

	whole, err := EstimateRepo(context.Background(), cfg, nil, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if !whole.Complete || whole.Bytes != 315 || whole.Files != 4 {
		t.Fatalf("whole repo: %+v, want bytes=315 files=4 complete", whole)
	}

	glob, err := EstimateRepo(context.Background(), cfg, []string{"*.gguf"}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if !glob.Complete || glob.Bytes != 200 || glob.Files != 1 {
		t.Fatalf("glob: %+v, want bytes=200 files=1", glob)
	}

	subdir, err := EstimateRepo(context.Background(), cfg, nil, nil, "onnx")
	if err != nil {
		t.Fatal(err)
	}
	if !subdir.Complete || subdir.Bytes != 105 || subdir.Files != 2 {
		t.Fatalf("subdir: %+v, want bytes=105 files=2", subdir)
	}

	excl, err := EstimateRepo(context.Background(), cfg, nil, []string{"*.md"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !excl.Complete || excl.Bytes != 305 || excl.Files != 3 {
		t.Fatalf("exclude: %+v, want bytes=305 files=3", excl)
	}

	missing, err := EstimateRepo(context.Background(), cfg, []string{"absent.gguf"}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if missing.Complete || missing.Files != 0 {
		t.Fatalf("missing exact include must not complete: %+v", missing)
	}

	partial, err := EstimateRepo(context.Background(), cfg, []string{"weights/q4.gguf", "nope.gguf"}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if partial.Complete || partial.Files != 1 || partial.Bytes != 200 {
		t.Fatalf("one of two exact includes: %+v, want files=1 bytes=200 incomplete", partial)
	}

	both, err := EstimateRepo(context.Background(), cfg, []string{"*.gguf"}, []string{"*.md"}, "weights")
	if err != nil {
		t.Fatal(err)
	}
	if !both.Complete || both.Bytes != 200 || both.Files != 1 {
		t.Fatalf("glob+exclude+subdir: %+v, want bytes=200 files=1", both)
	}
}

func TestResolveCommit(t *testing.T) {
	t.Parallel()
	const commit = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		switch r.URL.Path {
		case "/api/models/o/r/revision/main":
			_, _ = w.Write([]byte(`{"id":"o/r","sha":"` + commit + `"}`))
		case "/api/models/o/r/revision/bad":
			_, _ = w.Write([]byte(`{"sha":"../../x"}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	ctx := context.Background()

	if got, err := ResolveCommit(ctx, Config{Repo: "o/r", Endpoint: srv.URL}); err != nil || got != commit {
		t.Fatalf("branch: %q %v, want %s", got, err, commit)
	}
	if got, err := ResolveCommit(ctx, Config{Repo: "o/r", Revision: commit, Endpoint: srv.URL}); err != nil || got != commit || calls != 1 {
		t.Fatalf("pinned: %q %v calls=%d, want %s and no request", got, err, calls, commit)
	}
	if _, err := ResolveCommit(ctx, Config{Repo: "o/r", Revision: "bad", Endpoint: srv.URL}); err == nil {
		t.Error("a non-commit sha must be an error")
	}
	if _, err := ResolveCommit(ctx, Config{Repo: "o/r", Revision: "gone", Endpoint: srv.URL}); err == nil {
		t.Error("a missing revision API must be an error")
	}
}

func TestEstimateRepo_CommitOnlyWhenListedAtOne(t *testing.T) {
	t.Parallel()
	const commit = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/o/r/tree/main", "/api/models/o/r/tree/" + commit:
			_, _ = w.Write([]byte(`[{"type":"file","path":"m.gguf","size":9}]`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)

	at, err := EstimateRepo(context.Background(), Config{Repo: "o/r", Revision: commit, Endpoint: srv.URL}, nil, nil, "")
	if err != nil || at.Commit != commit || at.Bytes != 9 {
		t.Fatalf("at commit: %+v %v", at, err)
	}
	branch, err := EstimateRepo(context.Background(), Config{Repo: "o/r", Endpoint: srv.URL}, nil, nil, "")
	if err != nil || branch.Commit != "" || branch.Bytes != 9 {
		t.Fatalf("branch: %+v %v, want no commit", branch, err)
	}
}

func TestWriteRefCommit(t *testing.T) {
	t.Parallel()
	const commit = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
	cache := t.TempDir()
	for _, ref := range []string{"", "v1.0", "refs/pr/1"} {
		if err := WriteRefCommit(cache, "o/r", ref, commit); err != nil {
			t.Fatalf("ref %q: %v", ref, err)
		}
		if got, err := ReadRefCommit(cache, "o/r", ref); err != nil || got != commit {
			t.Errorf("ref %q read back %q %v", ref, got, err)
		}
	}
	if err := WriteRefCommit(cache, "o/r", "../../escape", commit); err == nil {
		t.Error("a ref leaving refs/ must be refused")
	}
	if err := WriteRefCommit(cache, "o/r", "main", "not-a-commit"); err == nil {
		t.Error("a non-commit must be refused")
	}
	if err := WriteRefCommit(cache, "o/r", commit, commit); err != nil {
		t.Errorf("a commit revision needs no ref: %v", err)
	}
}

func TestWriteRefCommit_ReadableByOtherUsers(t *testing.T) {
	t.Parallel()
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permission bits")
	}
	const commit = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
	cache := t.TempDir()
	refs := filepath.Join(HFRepoCacheDir(cache, "o/r"), "refs")

	if err := WriteRefCommit(cache, "o/r", "main", commit); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Stat(filepath.Join(refs, "main")); err != nil || info.Mode().Perm() != 0o644 {
		t.Fatalf("new ref mode = %v (%v), want 0644", info.Mode().Perm(), err)
	}

	existing := filepath.Join(refs, "v1")
	if err := os.WriteFile(existing, []byte("old"), 0o640); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(existing, 0o664); err != nil {
		t.Fatal(err)
	}
	if err := WriteRefCommit(cache, "o/r", "v1", commit); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Stat(existing); err != nil || info.Mode().Perm() != 0o664 {
		t.Fatalf("overwritten ref mode = %v (%v), want the existing 0664", info.Mode().Perm(), err)
	}
}

func TestEstimateRepo_EmptyRepoAndNoRepo(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[]`))
	}))
	t.Cleanup(srv.Close)

	empty, err := EstimateRepo(context.Background(), Config{Repo: "o/r", Endpoint: srv.URL}, nil, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if empty.Complete || empty.Files != 0 {
		t.Fatalf("empty tree: %+v", empty)
	}

	none, err := EstimateRepo(context.Background(), Config{}, nil, nil, "")
	if err != nil || none.Complete {
		t.Fatalf("no repo: %+v err=%v", none, err)
	}
}

func TestEstimateRepo_TreeError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "gated", http.StatusUnauthorized)
	}))
	t.Cleanup(srv.Close)

	_, err := EstimateRepo(context.Background(), Config{Repo: "o/r", Endpoint: srv.URL}, nil, nil, "")
	if err == nil {
		t.Fatal("expected tree error")
	}
}

func TestEstimateRepoFiles_EmptyPatterns(t *testing.T) {
	t.Parallel()
	total, all, err := EstimateRepoFiles(context.Background(), Config{Repo: "o/r"}, nil)
	if err != nil || total != 0 || all {
		t.Fatalf("empty patterns: total=%d all=%v err=%v", total, all, err)
	}
}
