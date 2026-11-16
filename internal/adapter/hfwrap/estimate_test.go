package hfwrap

import (
	"context"
	"net/http"
	"net/http/httptest"
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

func TestEstimateRepo_ListsAtResolvedCommit(t *testing.T) {
	t.Parallel()
	const commit = "50968a4468ef4233ed78cd7c3de230dd1d61a56b"
	var revisionCalls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/o/r/revision/main":
			revisionCalls++
			_, _ = w.Write([]byte(`{"id":"o/r","sha":"` + commit + `"}`))
		case "/api/models/o/r/tree/" + commit:
			_, _ = w.Write([]byte(`[{"type":"file","path":"m.gguf","size":9}]`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)

	est, err := EstimateRepo(context.Background(), Config{Repo: "o/r", Endpoint: srv.URL}, nil, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if est.Commit != commit || est.Bytes != 9 {
		t.Fatalf("branch: %+v, want commit %s and 9 bytes", est, commit)
	}

	pinned, err := EstimateRepo(context.Background(), Config{Repo: "o/r", Revision: commit, Endpoint: srv.URL}, nil, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if pinned.Commit != commit || revisionCalls != 1 {
		t.Fatalf("pinned: commit=%q revision calls=%d, want %s and no extra call", pinned.Commit, revisionCalls, commit)
	}
}

func TestEstimateRepo_UnresolvedRevisionFallsBack(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/o/r/tree/main" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`[{"type":"file","path":"m.gguf","size":9}]`))
	}))
	t.Cleanup(srv.Close)

	est, err := EstimateRepo(context.Background(), Config{Repo: "o/r", Endpoint: srv.URL}, nil, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if est.Commit != "" || est.Bytes != 9 || !est.Complete {
		t.Fatalf("fallback: %+v, want the branch tree and no commit", est)
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
