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
