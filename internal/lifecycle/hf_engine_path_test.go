package lifecycle

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter/hfwrap"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/sentinel"
)

func TestResolveHFEnginePath_NoSubdir(t *testing.T) {
	t.Parallel()
	main := config.ModelSource{HFInclude: []string{"a.gguf", "b.gguf"}}
	snap := "/cache/hf/hub/models--o--r/snapshots/abc"
	got, err := resolveHFEnginePath(main, hfwrap.Result{Path: snap})
	if err != nil {
		t.Fatal(err)
	}
	if got != snap {
		t.Fatalf("got %q want %q", got, snap)
	}
}

func TestResolveHFEnginePath_SingleIncludeFile(t *testing.T) {
	t.Parallel()
	main := config.ModelSource{HFInclude: []string{"model.gguf"}}
	file := "/cache/hf/hub/models--o--r/snapshots/abc/model.gguf"
	got, err := resolveHFEnginePath(main, hfwrap.Result{Path: file})
	if err != nil {
		t.Fatal(err)
	}
	if got != file {
		t.Fatalf("got %q want %q", got, file)
	}
}

func TestResolveHFEnginePath_Subdir(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	snap := filepath.Join(root, "snapshots", "abc")
	onnx := filepath.Join(snap, "onnx")
	if err := os.MkdirAll(onnx, 0o755); err != nil {
		t.Fatal(err)
	}

	main := config.ModelSource{HFSubdir: "onnx"}
	got, err := resolveHFEnginePath(main, hfwrap.Result{Path: snap})
	if err != nil {
		t.Fatal(err)
	}
	if got != onnx {
		t.Fatalf("got %q want %q", got, onnx)
	}
}

func TestResolveHFEnginePath_SubdirMissing(t *testing.T) {
	t.Parallel()
	snap := t.TempDir()
	main := config.ModelSource{HFSubdir: "onnx"}
	if _, err := resolveHFEnginePath(main, hfwrap.Result{Path: snap}); err == nil {
		t.Fatal("expected error for missing subdir")
	}
}

func TestEnsureHF_SubdirSentinel(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	snap := filepath.Join(root, "snapshots", "abc")
	onnx := filepath.Join(snap, "onnx")
	if err := os.MkdirAll(onnx, 0o755); err != nil {
		t.Fatal(err)
	}

	src := hfSource("beclab/embeddinggemma-300m", "abc")
	src.HFSubdir = "onnx"
	src.HFInclude = []string{"onnx/*", "onnx/**/*"}

	o := baseOptions(t, []config.ModelSource{src})
	o.Config.Runtime.RunDir = filepath.Join(root, "run")
	o.HFRun = func(context.Context, hfwrap.Config, progress.Sink) (hfwrap.Result, error) {
		return hfwrap.Result{Status: "ok", Path: snap, Commit: "abc"}, nil
	}

	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	got, err := sentinel.ReadModelPath(o.Config.Runtime.RunDir)
	if err != nil {
		t.Fatalf("ReadModelPath: %v", err)
	}
	if got != onnx {
		t.Fatalf("model_path = %q want %q", got, onnx)
	}
}
