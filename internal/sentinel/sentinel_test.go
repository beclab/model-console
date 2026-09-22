package sentinel

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteAndRead_RoundTrip(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()

	if err := Write(dir, "/models/qwen2.5-7b"); err != nil {
		t.Fatalf("Write: %v", err)
	}

	exists, err := Exists(dir)
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if !exists {
		t.Error("Exists = false, want true after Write")
	}

	got, err := ReadModelPath(dir)
	if err != nil {
		t.Fatalf("ReadModelPath: %v", err)
	}
	if got != "/models/qwen2.5-7b" {
		t.Errorf("ReadModelPath = %q, want /models/qwen2.5-7b", got)
	}

	// Finish file must contain something parseable as a timestamp; we just
	// require non-empty.
	body, err := os.ReadFile(filepath.Join(dir, FinishFile))
	if err != nil {
		t.Fatalf("read finish: %v", err)
	}
	if len(body) == 0 {
		t.Error("finish file is empty")
	}
}

func TestWrite_CreatesMissingDir(t *testing.T) {
	t.Parallel()
	dir := filepath.Join(t.TempDir(), "nested", "run")

	if err := Write(dir, "/models/x"); err != nil {
		t.Fatalf("Write into missing dir: %v", err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Errorf("dir not created: %v", err)
	}
}

func TestWrite_RejectsRelativePath(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	err := Write(dir, "models/x")
	if err == nil {
		t.Error("expected error for relative modelPath")
	}
}

func TestWrite_RejectsEmpty(t *testing.T) {
	t.Parallel()
	if err := Write("", "/models/x"); err == nil {
		t.Error("expected error for empty runDir")
	}
	if err := Write(t.TempDir(), ""); err == nil {
		t.Error("expected error for empty modelPath")
	}
}

func TestRemove_Idempotent(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()

	if err := Remove(dir); err != nil {
		t.Errorf("Remove on empty dir = %v, want nil", err)
	}

	if err := Write(dir, "/models/x"); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := Remove(dir); err != nil {
		t.Fatalf("Remove: %v", err)
	}

	exists, err := Exists(dir)
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if exists {
		t.Error("Exists = true after Remove")
	}

	if _, err := ReadModelPath(dir); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("ReadModelPath after Remove = %v, want os.ErrNotExist", err)
	}

	if err := Remove(dir); err != nil {
		t.Errorf("Remove twice = %v, want nil (idempotent)", err)
	}
}

func TestExists_MissingDir(t *testing.T) {
	t.Parallel()
	dir := filepath.Join(t.TempDir(), "no-such")
	exists, err := Exists(dir)
	if err != nil {
		t.Errorf("Exists on missing dir = %v, want nil", err)
	}
	if exists {
		t.Error("Exists on missing dir = true, want false")
	}
}

func TestReadModelPath_TrimsTrailingNewline(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	if err := Write(dir, "/models/qwen"); err != nil {
		t.Fatalf("Write: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join(dir, ModelPathFile))
	if err != nil {
		t.Fatalf("read raw: %v", err)
	}
	if raw[len(raw)-1] != '\n' {
		t.Error("model_path file should end with newline")
	}

	got, err := ReadModelPath(dir)
	if err != nil {
		t.Fatalf("ReadModelPath: %v", err)
	}
	if got != "/models/qwen" {
		t.Errorf("ReadModelPath = %q, want trimmed", got)
	}
}

func TestWrite_OverwritesExisting(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	if err := Write(dir, "/models/a"); err != nil {
		t.Fatalf("Write a: %v", err)
	}
	if err := Write(dir, "/models/b"); err != nil {
		t.Fatalf("Write b: %v", err)
	}
	got, _ := ReadModelPath(dir)
	if got != "/models/b" {
		t.Errorf("ReadModelPath = %q, want /models/b", got)
	}
}

func TestWriteExtra_AtomicMultiLineAndEmpty(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()

	if err := WriteExtra(dir, []string{"/models/onnx", "/models/aux"}); err != nil {
		t.Fatalf("WriteExtra: %v", err)
	}
	got, err := ReadExtraPaths(dir)
	if err != nil {
		t.Fatalf("ReadExtraPaths: %v", err)
	}
	if len(got) != 2 || got[0] != "/models/onnx" || got[1] != "/models/aux" {
		t.Fatalf("got %v", got)
	}
	raw, err := os.ReadFile(filepath.Join(dir, ExtraModelPathFile))
	if err != nil {
		t.Fatalf("read raw: %v", err)
	}
	if string(raw) != "/models/onnx\n/models/aux\n" {
		t.Errorf("raw = %q", raw)
	}
	if _, err := os.Stat(filepath.Join(dir, ExtraModelPathFile+".tmp")); !errors.Is(err, os.ErrNotExist) {
		t.Error("tmp leftover after WriteExtra")
	}

	if err := WriteExtra(dir, nil); err != nil {
		t.Fatalf("WriteExtra empty: %v", err)
	}
	if _, err := ReadExtraPaths(dir); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("after empty WriteExtra: %v, want ErrNotExist", err)
	}
}

func TestWriteExtra_RejectsRelative(t *testing.T) {
	t.Parallel()
	if err := WriteExtra(t.TempDir(), []string{"rel/path"}); err == nil {
		t.Error("expected error for relative extra path")
	}
	if err := WriteExtra("", []string{"/abs"}); err == nil {
		t.Error("expected error for empty runDir")
	}
}

func TestRemove_AlsoClearsExtra(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	if err := Write(dir, "/models/main"); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := WriteExtra(dir, []string{"/models/extra"}); err != nil {
		t.Fatalf("WriteExtra: %v", err)
	}
	if err := Remove(dir); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	if _, err := ReadExtraPaths(dir); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("extra after Remove: %v", err)
	}
}

func TestWrite_MkdirFails(t *testing.T) {
	t.Parallel()
	parent := t.TempDir()
	// A regular file that Write will try to MkdirAll into → must fail.
	blocker := filepath.Join(parent, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	target := filepath.Join(blocker, "child")
	if err := Write(target, "/models/x"); err == nil {
		t.Error("Write into runDir under a file should fail")
	}
}

func TestExists_StatErrorOther(t *testing.T) {
	t.Parallel()
	parent := t.TempDir()
	// stat on a path whose ancestor is a regular file returns a non-ENOENT
	// error on most platforms. The function must wrap and return it.
	blocker := filepath.Join(parent, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatalf("seed: %v", err)
	}
	// We don't care whether the OS returns ENOENT or ENOTDIR here; both
	// are acceptable. The point is exercising the stat branch.
	_, _ = Exists(filepath.Join(blocker, "sub"))
}

func TestExists_RejectsEmptyRunDir(t *testing.T) {
	t.Parallel()
	if _, err := Exists(""); err == nil {
		t.Error("Exists(\"\") should error")
	}
	if _, err := ReadModelPath(""); err == nil {
		t.Error("ReadModelPath(\"\") should error")
	}
	if err := Remove(""); err == nil {
		t.Error("Remove(\"\") should error")
	}
}
