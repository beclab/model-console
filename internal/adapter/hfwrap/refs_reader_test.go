package hfwrap

import (
	"os"
	"path/filepath"
	"testing"
)

func TestHFRepoCacheDirName(t *testing.T) {
	for _, tc := range []struct {
		repo string
		want string
	}{
		{"Qwen/Qwen3-27B", "models--Qwen--Qwen3-27B"},
		{"unsloth/Llama-3.2-3B-Instruct-Q4_K_M-GGUF", "models--unsloth--Llama-3.2-3B-Instruct-Q4_K_M-GGUF"},
	} {
		got := hfRepoCacheDirName(tc.repo)
		if got != tc.want {
			t.Errorf("repo=%q got %q want %q", tc.repo, got, tc.want)
		}
	}
}

func TestSnapshotPath(t *testing.T) {
	got := SnapshotPath("/cache/hf/hub", "Qwen/Qwen3-27B", "abc123def456")
	want := filepath.Join("/cache/hf/hub", "models--Qwen--Qwen3-27B", "snapshots", "abc123def456")
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestReadRefCommit_FromFile(t *testing.T) {
	cache := t.TempDir()
	repoDir := filepath.Join(cache, "models--x--y", "refs")
	if err := os.MkdirAll(repoDir, 0o755); err != nil {
		t.Fatal(err)
	}
	sha := "abcdef0123456789abcdef0123456789abcdef01"
	if err := os.WriteFile(filepath.Join(repoDir, "main"), []byte(sha+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadRefCommit(cache, "x/y", "main")
	if err != nil {
		t.Fatalf("ReadRefCommit: %v", err)
	}
	if got != sha {
		t.Errorf("got %q want %q", got, sha)
	}
}

func TestReadRefCommit_HexShortCircuit(t *testing.T) {
	sha := "abcdef0123456789abcdef0123456789abcdef01"
	got, err := ReadRefCommit("/nonexistent", "x/y", sha)
	if err != nil {
		t.Fatalf("hex SHA must short-circuit without I/O: %v", err)
	}
	if got != sha {
		t.Errorf("got %q want %q", got, sha)
	}
}

func TestReadRefCommit_DefaultsToMain(t *testing.T) {
	cache := t.TempDir()
	repoDir := filepath.Join(cache, "models--x--y", "refs")
	if err := os.MkdirAll(repoDir, 0o755); err != nil {
		t.Fatal(err)
	}
	sha := "0011223344556677001122334455667700112233"
	if err := os.WriteFile(filepath.Join(repoDir, "main"), []byte(sha), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := ReadRefCommit(cache, "x/y", "")
	if err != nil {
		t.Fatalf("ReadRefCommit: %v", err)
	}
	if got != sha {
		t.Errorf("got %q want %q", got, sha)
	}
}

func TestReadRefCommit_MissingReturnsEmpty(t *testing.T) {
	cache := t.TempDir()
	got, err := ReadRefCommit(cache, "missing/repo", "main")
	if err != nil {
		t.Errorf("missing ref should be soft (got err %v); xet path occasionally skips refs write", err)
	}
	if got != "" {
		t.Errorf("missing ref must yield empty SHA; got %q", got)
	}
}

func TestIsHexSHA(t *testing.T) {
	for _, tc := range []struct {
		s    string
		want bool
	}{
		{"abcdef0123456789abcdef0123456789abcdef01", true},
		{"ABCDEF0123456789ABCDEF0123456789ABCDEF01", false}, // hf canonicalises to lower hex
		{"abcdef0123456789abcdef0123456789abcdef0", false},  // too short
		{"main", false},
		{"v1.0", false},
		{"", false},
	} {
		got := isHexSHA(tc.s)
		if got != tc.want {
			t.Errorf("isHexSHA(%q)=%v want %v", tc.s, got, tc.want)
		}
	}
}
