package handoff

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublishWrapperHelpers_CopiesTheImageSCopy(t *testing.T) {
	t.Parallel()
	src := t.TempDir()
	run := t.TempDir()
	want := "# helpers\nlog() { :; }\n"
	if err := os.WriteFile(filepath.Join(src, HelpersName), []byte(want), 0o644); err != nil {
		t.Fatalf("seed source: %v", err)
	}

	path, err := PublishWrapperHelpers(run, src)
	if err != nil {
		t.Fatalf("PublishWrapperHelpers: %v", err)
	}
	if path != filepath.Join(run, WrappersDirName, HelpersName) {
		t.Errorf("published to %q, want ${RUN_DIR}/%s/%s", path, WrappersDirName, HelpersName)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read published: %v", err)
	}
	// Byte-identical matters: the wrapper sources this, and the copy in
	// this repository is the one shellcheck and tests/wrappers cover.
	if string(got) != want {
		t.Errorf("published contents = %q, want the source verbatim %q", got, want)
	}

	// Republished every boot, so an image upgrade is how a wrapper bug
	// gets fixed. A second call must overwrite rather than fail.
	next := want + "log2() { :; }\n"
	if err := os.WriteFile(filepath.Join(src, HelpersName), []byte(next), 0o644); err != nil {
		t.Fatalf("update source: %v", err)
	}
	if _, err := PublishWrapperHelpers(run, src); err != nil {
		t.Fatalf("second publish: %v", err)
	}
	got, err = os.ReadFile(path)
	if err != nil {
		t.Fatalf("read republished: %v", err)
	}
	if string(got) != next {
		t.Errorf("republished contents = %q, want the updated source %q", got, next)
	}

	// The temp file the atomic write goes through must not be left where
	// a wrapper globbing the directory could source it.
	entries, err := os.ReadDir(filepath.Join(run, WrappersDirName))
	if err != nil {
		t.Fatalf("read published dir: %v", err)
	}
	for _, e := range entries {
		if e.Name() != HelpersName {
			t.Errorf("published dir holds %q, want only %s", e.Name(), HelpersName)
		}
	}
}

func TestPublishWrapperHelpers_MissingSourceIsNotAnError(t *testing.T) {
	t.Parallel()
	// `go run ./cmd/llm-init` has no /usr/share tree, and a deployment
	// whose wrapper sources its own copy does not care either way.
	path, err := PublishWrapperHelpers(t.TempDir(), filepath.Join(t.TempDir(), "absent"))
	if err != nil {
		t.Fatalf("missing source dir: %v", err)
	}
	if path != "" {
		t.Errorf("published %q, want nothing", path)
	}
}

func TestPublishWrapperHelpers_NoRunDirIsANoOp(t *testing.T) {
	t.Parallel()
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, HelpersName), []byte("x"), 0o644); err != nil {
		t.Fatalf("seed source: %v", err)
	}
	path, err := PublishWrapperHelpers("", src)
	if err != nil {
		t.Fatalf("empty runDir: %v", err)
	}
	if path != "" {
		t.Errorf("published %q, want nothing", path)
	}
}

// TestImageShipsTheWrapperHelpers pins the Dockerfile line that puts
// common.sh where DefaultHelpersSourceDir looks for it. Without it the
// publish silently does nothing -- the missing-source path above is
// deliberately not an error -- and the only place that shows up is a
// deployment whose engine waits an hour for a file nobody writes.
func TestImageShipsTheWrapperHelpers(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("..", "..", "Dockerfile"))
	if err != nil {
		t.Fatalf("read Dockerfile: %v", err)
	}
	want := "deploy/wrappers/" + HelpersName + " " + DefaultHelpersSourceDir + "/" + HelpersName
	if !strings.Contains(string(data), want) {
		t.Errorf("Dockerfile does not COPY %q; PublishWrapperHelpers would find nothing at runtime", want)
	}
}
