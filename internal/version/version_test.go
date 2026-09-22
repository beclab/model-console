package version

import (
	"bytes"
	"runtime"
	"strings"
	"testing"
)

func TestGetDefaults(t *testing.T) {
	t.Parallel()
	info := Get()
	if info.Version == "" {
		t.Errorf("Version should not be empty; got %q", info.Version)
	}
	if info.Commit == "" {
		t.Errorf("Commit should not be empty; got %q", info.Commit)
	}
	if info.BuildTime == "" {
		t.Errorf("BuildTime should not be empty; got %q", info.BuildTime)
	}
	if info.GoVersion != runtime.Version() {
		t.Errorf("GoVersion = %q, want %q", info.GoVersion, runtime.Version())
	}
	wantPlatform := runtime.GOOS + "/" + runtime.GOARCH
	if info.Platform != wantPlatform {
		t.Errorf("Platform = %q, want %q", info.Platform, wantPlatform)
	}
}

func TestInfoStringContainsAllFields(t *testing.T) {
	t.Parallel()
	info := Info{
		Version:   "v1.2.3",
		Commit:    "abcdef",
		BuildTime: "2026-05-13T22:30:00Z",
		GoVersion: "go1.22.0",
		Platform:  "linux/amd64",
	}
	s := info.String()
	for _, want := range []string{
		"llm-init v1.2.3",
		"abcdef",
		"2026-05-13T22:30:00Z",
		"go1.22.0",
		"linux/amd64",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("String() missing %q\noutput:\n%s", want, s)
		}
	}
}

func TestPrintToWritesNewline(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	if err := PrintTo(&buf); err != nil {
		t.Fatalf("PrintTo: %v", err)
	}
	out := buf.String()
	if !strings.HasSuffix(out, "\n") {
		t.Errorf("PrintTo output missing trailing newline: %q", out)
	}
	if !strings.Contains(out, "llm-init") {
		t.Errorf("PrintTo output missing prefix: %q", out)
	}
}

// TestLDFlagsOverride documents the ldflags mechanism: package-level vars
// are mutated via -X at link time. This test ensures the vars are exported
// strings (compile-time guarantee for ldflags).
func TestLDFlagsOverride(t *testing.T) {
	original := Version
	t.Cleanup(func() { Version = original })
	Version = "v9.9.9-test"
	if got := Get().Version; got != "v9.9.9-test" {
		t.Errorf("Get().Version after override = %q, want v9.9.9-test", got)
	}
}
