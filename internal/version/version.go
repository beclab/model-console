// Package version exposes build-time metadata.
//
// Values are injected at link time via -ldflags, see Makefile target `build`:
//
//	go build -ldflags "\
//	  -X github.com/llm-init/llm-init/internal/version.Version=$(VERSION) \
//	  -X github.com/llm-init/llm-init/internal/version.Commit=$(COMMIT) \
//	  -X github.com/llm-init/llm-init/internal/version.BuildTime=$(BUILD_TIME)"
package version

import (
	"fmt"
	"io"
	"runtime"
)

// Default values are used when ldflags are not supplied (e.g., `go run`).
var (
	Version   = "dev"
	Commit    = "none"
	BuildTime = "unknown"
)

// Info bundles the version metadata for structured callers (logs, /api/config).
type Info struct {
	Version   string `json:"version"`
	Commit    string `json:"commit"`
	BuildTime string `json:"build_time"`
	GoVersion string `json:"go_version"`
	Platform  string `json:"platform"`
}

// Get returns a snapshot of the current build's version info.
func Get() Info {
	return Info{
		Version:   Version,
		Commit:    Commit,
		BuildTime: BuildTime,
		GoVersion: runtime.Version(),
		Platform:  fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
	}
}

// String renders the version info as a human-readable multi-line block.
func (i Info) String() string {
	return fmt.Sprintf(
		"llm-init %s\ncommit:     %s\nbuild time: %s\ngo version: %s\nplatform:   %s",
		i.Version, i.Commit, i.BuildTime, i.GoVersion, i.Platform,
	)
}

// PrintTo writes the human-readable version block to w followed by a newline.
func PrintTo(w io.Writer) error {
	_, err := fmt.Fprintln(w, Get().String())
	return err
}
