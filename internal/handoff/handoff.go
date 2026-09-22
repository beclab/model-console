// Package handoff owns the files llm-init writes into RUN_DIR for the
// sibling engine wrapper: the engine_args it should launch with, the
// generation counter that asks it to relaunch, and llm-init's own record
// of whether anything ever acted on that counter.
//
// The channel is one-way, and that is a deployment fact rather than a
// design preference: every Market chart mounts the run dir into the
// engine container with readOnly: true, and deploy/wrappers/common.sh
// only ever reads from it. A wrapper therefore cannot declare what it
// supports or acknowledge a request. Whether a restart happened is
// something llm-init observes (see Tracker), never something it is told.
package handoff

import (
	"bytes"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"
)

const (
	// EngineArgsName carries the raw engine_args string. Wrappers prefer
	// this file over the chart's ENGINE_ARGS env so the model card is the
	// single source of truth.
	EngineArgsName = "engine_args"

	// RestartName holds an opaque generation string. A supervising
	// wrapper polls it and, on change, stops the engine child and
	// relaunches with the current EngineArgsName contents.
	RestartName = "engine_restart"

	// StateName is llm-init's own notebook, not part of the wrapper
	// contract: nothing reads it but us. It records what we have
	// observed about past restart requests so a receipt can distinguish
	// "nobody is listening" from "we have not looked yet".
	StateName = "engine_restart_state"

	// WrappersDirName is the subdirectory of RUN_DIR where the wrapper
	// helpers are published.
	WrappersDirName = "wrappers"

	// HelpersName is the only file published there. The engine scripts
	// themselves stay with the chart, which is where their per-application
	// customisation lives; the helpers are the part that has to be one
	// implementation everywhere, because supervise_engine's correctness
	// is not something 40 hand-copies can hold.
	HelpersName = "common.sh"

	// DefaultHelpersSourceDir is where the image keeps its copy. The
	// Dockerfile COPYs deploy/wrappers/common.sh here from the build
	// context, so the file published at runtime is byte-identical to the
	// one in this repository and shellcheck / tests/wrappers cover it.
	DefaultHelpersSourceDir = "/usr/share/llm-init/wrappers"
)

// WriteEngineArgs atomically writes raw to ${runDir}/engine_args. Empty
// runDir is a no-op. The file is always written, including when raw is
// empty, so a wrapper can tell "Model Console says no flags" from "Model
// Console has not spoken, fall back to the chart env".
func WriteEngineArgs(runDir, raw string) error {
	if runDir == "" {
		return nil
	}
	if err := writeFileAtomic(runDir, EngineArgsName, []byte(raw)); err != nil {
		return fmt.Errorf("engine_args handoff: %w", err)
	}
	return nil
}

// ReadEngineArgs returns the current handoff contents. missing is true
// when the file is absent or unreadable, which callers treat as "differs
// from anything we might write" so the first write still bumps a restart.
func ReadEngineArgs(runDir string) (raw string, missing bool) {
	if runDir == "" {
		return "", true
	}
	data, err := os.ReadFile(filepath.Join(runDir, EngineArgsName))
	if err != nil {
		return "", true
	}
	return string(data), false
}

// Bump writes a new generation to ${runDir}/engine_restart and returns
// it. Empty runDir yields ("", nil): there is nowhere to signal, and the
// caller decides whether that is an error in its context.
func Bump(runDir string) (generation string, err error) {
	if runDir == "" {
		return "", nil
	}
	path := filepath.Join(runDir, RestartName)
	prev := []byte("0")
	if data, readErr := os.ReadFile(path); readErr == nil {
		prev = bytes.TrimSpace(data)
	}
	next := fmt.Sprintf("%d", time.Now().UnixNano())
	if string(prev) == next {
		next += ".1"
	}
	if err := writeFileAtomic(runDir, RestartName, []byte(next+"\n")); err != nil {
		return "", fmt.Errorf("engine_restart handoff: %w", err)
	}
	slog.Info("engine restart requested", "path", path, "gen", next)
	return next, nil
}

// ReadGeneration returns the generation currently on disk, or "" when
// there is none.
func ReadGeneration(runDir string) string {
	if runDir == "" {
		return ""
	}
	data, err := os.ReadFile(filepath.Join(runDir, RestartName))
	if err != nil {
		return ""
	}
	return string(bytes.TrimSpace(data))
}

// PublishWrapperHelpers copies the engine wrappers' shared helpers from
// srcDir into ${runDir}/wrappers so the sibling engine container sources
// one implementation instead of the copy its chart happened to inline.
// Returns the published path, or "" when there was nothing to do.
//
// This is the one file that travels the other way round from the rest of
// this package: the others are state llm-init produces, this one is code
// it merely forwards from its own image. Publishing it every boot is what
// makes an image upgrade the way to fix a wrapper bug.
//
// A missing srcDir is not an error. `go run ./cmd/llm-init` has no
// /usr/share tree, and a deployment whose engine wrapper does not source
// the published copy is unaffected by its absence.
func PublishWrapperHelpers(runDir, srcDir string) (string, error) {
	if runDir == "" {
		return "", nil
	}
	if srcDir == "" {
		srcDir = DefaultHelpersSourceDir
	}
	src := filepath.Join(srcDir, HelpersName)
	data, err := os.ReadFile(src)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Info("wrapper helpers not shipped in this image; skipping publish", "path", src)
			return "", nil
		}
		return "", fmt.Errorf("read wrapper helpers %s: %w", src, err)
	}
	dir := filepath.Join(runDir, WrappersDirName)
	if err := writeFileAtomic(dir, HelpersName, data); err != nil {
		return "", fmt.Errorf("publish wrapper helpers: %w", err)
	}
	dst := filepath.Join(dir, HelpersName)
	slog.Info("published wrapper helpers", "path", dst, "bytes", len(data))
	return dst, nil
}

// writeFileAtomic writes name under dir via tmp+rename so a wrapper
// polling the file never reads a partial write.
func writeFileAtomic(dir, name string, data []byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}
	path := filepath.Join(dir, name)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("rename %s: %w", path, err)
	}
	return nil
}
