// Package sentinel manages the readiness sentinel files that engine wrapper
// scripts poll to discover when llm-init has finished preparing the model.
//
// Files are written into RUN_DIR (default /run/llm-init, a volume shared
// with the engine container:
//
//	model_download_finish  ISO-8601 timestamp, presence == ready
//	model_path             absolute path the engine should load (single line)
//	extra_model_path       optional; ROLE=extra paths, one absolute path per line
//	                       (e.g. ocrlayout reads line 1 for an ONNX dir)
//
// Wrappers do `until [ -f $RUN_DIR/model_download_finish ]; do sleep 5; done`,
// then `exec engine $(cat $RUN_DIR/model_path)`. Writes are atomic: data goes
// to <name>.tmp first and is then os.Rename'd, so a wrapper can never observe
// a half-written model_path.
package sentinel

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// FinishFile is the marker file whose presence signals readiness.
	FinishFile = "model_download_finish"
	// ModelPathFile contains the absolute path the engine should load.
	ModelPathFile = "model_path"
	// ExtraModelPathFile lists ROLE=extra absolute paths (one per line).
	// ROLE=mmproj must not appear here; mmproj is handled inline by ensureHF.
	ExtraModelPathFile = "extra_model_path"
)

// Write atomically creates FinishFile and ModelPathFile inside runDir.
//
// The directory is created (mode 0o755) if missing. modelPath must be a
// non-empty absolute path; relative or empty inputs return an error to avoid
// engines being told to load "" or some unintended cwd-relative location.
func Write(runDir, modelPath string) error {
	if runDir == "" {
		return errors.New("sentinel: runDir is empty")
	}
	if modelPath == "" {
		return errors.New("sentinel: modelPath is empty")
	}
	if !filepath.IsAbs(modelPath) && !strings.HasPrefix(modelPath, "/") {
		// Engine wrappers exec the path; relative paths would resolve
		// against the engine's cwd, which is rarely correct.
		return fmt.Errorf("sentinel: modelPath must be absolute, got %q", modelPath)
	}
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return fmt.Errorf("sentinel: mkdir %s: %w", runDir, err)
	}
	now := time.Now().UTC().Format(time.RFC3339)
	if err := atomicWrite(filepath.Join(runDir, ModelPathFile), []byte(modelPath+"\n")); err != nil {
		return err
	}
	if err := atomicWrite(filepath.Join(runDir, FinishFile), []byte(now+"\n")); err != nil {
		return err
	}
	return nil
}

// WriteExtra atomically writes ExtraModelPathFile with one absolute path per
// line (ROLE=extra only; callers must not pass mmproj paths).
//
// When paths is empty, the file is removed (or left absent) so a retry after
// dropping extras cannot leave a stale ONNX / preload path for consumers
// such as ocrlayout.
func WriteExtra(runDir string, paths []string) error {
	if runDir == "" {
		return errors.New("sentinel: runDir is empty")
	}
	target := filepath.Join(runDir, ExtraModelPathFile)
	if len(paths) == 0 {
		if err := os.Remove(target); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("sentinel: remove %s: %w", target, err)
		}
		return nil
	}
	for i, p := range paths {
		if p == "" {
			return fmt.Errorf("sentinel: extra path[%d] is empty", i)
		}
		if !filepath.IsAbs(p) && !strings.HasPrefix(p, "/") {
			return fmt.Errorf("sentinel: extra path[%d] must be absolute, got %q", i, p)
		}
	}
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return fmt.Errorf("sentinel: mkdir %s: %w", runDir, err)
	}
	body := strings.Join(paths, "\n") + "\n"
	return atomicWrite(target, []byte(body))
}

// ReadExtraPaths returns the absolute paths from ExtraModelPathFile (blank
// lines dropped). Returns os.ErrNotExist when the file is missing.
func ReadExtraPaths(runDir string) ([]string, error) {
	if runDir == "" {
		return nil, errors.New("sentinel: runDir is empty")
	}
	data, err := os.ReadFile(filepath.Join(runDir, ExtraModelPathFile))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, os.ErrNotExist
		}
		return nil, fmt.Errorf("sentinel: read extra_model_path: %w", err)
	}
	var out []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out, nil
}

// Remove deletes FinishFile, ModelPathFile, and ExtraModelPathFile from
// runDir. Missing files are not an error.
//
// What this can and cannot do is worth being exact about, because the
// file name suggests more power than it has. Every wrapper waits for the
// sentinel once and then execs the engine; nothing re-reads it. So
// removing it stops no engine that is already running, and an operator
// who deletes it expecting the model to go offline has done nothing.
//
// It matters for the process that has not read it yet: a container
// restarted by the kubelet, a CrashLoop, or a supervise loop launching
// for the first time. That is exactly the window a force re-download
// opens — the bytes model_path names are deleted while the sentinel
// still says "ready, load this" — and an engine that restarts in it
// execs a path that is gone. Removing the sentinel first makes it wait
// instead, until the pass rewrites it.
func Remove(runDir string) error {
	if runDir == "" {
		return errors.New("sentinel: runDir is empty")
	}
	for _, name := range []string{FinishFile, ModelPathFile, ExtraModelPathFile} {
		p := filepath.Join(runDir, name)
		if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("sentinel: remove %s: %w", p, err)
		}
	}
	return nil
}

// Exists reports whether FinishFile is present in runDir.
func Exists(runDir string) (bool, error) {
	if runDir == "" {
		return false, errors.New("sentinel: runDir is empty")
	}
	_, err := os.Stat(filepath.Join(runDir, FinishFile))
	if err == nil {
		return true, nil
	}
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	return false, fmt.Errorf("sentinel: stat: %w", err)
}

// ReadModelPath reads the absolute path from ModelPathFile (whitespace
// trimmed). Returns os.ErrNotExist when the file is missing.
func ReadModelPath(runDir string) (string, error) {
	if runDir == "" {
		return "", errors.New("sentinel: runDir is empty")
	}
	data, err := os.ReadFile(filepath.Join(runDir, ModelPathFile))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", os.ErrNotExist
		}
		return "", fmt.Errorf("sentinel: read model_path: %w", err)
	}
	return strings.TrimSpace(string(data)), nil
}

// atomicWrite writes data to path via a sibling .tmp file followed by
// os.Rename so readers (engine wrappers) never observe a partial write.
// fsync is intentionally skipped: the sentinel lives on tmpfs/emptyDir, so
// crash safety is irrelevant; survivability requires re-running llm-init
// anyway.
func atomicWrite(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("sentinel: write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("sentinel: rename %s: %w", path, err)
	}
	return nil
}
