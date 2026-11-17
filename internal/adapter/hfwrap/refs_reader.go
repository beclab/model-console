package hfwrap

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// defaultHFRef is HF's default branch, used when no revision is pinned.
const defaultHFRef = "main"

// hfRepoCacheDirName mirrors huggingface_hub's cache layout: every
// "<owner>/<repo>" maps to a single directory named
// "models--<owner>--<repo>" inside HF_HUB_CACHE. Slashes are doubled
// to dashes; the function expects exactly one "/" in the input.
func hfRepoCacheDirName(repo string) string {
	return "models--" + strings.ReplaceAll(repo, "/", "--")
}

// HFRepoCacheDir returns the absolute path to the per-repo cache
// directory inside the supplied HF_HUB_CACHE root. Exported so
// lifecycle can synthesise the "snapshots/<sha>" model path without
// duplicating the layout knowledge.
func HFRepoCacheDir(cacheRoot, repo string) string {
	return filepath.Join(cacheRoot, hfRepoCacheDirName(repo))
}

// SnapshotPath returns the engine-facing path to the resolved snapshot
// directory: "<HF_HUB_CACHE>/models--<owner>--<repo>/snapshots/<sha>".
// SHA must be the resolved 40-char commit hex.
func SnapshotPath(cacheRoot, repo, sha string) string {
	return filepath.Join(HFRepoCacheDir(cacheRoot, repo), "snapshots", sha)
}

// ReadRefCommit reads the resolved commit SHA huggingface_hub wrote to
// `<HF_HUB_CACHE>/models--<owner>--<repo>/refs/<ref>`. ref defaults to
// "main" when empty, mirroring HF's own default branch.
//
// If the ref file is missing or empty (xet path occasionally skips
// the write) the function returns ("", nil) so callers can fall back
// to whatever they passed in. A read error is wrapped and returned.
func ReadRefCommit(cacheRoot, repo, ref string) (string, error) {
	if isHexSHA(ref) {
		return ref, nil
	}
	if ref == "" {
		ref = defaultHFRef
	}
	path := filepath.Join(HFRepoCacheDir(cacheRoot, repo), "refs", ref)
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", nil
		}
		return "", err
	}
	sha := strings.TrimSpace(string(b))
	if sha == "" {
		return "", nil
	}
	return sha, nil
}

// WriteRefCommit records sha under refs/<ref>, the file `hf download`
// writes when it resolves a branch or tag itself. A download run at a
// commit resolved beforehand leaves that write to the caller, and
// without it the next boot cannot find the snapshot by its branch.
func WriteRefCommit(cacheRoot, repo, ref, sha string) error {
	if isHexSHA(ref) {
		return nil
	}
	if ref == "" {
		ref = defaultHFRef
	}
	if !isHexSHA(sha) {
		return fmt.Errorf("hfwrap refs: %q is not a commit", sha)
	}
	rel := filepath.FromSlash(ref)
	if !filepath.IsLocal(rel) {
		return fmt.Errorf("hfwrap refs: unsafe ref %q", ref)
	}
	path := filepath.Join(HFRepoCacheDir(cacheRoot, repo), "refs", rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	// CreateTemp makes the file 0600; an engine or app reading a shared
	// cache under another uid needs the ref readable, as hf leaves it.
	perm := os.FileMode(0o644)
	if info, err := os.Stat(path); err == nil {
		perm = info.Mode().Perm()
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".ref-*")
	if err != nil {
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return err
	}
	if _, err := tmp.WriteString(sha); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return err
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		_ = os.Remove(tmp.Name())
		return err
	}
	return nil
}

// isHexSHA reports whether s is a 40-char lowercase hex string.
// huggingface_hub canonicalises commit SHAs to lower hex, so we accept
// only that one form (uppercase would be the operator's typo, not the
// hub's output).
func isHexSHA(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, r := range s {
		switch {
		case r >= '0' && r <= '9':
		case r >= 'a' && r <= 'f':
		default:
			return false
		}
	}
	return true
}
