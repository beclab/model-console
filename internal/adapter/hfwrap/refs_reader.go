package hfwrap

import (
	"errors"
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
