package hfwrap

import (
	"path"
	"strings"
)

// MatchPatterns reports whether rel (repository-relative, slash-separated)
// is selected by huggingface_hub-style --include / --exclude.
//
// No include patterns means the request asked for the whole repository.
// A pattern is tried against the whole path and against its base name,
// because huggingface_hub matches with fnmatch, whose `*` crosses a
// separator, while Go's path.Match stops at one — and `--include *.gguf`
// has to keep matching `onnx/model.gguf`.
func MatchPatterns(rel string, include, exclude []string) bool {
	rel = strings.TrimPrefix(rel, "/")
	for _, pattern := range exclude {
		if matchPattern(pattern, rel) {
			return false
		}
	}
	if len(include) == 0 {
		return true
	}
	for _, pattern := range include {
		if matchPattern(pattern, rel) {
			return true
		}
	}
	return false
}

func matchPattern(pattern, rel string) bool {
	pattern = strings.TrimSpace(pattern)
	if pattern == "" {
		return false
	}
	if ok, err := path.Match(pattern, rel); err == nil && ok {
		return true
	}
	ok, err := path.Match(pattern, path.Base(rel))
	return err == nil && ok
}

// UnderSubdir reports whether rel sits at or under subdir. An empty
// subdir accepts every path.
func UnderSubdir(rel, subdir string) bool {
	subdir = strings.Trim(strings.TrimSpace(subdir), "/")
	if subdir == "" {
		return true
	}
	rel = strings.TrimPrefix(rel, "/")
	return rel == subdir || strings.HasPrefix(rel, subdir+"/")
}

func looksLikeGlob(p string) bool {
	return strings.ContainsAny(p, "*?[")
}
