package config

import (
	"fmt"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"
)

// Getenv abstracts environment lookup so tests can inject deterministic maps
// without touching os.Environ.
type Getenv func(string) string

// stringDefault returns g(key) if non-empty, else def.
func stringDefault(g Getenv, key, def string) string {
	if v := strings.TrimSpace(g(key)); v != "" {
		return v
	}
	return def
}

// requireString returns g(key) trimmed; empty values produce an error.
func requireString(g Getenv, key string) (string, error) {
	v := strings.TrimSpace(g(key))
	if v == "" {
		return "", fmt.Errorf("%s: required but not set", key)
	}
	return v, nil
}

// parseEnum validates that v is one of allowed and returns the
// canonical form (the matching entry from allowed) so downstream
// code sees a single normalized value regardless of how operators
// spelled the env var. Matching is case-insensitive via
// strings.EqualFold; this aligns with parseBool's pre-existing
// case-insensitive contract and means operators can use INFO /
// Info / info interchangeably. All allowed slices in this package
// are authored lowercase, so the canonical form returned is the
// lowercase variant — callers (e.g. cfg.Log.Level == "debug")
// continue to compare against the same string literals they
// always have.
//
// An empty allowed slice is treated as "no restriction"; v is
// returned verbatim (no normalization) since there is no canonical
// set to project onto.
func parseEnum(key, v string, allowed []string) (string, error) {
	if len(allowed) == 0 {
		return v, nil
	}
	for _, a := range allowed {
		if strings.EqualFold(v, a) {
			return a, nil
		}
	}
	return "", fmt.Errorf("%s: %q is not one of %v (case-insensitive)", key, v, allowed)
}

// parseInt parses an int; empty string returns def.
func parseInt(key, v string, def int) (int, error) {
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not an integer", key, v)
	}
	return n, nil
}

// parseIntInRange parses an int and asserts min <= n <= max.
func parseIntInRange(key, v string, def, min, max int) (int, error) {
	n, err := parseInt(key, v, def)
	if err != nil {
		return 0, err
	}
	if n < min || n > max {
		return 0, fmt.Errorf("%s: %d out of range [%d, %d]", key, n, min, max)
	}
	return n, nil
}

// parseFloat parses a float64; empty string returns def.
func parseFloat(key, v string, def float64) (float64, error) {
	if v == "" {
		return def, nil
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not a float", key, v)
	}
	return f, nil
}

// parseDuration parses a Go duration; empty string returns def (which may be 0).
func parseDuration(key, v string, def time.Duration) (time.Duration, error) {
	if v == "" {
		return def, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %q is not a duration", key, v)
	}
	return d, nil
}

// parseBool accepts {1,true,yes,on,0,false,no,off}.
// Empty string returns def. Comparison is case-insensitive.
func parseBool(key, v string, def bool) (bool, error) {
	if v == "" {
		return def, nil
	}
	switch strings.ToLower(v) {
	case "1", "true", "yes", "on", "y", "t":
		return true, nil
	case "0", "false", "no", "off", "n", "f":
		return false, nil
	}
	return false, fmt.Errorf("%s: %q is not a bool", key, v)
}

// parseHTTPURL ensures v is a valid http/https URL.
func parseHTTPURL(key, v string) (string, error) {
	u, err := url.Parse(v)
	if err != nil {
		return "", fmt.Errorf("%s: %q is not a valid URL: %w", key, v, err)
	}
	if u.Scheme != schemeHTTP && u.Scheme != schemeHTTPS {
		return "", fmt.Errorf("%s: scheme must be http or https; got %q", key, u.Scheme)
	}
	if u.Host == "" {
		return "", fmt.Errorf("%s: host is empty in %q", key, v)
	}
	return v, nil
}

// parseAbsPath ensures v is non-empty and absolute.
//
// Production deployments are Linux containers so we keep paths POSIX-style
// (forward slashes) regardless of the host OS running the binary. Windows
// drive paths (`C:\...`) are accepted for test ergonomics but not normalised.
func parseAbsPath(key, v string) (string, error) {
	if v == "" {
		return "", fmt.Errorf("%s: path is empty", key)
	}
	if !isAbsPath(v) {
		return "", fmt.Errorf("%s: %q is not an absolute path", key, v)
	}
	if strings.HasPrefix(v, "/") {
		return path.Clean(v), nil
	}
	return v, nil
}

// isAbsPath returns true for POSIX absolute paths (`/...`) or Windows drive
// paths (`C:\...`). We accept both because tests run on Windows.
func isAbsPath(p string) bool {
	if strings.HasPrefix(p, "/") {
		return true
	}
	if len(p) >= 3 && p[1] == ':' && (p[2] == '/' || p[2] == '\\') {
		c := p[0]
		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
	}
	return false
}

// parseHFRepo enforces "owner/repo" format used by Hugging Face.
func parseHFRepo(key, v string) (string, error) {
	parts := strings.Split(v, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", fmt.Errorf("%s: %q must be in owner/repo format", key, v)
	}
	return v, nil
}

// parseHFCommitSHA enforces a 40-char lowercase hex commit SHA. Used by
// HF_REVISION since llm-init now requires charts to pin a specific commit
// rather than using a moving ref like "main" or a tag (which can be
// retargeted on the HF side, leading to drift across deployments).
func parseHFCommitSHA(key, v string) (string, error) {
	if len(v) != 40 {
		return "", fmt.Errorf(
			"%s: must be a 40-char commit SHA pinned from huggingface.co/<repo>/commits; got %d chars",
			key, len(v))
	}
	for _, c := range v {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return "", fmt.Errorf("%s: %q is not lowercase hex", key, v)
		}
	}
	return v, nil
}

// parseSHA256 verifies v is exactly 64 lowercase hex characters.
func parseSHA256(key, v string) (string, error) {
	if len(v) != 64 {
		return "", fmt.Errorf("%s: sha256 must be 64 hex characters; got %d", key, len(v))
	}
	for _, c := range v {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return "", fmt.Errorf("%s: %q is not lowercase hex", key, v)
		}
	}
	return v, nil
}

// parseModelName allows letters, digits and a small set of separators.
func parseModelName(key, v string) (string, error) {
	if v == "" {
		return "", fmt.Errorf("%s: required but not set", key)
	}
	for _, c := range v {
		ok := (c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == '.' || c == '_' || c == ':' || c == '-' || c == '/'
		if !ok {
			return "", fmt.Errorf("%s: %q contains illegal character %q", key, v, string(c))
		}
	}
	return v, nil
}
