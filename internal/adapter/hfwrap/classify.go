package hfwrap

import (
	"regexp"
	"strings"
)

// CodePermission is the v1.1 permission-error code (EACCES / EROFS on
// the cache root). Permanent: lifecycle parks until the operator fixes
// the mount.
const CodePermission = 51

// classifyExit maps a subprocess exit code + stderr tail into the
// (Code, Retriable) tuple Run consumes. tail is the last few stderr
// lines (non-empty after stripping the tqdm progress noise); the
// pattern matching below is intentionally substring-based so hub
// version bumps that rename exception classes show up as test
// failures rather than silent reclassifications.
//
// Unknown errors fall through to (CodeNetwork, retriable=true) by
// design: the lifecycle backoff is capped so a true "permanent unknown"
// will eventually surface to the operator via /api/progress, while a
// transient unknown gets a free retry.
func classifyExit(exit int, tail string) (code int, retriable bool) {
	if exit == CodeOK {
		return CodeOK, false
	}
	if exit < 0 || exit == CodeProcessKilled {
		return CodeProcessKilled, true
	}

	switch {
	case strings.Contains(tail, "RepositoryNotFoundError"):
		return CodeRepoNotFound, false
	case strings.Contains(tail, "GatedRepoError"),
		strings.Contains(tail, "Cannot access gated repo"):
		return CodeGated, false
	case strings.Contains(tail, "RevisionNotFoundError"):
		return CodeRevision, false
	case strings.Contains(tail, "LocalTokenNotFoundError"),
		strings.Contains(tail, "401 Client Error"),
		strings.Contains(tail, "Invalid credentials"):
		return CodeTokenMissing, false
	case strings.Contains(tail, "ENOSPC"),
		strings.Contains(tail, "No space left on device"):
		return CodeDiskFull, false
	case strings.Contains(tail, "EACCES"),
		strings.Contains(tail, "Permission denied"),
		strings.Contains(tail, "EROFS"),
		strings.Contains(tail, "Read-only file system"):
		return CodePermission, false
	case strings.Contains(tail, "ConnectionError"),
		strings.Contains(tail, "ReadTimeout"),
		strings.Contains(tail, "Read timed out"),
		strings.Contains(tail, "ChunkedEncodingError"),
		strings.Contains(tail, "XetDownloadError"),
		strings.Contains(tail, "Name or service not known"),
		strings.Contains(tail, "Connection refused"),
		strings.Contains(tail, "429"):
		return CodeNetwork, true
	case http5xxRE.MatchString(tail):
		return CodeNetwork, true
	case http4xxRE.MatchString(tail):
		return CodeInternal, false
	case strings.Contains(tail, "unrecognized arguments"),
		strings.Contains(tail, "no such option"),
		strings.Contains(tail, "executable file not found"):
		return CodeBadArgs, false
	}
	return CodeNetwork, true
}

var (
	http5xxRE = regexp.MustCompile(`\b5\d{2}\b`)
	http4xxRE = regexp.MustCompile(`\b4\d{2}\b`)
)
