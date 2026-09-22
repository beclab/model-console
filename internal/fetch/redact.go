package fetch

import "regexp"

// Secret-bearing substrings that an upstream error body can echo back.
// The worst offender is an S3/GCS "SignatureDoesNotMatch" 403, whose XML
// reproduces the signed CanonicalRequest verbatim — including the
// Authorization header value and the X-Amz-Signature. That body flows
// into HTTPStatusError.Body, then into logs and /api/progress last_error,
// so it must be scrubbed before storage.
var redactPatterns = []*regexp.Regexp{
	// URL userinfo: scheme://user:password@host -> scheme://user:***@host
	regexp.MustCompile(`([a-zA-Z][a-zA-Z0-9+.\-]*://[^/\s:@]+):[^/\s@]+@`),
	// Authorization header echoed in a body ("Authorization: Bearer x",
	// "authorization=AWS4-HMAC-SHA256 ...").
	regexp.MustCompile(`(?i)(authorization\s*[:=]\s*)\S+`),
	// Bare bearer tokens not preceded by the header name.
	regexp.MustCompile(`(?i)(bearer\s+)[A-Za-z0-9._\-]+`),
	// Signed-URL / credential query params and AWS sigv4 fields.
	regexp.MustCompile(`(?i)([?&;]?(?:x-amz-signature|x-amz-credential|x-amz-security-token|signature|sig|credential|access_token|api[_-]?key|token)=)[^&\s"'<>]+`),
}

// redactSecrets masks credentials an upstream may have echoed into an
// error body. It is intentionally conservative — over-masking a
// diagnostic snippet is preferable to leaking a token into a log line or
// the /api/progress last_error field.
func redactSecrets(s string) string {
	for i, re := range redactPatterns {
		switch i {
		case 0:
			s = re.ReplaceAllString(s, `$1:***@`)
		default:
			s = re.ReplaceAllString(s, `${1}***`)
		}
	}
	return s
}
