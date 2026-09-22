package fetch

import (
	"strings"
	"testing"
)

func TestRedactSecrets(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		in       string
		mustHave []string // substrings that must survive (non-secret context)
		mustnt   []string // secret substrings that must be gone
	}{
		{
			name:     "url userinfo",
			in:       "failed GET https://alice:s3cr3t-token@host/path",
			mustnt:   []string{"s3cr3t-token"},
			mustHave: []string{"alice", "host/path"},
		},
		{
			name:   "authorization header echoed",
			in:     "CanonicalRequest:\nauthorization:AWS4-HMAC-SHA256 Credential=AKIA/x, Signature=deadbeefcafe",
			mustnt: []string{"AWS4-HMAC-SHA256", "AKIA/x", "deadbeefcafe"},
		},
		{
			name:   "bearer token",
			in:     "401 Unauthorized: Bearer eyJhbGciOi.payload.sig",
			mustnt: []string{"eyJhbGciOi.payload.sig"},
		},
		{
			name:     "presigned query params",
			in:       "denied for s3.amazonaws.com/key?X-Amz-Signature=abc123&X-Amz-Credential=AKIA%2Fz",
			mustnt:   []string{"abc123", "AKIA%2Fz"},
			mustHave: []string{"s3.amazonaws.com/key"},
		},
		{
			name:     "innocuous body untouched",
			in:       "404 Not Found: object does not exist",
			mustHave: []string{"404 Not Found", "object does not exist"},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := redactSecrets(tc.in)
			for _, s := range tc.mustnt {
				if strings.Contains(got, s) {
					t.Errorf("secret %q survived redaction: %q", s, got)
				}
			}
			for _, s := range tc.mustHave {
				if !strings.Contains(got, s) {
					t.Errorf("expected %q to survive, got %q", s, got)
				}
			}
		})
	}
}
