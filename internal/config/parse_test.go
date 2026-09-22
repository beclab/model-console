package config

import (
	"strings"
	"testing"
	"time"
)

func mapEnv(m map[string]string) Getenv {
	return func(k string) string { return m[k] }
}

func TestStringDefault(t *testing.T) {
	t.Parallel()
	g := mapEnv(map[string]string{"A": "x", "B": "  ", "C": ""})
	cases := []struct{ key, def, want string }{
		{"A", "fallback", "x"},
		{"B", "fallback", "fallback"}, // whitespace treated as empty
		{"C", "fallback", "fallback"},
		{"missing", "fallback", "fallback"},
	}
	for _, tc := range cases {
		if got := stringDefault(g, tc.key, tc.def); got != tc.want {
			t.Errorf("stringDefault(%q, %q) = %q, want %q", tc.key, tc.def, got, tc.want)
		}
	}
}

func TestRequireString(t *testing.T) {
	t.Parallel()
	g := mapEnv(map[string]string{"A": "x", "B": "   "})

	if v, err := requireString(g, "A"); err != nil || v != "x" {
		t.Errorf("requireString(A) = (%q, %v), want (\"x\", nil)", v, err)
	}
	if _, err := requireString(g, "B"); err == nil {
		t.Error("requireString(B): expected error for whitespace-only value")
	}
	if _, err := requireString(g, "missing"); err == nil {
		t.Error("requireString(missing): expected error")
	}
}

func TestParseEnum(t *testing.T) {
	t.Parallel()
	allowed := []string{"a", "b", "c"}
	if v, err := parseEnum("X", "b", allowed); err != nil || v != "b" {
		t.Errorf("parseEnum(b): got (%q, %v)", v, err)
	}
	if _, err := parseEnum("X", "d", allowed); err == nil {
		t.Error("parseEnum(d): expected rejection")
	}
	if v, err := parseEnum("X", "anything", nil); err != nil || v != "anything" {
		t.Errorf("parseEnum with nil allowed = (%q, %v); want pass-through", v, err)
	}
}

// TestParseEnum_CaseInsensitive locks in the contract introduced by
// the parseEnum normalization change: operators may spell env values
// in any case (INFO, Info, info, iNfO) and parseEnum returns the
// canonical form from `allowed`. This means downstream consumers like
// cfg.Log.Level can keep comparing against the lowercase literals
// they always have. Pass-through (nil allowed) is intentionally NOT
// normalized — without a canonical set there is nothing to project
// onto.
func TestParseEnum_CaseInsensitive(t *testing.T) {
	t.Parallel()
	allowed := []string{"debug", "info", "warn", "error"}
	for _, tc := range []struct {
		in   string
		want string
	}{
		{in: "info", want: "info"},
		{in: "INFO", want: "info"},
		{in: "Info", want: "info"},
		{in: "iNfO", want: "info"},
		{in: "DEBUG", want: "debug"},
	} {
		got, err := parseEnum("LOG_LEVEL", tc.in, allowed)
		if err != nil {
			t.Errorf("parseEnum(%q): unexpected error %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseEnum(%q): got %q, want %q (canonical)", tc.in, got, tc.want)
		}
	}
	// Unknown value still rejected; the new error message should
	// hint at the case-insensitive semantics so operators stop
	// double-checking their capitalisation when the real issue is
	// a typo.
	_, err := parseEnum("LOG_LEVEL", "trace", allowed)
	if err == nil {
		t.Fatal("parseEnum(trace): expected rejection")
	}
	if !strings.Contains(err.Error(), "case-insensitive") {
		t.Errorf("parseEnum error message should hint case-insensitive: %v", err)
	}
	// Pass-through path is unchanged: no allowed list, no normalization.
	if v, err := parseEnum("X", "MIXED", nil); err != nil || v != "MIXED" {
		t.Errorf("parseEnum nil-allowed should pass-through verbatim, got (%q, %v)", v, err)
	}
}

func TestParseInt(t *testing.T) {
	t.Parallel()
	cases := []struct {
		v       string
		def     int
		want    int
		wantErr bool
	}{
		{"", 7, 7, false},
		{"0", 7, 0, false},
		{"-3", 7, -3, false},
		{"abc", 7, 0, true},
		{"3.5", 7, 0, true},
	}
	for _, tc := range cases {
		got, err := parseInt("K", tc.v, tc.def)
		if (err != nil) != tc.wantErr {
			t.Errorf("parseInt(%q): err=%v wantErr=%v", tc.v, err, tc.wantErr)
		}
		if !tc.wantErr && got != tc.want {
			t.Errorf("parseInt(%q) = %d, want %d", tc.v, got, tc.want)
		}
	}
}

func TestParseIntInRange(t *testing.T) {
	t.Parallel()
	if _, err := parseIntInRange("PORT", "70000", 8080, 1, 65535); err == nil {
		t.Error("expected out-of-range error")
	}
	if v, err := parseIntInRange("PORT", "", 8080, 1, 65535); err != nil || v != 8080 {
		t.Errorf("default path: (%d, %v)", v, err)
	}
}

func TestParseFloat(t *testing.T) {
	t.Parallel()
	if v, err := parseFloat("K", "1.25", 0); err != nil || v != 1.25 {
		t.Errorf("parseFloat(1.25) = (%v, %v)", v, err)
	}
	if _, err := parseFloat("K", "abc", 0); err == nil {
		t.Error("parseFloat(abc): expected error")
	}
	if v, err := parseFloat("K", "", 0.5); err != nil || v != 0.5 {
		t.Errorf("parseFloat empty default: (%v, %v)", v, err)
	}
}

func TestParseDuration(t *testing.T) {
	t.Parallel()
	if v, err := parseDuration("K", "1h30m", 0); err != nil || v != 90*time.Minute {
		t.Errorf("parseDuration(1h30m) = (%v, %v)", v, err)
	}
	if _, err := parseDuration("K", "abc", 0); err == nil {
		t.Error("parseDuration(abc): expected error")
	}
	if v, err := parseDuration("K", "", time.Second); err != nil || v != time.Second {
		t.Errorf("parseDuration empty default: (%v, %v)", v, err)
	}
}

func TestParseBool(t *testing.T) {
	t.Parallel()
	truthy := []string{"1", "true", "yes", "on", "Y", "T"}
	falsy := []string{"0", "false", "no", "off", "n", "F"}
	for _, v := range truthy {
		if b, err := parseBool("K", v, false); err != nil || !b {
			t.Errorf("parseBool(%q) want true", v)
		}
	}
	for _, v := range falsy {
		if b, err := parseBool("K", v, true); err != nil || b {
			t.Errorf("parseBool(%q) want false", v)
		}
	}
	if b, err := parseBool("K", "", true); err != nil || !b {
		t.Errorf("parseBool empty default true: (%v, %v)", b, err)
	}
	if _, err := parseBool("K", "maybe", false); err == nil {
		t.Error("parseBool(maybe): expected error")
	}
}

func TestParseHTTPURL(t *testing.T) {
	t.Parallel()
	good := []string{
		"http://localhost",
		"http://localhost:11434",
		"https://huggingface.co",
		"https://hf-mirror.com/some/path",
	}
	for _, v := range good {
		if got, err := parseHTTPURL("K", v); err != nil || got != v {
			t.Errorf("parseHTTPURL(%q): (%q, %v)", v, got, err)
		}
	}
	bad := []string{
		"",            // empty (parser will accept empty url; we reject via host check)
		"ftp://x",     // wrong scheme
		"localhost",   // missing scheme
		"http://",     // missing host
		"://noscheme", // garbage
	}
	for _, v := range bad {
		if _, err := parseHTTPURL("K", v); err == nil {
			t.Errorf("parseHTTPURL(%q): expected error", v)
		}
	}
}

func TestParseAbsPath(t *testing.T) {
	t.Parallel()
	good := []string{"/models", "/run/llm-init", "C:/data", `D:\models`}
	for _, v := range good {
		if _, err := parseAbsPath("K", v); err != nil {
			t.Errorf("parseAbsPath(%q): %v", v, err)
		}
	}
	bad := []string{"", "models", "./relative", "../up"}
	for _, v := range bad {
		if _, err := parseAbsPath("K", v); err == nil {
			t.Errorf("parseAbsPath(%q): expected error", v)
		}
	}
}

func TestParseHFRepo(t *testing.T) {
	t.Parallel()
	if _, err := parseHFRepo("K", "Qwen/Qwen2.5-7B-Instruct"); err != nil {
		t.Errorf("expected accept, got %v", err)
	}
	bad := []string{"noslash", "/leading", "trailing/", "a/b/c"}
	for _, v := range bad {
		if _, err := parseHFRepo("K", v); err == nil {
			t.Errorf("parseHFRepo(%q): expected error", v)
		}
	}
}

// FuzzParseHFRepo throws random inputs at the owner/repo parser. The
// only contract being asserted is that the parser never panics on a
// hostile input — we don't constrain accept/reject decisions because
// the production code is intentionally lenient about the alphabet
// (HF allows dots, dashes, dots-with-version, unicode in the future).
//
// Run locally with: go test -fuzz=FuzzParseHFRepo -fuzztime=30s ./internal/config/
func FuzzParseHFRepo(f *testing.F) {
	seeds := []string{
		"Qwen/Qwen2.5-7B",
		"meta-llama/Meta-Llama-3-8B-Instruct",
		"",
		"/",
		"a/",
		"/b",
		"a/b/c",
		"../etc/passwd",
		"a/../b",
		strings.Repeat("a", 1024) + "/" + strings.Repeat("b", 1024),
		"日本/モデル",
		"a/b\x00",
		"a\n/b",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, v string) {
		// Property: must not panic. Errors are fine; nil errors are fine.
		_, _ = parseHFRepo("K", v)
	})
}

func TestParseSHA256(t *testing.T) {
	t.Parallel()
	good := strings.Repeat("a", 64)
	if _, err := parseSHA256("K", good); err != nil {
		t.Errorf("good sha: %v", err)
	}
	bad := []string{
		strings.Repeat("a", 63), // short
		strings.Repeat("a", 65), // long
		strings.Repeat("A", 64), // uppercase rejected
		strings.Repeat("z", 64), // invalid hex char
	}
	for _, v := range bad {
		if _, err := parseSHA256("K", v); err == nil {
			t.Errorf("parseSHA256(%q): expected error", v)
		}
	}
}

func TestParseHFCommitSHA(t *testing.T) {
	t.Parallel()
	// 40 lower-hex chars: accepted
	good := []string{
		"0123456789abcdef0123456789abcdef01234567",
		strings.Repeat("a", 40),
		strings.Repeat("0", 40),
	}
	for _, v := range good {
		if _, err := parseHFCommitSHA("HF_REVISION", v); err != nil {
			t.Errorf("parseHFCommitSHA(%q): unexpected error %v", v, err)
		}
	}
	bad := []string{
		"main",                  // moving ref
		"v1.0.0",                // tag
		"refs/heads/main",       // ref form
		strings.Repeat("a", 39), // short
		strings.Repeat("a", 41), // long
		strings.Repeat("A", 40), // uppercase rejected
		strings.Repeat("g", 40), // non-hex char
	}
	for _, v := range bad {
		if _, err := parseHFCommitSHA("HF_REVISION", v); err == nil {
			t.Errorf("parseHFCommitSHA(%q): expected error", v)
		}
	}
}

func TestParseModelName(t *testing.T) {
	t.Parallel()
	good := []string{"qwen2.5-7b", "qwen2.5:7b-instruct", "ollama/library/x", "embed_v1"}
	for _, v := range good {
		if _, err := parseModelName("K", v); err != nil {
			t.Errorf("parseModelName(%q): %v", v, err)
		}
	}
	bad := []string{"", "has space", "has*star", "has@at"}
	for _, v := range bad {
		if _, err := parseModelName("K", v); err == nil {
			t.Errorf("parseModelName(%q): expected error", v)
		}
	}
}
