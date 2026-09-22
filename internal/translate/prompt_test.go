package translate

import (
	"strings"
	"testing"
)

func TestBuildPrompt_AutoFrom(t *testing.T) {
	t.Parallel()
	to, ok := ResolveLanguage("en")
	if !ok {
		t.Fatal("en missing")
	}
	got := DefaultPrompt(Language{}, to, "你好", true)
	if !strings.Contains(got, "into English") {
		t.Fatalf("got %q", got)
	}
	if !strings.Contains(got, "你好") {
		t.Fatalf("missing source text: %q", got)
	}
}

func TestBuildPrompt_ExplicitFrom(t *testing.T) {
	t.Parallel()
	from, ok := ResolveLanguage(langZhHans)
	if !ok {
		t.Fatal("zh-Hans missing")
	}
	to, _ := ResolveLanguage("en")
	got := DefaultPrompt(from, to, "hello", false)
	if !strings.Contains(got, "Chinese text into English") {
		t.Fatalf("got %q", got)
	}
}

func TestResolveLanguage_MTranAliases(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"zh":       langZhHans,
		"zh-CN":    langZhHans,
		langZhHans: langZhHans,
		"zh-TW":    langZhHant,
		"EN":       "en",
		"en-US":    "en",
		"nb":       "nb",
		"jp":       "ja",
		"kr":       "ko",
	}
	for in, want := range cases {
		got, ok := ResolveLanguage(in)
		if !ok || got.Code != want {
			t.Errorf("ResolveLanguage(%q) = %+v ok=%v; want code %q", in, got, ok, want)
		}
	}
}

func TestLanguageCodes_MatchMTran(t *testing.T) {
	t.Parallel()
	codes := LanguageCodes()
	if len(codes) != 54 {
		t.Fatalf("languages=%d want 54 (Mozilla/MTran)", len(codes))
	}
	seen := map[string]bool{}
	for _, c := range codes {
		seen[c] = true
	}
	for _, must := range []string{langZhHans, langZhHant, "en", "ja", "de"} {
		if !seen[must] {
			t.Errorf("missing %q in /languages", must)
		}
	}
	if seen["zh"] {
		t.Error("canonical list must use zh-Hans, not zh (MTran wire)")
	}
	pairs := LanguagePairs()
	if len(pairs) != 102 {
		t.Fatalf("pairs=%d want 102", len(pairs))
	}
}

func TestNormalizeLanguageCode_MatchesMTran(t *testing.T) {
	t.Parallel()
	if got := NormalizeLanguageCode("zh"); got != langZhHans {
		t.Fatalf("zh → %q", got)
	}
	if got := NormalizeLanguageCode("nb"); got != "no" {
		t.Fatalf("nb → %q (MTran alias table returns no)", got)
	}
}
