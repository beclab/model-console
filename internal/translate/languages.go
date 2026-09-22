package translate

import "strings"

// Language is one MTran/Bergamot supported language. Code matches
// Mozilla translations-models-v2 records (same wire as MTranServer
// GET /languages). NameEN is used in the default English translation prompt.
type Language struct {
	Code   string
	NameEN string
}

// languagePair is the MTranServer LanguagePair wire shape.
type languagePair struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// Snapshot of firefox.settings.services.mozilla.com
// .../translations-models-v2/records (fetched 2026-07-15).
// Keep in lockstep with MTranServer getSupportedLanguages /
// getLanguagePairs so existing MTran clients see the same codes.
var mtranLanguages = []Language{
	{Code: "ar", NameEN: "Arabic"},
	{Code: "az", NameEN: "Azerbaijani"},
	{Code: "be", NameEN: "Belarusian"},
	{Code: "bg", NameEN: "Bulgarian"},
	{Code: "bn", NameEN: "Bengali"},
	{Code: "bs", NameEN: "Bosnian"},
	{Code: "ca", NameEN: "Catalan"},
	{Code: "cs", NameEN: "Czech"},
	{Code: "da", NameEN: "Danish"},
	{Code: "de", NameEN: "German"},
	{Code: "el", NameEN: "Greek"},
	{Code: "en", NameEN: "English"},
	{Code: "es", NameEN: "Spanish"},
	{Code: "et", NameEN: "Estonian"},
	{Code: "eu", NameEN: "Basque"},
	{Code: "fa", NameEN: "Persian"},
	{Code: "fi", NameEN: "Finnish"},
	{Code: "fr", NameEN: "French"},
	{Code: "gl", NameEN: "Galician"},
	{Code: "gu", NameEN: "Gujarati"},
	{Code: "he", NameEN: "Hebrew"},
	{Code: "hi", NameEN: "Hindi"},
	{Code: "hr", NameEN: "Croatian"},
	{Code: "hu", NameEN: "Hungarian"},
	{Code: "id", NameEN: "Indonesian"},
	{Code: "is", NameEN: "Icelandic"},
	{Code: "it", NameEN: "Italian"},
	{Code: "ja", NameEN: "Japanese"},
	{Code: "kn", NameEN: "Kannada"},
	{Code: "ko", NameEN: "Korean"},
	{Code: "lt", NameEN: "Lithuanian"},
	{Code: "lv", NameEN: "Latvian"},
	{Code: "ml", NameEN: "Malayalam"},
	{Code: "ms", NameEN: "Malay"},
	{Code: "nb", NameEN: "Norwegian Bokmål"},
	{Code: "nl", NameEN: "Dutch"},
	{Code: "nn", NameEN: "Norwegian Nynorsk"},
	{Code: "pl", NameEN: "Polish"},
	{Code: "pt", NameEN: "Portuguese"},
	{Code: "ro", NameEN: "Romanian"},
	{Code: "ru", NameEN: "Russian"},
	{Code: "sk", NameEN: "Slovak"},
	{Code: "sl", NameEN: "Slovenian"},
	{Code: "sq", NameEN: "Albanian"},
	{Code: "sr", NameEN: "Serbian"},
	{Code: "sv", NameEN: "Swedish"},
	{Code: "ta", NameEN: "Tamil"},
	{Code: "te", NameEN: "Telugu"},
	{Code: "th", NameEN: "Thai"},
	{Code: "tr", NameEN: "Turkish"},
	{Code: "uk", NameEN: "Ukrainian"},
	{Code: "vi", NameEN: "Vietnamese"},
	{Code: langZhHans, NameEN: "Chinese"},
	{Code: langZhHant, NameEN: "Traditional Chinese"},
}

var mtranPairs = []languagePair{
	{From: "ar", To: "en"},
	{From: "az", To: "en"},
	{From: "be", To: "en"},
	{From: "bg", To: "en"},
	{From: "bn", To: "en"},
	{From: "bs", To: "en"},
	{From: "ca", To: "en"},
	{From: "cs", To: "en"},
	{From: "da", To: "en"},
	{From: "de", To: "en"},
	{From: "el", To: "en"},
	{From: "en", To: "ar"},
	{From: "en", To: "az"},
	{From: "en", To: "bg"},
	{From: "en", To: "bn"},
	{From: "en", To: "bs"},
	{From: "en", To: "ca"},
	{From: "en", To: "cs"},
	{From: "en", To: "da"},
	{From: "en", To: "de"},
	{From: "en", To: "el"},
	{From: "en", To: "es"},
	{From: "en", To: "et"},
	{From: "en", To: "eu"},
	{From: "en", To: "fa"},
	{From: "en", To: "fi"},
	{From: "en", To: "fr"},
	{From: "en", To: "gl"},
	{From: "en", To: "gu"},
	{From: "en", To: "he"},
	{From: "en", To: "hi"},
	{From: "en", To: "hr"},
	{From: "en", To: "hu"},
	{From: "en", To: "id"},
	{From: "en", To: "is"},
	{From: "en", To: "it"},
	{From: "en", To: "ja"},
	{From: "en", To: "kn"},
	{From: "en", To: "ko"},
	{From: "en", To: "lt"},
	{From: "en", To: "lv"},
	{From: "en", To: "ml"},
	{From: "en", To: "ms"},
	{From: "en", To: "nb"},
	{From: "en", To: "nl"},
	{From: "en", To: "pl"},
	{From: "en", To: "pt"},
	{From: "en", To: "ro"},
	{From: "en", To: "ru"},
	{From: "en", To: "sk"},
	{From: "en", To: "sl"},
	{From: "en", To: "sr"},
	{From: "en", To: "sv"},
	{From: "en", To: "ta"},
	{From: "en", To: "te"},
	{From: "en", To: "th"},
	{From: "en", To: "tr"},
	{From: "en", To: "uk"},
	{From: "en", To: "vi"},
	{From: "en", To: langZhHans},
	{From: "en", To: langZhHant},
	{From: "es", To: "en"},
	{From: "et", To: "en"},
	{From: "eu", To: "en"},
	{From: "fa", To: "en"},
	{From: "fi", To: "en"},
	{From: "fr", To: "en"},
	{From: "gl", To: "en"},
	{From: "gu", To: "en"},
	{From: "he", To: "en"},
	{From: "hi", To: "en"},
	{From: "hr", To: "en"},
	{From: "hu", To: "en"},
	{From: "id", To: "en"},
	{From: "is", To: "en"},
	{From: "it", To: "en"},
	{From: "ja", To: "en"},
	{From: "kn", To: "en"},
	{From: "ko", To: "en"},
	{From: "lt", To: "en"},
	{From: "lv", To: "en"},
	{From: "ml", To: "en"},
	{From: "ms", To: "en"},
	{From: "nb", To: "en"},
	{From: "nl", To: "en"},
	{From: "nn", To: "en"},
	{From: "pl", To: "en"},
	{From: "pt", To: "en"},
	{From: "ro", To: "en"},
	{From: "ru", To: "en"},
	{From: "sk", To: "en"},
	{From: "sl", To: "en"},
	{From: "sq", To: "en"},
	{From: "sr", To: "en"},
	{From: "sv", To: "en"},
	{From: "te", To: "en"},
	{From: "th", To: "en"},
	{From: "tr", To: "en"},
	{From: "uk", To: "en"},
	{From: "vi", To: "en"},
	{From: langZhHans, To: "en"},
	{From: langZhHant, To: "en"},
}

// languageAliases mirrors MTranServer src/utils/lang-alias.ts.
var languageAliases = map[string]string{
	"zh":      langZhHans,
	"zh-cn":   langZhHans,
	"zh-sg":   langZhHans,
	"zh-hans": langZhHans,
	"cmn":     langZhHans,
	"chinese": langZhHans,
	"zh-tw":   langZhHant,
	"zh-hk":   langZhHant,
	"zh-mo":   langZhHant,
	"zh-hant": langZhHant,
	"cht":     langZhHant,
	"en-us":   "en",
	"en-gb":   "en",
	"en-au":   "en",
	"en-ca":   "en",
	"en-nz":   "en",
	"en-ie":   "en",
	"en-za":   "en",
	"en-jm":   "en",
	"en-bz":   "en",
	"en-tt":   "en",
	"fr-fr":   "fr",
	"fr-ca":   "fr",
	"fr-be":   "fr",
	"fr-ch":   "fr",
	"es-es":   "es",
	"es-mx":   "es",
	"es-ar":   "es",
	"es-co":   "es",
	"es-cl":   "es",
	"es-pe":   "es",
	"es-ve":   "es",
	"pt-pt":   "pt",
	"pt-br":   "pt",
	"de-de":   "de",
	"de-at":   "de",
	"de-ch":   "de",
	"it-it":   "it",
	"it-ch":   "it",
	"ja-jp":   "ja",
	"jp":      "ja",
	"ko-kr":   "ko",
	"kr":      "ko",
	"ru-ru":   "ru",
	"nb":      "no",
}

var byCode = func() map[string]Language {
	m := make(map[string]Language, len(mtranLanguages))
	for _, l := range mtranLanguages {
		m[strings.ToLower(l.Code)] = l
	}
	return m
}()

// NormalizeLanguageCode mirrors MTranServer NormalizeLanguageCode.
func NormalizeLanguageCode(code string) string {
	if code == "" {
		return ""
	}
	normalized := strings.ToLower(strings.ReplaceAll(code, "_", "-"))
	if v, ok := languageAliases[normalized]; ok {
		return v
	}
	main := strings.Split(normalized, "-")[0]
	if v, ok := languageAliases[main]; ok {
		return v
	}
	return main
}

// LanguageCodes returns MTran-compatible language codes for GET /languages.
func LanguageCodes() []string {
	out := make([]string, len(mtranLanguages))
	for i, l := range mtranLanguages {
		out[i] = l.Code
	}
	return out
}

// LanguagePairs returns MTran-compatible directed pairs for GET /languages.
func LanguagePairs() []languagePair {
	out := make([]languagePair, len(mtranPairs))
	copy(out, mtranPairs)
	return out
}

// ResolveLanguage maps a client code (via MTran NormalizeLanguageCode)
// onto a known Mozilla/MTran language. from=auto is handled by callers.
func ResolveLanguage(code string) (Language, bool) {
	n := NormalizeLanguageCode(strings.TrimSpace(code))
	if n == "" {
		return Language{}, false
	}
	// Alias table maps nb→no, but Mozilla records use nb.
	if n == "no" {
		n = "nb"
	}
	l, ok := byCode[strings.ToLower(n)]
	return l, ok
}
