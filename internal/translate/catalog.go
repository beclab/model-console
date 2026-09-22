package translate

import (
	"encoding/json"
	"os"
	"strings"
)

// ExtensionKey is the model-spec Extensions map key for translate catalog
// overrides (extensions.translate).
const ExtensionKey = "translate"

// EnvLanguages / EnvPairs seed catalog from chart env when
// extensions.translate is absent.
const (
	EnvLanguages = "TRANSLATE_LANGUAGES"
	EnvPairs     = "TRANSLATE_PAIRS"

	// maxCatalogLanguages caps declared language lists (and thus mesh
	// pair allocation). Real MT models are tens of languages; this
	// bound keeps n*(n-1) make capacity far from int overflow and
	// silences CodeQL go/allocation-size-overflow on config-sourced n.
	maxCatalogLanguages = 512
)

// Catalog is the language directory served by GET /languages and used to
// validate translate/detect requests for the deployed model.
type Catalog struct {
	Languages []string
	Pairs     []languagePair
}

// Extension is the wire shape of model-spec extensions.translate.
type Extension struct {
	Languages []string       `json:"languages,omitempty"`
	Pairs     []languagePair `json:"pairs,omitempty"`
}

// BuiltinCatalog returns the full MTran/Mozilla snapshot (default when
// the chart does not declare a model-specific directory).
func BuiltinCatalog() Catalog {
	return Catalog{
		Languages: LanguageCodes(),
		Pairs:     LanguagePairs(),
	}
}

// ResolveCatalog picks the catalog in priority order:
//  1. model-spec extensions.translate (non-empty languages and/or pairs)
//  2. TRANSLATE_LANGUAGES / TRANSLATE_PAIRS env (first-boot seed)
//  3. Builtin MTran catalog
//
// getenv may be nil (uses os.Getenv). Once a card exists it is the runtime
// source of truth, so an application restart adopts edits made through PUT.
func ResolveCatalog(extensions map[string]any, getenv func(string) string) Catalog {
	if getenv == nil {
		getenv = os.Getenv
	}
	if c, ok := catalogFromExtensions(extensions); ok {
		return c
	}
	if c, ok := catalogFromEnv(getenv); ok {
		return c
	}
	return BuiltinCatalog()
}

func catalogFromExtensions(extensions map[string]any) (Catalog, bool) {
	if extensions == nil {
		return Catalog{}, false
	}
	raw, ok := extensions[ExtensionKey]
	if !ok || raw == nil {
		return Catalog{}, false
	}
	var ext Extension
	switch v := raw.(type) {
	case Extension:
		ext = v
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return Catalog{}, false
		}
		if err := json.Unmarshal(b, &ext); err != nil {
			return Catalog{}, false
		}
	}
	return buildCatalog(ext.Languages, ext.Pairs)
}

func catalogFromEnv(getenv func(string) string) (Catalog, bool) {
	langs := splitCSV(getenv(EnvLanguages))
	pairs := parsePairsCSV(getenv(EnvPairs))
	return buildCatalog(langs, pairs)
}

// buildCatalog returns ok=false when both languages and pairs are empty
// (caller should fall through to the next source).
func buildCatalog(languages []string, pairs []languagePair) (Catalog, bool) {
	languages = normalizeLanguageList(languages)
	pairs = normalizePairList(pairs)
	if len(languages) == 0 && len(pairs) == 0 {
		return Catalog{}, false
	}

	if len(languages) == 0 {
		// Pairs-only: derive language set from pair endpoints.
		seen := map[string]struct{}{}
		var langs []string
		for _, p := range pairs {
			for _, code := range []string{p.From, p.To} {
				if _, ok := seen[code]; ok {
					continue
				}
				seen[code] = struct{}{}
				langs = append(langs, code)
				if len(langs) >= maxCatalogLanguages {
					break
				}
			}
			if len(langs) >= maxCatalogLanguages {
				break
			}
		}
		return Catalog{Languages: langs, Pairs: pairs}, true
	}

	if len(pairs) > 0 {
		return Catalog{Languages: languages, Pairs: pairs}, true
	}

	// Languages only: filter builtin pairs; if the filter cannot cover
	// every declared language (empty, or extras outside MTran), use a
	// full directed mesh (any-to-any models like Hy-MT2).
	langSet := map[string]struct{}{}
	for _, c := range languages {
		langSet[c] = struct{}{}
	}
	filtered := filterBuiltinPairs(langSet)
	if !pairsCoverLanguages(filtered, langSet) {
		filtered = fullMeshPairs(languages)
	}
	return Catalog{Languages: languages, Pairs: filtered}, true
}

func filterBuiltinPairs(langSet map[string]struct{}) []languagePair {
	var out []languagePair
	for _, p := range mtranPairs {
		if _, ok := langSet[p.From]; !ok {
			continue
		}
		if _, ok := langSet[p.To]; !ok {
			continue
		}
		out = append(out, p)
	}
	return out
}

func pairsCoverLanguages(pairs []languagePair, langSet map[string]struct{}) bool {
	if len(pairs) == 0 {
		return false
	}
	seen := map[string]struct{}{}
	for _, p := range pairs {
		seen[p.From] = struct{}{}
		seen[p.To] = struct{}{}
	}
	for code := range langSet {
		if _, ok := seen[code]; !ok {
			return false
		}
	}
	return true
}

func fullMeshPairs(languages []string) []languagePair {
	n := len(languages)
	if n < 2 || n > maxCatalogLanguages {
		return nil
	}
	// Append without a computed capacity: avoids CodeQL
	// go/allocation-size-overflow on n*(n-1) from config-sourced n.
	// n is already capped by maxCatalogLanguages.
	var out []languagePair
	for _, from := range languages {
		for _, to := range languages {
			if from == to {
				continue
			}
			out = append(out, languagePair{From: from, To: to})
		}
	}
	return out
}

func normalizeLanguageList(in []string) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, raw := range in {
		if len(out) >= maxCatalogLanguages {
			break
		}
		code := canonicalCatalogCode(raw)
		if code == "" {
			continue
		}
		if _, ok := seen[code]; ok {
			continue
		}
		seen[code] = struct{}{}
		out = append(out, code)
	}
	return out
}

func normalizePairList(in []languagePair) []languagePair {
	seen := map[string]struct{}{}
	var out []languagePair
	for _, p := range in {
		from := canonicalCatalogCode(p.From)
		to := canonicalCatalogCode(p.To)
		if from == "" || to == "" || from == to {
			continue
		}
		key := from + "\x00" + to
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, languagePair{From: from, To: to})
	}
	return out
}

// canonicalCatalogCode normalizes aliases then preserves MTran casing
// (zh-Hans) when the code is known; otherwise returns the normalized
// lowercase form so chart-declared extras (tl, yue, …) stay usable.
func canonicalCatalogCode(raw string) string {
	n := NormalizeLanguageCode(strings.TrimSpace(raw))
	if n == "" {
		return ""
	}
	if n == "no" {
		n = "nb"
	}
	if l, ok := byCode[strings.ToLower(n)]; ok {
		return l.Code
	}
	// Preserve zh-Hans / zh-Hant style if caller already used it.
	if strings.EqualFold(n, langZhHans) {
		return langZhHans
	}
	if strings.EqualFold(n, langZhHant) {
		return langZhHant
	}
	return n
}

func splitCSV(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// parsePairsCSV accepts "from:to,from:to" (also allows "from->to").
func parsePairsCSV(raw string) []languagePair {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	var out []languagePair
	for _, tok := range strings.Split(raw, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		var from, to string
		switch {
		case strings.Contains(tok, "->"):
			parts := strings.SplitN(tok, "->", 2)
			from, to = parts[0], parts[1]
		case strings.Contains(tok, ":"):
			parts := strings.SplitN(tok, ":", 2)
			from, to = parts[0], parts[1]
		default:
			continue
		}
		out = append(out, languagePair{
			From: strings.TrimSpace(from),
			To:   strings.TrimSpace(to),
		})
	}
	return out
}

// HasLanguage reports whether code (after alias normalize) is in the catalog.
func (c Catalog) HasLanguage(code string) bool {
	_, ok := c.Resolve(code)
	return ok
}

// Resolve maps a client language code onto a Catalog entry. Unknown codes
// outside the catalog return ok=false. Codes not in the builtin MTran table
// still resolve when declared in the catalog (NameEN falls back to Code).
func (c Catalog) Resolve(code string) (Language, bool) {
	canon := canonicalCatalogCode(code)
	if canon == "" {
		return Language{}, false
	}
	for _, listed := range c.Languages {
		if listed == canon || strings.EqualFold(listed, canon) {
			if l, ok := byCode[strings.ToLower(listed)]; ok {
				return l, true
			}
			return Language{Code: listed, NameEN: listed}, true
		}
	}
	return Language{}, false
}

// Codes returns the catalog language codes (for detect prompts).
func (c Catalog) Codes() []string {
	return append([]string(nil), c.Languages...)
}

// SeedExtensionFromEnv builds extensions.translate from TRANSLATE_* env
// for model-spec seeding. Returns nil when env is unset.
func SeedExtensionFromEnv(getenv func(string) string) map[string]any {
	if getenv == nil {
		getenv = os.Getenv
	}
	langs := splitCSV(getenv(EnvLanguages))
	pairs := parsePairsCSV(getenv(EnvPairs))
	if len(langs) == 0 && len(pairs) == 0 {
		return nil
	}
	ext := Extension{Languages: normalizeLanguageList(langs)}
	if len(pairs) > 0 {
		ext.Pairs = normalizePairList(pairs)
	}
	return map[string]any{ExtensionKey: ext}
}
