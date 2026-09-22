package translate

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func TestResolveCatalog_BuiltinDefault(t *testing.T) {
	t.Parallel()
	c := ResolveCatalog(nil, func(string) string { return "" })
	if len(c.Languages) != 54 {
		t.Fatalf("languages=%d want 54", len(c.Languages))
	}
	if len(c.Pairs) != 102 {
		t.Fatalf("pairs=%d want 102", len(c.Pairs))
	}
}

func TestResolveCatalog_EnvLanguagesFiltersPairs(t *testing.T) {
	t.Parallel()
	c := ResolveCatalog(nil, func(k string) string {
		if k == EnvLanguages {
			return "en,zh-Hans,ja"
		}
		return ""
	})
	if len(c.Languages) != 3 {
		t.Fatalf("languages=%v", c.Languages)
	}
	// en↔zh-Hans and en↔ja exist in mtranPairs; all three covered → filtered.
	if len(c.Pairs) == 0 {
		t.Fatal("want filtered builtin pairs")
	}
	for _, p := range c.Pairs {
		if p.From != "en" && p.From != "zh-Hans" && p.From != "ja" {
			t.Fatalf("unexpected pair %#v", p)
		}
	}
}

func TestResolveCatalog_EnvLanguagesMeshWhenExtras(t *testing.T) {
	t.Parallel()
	// tl is outside MTran; filtered pairs cannot cover it → full mesh.
	c := ResolveCatalog(nil, func(k string) string {
		if k == EnvLanguages {
			return "en,zh-Hans,tl"
		}
		return ""
	})
	wantPairs := 3 * 2
	if len(c.Pairs) != wantPairs {
		t.Fatalf("pairs=%d want mesh %d; pairs=%v", len(c.Pairs), wantPairs, c.Pairs)
	}
}

func TestResolveCatalog_ExplicitPairs(t *testing.T) {
	t.Parallel()
	c := ResolveCatalog(map[string]any{
		ExtensionKey: Extension{
			Languages: []string{"en", "zh-Hans"},
			Pairs:     []languagePair{{From: "en", To: "zh-Hans"}},
		},
	}, func(string) string { return "" }) // no env → extensions
	if len(c.Languages) != 2 || len(c.Pairs) != 1 {
		t.Fatalf("catalog=%+v", c)
	}
	if c.Pairs[0].From != "en" || c.Pairs[0].To != "zh-Hans" {
		t.Fatalf("pair=%+v", c.Pairs[0])
	}
}

func TestResolveCatalog_ExtensionsBeatEnv(t *testing.T) {
	t.Parallel()
	c := ResolveCatalog(map[string]any{
		ExtensionKey: map[string]any{
			"languages": []any{"en", "de"},
		},
	}, func(k string) string {
		if k == EnvLanguages {
			return "en,fr,ja"
		}
		return ""
	})
	if len(c.Languages) != 2 || c.Languages[1] != "de" {
		t.Fatalf("languages=%v", c.Languages)
	}
}

func TestCatalog_ResolveAliasAndExtra(t *testing.T) {
	t.Parallel()
	c, ok := buildCatalog([]string{"zh-Hans", "en", "yue"}, nil)
	if !ok {
		t.Fatal("buildCatalog")
	}
	lang, ok := c.Resolve("zh-CN")
	if !ok || lang.Code != langZhHans {
		t.Fatalf("zh-CN -> %+v ok=%v", lang, ok)
	}
	lang, ok = c.Resolve("yue")
	if !ok || lang.Code != "yue" || lang.NameEN != "yue" {
		t.Fatalf("yue -> %+v ok=%v", lang, ok)
	}
	if _, ok := c.Resolve("ja"); ok {
		t.Fatal("ja must be rejected outside catalog")
	}
}

func TestMount_CatalogOverrideAndReject(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	fakeChat := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "Hi"}},
			},
		})
	})
	Mount(mux, Options{
		Completer: Completer{Handler: fakeChat, ModelName: "m"},
		Catalog: Catalog{
			Languages: []string{"en", "zh-Hans"},
			Pairs:     []languagePair{{From: "zh-Hans", To: "en"}},
		},
	})

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/languages", nil))
	var langs languagesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &langs); err != nil {
		t.Fatal(err)
	}
	if len(langs.Languages) != 2 || len(langs.Pairs) != 1 {
		t.Fatalf("languages response=%+v", langs)
	}

	rec = httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/translate",
		strings.NewReader(`{"from":"zh-Hans","to":"ja","text":"你好"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "for this model") {
		t.Fatalf("body=%s", rec.Body.String())
	}

	rec = httptest.NewRecorder()
	req = httptest.NewRequest(http.MethodPost, "/translate",
		strings.NewReader(`{"from":"zh-Hans","to":"en","text":"你好"}`))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestNormalizeLanguageList_Caps(t *testing.T) {
	t.Parallel()
	in := make([]string, maxCatalogLanguages+10)
	for i := range in {
		in[i] = "l" + strconv.Itoa(i)
	}
	out := normalizeLanguageList(in)
	if len(out) != maxCatalogLanguages {
		t.Fatalf("len=%d want %d", len(out), maxCatalogLanguages)
	}
}

func TestSeedExtensionFromEnv(t *testing.T) {
	t.Parallel()
	if SeedExtensionFromEnv(func(string) string { return "" }) != nil {
		t.Fatal("want nil")
	}
	seed := SeedExtensionFromEnv(func(k string) string {
		if k == EnvLanguages {
			return "en, zh-CN"
		}
		return ""
	})
	ext, ok := seed[ExtensionKey].(Extension)
	if !ok || len(ext.Languages) != 2 || ext.Languages[1] != langZhHans {
		t.Fatalf("seed=%#v", seed)
	}
}
