package controlplane

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/moderoutes"
	"github.com/llm-init/llm-init/internal/obs"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/runtimecfg"
	"github.com/llm-init/llm-init/internal/version"
)

// endpointsFixture lets each test customise which registrars are wired
// so we can assert availability flags switching on/off. Mirrors fixture()
// but accepts a closure to tweak Options just before NewServer.
func endpointsFixture(t *testing.T, kind config.EngineKind, mutate func(*Options)) *Server {
	return endpointsFixtureMode(t, kind, "", mutate)
}

func endpointsFixtureMode(t *testing.T, kind config.EngineKind, mode string, mutate func(*Options)) *Server {
	t.Helper()
	env := map[string]string{
		"ENGINE_KIND": string(kind),
		"MODEL_NAME":  "demo-7b",
		"MODEL_MODE":  "chat",
	}
	switch kind {
	case config.EngineOllama:
		env["MODEL_SOURCE"] = "ollama://llama3:8b"
	case config.EngineLlamaCpp:
		env["MODEL_SOURCE"] = "hf://Qwen/Qwen2.5-7B-Instruct --include demo.gguf --revision 0123456789abcdef0123456789abcdef01234567"
	case config.EngineEmbed, config.EngineClipEmbed:
		env["MODEL_MODE"] = "embedding"
		env["MODEL_SOURCE"] = "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64)
		env["MODEL_SOURCE_LOCAL"] = "/data/model"
	case config.EngineRerank:
		env["MODEL_MODE"] = "rerank"
		env["MODEL_NAME"] = "bge-reranker-v2-m3"
		env["MODEL_SOURCE"] = "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64)
		env["MODEL_SOURCE_LOCAL"] = "/data/model"
	case config.EngineOCR:
		env["MODEL_MODE"] = "ocr"
		env["MODEL_SOURCE"] = "https://example.com/model.gguf#sha256=" + strings.Repeat("a", 64)
		env["MODEL_SOURCE_LOCAL"] = "/data/model.gguf"
	case config.EngineAudio:
		env["MODEL_MODE"] = "audio"
		env["MODEL_SUPPORTS"] = "supports_stt,supports_stt_stream"
		env["MODEL_SOURCE"] = "hf://Qwen/Qwen3-ASR-1.7B"
	case config.EngineMusic:
		env["MODEL_MODE"] = "music_generation"
		env["MODEL_SOURCE"] = "hf://ACE-Step/acestep-v15-xl-turbo"
	case config.EngineSystemOne:
		env["MODEL_MODE"] = "system_one"
		env["MODEL_NAME"] = "jevk5"
		env["MODEL_SOURCE"] = "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64)
		env["MODEL_SOURCE_LOCAL"] = "/data/model"
	default:
		env["MODEL_SOURCE"] = "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567"
	}
	if mode != "" {
		env["MODEL_MODE"] = mode
	}
	cfg, err := config.Load(func(k string) string { return env[k] })
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	opts := Options{
		Manager: progress.New(time.Now()),
		Metrics: obs.NewMetrics(),
		Config:  runtimecfg.New(cfg, nil),
		Version: version.Info{Version: "v1.0.9-test", Commit: "deadbeef"},
	}
	if mutate != nil {
		mutate(&opts)
	}
	return NewServer(opts)
}

// withCfg swaps in a store whose config has been tweaked. The store owns
// the config, so a test that wants a different one replaces the store
// rather than reaching into it.
func withCfg(o *Options, f func(*config.Config)) {
	cfg := o.Config.Snapshot()
	f(&cfg)
	o.Config = runtimecfg.New(cfg, nil)
}

// parseEndpoints decodes the catalog body, failing the test on shape
// drift so wire-shape regressions surface here rather than in JS.
func parseEndpoints(t *testing.T, body []byte) EndpointList {
	t.Helper()
	var got EndpointList
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v\nbody=%s", err, body)
	}
	if got.SchemaVersion == 0 {
		t.Errorf("schema_version unset; payload=%s", body)
	}
	if len(got.Endpoints) == 0 {
		t.Fatalf("endpoints list empty; payload=%s", body)
	}
	return got
}

// findByPath returns the first (method, path) match. Tests rely on
// path-level uniqueness within categories — /api/version appears in
// both control and ollama-native categories, so this picks whichever
// matches first, sufficient for the asserts we run.
func findByPath(t *testing.T, list EndpointList, method, path string) EndpointInfo {
	t.Helper()
	for _, e := range list.Endpoints {
		if e.Method == method && e.Path == path {
			return e
		}
	}
	t.Fatalf("endpoint %s %s not found in catalog (got %d entries)", method, path, len(list.Endpoints))
	return EndpointInfo{}
}

// findAllByPath returns every entry matching the path regardless of
// method. Used for /api/version because the route exists twice (control
// build-info vs ollama-native passthrough) at the catalog level once we
// rename to /api/build-info — left here for forward compatibility if
// future routes share paths across categories.
func findAllByPath(list EndpointList, path string) []EndpointInfo {
	var out []EndpointInfo
	for _, e := range list.Endpoints {
		if e.Path == path {
			out = append(out, e)
		}
	}
	return out
}

func assertAvail(t *testing.T, list EndpointList, method, path string, want bool) {
	t.Helper()
	e := findByPath(t, list, method, path)
	if e.Available != want {
		t.Errorf("%s %s: available=%v, want %v (reasons=%v)",
			method, path, e.Available, want, e.Reasons)
	}
}

func TestEndpoints_AllWired(t *testing.T) {
	t.Parallel()
	noop := func(_ *http.ServeMux) {}
	s := endpointsFixture(t, config.EngineOllama, func(o *Options) {
		o.DataPlane = noop
		o.Diag = noop
		o.OllamaNative = noop
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	list := parseEndpoints(t, r.Body)
	if list.EngineKind != string(config.EngineOllama) {
		t.Errorf("engine_kind = %q, want %q", list.EngineKind, config.EngineOllama)
	}
	checks := []struct{ method, path string }{
		{"POST", "/v1/chat/completions"},
		{"POST", "/v1/responses"},
		{"POST", "/v1/messages"},
		{"GET", "/api/diag/gpu"},
		{"GET", "/api/tags"},
		{"POST", "/api/chat"},
		{"POST", "/api/show"},
		{"GET", "/metrics"},
	}
	for _, c := range checks {
		e := findByPath(t, list, c.method, c.path)
		if !e.Available {
			t.Errorf("%s %s: available=false, want true (reasons=%v)", c.method, c.path, e.Reasons)
		}
	}
}

func TestEndpoints_TranslateWhenModeTranslate(t *testing.T) {
	t.Parallel()
	noop := func(_ *http.ServeMux) {}
	s := endpointsFixtureMode(t, config.EngineLlamaCpp, "translate", func(o *Options) {
		o.DataPlane = noop
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	list := parseEndpoints(t, r.Body)
	for _, c := range []struct{ method, path string }{
		{"POST", "/translate"},
		{"POST", "/translate/batch"},
		{"POST", "/translate/transcript"},
		{"GET", "/languages"},
		{"POST", "/detect"},
	} {
		e := findByPath(t, list, c.method, c.path)
		if !e.Available {
			t.Errorf("%s %s: available=false, want true (reasons=%v)", c.method, c.path, e.Reasons)
		}
		if e.Category != categoryTranslate {
			t.Errorf("%s %s: category=%q, want %q", c.method, c.path, e.Category, categoryTranslate)
		}
	}
}

func TestEndpoints_TranslateOffForOtherModes(t *testing.T) {
	t.Parallel()
	noop := func(_ *http.ServeMux) {}
	for _, tc := range []struct {
		name string
		kind config.EngineKind
		mode string
	}{
		{"chat", config.EngineLlamaCpp, "chat"},
		{"embedding", config.EngineEmbed, "embedding"},
		{"audio", config.EngineAudio, "audio"},
		{"ocr", config.EngineOCR, "ocr"},
		{"rerank", config.EngineRerank, "rerank"},
	} {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s := endpointsFixtureMode(t, tc.kind, tc.mode, func(o *Options) {
				o.DataPlane = noop
				withCfg(o, func(c *config.Config) {
					c.Spec.Supports = map[string]bool{"supports_translate": true}
				})
			})
			list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)
			for _, row := range []struct{ method, path string }{
				{mPOST, "/translate"},
				{mPOST, "/translate/batch"},
				{mPOST, "/translate/transcript"},
				{mGET, "/languages"},
				{mPOST, "/detect"},
			} {
				e := findByPath(t, list, row.method, row.path)
				if e.Available {
					t.Errorf("%s %s available for MODEL_MODE=%s", row.method, row.path, tc.mode)
				}
				joined := strings.Join(e.Reasons, " ")
				if !strings.Contains(joined, "MODEL_MODE=translate") ||
					strings.Contains(joined, "supports_"+"translate") {
					t.Errorf("%s %s reasons = %v", row.method, row.path, e.Reasons)
				}
			}
		})
	}
}

func TestEndpoints_TranslateModeWithoutDataPlane(t *testing.T) {
	t.Parallel()
	s := endpointsFixtureMode(t, config.EngineLlamaCpp, "translate", nil)
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)
	e := findByPath(t, list, mPOST, "/translate")
	if e.Available {
		t.Fatal("/translate available without data plane")
	}
	if len(e.Reasons) != 1 || e.Reasons[0] != "data plane registrar not wired" {
		t.Fatalf("reasons = %v", e.Reasons)
	}
}

func TestEndpoints_NoDataPlane(t *testing.T) {
	t.Parallel()
	// Bare control plane: no DataPlane, no Diag, no OllamaNative; only
	// Metrics inherited from fixture wiring. /v1/* must report
	// unavailable with the canonical "data plane registrar not wired"
	// reason; control + health + UI rows must stay available.
	s := endpointsFixture(t, config.EngineVLLM, nil)
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d", r.Status)
	}
	list := parseEndpoints(t, r.Body)

	for _, path := range []string{"/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/embeddings"} {
		e := findByPath(t, list, "POST", path)
		if e.Available {
			t.Errorf("%s: available=true with no DataPlane, want false", path)
		}
		if len(e.Reasons) == 0 || !strings.Contains(e.Reasons[0], "data plane") {
			t.Errorf("%s: reasons=%v, want first reason to mention 'data plane'", path, e.Reasons)
		}
	}

	if e := findByPath(t, list, lookupMethod(list, pathDiagGPU), pathDiagGPU); e.Available {
		t.Errorf("%s available=true with no Diag, want false", pathDiagGPU)
	}
	// Control plane endpoints stay green regardless.
	for _, path := range []string{"/api/progress", "/api/build-info", "/api/endpoints", "/livez", "/healthz"} {
		e := findByPath(t, list, lookupMethod(list, path), path)
		if !e.Available {
			t.Errorf("%s available=false, want true (always-on control surface)", path)
		}
	}
}

func TestEndpoints_NonOllamaHidesNativeRoutes(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineSGLang, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d", r.Status)
	}
	list := parseEndpoints(t, r.Body)
	if list.EngineKind != string(config.EngineSGLang) {
		t.Errorf("engine_kind = %q, want sglang", list.EngineKind)
	}
	// Every ollama-native row must report unavailable on a non-ollama
	// engine; the dashboard hides or greys these rows accordingly.
	for _, e := range list.Endpoints {
		if e.Category == categoryOllama && e.Available {
			t.Errorf("ollama-native %s %s available on engine=%s, want false",
				e.Method, e.Path, list.EngineKind)
		}
		if e.Category == categoryOllama && len(e.Reasons) == 0 {
			t.Errorf("ollama-native %s %s missing reason explaining unavailability", e.Method, e.Path)
		}
	}
	// /v1/responses on SGLang must surface the partial-compliance caveat
	// so the dashboard can warn operators without scraping CHANGELOG.
	resp := findByPath(t, list, "POST", "/v1/responses")
	if !resp.Available {
		t.Fatalf("/v1/responses available=false with DataPlane wired, want true")
	}
	foundCaveat := false
	for _, r := range resp.Reasons {
		if strings.Contains(strings.ToLower(r), "sglang") {
			foundCaveat = true
			break
		}
	}
	if !foundCaveat {
		t.Errorf("/v1/responses on sglang: reasons=%v, want one mentioning sglang", resp.Reasons)
	}
}

func TestEndpoints_ClipEmbedKind(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineClipEmbed, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d", r.Status)
	}
	list := parseEndpoints(t, r.Body)
	if list.EngineKind != string(config.EngineClipEmbed) {
		t.Errorf("engine_kind = %q, want clipembed", list.EngineKind)
	}
	for _, e := range list.Endpoints {
		if e.Category == categoryOllama && e.Available {
			t.Errorf("ollama-native %s %s available on engine=clipembed, want false",
				e.Method, e.Path)
		}
	}
}

func TestEndpoints_EmbedKind(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineEmbed, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d", r.Status)
	}
	list := parseEndpoints(t, r.Body)
	if list.EngineKind != string(config.EngineEmbed) {
		t.Errorf("engine_kind = %q, want embed", list.EngineKind)
	}
	for _, e := range list.Endpoints {
		if e.Category == categoryOllama && e.Available {
			t.Errorf("ollama-native %s %s available on engine=embed, want false",
				e.Method, e.Path)
		}
	}
}

func TestEndpoints_OCRKind(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineOCR, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d", r.Status)
	}
	list := parseEndpoints(t, r.Body)
	if list.EngineKind != string(config.EngineOCR) {
		t.Errorf("engine_kind = %q, want ocr", list.EngineKind)
	}
	for _, e := range list.Endpoints {
		if e.Category == categoryOllama && e.Available {
			t.Errorf("ollama-native %s %s available on engine=ocr, want false",
				e.Method, e.Path)
		}
	}
}

func TestEndpoints_EmbeddingModeCatalogSoftFilter(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineEmbed, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	assertAvail(t, list, mGET, pathModels, true)
	assertAvail(t, list, mPOST, pathEmbeddings, true)
	assertAvail(t, list, mPOST, pathChatCompletions, false)
	assertAvail(t, list, mPOST, pathMessages, false)
	assertAvail(t, list, mPOST, pathOCR, false)
}

func TestEndpoints_ClipEmbedModeCatalogSoftFilter(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineClipEmbed, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	assertAvail(t, list, mGET, pathModels, true)
	assertAvail(t, list, mPOST, pathEmbeddings, true)
	assertAvail(t, list, mPOST, pathChatCompletions, false)
	assertAvail(t, list, mPOST, pathMessages, false)
	assertAvail(t, list, mPOST, pathOCR, false)
}

func TestEndpoints_RerankModeCatalogSoftFilter(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineRerank, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	assertAvail(t, list, mGET, pathModels, true)
	assertAvail(t, list, mPOST, pathRerank, true)
	assertAvail(t, list, mPOST, pathChatCompletions, false)
	assertAvail(t, list, mPOST, pathEmbeddings, false)
	assertAvail(t, list, mPOST, pathOCR, false)
}

func TestEndpoints_SystemOneModeCatalogSoftFilter(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineSystemOne, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	assertAvail(t, list, mGET, pathModels, true)
	assertAvail(t, list, mPOST, pathSystemOne, true)
	assertAvail(t, list, mPOST, pathChatCompletions, false)
	assertAvail(t, list, mPOST, pathEmbeddings, false)
	assertAvail(t, list, mPOST, pathRerank, false)
}

func TestEndpoints_OCRModeCatalogSoftFilter(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineOCR, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	assertAvail(t, list, mGET, pathModels, true)
	assertAvail(t, list, mGET, pathOCRQueue, true)
	assertAvail(t, list, mPOST, pathOCR, true)
	assertAvail(t, list, mGET, pathOCRJob, true)
	assertAvail(t, list, mDELETE, pathOCRJob, true)
	assertAvail(t, list, mGET, pathTasks, true)
	assertAvail(t, list, mGET, pathTask, true)
	assertAvail(t, list, mGET, pathTaskResult, true)
	assertAvail(t, list, mDELETE, pathTask, true)
	assertAvail(t, list, mPOST, pathChatCompletions, false)
	assertAvail(t, list, mPOST, pathEmbeddings, false)
}

// selfServedCategories are the rows this package mounts itself, so a
// Server is enough to check them. The rest arrive through a registrar
// (data plane, diag, ollama-native) or from main (translate): a fixture
// can only supply those itself, which would be the test checking its own
// stub rather than the product.
var selfServedCategories = map[string]bool{
	categoryControl: true,
	categoryHealth:  true,
	categoryUI:      true,
	categoryMetrics: true,
}

// Every row this catalogue advertises has to be a route that is really
// there. The list is written by hand — descriptions and curl hints have
// to be — and nothing checked it against the mux, so a renamed path, a
// wrong verb, or a row outliving the handler it describes would be
// published as a working endpoint, with a copy-paste curl the operator
// finds out about by getting a 404.
//
// The mux is asked which pattern each row resolves to. Falling through
// to the dashboard's "/" catch-all counts as absent: that is what an
// unregistered path and a wrong method both do.
func TestEndpoints_EveryAdvertisedRouteIsRegistered(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineVLLM, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		o.Metrics = obs.NewMetrics()
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	checked := 0
	for _, e := range list.Endpoints {
		if !selfServedCategories[e.Category] || !e.Available {
			continue
		}
		path := strings.ReplaceAll(e.Path, "{id}", "x")
		path = strings.ReplaceAll(path, "/*", "/app.js")
		req := httptest.NewRequest(e.Method, path, nil)
		_, pattern := s.mux.Handler(req)
		switch {
		case pattern == "":
			t.Errorf("%s %s is advertised but matches no route", e.Method, e.Path)
		case pattern == "/" && e.Path != "/":
			t.Errorf("%s %s is advertised but only reaches the dashboard catch-all",
				e.Method, e.Path)
		}
		checked++
	}
	// A filter that quietly matched nothing would pass every assertion
	// above without checking a single route.
	if checked < 10 {
		t.Fatalf("checked only %d rows; the category filter is wrong", checked)
	}
}

// The other direction: a route mounted here and left out of the
// catalogue. It is not a lie, it is an absence — the dashboard's API tab
// and every operator who reads it are simply not told the route exists,
// and nothing goes wrong loudly enough for anyone to notice for a
// release.
//
// The mounted set is read out of the source, because http.ServeMux has
// no way to enumerate what was registered on it. The alternative is a
// list on Server whose only reader is this test, which is a worse trade:
// production state that exists to be asserted about tends to drift from
// the registrations beside it.
func TestEndpoints_EveryMountedRouteIsAdvertised(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineVLLM, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		o.Metrics = obs.NewMetrics()
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	// Which patterns the advertised rows reach. Asking the mux, rather
	// than comparing strings, is what lets a row spelled "/static/*"
	// account for a pattern registered as "/static/".
	covered := map[string]bool{}
	for _, e := range list.Endpoints {
		path := strings.ReplaceAll(e.Path, "{id}", "x")
		path = strings.ReplaceAll(path, "/*", "/app.js")
		if _, pattern := s.mux.Handler(httptest.NewRequest(e.Method, path, nil)); pattern != "" {
			covered[pattern] = true
		}
	}

	for _, pattern := range mountedPatterns(t) {
		if !covered[pattern] {
			t.Errorf("%q is mounted but no /api/endpoints row reaches it", pattern)
		}
	}
}

var registerRE = regexp.MustCompile(`m\.Handle(?:Func)?\("([^"]+)"`)

// mountedPatterns reads the patterns buildMux and attachUI register.
func mountedPatterns(t *testing.T) []string {
	t.Helper()
	var out []string
	for _, f := range []string{"server.go", "ui.go"} {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		for _, m := range registerRE.FindAllStringSubmatch(string(src), -1) {
			// The /v1/ stub stands in for the data plane, which
			// advertises its own rows and is mounted by a registrar.
			if m[1] == "/v1/" {
				continue
			}
			out = append(out, m[1])
		}
	}
	if len(out) < 10 {
		t.Fatalf("found %d registrations; the source scan is broken", len(out))
	}
	return out
}

// This catalogue must add no opinion of its own to the mode's route
// table: for the proxied categories, `available` is that table and
// nothing else. It is one link of a chain — moderoutes proves its two
// faces agree, and dataplane's mux tests prove the port obeys them —
// and the chain exists because these answers were once computed
// separately and came out different: the row for chat on an embedding
// application read `available: false` while the /v1/ catch-all forwarded
// the request to an engine with no such route.
//
// Only the restricted modes belong here. A pass-through mode has no route
// table for the catalogue to equal.
func TestEndpoints_CatalogIsTheModeRouteTable(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		kind config.EngineKind
		mode config.ModelType
	}{
		{config.EngineEmbed, config.ModelEmbedding},
		{config.EngineRerank, config.ModelRerank},
		{config.EngineOCR, config.ModelOCR},
		{config.EngineSystemOne, config.ModelSystemOne},
	} {
		s := endpointsFixture(t, tc.kind, func(o *Options) {
			o.DataPlane = func(_ *http.ServeMux) {}
		})
		list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

		for _, e := range list.Endpoints {
			if e.Category != categoryOpenAI && e.Category != categoryAnthropic {
				continue
			}
			declared := moderoutes.Declares(tc.mode, e.Method, e.Path)
			if e.Available != declared {
				t.Errorf("mode=%s %s %s: catalogue says available=%v, the route table says %v",
					tc.mode, e.Method, e.Path, e.Available, declared)
			}
		}
	}
}

// The async task contract is one surface across modalities, so audio advertises the same
// four rows ocr does — and chat, which runs nothing long enough to need them, advertises none.
func TestEndpoints_TaskRowsFollowTheModality(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		kind config.EngineKind
		want bool
	}{
		{config.EngineAudio, true},
		{config.EngineOCR, true},
		{config.EngineVLLM, false},
	} {
		s := endpointsFixture(t, tc.kind, func(o *Options) {
			o.DataPlane = func(_ *http.ServeMux) {}
		})
		list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)
		for _, row := range []struct {
			method, path string
		}{
			{mGET, pathTasks},
			{mGET, pathTask},
			{mGET, pathTaskResult},
			{mDELETE, pathTask},
		} {
			e := findByPath(t, list, row.method, row.path)
			if e.Available != tc.want {
				t.Errorf("engine=%s: %s %s available=%v, want %v (reasons=%v)",
					tc.kind, row.method, row.path, e.Available, tc.want, e.Reasons)
			}
			if e.Group != groupTasks {
				t.Errorf("engine=%s: %s %s group=%q, want %q",
					tc.kind, row.method, row.path, e.Group, groupTasks)
			}
		}
	}
}

// tts and audio share ENGINE_KIND, so the task rows have to key off the mode
// as well. Design and clone are the slowest things any engine here does and
// the reason async=1 exists on them; a catalogue that hid /v1/tasks would
// leave a 202 with nowhere to poll.
func TestEndpoints_TaskRowsAreAvailableForTTS(t *testing.T) {
	t.Parallel()
	s := endpointsFixtureMode(t, config.EngineAudio, "tts", func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)
	for _, row := range []struct {
		method, path string
	}{
		{mGET, pathTasks},
		{mGET, pathTask},
		{mGET, pathTaskResult},
		{mDELETE, pathTask},
	} {
		e := findByPath(t, list, row.method, row.path)
		if !e.Available {
			t.Errorf("mode=tts: %s %s available=false, want true (reasons=%v)",
				row.method, row.path, e.Reasons)
		}
	}
}

// An engine that declares the contract keeps llm-init's own row (with its curl hint) rather
// than getting a second, engine-reported one for the same path.
func TestEndpoints_EngineDeclaredTaskRowsDoNotDuplicate(t *testing.T) {
	t.Parallel()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/engine-spec" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(`{"schema_version":1,"model":"demo-7b",
			"implements":["ocr"],"declares":["ocr"],"serves":["ocr"],"endpoints":[
			{"capability":"ocr","method":"GET","path":"/v1/models","available":true},
			{"capability":"ocr","method":"GET","path":"/v1/tasks",
			 "description":"List tasks","available":true},
			{"capability":"ocr","method":"GET","path":"/v1/tasks/{id}",
			 "description":"Poll one task","available":true},
			{"capability":"ocr","method":"GET","path":"/v1/audio/tasks/{id}",
			 "description":"deprecated alias","available":true,"deprecated":true}]}`))
	}))
	defer engine.Close()

	s := endpointsFixture(t, config.EngineOCR, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
		o.Readiness = func() Readiness { return Readiness{EngineAlive: true} }
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	for _, p := range []string{pathTasks, pathTask} {
		var rows []EndpointInfo
		for _, e := range list.Endpoints {
			if e.Path == p && e.Method == mGET {
				rows = append(rows, e)
			}
		}
		if len(rows) != 1 {
			t.Fatalf("GET %s: got %d rows, want exactly llm-init's own", p, len(rows))
		}
		if rows[0].Group != groupTasks || rows[0].CurlHint == "" {
			t.Errorf("GET %s = %+v, want llm-init's own row with its curl hint", p, rows[0])
		}
	}
	// The engine's own deprecated alias still shows up; only paths llm-init documents are
	// deduped. It has to read as superseded, or an operator cannot tell it from a live route.
	alias := findByPath(t, list, mGET, "/v1/audio/tasks/{id}")
	if alias.Group != groupEngine {
		t.Errorf("alias row = %+v, want it relayed as %q", alias, groupEngine)
	}
	if !strings.Contains(alias.Description, "deprecated") {
		t.Errorf("alias description = %q, want the engine's deprecated flag relayed", alias.Description)
	}
	// Undeclared proxy rows stay dropped: declaring the contract does not resurrect chat.
	for _, e := range list.Endpoints {
		if e.Path == pathChatCompletions {
			t.Errorf("%s must be dropped: the engine's report does not declare it", e.Path)
		}
	}
	// Undeclared means the method too. This engine answers GET /v1/tasks/{id} and not DELETE,
	// and a path-only check would have shown the DELETE llm-init would then proxy to a 404.
	for _, e := range list.Endpoints {
		if e.Path == pathTask && e.Method == mDELETE {
			t.Errorf("DELETE %s must be dropped: the engine declares only GET on it", e.Path)
		}
	}
}

func TestEndpoints_ChatModeCatalogUnchanged(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineLlamaCpp, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	assertAvail(t, list, mPOST, pathChatCompletions, true)
	assertAvail(t, list, mGET, pathModels, true)
	assertAvail(t, list, mPOST, pathMessages, true)
	assertAvail(t, list, mPOST, pathOCR, false)
}

// TestEndpoints_BuildInfoReachable confirms the new /api/build-info
// route resolves and returns the version snapshot — both the catalog
// row AND the handler need to ship together for the Overview tab to
// render the build banner.
func TestEndpoints_BuildInfoReachable(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineLlamaCpp, nil)
	r := get(t, s, "/api/build-info")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	var got version.Info
	if err := json.Unmarshal(r.Body, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Version != "v1.0.9-test" || got.Commit != "deadbeef" {
		t.Errorf("build-info = %+v, want injected fixture values", got)
	}
}

// TestEndpoints_CatalogSchemaStable pins the response top-level keys so
// dashboards built against schema_version=1 can rely on them. Adding
// fields is fine; renaming/removing requires bumping schema_version
// and updating every client.
func TestEndpoints_CatalogSchemaStable(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineVLLM, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	r := get(t, s, "/api/endpoints")
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(r.Body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	for _, k := range []string{"schema_version", "engine_kind", "endpoints"} {
		if _, ok := raw[k]; !ok {
			t.Errorf("top-level key %q missing", k)
		}
	}
}

// lookupMethod returns the method of the first row matching path. Used
// only by tests to keep call sites concise when the caller doesn't care
// which method-of-a-multi-method path they hit (we don't have any in
// the current catalog, but the helper future-proofs the asserts).
func lookupMethod(list EndpointList, path string) string {
	for _, e := range list.Endpoints {
		if e.Path == path {
			return e.Method
		}
	}
	return "GET"
}

// silence unused warnings: findAllByPath is exported logic for future
// multi-method paths; keep it referenced from tests so refactors don't
// silently drop it.
var _ = findAllByPath

// An engine nobody has heard from yet leaves its own rows in place, and
// withholds the ones that belong to another modality: an audio application
// will not answer chat or embeddings whatever its engine turns out to
// declare, and saying otherwise for the length of a boot is a claim a
// dashboard and a Router model sync both read as settled.
func TestEndpoints_NoEngineSpecKeepsProxiedRows(t *testing.T) {
	t.Parallel()
	s := endpointsFixture(t, config.EngineAudio, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
	})
	r := get(t, s, "/api/endpoints")
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	list := parseEndpoints(t, r.Body)
	if list.EngineKind != string(config.EngineAudio) {
		t.Errorf("engine_kind = %q, want audio", list.EngineKind)
	}
	// No audio-specific data-plane category exists anymore.
	for _, e := range list.Endpoints {
		if e.Category == "data-plane-audio" {
			t.Errorf("audio catalog must not carry a data-plane-audio row: %s %s", e.Method, e.Path)
		}
	}
	// Every mode serves its own model list, whatever the engine says.
	models := findByPath(t, list, mGET, pathModels)
	if models.Category != categoryOpenAI || !models.Available {
		t.Errorf("%s %s: category=%q available=%v (reasons=%v)",
			mGET, pathModels, models.Category, models.Available, models.Reasons)
	}
	// Chat and embeddings belong to other modalities. They stay in the
	// catalog -- the row is where "not yet" gets said -- but not as
	// capabilities of this application.
	for _, c := range []struct{ method, path string }{
		{mPOST, pathChatCompletions},
		{mPOST, pathEmbeddings},
	} {
		e := findByPath(t, list, c.method, c.path)
		if e.Category != categoryOpenAI {
			t.Errorf("%s %s: category=%q, want %q", c.method, c.path, e.Category, categoryOpenAI)
		}
		if e.Available {
			t.Errorf("%s %s: an audio application advertised it before its engine said anything",
				c.method, c.path)
		}
		if !slices.Contains(e.Reasons, reasonSpecPending) {
			t.Errorf("%s %s: reasons=%v, want one naming the missing engine report",
				c.method, c.path, e.Reasons)
		}
	}
}

// An engine serving /api/engine-spec gets its capability rows into the one directory, without displacing ours.
func TestEndpoints_RelaysEngineReportedRows(t *testing.T) {
	t.Parallel()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/engine-spec" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(`{"schema_version":1,"base":"qwen3-asr","model":"demo-7b",
			"implements":["stt","align"],"declares":["stt"],"serves":["stt"],"endpoints":[
			{"capability":"stt","method":"POST","path":"/v1/audio/transcriptions",
			 "description":"Offline transcription","available":true},
			{"capability":"align","method":"POST","path":"/v1/audio/align",
			 "description":"Forced alignment","available":false,"reason":"not declared in MODEL_SUPPORTS"},
			{"method":"GET","path":"/v1/models","description":"engine's own model list","available":true}]}`))
	}))
	defer engine.Close()

	s := endpointsFixture(t, config.EngineAudio, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
		o.Readiness = func() Readiness { return Readiness{EngineAlive: true} }
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)

	stt := findByPath(t, list, mPOST, "/v1/audio/transcriptions")
	if !stt.Available || stt.Category != categoryOpenAI || stt.Group != groupEngine {
		t.Errorf("stt row = %+v, want an available OpenAI row grouped as %q", stt, groupEngine)
	}
	if !strings.HasPrefix(stt.Description, "stt: ") {
		t.Errorf("description = %q, want the capability prefixed", stt.Description)
	}
	align := findByPath(t, list, mPOST, "/v1/audio/align")
	if align.Available || len(align.Reasons) != 1 {
		t.Errorf("align row = %+v, want unavailable with one reason", align)
	}
	if own := findByPath(t, list, mGET, pathModels); own.Group == groupEngine {
		t.Errorf("%s must keep llm-init's own row, got %+v", pathModels, own)
	}
	// A self-reporting engine is authoritative: rows we would only proxy, and it never declared, must not be advertised.
	for _, p := range []string{pathChatCompletions, pathEmbeddings, pathResponses, pathMessages} {
		for _, e := range list.Endpoints {
			if e.Path == p {
				t.Errorf("%s %s must be dropped: the engine's report does not declare it", e.Method, p)
			}
		}
	}
}

// An engine without the spec leaves the catalog exactly as the binary declares it.
func TestEndpoints_EngineSpecAbsentIsSilent(t *testing.T) {
	t.Parallel()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer engine.Close()

	s := endpointsFixture(t, config.EngineAudio, func(o *Options) {
		o.DataPlane = func(_ *http.ServeMux) {}
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
		o.Readiness = func() Readiness { return Readiness{EngineAlive: true} }
	})
	list := parseEndpoints(t, get(t, s, "/api/endpoints").Body)
	for _, e := range list.Endpoints {
		if e.Group == groupEngine {
			t.Errorf("unexpected engine-reported row: %s %s", e.Method, e.Path)
		}
	}
}
