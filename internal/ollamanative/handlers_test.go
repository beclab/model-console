package ollamanative

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/ollama"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

// fakeUpstream is a configurable httptest.Server that records inbound
// /api/version, /api/tags, /api/ps, /api/show requests so tests can
// assert what llm-init *sent* upstream (header value, JSON body
// rewriting) without needing to spin up a real Ollama daemon. Each
// field is exactly one handler; the test sets the ones it cares about
// and leaves the rest at their zero-value (which serves a 200 + small
// canned reply, sufficient for the assertions that don't exercise that
// code path).
type fakeUpstream struct {
	version func(w http.ResponseWriter, r *http.Request)
	tags    func(w http.ResponseWriter, r *http.Request)
	ps      func(w http.ResponseWriter, r *http.Request)
	show    func(w http.ResponseWriter, r *http.Request)
	native  func(path string, w http.ResponseWriter, r *http.Request)
}

func (f *fakeUpstream) handler() http.Handler {
	m := http.NewServeMux()
	m.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		if f.version != nil {
			f.version(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"version":"0.5.4"}`))
	})
	m.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		if f.tags != nil {
			f.tags(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[]}`))
	})
	m.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		if f.ps != nil {
			f.ps(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[]}`))
	})
	m.HandleFunc("/api/show", func(w http.ResponseWriter, r *http.Request) {
		if f.show != nil {
			f.show(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"modelfile":"FROM x"}`))
	})
	for _, path := range nativeInferencePaths {
		p := path
		m.HandleFunc("POST "+p, func(w http.ResponseWriter, r *http.Request) {
			if f.native != nil {
				f.native(p, w, r)
				return
			}
			w.WriteHeader(http.StatusNotFound)
		})
	}
	return m
}

// build constructs a *Mount + serving mux for the given config and
// upstream. `ready` defaults to always-true so callers don't have to
// thread it through every table case; gate-specific tests override it.
func build(t *testing.T, cfg config.Config, up *fakeUpstream, ready func() bool) (*httptest.Server, *httptest.Server) {
	t.Helper()
	upSrv := httptest.NewServer(up.handler())
	t.Cleanup(upSrv.Close)

	cli := ollama.NewClient(upSrv.URL)
	mgr := progress.New(time.Now())
	if ready == nil {
		ready = func() bool { return true }
	}
	mux := http.NewServeMux()
	New(cfg, cli, ready, mgr).Apply(mux)

	frontSrv := httptest.NewServer(mux)
	t.Cleanup(frontSrv.Close)
	return upSrv, frontSrv
}

// cfgFor builds a minimal config that exercises both alias and
// upstream tag. Defaults: MODEL_NAME=qwen-prod, MODEL_SOURCE=ollama://qwen2.5:7b.
// Callers can mutate fields before passing to build().
func cfgFor() config.Config {
	return config.Config{
		Engine: config.Engine{Kind: config.EngineOllama},
		Model:  config.Model{Name: "qwen-prod"},
		Sources: []config.ModelSource{{
			Index:     1,
			Kind:      config.KindOllama,
			Role:      config.RoleMain,
			OllamaTag: "qwen2.5:7b",
		}},
	}
}

func TestVersion_Passthrough(t *testing.T) {
	up := &fakeUpstream{
		version: func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Upstream-Marker", "kept")
			_, _ = w.Write([]byte(`{"version":"0.5.4"}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	resp, err := http.Get(front.URL + "/api/version")
	if err != nil {
		t.Fatalf("GET /api/version: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if got := string(body); got != `{"version":"0.5.4"}` {
		t.Errorf("body = %q, want exact upstream bytes", got)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
}

func TestVersion_UpstreamErrorPropagated(t *testing.T) {
	up := &fakeUpstream{
		version: func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(502)
			_, _ = w.Write([]byte(`upstream sad`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	resp, err := http.Get(front.URL + "/api/version")
	if err != nil {
		t.Fatalf("GET /api/version: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 502 {
		t.Errorf("status = %d, want upstream 502 mirrored", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if got := string(body); got != "upstream sad" {
		t.Errorf("body = %q, want passthrough of upstream bytes", got)
	}
}

func TestVersion_UngatedDuringPhaseInit(t *testing.T) {
	// ready=false simulates phase != ready. /api/version must still
	// answer (it has no gate, by design) so client handshakes complete
	// before lifecycle reaches PhaseReady.
	up := &fakeUpstream{}
	_, front := build(t, cfgFor(), up, func() bool { return false })
	resp, err := http.Get(front.URL + "/api/version")
	if err != nil {
		t.Fatalf("GET /api/version: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200 even when not ready", resp.StatusCode)
	}
}

func TestTags_FilterAndRewrite(t *testing.T) {
	up := &fakeUpstream{
		tags: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"models":[
				{"name":"qwen2.5:7b","modified_at":"2026-05-10T00:00:00Z","size":4683073568},
				{"name":"llama3:8b","modified_at":"2026-05-08T00:00:00Z","size":4661211424}
			]}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	resp, err := http.Get(front.URL + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var got struct {
		Models []ollama.Model `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Models) != 1 {
		t.Fatalf("got %d models, want 1 (llama3 should be filtered)", len(got.Models))
	}
	if got.Models[0].Name != "qwen-prod" {
		t.Errorf("name = %q, want alias rewrite to qwen-prod", got.Models[0].Name)
	}
	if got.Models[0].Model != "qwen-prod" {
		t.Errorf("model = %q, want alias rewrite to qwen-prod", got.Models[0].Model)
	}
	if got.Models[0].Size != 4683073568 {
		t.Errorf("size mutated to %d, want upstream value preserved", got.Models[0].Size)
	}
}

func TestTags_PreservesUpstreamSchema(t *testing.T) {
	up := &fakeUpstream{
		tags: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"models":[{
				"name":"qwen2.5:7b",
				"model":"qwen2.5:7b",
				"modified_at":"2026-05-10T00:00:00Z",
				"size":4683073568,
				"digest":"abc123",
				"details":{
					"parent_model":"",
					"format":"gguf",
					"family":"qwen2",
					"families":["qwen2"],
					"parameter_size":"7.6B",
					"quantization_level":"Q4_K_M"
				}
			}]}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	resp, err := http.Get(front.URL + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()
	var got struct {
		Models []ollama.Model `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Models) != 1 {
		t.Fatalf("got %d models, want 1", len(got.Models))
	}
	m := got.Models[0]
	if m.Name != "qwen-prod" || m.Model != "qwen-prod" {
		t.Errorf("name/model = %q/%q, want qwen-prod/qwen-prod", m.Name, m.Model)
	}
	if m.Digest != "abc123" {
		t.Errorf("digest = %q, want abc123", m.Digest)
	}
	if m.Details == nil || m.Details.Family != "qwen2" || m.Details.Format != "gguf" {
		t.Errorf("details = %#v, want family/format preserved", m.Details)
	}
}

func TestTags_NoMatchReturnsEmptyModels(t *testing.T) {
	up := &fakeUpstream{
		tags: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"models":[{"name":"gemma2:9b","modified_at":"2026-05-01T00:00:00Z","size":1}]}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	resp, err := http.Get(front.URL + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	// Decode rather than literal-string-match: encoding/json may
	// vary in field order / spacing across Go versions, and the
	// contract is "valid Ollama shape with no models", not byte
	// equality.
	var got struct {
		Models []ollama.Model `json:"models"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode: %v (body=%s)", err, body)
	}
	if len(got.Models) != 0 {
		t.Errorf("got %d models, want 0", len(got.Models))
	}
}

func TestTags_AliasEqualsUpstream_NoDoubleCount(t *testing.T) {
	// OLLAMA_MODEL unset → upstream tag falls back to MODEL_NAME.
	// Single upstream entry must surface as a single result, not
	// double-counted by both matchers.
	cfg := cfgFor()
	cfg.Sources = nil
	cfg.Model.Name = "qwen2.5:7b"
	cfg.ModelName = "qwen2.5:7b"

	up := &fakeUpstream{
		tags: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:7b","modified_at":"2026-05-10T00:00:00Z","size":1}]}`))
		},
	}
	_, front := build(t, cfg, up, nil)
	resp, err := http.Get(front.URL + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()
	var got struct {
		Models []ollama.Model `json:"models"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&got)
	if len(got.Models) != 1 {
		t.Errorf("got %d models, want exactly 1", len(got.Models))
	}
}

func TestPS_FilterAndRewrite_BothNameAndModelFields(t *testing.T) {
	up := &fakeUpstream{
		ps: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"models":[
				{"name":"qwen2.5:7b","model":"qwen2.5:7b","size":4683073568,"size_vram":4000000000},
				{"name":"llama3:8b","model":"llama3:8b","size":1,"size_vram":1}
			]}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	resp, err := http.Get(front.URL + "/api/ps")
	if err != nil {
		t.Fatalf("GET /api/ps: %v", err)
	}
	defer resp.Body.Close()
	var got struct {
		Models []ollama.PSModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got.Models) != 1 {
		t.Fatalf("got %d models, want 1", len(got.Models))
	}
	m := got.Models[0]
	if m.Name != "qwen-prod" || m.Model != "qwen-prod" {
		t.Errorf("Name=%q Model=%q, want both rewritten to qwen-prod", m.Name, m.Model)
	}
	if m.Size != 4683073568 || m.SizeVRAM != 4000000000 {
		t.Errorf("size=%d size_vram=%d, must preserve upstream values", m.Size, m.SizeVRAM)
	}
}

func TestShow_InvalidNameReturns404OllamaShape(t *testing.T) {
	upstreamHit := false
	up := &fakeUpstream{
		show: func(http.ResponseWriter, *http.Request) {
			upstreamHit = true
		},
	}
	_, front := build(t, cfgFor(), up, nil)

	body := bytes.NewBufferString(`{"name":"random/other"}`)
	resp, err := http.Post(front.URL+"/api/show", "application/json", body)
	if err != nil {
		t.Fatalf("POST /api/show: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 404 {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
	if upstreamHit {
		t.Error("upstream should NOT be contacted for an invalid name")
	}
	var env map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&env)
	msg, ok := env["error"].(string)
	if !ok {
		t.Fatalf("error field type = %T, want string (Ollama-native shape)", env["error"])
	}
	if !strings.Contains(msg, "random/other") {
		t.Errorf("error msg = %q, want it to echo the asked name", msg)
	}
}

func TestShow_ValidAliasRewritesRequestPassthroughResponse(t *testing.T) {
	var seenUpstream map[string]any
	up := &fakeUpstream{
		show: func(w http.ResponseWriter, r *http.Request) {
			raw, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(raw, &seenUpstream)
			w.Header().Set("Content-Type", "application/json")
			// Upstream response intentionally references the *upstream*
			// tag in its body to exercise true-passthrough (response
			// must NOT be rewritten).
			_, _ = w.Write([]byte(`{"model":"qwen2.5:7b","details":{"family":"qwen"}}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)

	body := bytes.NewBufferString(`{"name":"qwen-prod","verbose":true}`)
	resp, err := http.Post(front.URL+"/api/show", "application/json", body)
	if err != nil {
		t.Fatalf("POST /api/show: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if seenUpstream["name"] != "qwen2.5:7b" || seenUpstream["model"] != "qwen2.5:7b" {
		t.Errorf("upstream saw name=%v model=%v, want both rewritten to qwen2.5:7b",
			seenUpstream["name"], seenUpstream["model"])
	}
	if seenUpstream["verbose"] != true {
		t.Errorf("unrelated fields dropped: verbose=%v, want true", seenUpstream["verbose"])
	}
	respBody, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(respBody), `"model":"qwen2.5:7b"`) {
		t.Errorf("response body = %q, want true-passthrough (upstream tag intact)", respBody)
	}
}

func TestShow_ValidUpstreamTagAccepted(t *testing.T) {
	up := &fakeUpstream{
		show: func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	body := bytes.NewBufferString(`{"name":"qwen2.5:7b"}`)
	resp, err := http.Post(front.URL+"/api/show", "application/json", body)
	if err != nil {
		t.Fatalf("POST /api/show: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200 for upstream-tag identity", resp.StatusCode)
	}
}

func TestShow_AcceptsModelFieldWhenNameMissing(t *testing.T) {
	up := &fakeUpstream{
		show: func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{"ok":true}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	body := bytes.NewBufferString(`{"model":"qwen-prod"}`)
	resp, err := http.Post(front.URL+"/api/show", "application/json", body)
	if err != nil {
		t.Fatalf("POST /api/show: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200 when only `model` is present", resp.StatusCode)
	}
}

func TestNativeChat_RewritesModelAndProxies(t *testing.T) {
	var upstreamBody []byte
	up := &fakeUpstream{
		native: func(path string, w http.ResponseWriter, r *http.Request) {
			if path != "/api/chat" {
				t.Errorf("unexpected path %q", path)
			}
			upstreamBody, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"ok"}}`))
		},
	}
	_, front := build(t, cfgFor(), up, nil)
	body := bytes.NewBufferString(`{"model":"qwen-prod","messages":[{"role":"user","content":"hi"}],"stream":false}`)
	resp, err := http.Post(front.URL+"/api/chat", "application/json", body)
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	var sent map[string]any
	if err := json.Unmarshal(upstreamBody, &sent); err != nil {
		t.Fatalf("upstream body: %v", err)
	}
	if sent["model"] != "qwen2.5:7b" {
		t.Errorf("upstream model = %v, want qwen2.5:7b", sent["model"])
	}
}

func TestGatedRoutesReturn503WhenNotReady(t *testing.T) {
	up := &fakeUpstream{}
	_, front := build(t, cfgFor(), up, func() bool { return false })

	for _, path := range []string{"/api/tags", "/api/ps"} {
		resp, err := http.Get(front.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		if resp.StatusCode != 503 {
			t.Errorf("%s: status=%d, want 503 when not ready", path, resp.StatusCode)
		}
		if resp.Header.Get("Retry-After") == "" {
			t.Errorf("%s: missing Retry-After header in 503", path)
		}
		resp.Body.Close()
	}

	// /api/show is POST and also gated.
	resp, err := http.Post(front.URL+"/api/show", "application/json", strings.NewReader(`{"name":"qwen-prod"}`))
	if err != nil {
		t.Fatalf("POST /api/show: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Errorf("/api/show: status=%d, want 503 when not ready", resp.StatusCode)
	}

	resp, err = http.Post(front.URL+"/api/chat", "application/json", strings.NewReader(`{"model":"qwen-prod"}`))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 503 {
		t.Errorf("/api/chat: status=%d, want 503 when not ready", resp.StatusCode)
	}
}

func TestVersion_UpstreamUnreachable_Returns502(t *testing.T) {
	// Build an upstream then close it immediately so the client gets a
	// connection-refused (or equivalent) on its next request.
	upSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(500)
	}))
	upSrv.Close()

	cli := ollama.NewClient(upSrv.URL)
	mgr := progress.New(time.Now())
	mux := http.NewServeMux()
	New(cfgFor(), cli, func() bool { return true }, mgr).Apply(mux)
	front := httptest.NewServer(mux)
	t.Cleanup(front.Close)

	resp, err := http.Get(front.URL + "/api/version")
	if err != nil {
		t.Fatalf("GET /api/version: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 502 {
		t.Errorf("status=%d, want 502 (bad gateway / upstream unreachable)", resp.StatusCode)
	}
	var env map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&env)
	if _, ok := env["error"].(string); !ok {
		t.Errorf("error field shape unexpected: %#v", env["error"])
	}
}
