package controlplane

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/handoff"
	"github.com/llm-init/llm-init/internal/obs"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/runtimecfg"
	"github.com/llm-init/llm-init/internal/version"
)

// fixtureWithSpec builds a Server whose model-spec store is seeded from
// modelSpecJSON (parsed with the production validator) and whose PUT
// target points at a temp file. Empty modelSpecJSON falls back to a
// minimal {name, mode:chat, supports.supports_reasoning:false} so the
// GET tests keep exercising the same snapshot fields.
func fixtureWithSpec(t *testing.T, modelSpecJSON string) *Server {
	t.Helper()
	return fixtureWithSpecSeeded(t, modelSpecJSON, nil)
}

// fixtureWithSpecSeeded lets a test write into the run dir, or blank it
// out, before the store is built. Both matter because the store reads the
// dir once at construction: prior supervision evidence is what a
// redeployed llm-init finds already on disk, and an empty RUN_DIR is a
// deployment that gave it nowhere to write.
func fixtureWithSpecSeeded(t *testing.T, modelSpecJSON string, seed func(cfg *config.Config)) *Server {
	t.Helper()
	if modelSpecJSON == "" {
		modelSpecJSON = `{"name":"qwen2.5-7b","mode":"chat","supports":{"supports_reasoning":false}}`
	}
	spec, err := config.ParseModelSpecBytes([]byte(modelSpecJSON), "test")
	if err != nil {
		t.Fatalf("parse spec: %v", err)
	}

	env := map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
		"HF_TOKEN":     "hf_abcdefghijklmnopqrstuvwxyz1234",
	}
	cfg, err := config.Load(func(k string) string { return env[k] })
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	cfg.Spec = spec
	runDir := t.TempDir()
	specPath := filepath.Join(runDir, "model-spec.json")
	cfg.Runtime.ModelSpecPath = specPath
	cfg.Runtime.RunDir = runDir
	if seed != nil {
		seed(&cfg)
	}

	mgr := progress.New(time.Now())
	return NewServer(Options{
		Manager:        mgr,
		Metrics:        obs.NewMetrics(),
		Config:         runtimecfg.New(cfg, nil),
		Version:        version.Info{Version: "v1.2.3", Commit: "abc"},
		RetryRateLimit: cfg.Runtime.RetryRateLimit,
	})
}

func putSpec(t *testing.T, s *Server, body string) *httpResp {
	t.Helper()
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	req, err := http.NewRequest(http.MethodPut, srv.URL+"/api/model-spec", strings.NewReader(body))
	if err != nil {
		t.Fatalf("new req: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PUT /api/model-spec: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return &httpResp{Status: resp.StatusCode, Header: resp.Header.Clone(), Body: b}
}

func TestModelSpecRawFileEndpointIsNotExposed(t *testing.T) {
	srv := httptest.NewServer(fixtureWithSpec(t, ""))
	t.Cleanup(srv.Close)

	resp, err := http.Get(srv.URL + "/api/model-spec/file")
	if err != nil {
		t.Fatalf("GET removed model-spec file endpoint: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

func TestModelSpec_DefaultSynthesisedFromScalars(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")
	r := get(t, s, "/api/model-spec")
	if r.Status != 200 {
		t.Fatalf("status = %d, want 200; body=%s", r.Status, r.Body)
	}
	if ct := r.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var got config.ModelSpec
	if err := json.Unmarshal(r.Body, &got); err != nil {
		t.Fatalf("decode: %v body=%s", err, r.Body)
	}
	if got.Name != "qwen2.5-7b" {
		t.Errorf("Name = %q, want qwen2.5-7b", got.Name)
	}
	if got.Mode != "chat" {
		t.Errorf("Mode = %q, want chat", got.Mode)
	}
	if v, ok := got.Supports["supports_reasoning"]; !ok || v {
		t.Errorf("Supports[supports_reasoning] = %v / %v; want present and false", v, ok)
	}
}

func TestModelSpecOverlaysMatchingEngineExtensions(t *testing.T) {
	t.Parallel()
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != pathEngineSpec {
			http.NotFound(w, r)
			return
		}
		_, _ = io.WriteString(w, `{
			"schema_version":1,"model":"qwen2.5-7b","implements":["music.repaint"],
			"declares":["music.repaint"],"serves":["music.repaint"],
			"extensions":{"creative":{"media":"music","operations":["generate","repaint"]},"capacity":{"max_concurrency":99}},
			"endpoints":[{"method":"POST","path":"/v1/music/generations","available":true}]
		}`)
	}))
	t.Cleanup(engine.Close)

	s := fixtureWithSpec(t, `{
		"name":"qwen2.5-7b","mode":"chat","engine_args":"",
		"extensions":{"capacity":{"max_concurrency":1,"source":"engine_args","reported_at":"now"}}
	}`)
	cfg := s.config()
	cfg.Engine.URL = engine.URL
	s.opts.Config = runtimecfg.New(cfg, nil)
	s.opts.Readiness = func() Readiness { return Readiness{EngineAlive: true} }

	recorder := httptest.NewRecorder()
	s.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/model-spec", http.NoBody))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	var got config.ModelSpec
	if err := json.Unmarshal(recorder.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	creative, ok := got.Extensions["creative"].(map[string]any)
	if !ok {
		t.Fatalf("creative extension=%v", got.Extensions["creative"])
	}
	operations, ok := creative["operations"].([]any)
	if !ok || len(operations) != 2 || operations[1] != "repaint" {
		t.Fatalf("creative extension=%v", creative)
	}
	capacity, ok := got.Extensions["capacity"].(map[string]any)
	if !ok || capacity["max_concurrency"] != float64(1) {
		t.Fatalf("engine overwrote Model Console capacity: %v", got.Extensions["capacity"])
	}
}

func TestModelSpec_FullJSONRoundtrip(t *testing.T) {
	t.Parallel()
	specJSON := `{
		"name": "qwen2.5-7b-instruct",
		"mode": "chat",
		"label": {"en_US": "Qwen2.5 7B Instruct"},
		"supports": {
			"supports_function_calling": true,
			"supports_native_streaming": true
		},
		"pricing": {
			"input_cost_per_token": "0.0000015",
			"output_cost_per_token": "0.0000060"
		},
		"parameter_rules": [
			{"name":"temperature","type":"float","default":0.7,"min":0,"max":2}
		],
		"context_size": 32768,
		"max_output_tokens": 4096
	}`
	s := fixtureWithSpec(t, specJSON)
	r := get(t, s, "/api/model-spec")
	if r.Status != 200 {
		t.Fatalf("status = %d", r.Status)
	}

	var got config.ModelSpec
	if err := json.Unmarshal(r.Body, &got); err != nil {
		t.Fatalf("decode: %v body=%s", err, r.Body)
	}
	if got.Name != "qwen2.5-7b-instruct" {
		t.Errorf("Name = %q", got.Name)
	}
	if got.Label["en_US"] != "Qwen2.5 7B Instruct" {
		t.Errorf("Label = %#v", got.Label)
	}
	if !got.Supports["supports_function_calling"] {
		t.Errorf("supports_function_calling not preserved: %#v", got.Supports)
	}
	if got.Pricing["input_cost_per_token"] != "0.0000015" {
		t.Errorf("pricing precision dropped: %#v", got.Pricing)
	}
	if got.ContextSize != 32768 || got.MaxOutputToks != 4096 {
		t.Errorf("token budgets = (%d, %d)", got.ContextSize, got.MaxOutputToks)
	}
	if len(got.ParameterRules) != 1 || got.ParameterRules[0].Name != "temperature" {
		t.Errorf("ParameterRules = %#v", got.ParameterRules)
	}
}

func TestModelSpec_ReachableBeforeReadyzFlips(t *testing.T) {
	t.Parallel()
	// /api/model-spec must answer 200 while the lifecycle is still in
	// PhaseInit — Router uses it during boot to stage the upsert
	// before /readyz flips, so a ready-gate here would cause
	// Router to retry the upsert post-ready instead of pre-ready.
	s := fixtureWithSpec(t, "")
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)

	r := get(t, s, "/api/model-spec")
	if r.Status != 200 {
		t.Fatalf("/api/model-spec status = %d, want 200 even in PhaseInit", r.Status)
	}
}

func TestModelSpec_PutValidatesPersistsAndServes(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	body := `{"name":"qwen2.5-7b","mode":"embedding","supports":{"supports_vision":true}}`
	r := putSpec(t, s, body)
	if r.Status != 200 {
		t.Fatalf("PUT status = %d, want 200; body=%s", r.Status, r.Body)
	}

	// GET must reflect the new spec immediately.
	g := get(t, s, "/api/model-spec")
	var got config.ModelSpec
	if err := json.Unmarshal(g.Body, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Mode != "embedding" || !got.Supports["supports_vision"] {
		t.Errorf("served spec not updated: %+v", got)
	}

	// Disk must be written and re-parseable.
	disk, err := config.LoadModelSpecFile(s.config().Runtime.ModelSpecPath)
	if err != nil {
		t.Fatalf("disk not persisted: %v", err)
	}
	if disk.Mode != "embedding" {
		t.Errorf("disk spec = %+v", disk)
	}
}

func TestModelSpec_PutEngineArgsWritesHandoff(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	body := `{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len 4096"}`
	r := putSpec(t, s, body)
	if r.Status != 200 {
		t.Fatalf("PUT status = %d, want 200; body=%s", r.Status, r.Body)
	}
	if got := r.Header.Get(headerRestart); got != restartSignaled {
		t.Errorf("%s = %q, want %q", headerRestart, got, restartSignaled)
	}
	if got := r.Header.Get(headerRestartSupervision); got != string(handoff.SupervisionUnknown) {
		t.Errorf("%s = %q, want unknown: no supervisor has been observed here",
			headerRestartSupervision, got)
	}
	if r.Header.Get(headerRestartGeneration) == "" {
		t.Errorf("%s must identify the signal that was sent", headerRestartGeneration)
	}
	// The legacy header claims a completed restart, so it may only appear
	// once a supervisor has been seen acting on an earlier signal.
	if got := r.Header.Get(headerRestarted); got != "" {
		t.Errorf("%s = %q, want absent while supervision is unknown", headerRestarted, got)
	}
	cfg := s.config()
	args, err := os.ReadFile(filepath.Join(cfg.Runtime.RunDir, handoff.EngineArgsName))
	if err != nil {
		t.Fatalf("handoff: %v", err)
	}
	if string(args) != "--max-model-len 4096" {
		t.Errorf("handoff = %q", args)
	}
	if cfg.Engine.Args.Known["max_model_len"] != "4096" {
		t.Errorf("Engine.Args.Known = %#v", cfg.Engine.Args.Known)
	}
	if _, err := os.Stat(filepath.Join(cfg.Runtime.RunDir, handoff.RestartName)); err != nil {
		t.Errorf("engine_restart missing: %v", err)
	}
}

func TestEngineRestart_POST(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")
	srv := httptest.NewServer(s)
	t.Cleanup(srv.Close)
	resp, err := http.Post(srv.URL+"/api/engine/restart", "application/json", http.NoBody)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if _, err := os.Stat(filepath.Join(s.config().Runtime.RunDir, handoff.RestartName)); err != nil {
		t.Errorf("engine_restart missing: %v", err)
	}
}

func TestModelSpec_PutEngineArgsEmptyRunDirNoRestartHeader(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpecSeeded(t, "", func(cfg *config.Config) {
		cfg.Runtime.RunDir = ""
	})

	body := `{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len 2048"}`
	r := putSpec(t, s, body)
	if r.Status != http.StatusServiceUnavailable {
		t.Fatalf("PUT status = %d, want 503; body=%s", r.Status, r.Body)
	}
	if r.Header.Get(headerRestarted) != "" {
		t.Errorf("%s must be absent when RUN_DIR is empty, got %q",
			headerRestarted, r.Header.Get(headerRestarted))
	}
	// Spec is still persisted and served — only the restart signal failed.
	g := get(t, s, "/api/model-spec")
	var got config.ModelSpec
	if err := json.Unmarshal(g.Body, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.EngineArgs != "--max-model-len 2048" {
		t.Errorf("served EngineArgs = %q after failed restart signal", got.EngineArgs)
	}
}

// A rename cannot be half-applied: the data plane rewrites to the alias
// it was built with, so accepting one would have Router route a name
// this process answers 404 for.
func TestModelSpec_PutRejectsRename(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	r := putSpec(t, s, `{"name":"qwen2.5-7b-instruct","mode":"chat","supports":{}}`)
	if r.Status != 400 {
		t.Fatalf("PUT status = %d, want 400; body=%s", r.Status, r.Body)
	}
	if !strings.Contains(string(r.Body), "model name is fixed") {
		t.Errorf("body should say why the rename was refused: %s", r.Body)
	}
	if _, err := os.Stat(s.config().Runtime.ModelSpecPath); !os.IsNotExist(err) {
		t.Errorf("rejected rename should not have written disk (stat err=%v)", err)
	}
}

func TestModelSpec_PutRejectsAudioSupportsChanges(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpecSeeded(t,
		`{"name":"qwen2.5-7b","mode":"audio","supports":{"supports_stt":true}}`,
		func(cfg *config.Config) {
			cfg.Engine.Kind = config.EngineAudio
		})

	r := putSpec(t, s,
		`{"name":"qwen2.5-7b","mode":"audio","supports":{"supports_stt":true,"supports_stt_stream":true}}`)
	if r.Status != http.StatusBadRequest {
		t.Fatalf("PUT status = %d, want 400; body=%s", r.Status, r.Body)
	}
	if !strings.Contains(string(r.Body), "audio supports are fixed at startup") {
		t.Fatalf("body should explain why supports are immutable: %s", r.Body)
	}
}

// Mode is accepted and stored, but the routes it selects were mounted at
// boot, so the receipt has to say the running process is not obeying it.
func TestModelSpec_PutReportsFieldsPendingAnAppRestart(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	r := putSpec(t, s, `{"name":"qwen2.5-7b","mode":"translate","supports":{}}`)
	if r.Status != 200 {
		t.Fatalf("PUT status = %d, want 200; body=%s", r.Status, r.Body)
	}
	if got := r.Header.Get(headerPendingAppRestart); got != "mode" {
		t.Errorf("%s = %q, want %q", headerPendingAppRestart, got, "mode")
	}
	// The translate routes are not there, and the catalog must not claim
	// they are just because the card now asks for them.
	e := get(t, s, "/api/endpoints")
	if strings.Contains(string(e.Body), `"/v1/translate"`) &&
		strings.Contains(string(e.Body), `"available":true`) {
		t.Error("/api/endpoints advertised translate routes that were never mounted")
	}

	// An edit that only touches metadata leaves the header off entirely.
	quiet := putSpec(t, s, `{"name":"qwen2.5-7b","mode":"translate","supports":{},"max_output_tokens":512}`)
	if got := quiet.Header.Get(headerPendingAppRestart); got != "" {
		t.Errorf("%s = %q, want absent on a pure-metadata edit", headerPendingAppRestart, got)
	}
}

func TestModelSpec_PutRejectsInvalid(t *testing.T) {
	t.Parallel()
	s := fixtureWithSpec(t, "")

	r := putSpec(t, s, `{"name":"x"}`) // missing required mode
	if r.Status != 400 {
		t.Fatalf("PUT status = %d, want 400; body=%s", r.Status, r.Body)
	}

	// Served spec is unchanged, and no file was written.
	g := get(t, s, "/api/model-spec")
	var got config.ModelSpec
	if err := json.Unmarshal(g.Body, &got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.Mode != "chat" {
		t.Errorf("served spec mutated by invalid PUT: %+v", got)
	}
	if _, err := os.Stat(s.config().Runtime.ModelSpecPath); !os.IsNotExist(err) {
		t.Errorf("invalid PUT should not have written disk (stat err=%v)", err)
	}
}
