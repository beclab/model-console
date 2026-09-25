package config

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/handoff"
)

// writeSpec is a thin helper for spec-only tests: writes body to a
// temp file and returns the path. Callers pass the path directly to
// LoadModelSpecFile; no ModelSpecPath swap needed.
func writeSpec(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "model-spec.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	return path
}

func TestLoadModelSpecFile_Minimal(t *testing.T) {
	t.Parallel()
	path := writeSpec(t, `{"name":"m","mode":"chat"}`)
	spec, err := LoadModelSpecFile(path)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if spec.Name != "m" || spec.Mode != "chat" {
		t.Errorf("Name/Mode = %q/%q", spec.Name, spec.Mode)
	}
	// Empty supports map should be initialised to non-nil so callers
	// can do `spec.Supports[k] = true` without a nil-map panic.
	if spec.Supports == nil {
		t.Error("Supports must be non-nil after parse")
	}
}

func TestLoadModelSpecFile_ModeSystemOne(t *testing.T) {
	t.Parallel()
	spec, err := LoadModelSpecFile(writeSpec(t, `{
		"name":"jevk5","mode":"system_one","context_size":4096,
		"extensions":{"system_one":{"max_choice_options":16,"max_score_levels":7,"languages":["en","zh"]}}
	}`))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if spec.Mode != "system_one" || spec.Name != "jevk5" {
		t.Errorf("Name/Mode = %q/%q, want jevk5/system_one", spec.Name, spec.Mode)
	}
}

func TestLoadModelSpecFile_RejectsInvalidSystemOneLimits(t *testing.T) {
	t.Parallel()
	cases := []string{
		`{"name":"jevk5","mode":"system_one"}`,
		`{"name":"jevk5","mode":"system_one","context_size":4096,"extensions":{"system_one":{"max_choice_options":1,"max_score_levels":7,"languages":["en"]}}}`,
		`{"name":"jevk5","mode":"system_one","context_size":4096,"extensions":{"system_one":{"max_choice_options":16,"max_score_levels":11,"languages":["en"]}}}`,
		`{"name":"jevk5","mode":"system_one","context_size":4096,"extensions":{"system_one":{"max_choice_options":16,"max_score_levels":7,"languages":["*"]}}}`,
	}
	for _, body := range cases {
		if _, err := LoadModelSpecFile(writeSpec(t, body)); err == nil || !strings.Contains(err.Error(), "system_one") {
			t.Fatalf("body %s: err = %v", body, err)
		}
	}
}

func TestLoadModelSpecFile_FullPayload(t *testing.T) {
	t.Parallel()
	body := `{
		"name": "qwen-7b",
		"mode": "chat",
		"label": {"en_US": "Qwen 7B"},
		"supports": {"supports_reasoning": true, "supports_vision": false},
		"pricing": {"input": "0.0000015", "output": "0.000002"},
		"parameter_rules": [
			{"name": "temperature", "type": "float", "default": 0.7}
		],
		"context_size": 32768,
		"max_output_tokens": 8192
	}`
	spec, err := LoadModelSpecFile(writeSpec(t, body))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if spec.ContextSize != 32768 || spec.MaxOutputToks != 8192 {
		t.Errorf("ctx/maxout = %d/%d", spec.ContextSize, spec.MaxOutputToks)
	}
	if spec.Pricing["input"] != "0.0000015" {
		t.Errorf("pricing = %v", spec.Pricing)
	}
	if len(spec.ParameterRules) != 1 || spec.ParameterRules[0].Name != "temperature" {
		t.Errorf("parameter_rules = %+v", spec.ParameterRules)
	}
}

func TestLoadModelSpecFile_TTSSpeedRange(t *testing.T) {
	t.Parallel()
	spec, err := LoadModelSpecFile(writeSpec(t, `{
		"name":"FireRedTeam/FireRedTTS3","mode":"tts",
		"extensions":{"tts":{"voice_settings":{"speed":{"min":0.25,"default":1,"max":4}}}}
	}`))
	if err != nil {
		t.Fatalf("LoadModelSpecFile: %v", err)
	}
	tts := spec.Extensions["tts"].(map[string]any)
	settings := tts["voice_settings"].(map[string]any)
	speed := settings["speed"].(map[string]any)
	if speed["min"] != 0.25 || speed["default"] != float64(1) || speed["max"] != float64(4) {
		t.Fatalf("speed = %#v", speed)
	}
}

func TestLoadModelSpecFile_RejectsInvalidTTSSpeedRange(t *testing.T) {
	t.Parallel()
	for _, body := range []string{
		`{"mode":"tts","extensions":{"tts":{"voice_settings":{"speed":{"min":0,"default":1,"max":4}}}}}`,
		`{"mode":"tts","extensions":{"tts":{"voice_settings":{"speed":{"min":2,"default":1,"max":4}}}}}`,
		`{"mode":"tts","extensions":{"tts":{"voice_settings":{"speed":{"min":0.25,"max":4}}}}}`,
	} {
		if _, err := LoadModelSpecFile(writeSpec(t, body)); err == nil || !strings.Contains(err.Error(), "voice_settings.speed") {
			t.Fatalf("body %s: err = %v", body, err)
		}
	}
}

func TestLoadModelSpecFile_NotFound(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSpecFile(filepath.Join(t.TempDir(), "missing.json"))
	if err == nil || !strings.Contains(err.Error(), "model-spec.json") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadModelSpecFile_BadJSON(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSpecFile(writeSpec(t, `{"mode":}`))
	if err == nil || !strings.Contains(err.Error(), "invalid JSON") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadModelSpecFile_MissingMode(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSpecFile(writeSpec(t, `{"name":"x"}`))
	if err == nil || !strings.Contains(err.Error(), "mode is required") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadModelSpecFile_BadMode(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSpecFile(writeSpec(t, `{"name":"x","mode":"bogus"}`))
	if err == nil || !strings.Contains(err.Error(), `must be`) {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadModelSpecFile_ModeOCR(t *testing.T) {
	t.Parallel()
	spec, err := LoadModelSpecFile(writeSpec(t, `{"name":"dots-ocr","mode":"ocr"}`))
	if err != nil {
		t.Fatalf("LoadModelSpecFile: %v", err)
	}
	if spec.Mode != string(ModelOCR) {
		t.Errorf("Mode = %q, want ocr", spec.Mode)
	}
	if spec.Name != "dots-ocr" {
		t.Errorf("Name = %q", spec.Name)
	}
}

func TestLoadModelSpecFile_ModeRerank(t *testing.T) {
	t.Parallel()
	spec, err := LoadModelSpecFile(writeSpec(t, `{"name":"bge-reranker-v2-m3","mode":"rerank"}`))
	if err != nil {
		t.Fatalf("LoadModelSpecFile: %v", err)
	}
	if spec.Mode != string(ModelRerank) {
		t.Errorf("Mode = %q, want rerank", spec.Mode)
	}
	if spec.Name != "bge-reranker-v2-m3" {
		t.Errorf("Name = %q", spec.Name)
	}
}

func TestLoadModelSpecFile_ModeTranslate(t *testing.T) {
	t.Parallel()
	spec, err := LoadModelSpecFile(writeSpec(t, `{"name":"hunyuan-mt","mode":"translate"}`))
	if err != nil {
		t.Fatalf("LoadModelSpecFile: %v", err)
	}
	if spec.Mode != string(ModelTranslate) {
		t.Errorf("Mode = %q, want translate", spec.Mode)
	}
}

func TestLoadModelSpecFile_NegativeContextSize(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSpecFile(writeSpec(t, `{"mode":"chat","context_size":-1}`))
	if err == nil || !strings.Contains(err.Error(), "context_size") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadModelSpecFile_BadPricing(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSpecFile(writeSpec(t, `{"mode":"chat","pricing":{"input":"abc"}}`))
	if err == nil || !strings.Contains(err.Error(), "is not a decimal") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadModelSpecFile_ParameterRulesValidation(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "missing name",
			body:    `{"mode":"chat","parameter_rules":[{"type":"float"}]}`,
			wantErr: "name is required",
		},
		{
			name:    "missing type",
			body:    `{"mode":"chat","parameter_rules":[{"name":"temp"}]}`,
			wantErr: "type is required",
		},
		{
			name:    "bad type",
			body:    `{"mode":"chat","parameter_rules":[{"name":"temp","type":"weird"}]}`,
			wantErr: "must be one of int|float|string|boolean|tag",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := LoadModelSpecFile(writeSpec(t, tc.body))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

func TestLoadModelSpecFile_UnknownSupportsTagged(t *testing.T) {
	t.Parallel()
	spec, err := LoadModelSpecFile(writeSpec(t,
		`{"mode":"chat","supports":{"supports_telepathy":true}}`))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !spec.Supports["supports_telepathy"] {
		t.Error("unknown supports key should be preserved verbatim")
	}
	tag, ok := spec.Extensions["_unknown_flag"]
	if !ok {
		t.Fatal("_unknown_flag tag missing")
	}
	keys, ok := tag.([]string)
	if !ok || len(keys) != 1 || keys[0] != "supports_telepathy" {
		t.Errorf("_unknown_flag = %v", tag)
	}
}

func TestLoadModelSpecFile_SupportsTranslateIsUnknown(t *testing.T) {
	t.Parallel()
	key := "supports_" + "translate"
	spec, err := LoadModelSpecFile(writeSpec(t,
		`{"mode":"translate","supports":{"`+key+`":true}}`))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !spec.Supports[key] {
		t.Error("unknown supports key should be preserved verbatim")
	}
	keys, ok := spec.Extensions["_unknown_flag"].([]string)
	if !ok || len(keys) != 1 || keys[0] != key {
		t.Errorf("_unknown_flag = %v", spec.Extensions["_unknown_flag"])
	}
}

func TestLoadModelSpecFile_AllCapabilitiesAccepted(t *testing.T) {
	t.Parallel()
	parts := make([]string, 0, len(allSupports))
	for _, k := range allSupports {
		parts = append(parts, `"`+k+`": true`)
	}
	body := `{"mode":"chat","supports":{` + strings.Join(parts, ",") + `}}`
	spec, err := LoadModelSpecFile(writeSpec(t, body))
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, ok := spec.Extensions["_unknown_flag"]; ok {
		t.Errorf("no key should be marked unknown; got %v", spec.Extensions["_unknown_flag"])
	}
}

// The two mirrors of Router's AllSupports are pinned to a literal here, in
// the order and at the split Router publishes them: entries 0..31 are
// allSupports, 32..47 are allAudioSupports. Router's own
// TestAllSupportsSplitBoundary pins that split from its side, so the two
// halves stay separable; drift in the keys themselves is caught by review,
// since nothing here can read the Router source at test time.
//
// The non-audio half went unpinned for longer than the audio half, which is
// the wrong way round: these 32 track LiteLLM upstream and turn over more
// often than the audio vocabulary this project defines itself.
func TestAllSupports_RouterOrder(t *testing.T) {
	t.Parallel()
	assertMirrorOrder(t, "allSupports", allSupports, []string{
		"supports_vision",
		"supports_function_calling",
		"supports_parallel_function_calling",
		"supports_native_streaming",
		"supports_response_schema",
		"supports_reasoning",
		"supports_prompt_caching",
		"supports_web_search",
		"supports_audio_input",
		"supports_audio_output",
		"supports_video_input",
		"supports_pdf_input",
		"supports_computer_use",
		"supports_url_context",
		"supports_reasoning_effort",
		"supports_assistant_prefill",
		"supports_tool_choice",
		"supports_tokenizer",
		"supports_system_messages",
		"supports_temperature",
		"supports_top_p",
		"supports_top_k",
		"supports_stop_sequences",
		"supports_frequency_penalty",
		"supports_presence_penalty",
		"supports_n",
		"supports_logprobs",
		"supports_seed",
		"supports_response_format",
		"supports_logit_bias",
		"supports_user",
		"supports_embedding_image_input",
	})
}

func TestAllAudioSupports_RouterOrder(t *testing.T) {
	t.Parallel()
	assertMirrorOrder(t, "allAudioSupports", allAudioSupports, []string{
		"supports_stt",
		"supports_stt_stream",
		"supports_align",
		"supports_diar",
		"supports_diar_stream",
		"supports_speaker_embed",
		"supports_vad",
		"supports_enhance",
		"supports_tts",
		"supports_tts_clone",
		"supports_tts_design",
		"supports_tts_custom",
		"supports_tts_dialogue",
		"supports_audio_llm",
		"supports_audio_s2s",
		"supports_sound_fx",
	})
}

func assertMirrorOrder(t *testing.T, name string, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%s[%d] = %q, want %q", name, i, got[i], want[i])
		}
	}
}

// TestLoadModelSpecFile_ShippedExamples holds the documented promise that
// every file under examples/local/specs/ can be mounted at MODEL_SPEC_PATH
// as-is. The loader disallows unknown top-level fields, so a rename in the
// struct silently invalidates the examples the docs point readers at, and
// nothing else in the tree reads them.
func TestLoadModelSpecFile_ShippedExamples(t *testing.T) {
	t.Parallel()
	dir := filepath.Join("..", "..", "examples", "local", "specs")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read examples dir: %v", err)
	}
	found := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		found++
		t.Run(e.Name(), func(t *testing.T) {
			t.Parallel()
			spec, err := LoadModelSpecFile(filepath.Join(dir, e.Name()))
			if err != nil {
				t.Fatalf("LoadModelSpecFile: %v", err)
			}
			if spec.Name == "" || spec.Mode == "" {
				t.Errorf("example lacks name/mode: %q/%q", spec.Name, spec.Mode)
			}
		})
	}
	if found == 0 {
		t.Fatal("no example specs found; the docs link to this directory")
	}
}

// TestLoadModelSpecFile_ReasoningEffortRule pins the shape the docs
// prescribe for declaring which reasoning levels a model×engine pair
// actually has. Router projects `options` onto /v1/models, so a client
// that reads no options assumes on/off only.
func TestLoadModelSpecFile_ReasoningEffortRule(t *testing.T) {
	t.Parallel()
	spec, err := LoadModelSpecFile(writeSpec(t, `{
		"name": "gpt-oss", "mode": "chat",
		"supports": {"supports_reasoning_effort": true},
		"parameter_rules": [
			{"name": "reasoning_effort", "type": "string",
			 "default": "medium", "options": ["low", "medium", "high"]}
		]
	}`))
	if err != nil {
		t.Fatalf("LoadModelSpecFile: %v", err)
	}
	if len(spec.ParameterRules) != 1 {
		t.Fatalf("parameter_rules = %+v", spec.ParameterRules)
	}
	rule := spec.ParameterRules[0]
	if rule.Name != "reasoning_effort" || rule.Type != "string" {
		t.Errorf("rule = %+v", rule)
	}
	if got := len(rule.Options); got != 3 {
		t.Errorf("options = %v", rule.Options)
	}
	if rule.Default != "medium" {
		t.Errorf("default = %v", rule.Default)
	}
}

func TestLoadModelSpecFile_DisallowsUnknownTopLevel(t *testing.T) {
	t.Parallel()
	_, err := LoadModelSpecFile(writeSpec(t, `{"mode":"chat","oops":42}`))
	if err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("err = %v", err)
	}
}

func TestReconcileModelSpecFile_SeedsWhenAbsent(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "sub", "model-spec.json")
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path},
		Spec:    ModelSpec{Name: "seed", Mode: "chat", Supports: map[string]bool{supportsReasoningKey: true}},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	got, err := LoadModelSpecFile(path)
	if err != nil {
		t.Fatalf("file not written/parseable: %v", err)
	}
	if got.Name != "seed" || got.Mode != "chat" || !got.Supports[supportsReasoningKey] {
		t.Errorf("seeded spec = %+v", got)
	}
}

func TestReconcileModelSpecFile_DiskWins(t *testing.T) {
	t.Parallel()
	path := writeSpec(t, `{"name":"disk","mode":"embedding","supports":{"supports_reasoning":true}}`)
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path},
		Spec:    ModelSpec{Name: "seed", Mode: "chat"},
		Model:   Model{Type: ModelChat},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if cfg.Spec.Name != "disk" || cfg.Spec.Mode != "embedding" {
		t.Errorf("disk should win, got %+v", cfg.Spec)
	}
	if cfg.Model.Type != ModelEmbedding {
		t.Errorf("Model.Type = %q, want re-derived embedding", cfg.Model.Type)
	}
	if !cfg.Model.ThinkSupported {
		t.Error("ThinkSupported should be re-derived true from disk supports")
	}
	if cfg.Model.Name != "disk" {
		t.Errorf("Model.Name = %q, want the card's name: the alias clients "+
			"call has to be the one Router read off the card", cfg.Model.Name)
	}
}

func TestReconcileModelSpecFile_TranslateCatalogSurvivesRestart(t *testing.T) {
	path := writeSpec(t, `{
		"name":"translator",
		"mode":"translate",
		"engine_args":"",
		"extensions":{"translate":{"languages":["zh-Hant","zh-Hans","en"]}}
	}`)
	t.Setenv("TRANSLATE_LANGUAGES", "zh-Hans,zh-Hant,en")

	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path},
		Spec: ModelSpec{
			Name: "translator",
			Mode: "translate",
			Extensions: map[string]any{
				"translate": map[string]any{"languages": []string{"zh-Hans", "zh-Hant", "en"}},
			},
		},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}

	assertLanguages := func(label string, extensions map[string]any) {
		t.Helper()
		raw, err := json.Marshal(extensions["translate"])
		if err != nil {
			t.Fatalf("%s marshal: %v", label, err)
		}
		if !bytes.Contains(raw, []byte(`["zh-Hant","zh-Hans","en"]`)) {
			t.Errorf("%s translate extension = %s", label, raw)
		}
	}
	assertLanguages("memory", cfg.Spec.Extensions)

	persisted, err := LoadModelSpecFile(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	assertLanguages("disk", persisted.Extensions)
}

// The card lives on the shared cache PVC and outlives the application, so
// a chart that changes MODEL_NAME meets a card still carrying the old
// one. Disk wins, and the log line is the only place that is visible.
// Not parallel: it swaps the default logger.
func TestReconcileModelSpecFile_WarnsWhenDiskCardOverridesChartName(t *testing.T) {
	path := writeSpec(t, `{"name":"disk","mode":"chat","engine_args":""}`)
	cfg := &Config{
		Runtime:   Runtime{ModelSpecPath: path},
		ModelName: "chart",
		Spec:      ModelSpec{Name: "chart", Mode: "chat"},
		Model:     Model{Name: "chart", Type: ModelChat},
	}
	logs := captureSlog(t, func() {
		if err := ReconcileModelSpecFile(cfg); err != nil {
			t.Fatalf("reconcile: %v", err)
		}
	})
	if !strings.Contains(logs, "MODEL_NAME from the chart is ignored") {
		t.Errorf("no warning about the ignored chart name; logs=%s", logs)
	}
	if cfg.Model.Name != "disk" {
		t.Errorf("Model.Name = %q, want the card to win", cfg.Model.Name)
	}
}

func TestReconcileModelSpecFile_SeedsEngineArgsAndHandoff(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "model-spec.json")
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path, RunDir: dir},
		Engine:  Engine{Kind: EngineLlamaCpp},
		Spec: ModelSpec{
			Name: "seed", Mode: "chat",
			EngineArgs: "-c 4096 -ngl all",
			Supports:   map[string]bool{},
		},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	got, err := LoadModelSpecFile(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.EngineArgs != "-c 4096 -ngl all" {
		t.Errorf("seeded engine_args = %q", got.EngineArgs)
	}
	if cfg.Engine.Args.Known["ctx_size"] != "4096" {
		t.Errorf("Engine.Args.Known = %#v", cfg.Engine.Args.Known)
	}
	handoff, err := os.ReadFile(filepath.Join(dir, handoff.EngineArgsName))
	if err != nil {
		t.Fatalf("handoff: %v", err)
	}
	if string(handoff) != "-c 4096 -ngl all" {
		t.Errorf("handoff = %q", handoff)
	}
}

func TestReconcileModelSpecFile_BackfillsEngineArgsFromEnvSeed(t *testing.T) {
	t.Parallel()
	path := writeSpec(t, `{"name":"disk","mode":"chat","supports":{}}`)
	// Relocate into a RunDir so handoff is written beside the spec.
	runDir := t.TempDir()
	newPath := filepath.Join(runDir, "model-spec.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(newPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: newPath, RunDir: runDir},
		Engine:  Engine{Kind: EngineVLLM},
		Spec:    ModelSpec{Name: "seed", Mode: "chat", EngineArgs: "--max-model-len 8192"},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if cfg.Spec.EngineArgs != "--max-model-len 8192" {
		t.Errorf("backfill engine_args = %q", cfg.Spec.EngineArgs)
	}
	got, err := LoadModelSpecFile(newPath)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if got.EngineArgs != "--max-model-len 8192" {
		t.Errorf("disk engine_args = %q", got.EngineArgs)
	}
}

func TestReconcileModelSpecFile_DoesNotBackfillExplicitEmptyEngineArgs(t *testing.T) {
	t.Parallel()
	runDir := t.TempDir()
	newPath := filepath.Join(runDir, "model-spec.json")
	// Key present with empty string — intentional clear.
	if err := os.WriteFile(newPath, []byte(`{"name":"disk","mode":"chat","supports":{},"engine_args":""}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: newPath, RunDir: runDir},
		Engine:  Engine{Kind: EngineVLLM},
		Spec:    ModelSpec{Name: "seed", Mode: "chat", EngineArgs: "--max-model-len 8192"},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if cfg.Spec.EngineArgs != "" {
		t.Errorf("explicit empty should win, got %q", cfg.Spec.EngineArgs)
	}
}

// An app installed before the env existed has a card on a hostPath that
// no seed will ever touch, so the chart's ladder has to reach it through
// reconcile. No t.Parallel here or below: the overlay reads os.Getenv.
func TestReconcileModelSpecFile_OverlaysReasoningEffortOntoExistingCard(t *testing.T) {
	runDir := t.TempDir()
	path := filepath.Join(runDir, "model-spec.json")
	if err := os.WriteFile(path,
		[]byte(`{"name":"disk","mode":"chat","supports":{"supports_reasoning":true}}`+"\n"),
		0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MODEL_REASONING_EFFORT", "low,medium,xhigh")
	t.Setenv("MODEL_REASONING_EFFORT_DEFAULT", "xhigh")

	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path, RunDir: runDir},
		Engine:  Engine{Kind: EngineLlamaCpp},
		Spec:    ModelSpec{Name: "seed", Mode: "chat"},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	got, err := LoadModelSpecFile(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(got.ParameterRules) != 1 || got.ParameterRules[0].Name != "reasoning_effort" {
		t.Fatalf("disk rules = %+v, want the ladder written through", got.ParameterRules)
	}
	if got.ParameterRules[0].Default != "xhigh" {
		t.Errorf("disk default = %v, want xhigh", got.ParameterRules[0].Default)
	}
	if len(cfg.Spec.ParameterRules) != 1 {
		t.Errorf("in-memory rules = %+v, want the overlaid ladder", cfg.Spec.ParameterRules)
	}
}

func TestOverlayReasoningEffortRule_KeepsSiblingsAndProse(t *testing.T) {
	t.Setenv("MODEL_REASONING_EFFORT", "low,medium")
	spec := ModelSpec{
		ParameterRules: []ParameterRule{
			{Name: "temperature", Type: "float"},
			{
				Name:    "reasoning_effort",
				Type:    "string",
				Options: []string{"none", "high"},
				Label:   map[string]string{"en_US": "Reasoning effort"},
				Help:    map[string]string{"en_US": "How hard to think."},
			},
		},
	}
	if !overlayReasoningEffortRuleFromEnv(&spec) {
		t.Fatal("stale options should have been replaced")
	}
	if len(spec.ParameterRules) != 2 || spec.ParameterRules[0].Name != "temperature" {
		t.Fatalf("rules = %+v, want the sibling rule untouched", spec.ParameterRules)
	}
	rule := spec.ParameterRules[1]
	if len(rule.Options) != 2 || rule.Options[0] != "low" || rule.Options[1] != "medium" {
		t.Errorf("options = %v, want the chart's ladder", rule.Options)
	}
	if rule.Label["en_US"] == "" || rule.Help["en_US"] == "" {
		t.Errorf("rule = %+v, want the card's prose kept", rule)
	}
	// A second pass has nothing to say, so nothing gets rewritten.
	if overlayReasoningEffortRuleFromEnv(&spec) {
		t.Error("overlay should be idempotent")
	}
}

func TestOverlayReasoningEffortRule_NoEnvLeavesCardAlone(t *testing.T) {
	t.Setenv("MODEL_REASONING_EFFORT", "")
	t.Setenv("MODEL_REASONING_EFFORT_DEFAULT", "")
	spec := ModelSpec{ParameterRules: []ParameterRule{
		{Name: "reasoning_effort", Type: "string", Options: []string{"none", "high"}},
	}}
	if overlayReasoningEffortRuleFromEnv(&spec) {
		t.Fatal("a deployment that declares no ladder must not erase the card's")
	}
	if len(spec.ParameterRules[0].Options) != 2 {
		t.Errorf("options = %v, want untouched", spec.ParameterRules[0].Options)
	}
}

func TestWriteModelSpecFile_PersistsEmptyEngineArgsKey(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "model-spec.json")
	spec := ModelSpec{Name: "qwen", Mode: "chat", Supports: map[string]bool{}, EngineArgs: ""}
	if err := WriteModelSpecFile(path, spec); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !modelSpecJSONHasEngineArgsKey(raw) {
		t.Fatalf("empty engine_args must keep JSON key; body=%s", raw)
	}
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path, RunDir: t.TempDir()},
		Engine:  Engine{Kind: EngineVLLM},
		Spec:    ModelSpec{Name: "seed", Mode: "chat", EngineArgs: "--max-model-len 8192"},
	}
	if err := ReconcileModelSpecFile(cfg); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if cfg.Spec.EngineArgs != "" {
		t.Errorf("PUT/write empty must not be backfilled, got %q", cfg.Spec.EngineArgs)
	}
}

func TestEnsureLlamacppEmbeddingArgs_MergesMissingFlag(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Engine: Engine{Kind: EngineLlamaCpp},
		Spec:   ModelSpec{Name: "e", Mode: string(ModelEmbedding), EngineArgs: "-c 512"},
	}
	if !ensureLlamacppEmbeddingArgs(cfg) {
		t.Fatal("expected mutate")
	}
	if !strings.Contains(cfg.Spec.EngineArgs, "--embedding") {
		t.Fatalf("got %q", cfg.Spec.EngineArgs)
	}
	if ensureLlamacppEmbeddingArgs(cfg) {
		t.Fatal("second call should be no-op")
	}
}

func TestApplyEngineArgsFromSpec_RejectsNonLLM(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Engine: Engine{Kind: EngineEmbed},
		Spec:   ModelSpec{EngineArgs: "--oops"},
	}
	if err := ApplyEngineArgsFromSpec(cfg); err == nil {
		t.Fatal("expected error for non-empty engine_args on embed")
	}
}

func TestApplySpecSideEffects_BumpsRestartWhenHandoffChanges(t *testing.T) {
	t.Parallel()
	runDir := t.TempDir()
	specPath := filepath.Join(runDir, "model-spec.json")
	if err := os.WriteFile(filepath.Join(runDir, handoff.EngineArgsName), []byte("-c 512"), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: specPath, RunDir: runDir},
		Engine:  Engine{Kind: EngineVLLM},
		Spec:    ModelSpec{Name: "m", Mode: "chat", Supports: map[string]bool{}, EngineArgs: "--max-model-len 4096"},
	}
	if err := applySpecSideEffects(cfg); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(runDir, handoff.RestartName)); err != nil {
		t.Fatalf("expected engine_restart after handoff change: %v", err)
	}
	handoff, err := os.ReadFile(filepath.Join(runDir, handoff.EngineArgsName))
	if err != nil {
		t.Fatal(err)
	}
	if string(handoff) != "--max-model-len 4096" {
		t.Fatalf("handoff = %q", handoff)
	}
}

func TestApplySpecSideEffects_NoRestartWhenHandoffUnchanged(t *testing.T) {
	t.Parallel()
	runDir := t.TempDir()
	specPath := filepath.Join(runDir, "model-spec.json")
	const args = "--max-model-len 4096"
	if err := os.WriteFile(filepath.Join(runDir, handoff.EngineArgsName), []byte(args), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: specPath, RunDir: runDir},
		Engine:  Engine{Kind: EngineVLLM},
		Spec:    ModelSpec{Name: "m", Mode: "chat", Supports: map[string]bool{}, EngineArgs: args},
	}
	if err := applySpecSideEffects(cfg); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(runDir, handoff.RestartName)); !os.IsNotExist(err) {
		t.Fatalf("engine_restart should be absent when handoff unchanged; err=%v", err)
	}
}

// captureSlog swaps the default logger for the duration of fn and
// returns what it wrote at WARN or above.
func captureSlog(t *testing.T, fn func()) string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)
	fn()
	return buf.String()
}

// A card on the shared cache PVC outlives the application, so a chart
// that raises ENGINE_ARGS can be silently overruled by a file written
// before the upgrade. Not parallel: it swaps the default logger.
func TestReconcileModelSpecFile_WarnsWhenDiskCardOverridesChartSeed(t *testing.T) {
	path := writeSpec(t, `{"name":"disk","mode":"chat","supports":{},"engine_args":"-c 65536 -np 1"}`)
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path, RunDir: t.TempDir()},
		Engine:  Engine{Kind: EngineLlamaCpp},
		Spec:    ModelSpec{Name: "seed", Mode: "chat", EngineArgs: "-c 104448 -np 1"},
	}
	logs := captureSlog(t, func() {
		if err := ReconcileModelSpecFile(cfg); err != nil {
			t.Fatalf("reconcile: %v", err)
		}
	})
	if !strings.Contains(logs, "the model card on disk wins") {
		t.Errorf("expected a warning, got: %s", logs)
	}
	// Both numbers, or the operator still has to go and read the file.
	if !strings.Contains(logs, "65536") || !strings.Contains(logs, "104448") {
		t.Errorf("warning must name both values, got: %s", logs)
	}
	if cfg.Spec.ContextSize != 65536 {
		t.Errorf("ContextSize = %d, want the window the disk card actually serves", cfg.Spec.ContextSize)
	}
}

func TestReconcileModelSpecFile_NoWarnWhenCardMatchesChartSeed(t *testing.T) {
	path := writeSpec(t, `{"name":"disk","mode":"chat","supports":{},"engine_args":"-c 104448 -np 1"}`)
	cfg := &Config{
		Runtime: Runtime{ModelSpecPath: path, RunDir: t.TempDir()},
		Engine:  Engine{Kind: EngineLlamaCpp},
		Spec:    ModelSpec{Name: "seed", Mode: "chat", EngineArgs: "-c 104448 -np 1"},
	}
	logs := captureSlog(t, func() {
		if err := ReconcileModelSpecFile(cfg); err != nil {
			t.Fatalf("reconcile: %v", err)
		}
	})
	if strings.Contains(logs, "disk wins") {
		t.Errorf("identical args must not warn, got: %s", logs)
	}
}
