package config

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/translate"
)

// minimalEnv returns a base env map for any engine kind. Callers
// override or add fields as needed. Mirrors the user-facing v1.1 env
// surface (ENGINE_KIND + ENGINE_ARGS + MODEL_NAME + MODEL_MODE +
// MODEL_SOURCE). MODEL_MODE seeds the env-synthesised model-spec.
func minimalEnv(extras map[string]string) map[string]string {
	base := map[string]string{
		"ENGINE_KIND":  "ollama",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "ollama://qwen2.5:7b",
	}
	for k, v := range extras {
		base[k] = v
	}
	return base
}

func TestLoad_OllamaLibraryMinimal(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineOllama {
		t.Errorf("Kind = %q", cfg.Engine.Kind)
	}
	if cfg.Engine.URL != "http://ollama:11434" {
		t.Errorf("URL = %q", cfg.Engine.URL)
	}
	if len(cfg.Sources) != 1 || cfg.Sources[0].Kind != KindOllama {
		t.Fatalf("Sources = %+v", cfg.Sources)
	}
	if cfg.Sources[0].OllamaTag != "qwen2.5:7b" {
		t.Errorf("OllamaTag = %q", cfg.Sources[0].OllamaTag)
	}
	if cfg.ModelName != "qwen2.5-7b" || cfg.Spec.Name != "qwen2.5-7b" {
		t.Errorf("ModelName/Spec.Name = %q / %q", cfg.ModelName, cfg.Spec.Name)
	}
	if cfg.PrimaryOllamaTag() != "qwen2.5:7b" {
		t.Errorf("PrimaryOllamaTag = %q", cfg.PrimaryOllamaTag())
	}
	if cfg.PrimarySourceKind() != KindOllama {
		t.Errorf("PrimarySourceKind = %q", cfg.PrimarySourceKind())
	}
}

func TestPrimaryOllamaTag_UsesMainSourceOnly(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name    string
		sources []ModelSource
		want    string
	}{
		{
			name: "Ollama URL main ignores Ollama extra",
			sources: []ModelSource{
				{Kind: KindOllamaURL, Role: RoleMain, URL: "https://example.com/main.gguf"},
				{Kind: KindOllama, Role: RoleExtra, OllamaTag: "wrong-extra"},
			},
			want: "model-alias",
		},
		{
			name: "HF main ignores Ollama extra",
			sources: []ModelSource{
				{Kind: KindHF, Role: RoleMain, HFRepo: "owner/model"},
				{Kind: KindOllama, Role: RoleExtra, OllamaTag: "wrong-extra"},
			},
			want: "model-alias",
		},
		{
			name: "plain Ollama main uses its tag",
			sources: []ModelSource{
				{Kind: KindOllama, Role: RoleMain, OllamaTag: "main-tag"},
				{Kind: KindOllama, Role: RoleExtra, OllamaTag: "wrong-extra"},
			},
			want: "main-tag",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{
				ModelName: "model-alias",
				Model:     Model{Name: "model-fallback"},
				Sources:   tc.sources,
			}
			if got := cfg.PrimaryOllamaTag(); got != tc.want {
				t.Fatalf("PrimaryOllamaTag() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestLoad_HFRepoIntegralVLLM(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision " + strings.Repeat("a", 40),
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineVLLM || cfg.Engine.URL != "http://vllm:8000" {
		t.Errorf("Engine = %+v", cfg.Engine)
	}
	if len(cfg.Sources) != 1 || cfg.Sources[0].Kind != KindHF {
		t.Fatalf("Sources = %+v", cfg.Sources)
	}
	if cfg.Sources[0].HFRepo != "Qwen/Qwen2.5-7B-Instruct" {
		t.Errorf("HFRepo = %q", cfg.Sources[0].HFRepo)
	}
	if cfg.HFEndpoint != "https://huggingface.co" {
		t.Errorf("HFEndpoint = %q", cfg.HFEndpoint)
	}
}

func TestLoad_ClipEmbedEmbeddingMinimal(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "clipembed",
		"MODEL_NAME":         "jina-clip-v2",
		"MODEL_MODE":         "embedding",
		"MODEL_SOURCE":       "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64),
		"MODEL_SOURCE_LOCAL": "/data/model",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineClipEmbed {
		t.Errorf("Kind = %q", cfg.Engine.Kind)
	}
	if cfg.Engine.URL != "http://clipembed:8080" {
		t.Errorf("URL = %q", cfg.Engine.URL)
	}
	if cfg.Model.Type != ModelEmbedding {
		t.Errorf("Model.Type = %q", cfg.Model.Type)
	}
	if len(cfg.Engine.Args.Known) != 0 || cfg.Engine.Args.Raw != "" {
		t.Errorf("Engine.Args should be empty for clipembed, got %+v", cfg.Engine.Args)
	}
}

func TestLoad_EmbedEmbeddingMinimal(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "embed",
		"MODEL_NAME":         "embeddinggemma-300m",
		"MODEL_MODE":         "embedding",
		"MODEL_SOURCE":       "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64),
		"MODEL_SOURCE_LOCAL": "/data/model",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineEmbed {
		t.Errorf("Kind = %q", cfg.Engine.Kind)
	}
	if cfg.Engine.URL != "http://embed:8080" {
		t.Errorf("URL = %q", cfg.Engine.URL)
	}
	if cfg.Model.Type != ModelEmbedding {
		t.Errorf("Model.Type = %q", cfg.Model.Type)
	}
	if len(cfg.Engine.Args.Known) != 0 || cfg.Engine.Args.Raw != "" {
		t.Errorf("Engine.Args should be empty for embed, got %+v", cfg.Engine.Args)
	}
}

func TestLoad_EngineMaxConcurrency(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":            "audio",
		"ENGINE_MAX_CONCURRENCY": "3",
		"MODEL_NAME":             "audio-model",
		"MODEL_MODE":             "audio",
		"MODEL_SOURCE":           "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64),
		"MODEL_SOURCE_LOCAL":     "/data/model",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.MaxConcurrency != 3 {
		t.Fatalf("Engine.MaxConcurrency = %d, want 3", cfg.Engine.MaxConcurrency)
	}
}

func TestLoad_EngineMaxConcurrencyRejectsInvalidValues(t *testing.T) {
	for _, value := range []string{"0", "-1", "1.5", "many"} {
		t.Run(value, func(t *testing.T) {
			_, err := Load(mapEnv(map[string]string{
				"ENGINE_KIND":            "audio",
				"ENGINE_MAX_CONCURRENCY": value,
				"MODEL_NAME":             "audio-model",
				"MODEL_MODE":             "audio",
				"MODEL_SOURCE":           "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64),
				"MODEL_SOURCE_LOCAL":     "/data/model",
			}))
			if err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

func TestLoad_EmbedRejectsUnknownKind(t *testing.T) {
	_, err := Load(mapEnv(minimalEnv(map[string]string{
		"ENGINE_KIND": "not-a-real-engine",
	})))
	if err == nil {
		t.Fatal("expected error")
	}
}

// TestLoad_OCRKindAccepted: ENGINE_KIND=ocr + MODEL_MODE=ocr Load with
// sibling URL pointing at OCRAdapter (not llamacpp).
func TestLoad_OCRKindAccepted(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "ocr",
		"MODEL_NAME":         "dots-ocr",
		"MODEL_MODE":         "ocr",
		"MODEL_SOURCE":       "https://example.com/model.gguf#sha256=" + strings.Repeat("a", 64),
		"MODEL_SOURCE_LOCAL": "/data/model.gguf",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineOCR {
		t.Errorf("Kind = %q, want ocr", cfg.Engine.Kind)
	}
	if cfg.Model.Type != ModelOCR || cfg.Spec.Mode != "ocr" {
		t.Errorf("Model.Type/Spec.Mode = %q / %q, want ocr", cfg.Model.Type, cfg.Spec.Mode)
	}
	if cfg.Engine.URL != "http://ocradapter:8080" {
		t.Errorf("URL = %q, want http://ocradapter:8080", cfg.Engine.URL)
	}
	if len(cfg.Engine.Args.Known) != 0 || cfg.Engine.Args.Raw != "" {
		t.Errorf("Engine.Args should be empty for ocr, got %+v", cfg.Engine.Args)
	}
}

// TestLoad_RerankKindAccepted: ENGINE_KIND=rerank + MODEL_MODE=rerank Load.
// Engine URL and proxy wiring are covered in L3.
func TestLoad_RerankKindAccepted(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "rerank",
		"MODEL_NAME":         "bge-reranker-v2-m3",
		"MODEL_MODE":         "rerank",
		"MODEL_SOURCE":       "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64),
		"MODEL_SOURCE_LOCAL": "/data/model",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineRerank {
		t.Errorf("Kind = %q, want rerank", cfg.Engine.Kind)
	}
	if cfg.Model.Type != ModelRerank || cfg.Spec.Mode != "rerank" {
		t.Errorf("Model.Type/Spec.Mode = %q / %q, want rerank", cfg.Model.Type, cfg.Spec.Mode)
	}
	if cfg.Engine.URL != "http://rerank:8080" {
		t.Errorf("URL = %q, want http://rerank:8080", cfg.Engine.URL)
	}
	if len(cfg.Engine.Args.Known) != 0 || cfg.Engine.Args.Raw != "" {
		t.Errorf("Engine.Args should be empty for rerank, got %+v", cfg.Engine.Args)
	}
}

func TestLoad_SystemOneKindAccepted(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":                   "systemone",
		"MODEL_NAME":                    "jevk5",
		"MODEL_MODE":                    "system_one",
		"MODEL_SOURCE":                  "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64),
		"MODEL_SOURCE_LOCAL":            "/data/model",
		"SYSTEM_ONE_CONTEXT_SIZE":       "4096",
		"SYSTEM_ONE_MAX_CHOICE_OPTIONS": "16",
		"SYSTEM_ONE_MAX_SCORE_LEVELS":   "7",
		"SYSTEM_ONE_LANGUAGES":          "en, zh, en",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineSystemOne {
		t.Errorf("Kind = %q, want systemone", cfg.Engine.Kind)
	}
	if cfg.Model.Type != ModelSystemOne || cfg.Spec.Mode != "system_one" {
		t.Errorf("Model.Type/Spec.Mode = %q / %q, want system_one", cfg.Model.Type, cfg.Spec.Mode)
	}
	if cfg.Engine.URL != "http://systemone:8000" {
		t.Errorf("URL = %q, want http://systemone:8000", cfg.Engine.URL)
	}
	if len(cfg.Engine.Args.Known) != 0 || cfg.Engine.Args.Raw != "" {
		t.Errorf("Engine.Args should be empty for systemone, got %+v", cfg.Engine.Args)
	}
	if cfg.Spec.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096", cfg.Spec.ContextSize)
	}
	extension := cfg.Spec.Extensions["system_one"].(map[string]any)
	if got := extension["languages"]; !reflect.DeepEqual(got, []string{"en", "zh"}) {
		t.Errorf("languages = %#v, want [en zh]", got)
	}
}

func TestLoad_SystemOneRequiresCapacityDeclaration(t *testing.T) {
	_, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "systemone",
		"MODEL_NAME":         "jevk5",
		"MODEL_MODE":         "system_one",
		"MODEL_SOURCE":       "https://example.com/model.tgz#sha256=" + strings.Repeat("a", 64),
		"MODEL_SOURCE_LOCAL": "/data/model",
	}))
	if err == nil || !strings.Contains(err.Error(), systemOneContextSizeEnv) {
		t.Fatalf("err = %v, want missing System One capacity error", err)
	}
}

func TestLoad_MusicKindAccepted(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "music",
		"MODEL_NAME":   "ACE-Step/acestep-v15-xl-turbo",
		"MODEL_MODE":   "music_generation",
		"MODEL_SOURCE": "hf://ACE-Step/acestep-v15-xl-turbo",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != EngineMusic || cfg.Engine.URL != "http://music-engine:8001" {
		t.Errorf("engine = %q %q", cfg.Engine.Kind, cfg.Engine.URL)
	}
	if cfg.Model.Type != ModelMusicGeneration || cfg.Spec.Mode != "music_generation" {
		t.Errorf("mode = %q / %q", cfg.Model.Type, cfg.Spec.Mode)
	}
}

func TestLoad_HFSingleIncludeLlamaCpp(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "llamacpp",
		"MODEL_NAME":   "qwen-q4",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://bartowski/Qwen2.5-7B-Instruct-GGUF --include qwen2.5-7b-instruct-q4_k_m.gguf",
		"ENGINE_ARGS":  "-c 8192 -ngl all",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := cfg.Sources[0].HFInclude; len(got) != 1 || got[0] != "qwen2.5-7b-instruct-q4_k_m.gguf" {
		t.Errorf("HFInclude = %v", got)
	}
	if cfg.Engine.Args.Known["ctx_size"] != "8192" {
		t.Errorf("Args.ctx_size = %q", cfg.Engine.Args.Known["ctx_size"])
	}
	if v, ok := cfg.Engine.Args.GetInt("ctx_size"); !ok || v != 8192 {
		t.Errorf("Args.GetInt(ctx_size) = %d ok=%v", v, ok)
	}
	if v, ok := cfg.Engine.Args.GetString("n_gpu_layers"); !ok || v != "all" {
		t.Errorf("Args.n_gpu_layers = %q ok=%v", v, ok)
	}
}

func TestLoad_URLWithSha256AndLocal(t *testing.T) {
	sha := strings.Repeat("a", 64)
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "llamacpp",
		"MODEL_NAME":         "manual-q4",
		"MODEL_MODE":         "chat",
		"MODEL_SOURCE":       "https://example.com/q4.gguf#sha256=" + sha,
		"MODEL_SOURCE_LOCAL": "/cache/manual/q4.gguf",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Sources[0].Kind != KindURL || cfg.Sources[0].URLSha256 != sha {
		t.Fatalf("Source = %+v", cfg.Sources[0])
	}
	if cfg.Sources[0].LocalPath != "/cache/manual/q4.gguf" {
		t.Errorf("LocalPath = %q", cfg.Sources[0].LocalPath)
	}
	if cfg.PrimarySourceKind() != KindURL {
		t.Errorf("PrimarySourceKind = %q", cfg.PrimarySourceKind())
	}
}

func TestLoad_CommaSeparatedHFPlusURLExtra(t *testing.T) {
	sha := strings.Repeat("b", 64)
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "llamacpp",
		"MODEL_NAME":         "internvl3-8b",
		"MODEL_MODE":         "chat",
		"MODEL_SOURCE":       "hf://InternVL/InternVL3-8B-GGUF --include internvl3-8b-q4.gguf,https://example.com/aux.gguf#sha256=" + sha,
		"MODEL_SOURCE_LOCAL": ",/cache/internvl/aux.gguf",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Sources) != 2 {
		t.Fatalf("len(Sources) = %d", len(cfg.Sources))
	}
	if cfg.Sources[0].Role != RoleMain || cfg.Sources[1].Role != RoleExtra {
		t.Errorf("roles = %q,%q", cfg.Sources[0].Role, cfg.Sources[1].Role)
	}
}

func TestLoad_MODEL_SOURCE_NUMRemoved(t *testing.T) {
	_, err := Load(mapEnv(minimalEnv(map[string]string{
		removedModelSourceNumEnv: "2",
		"MODEL_SOURCE_1":         "hf://owner/main",
		"MODEL_SOURCE_2":         "hf://owner/extra",
	})))
	if err == nil {
		t.Fatal("expected MODEL_SOURCE_NUM removal error")
	}
	for _, want := range []string{removedModelSourceNumEnv, "removed", "comma-separated MODEL_SOURCE"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %q, want %q", err, want)
		}
	}
}

func TestLoad_EngineArgsParsedPerKind(t *testing.T) {
	cases := []struct {
		kind    string
		source  string
		raw     string
		wantKey string
		wantVal string
	}{
		{"ollama", "ollama://qwen3:0.6b", "OLLAMA_NUM_CTX=4096 OLLAMA_KEEP_ALIVE=10m", "num_ctx", "4096"},
		{"vllm", "hf://Qwen/Qwen --include x.gguf", "--max-model-len 8192 --gpu-memory-utilization 0.9", "max_model_len", "8192"},
		{"llamacpp", "hf://o/r --include x.gguf", "-c 4096 -ngl all", "n_gpu_layers", "all"},
		{"sglang", "hf://o/r --include x.gguf", "--mem-fraction-static 0.8 --max-running-requests 256", "mem_fraction_static", "0.8"},
	}
	for _, tc := range cases {
		t.Run(tc.kind, func(t *testing.T) {
			cfg, err := Load(mapEnv(map[string]string{
				"ENGINE_KIND":  tc.kind,
				"MODEL_NAME":   "m",
				"MODEL_MODE":   "chat",
				"MODEL_SOURCE": tc.source,
				"ENGINE_ARGS":  tc.raw,
			}))
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if got := cfg.Engine.Args.Known[tc.wantKey]; got != tc.wantVal {
				t.Errorf("Args.Known[%q] = %q, want %q", tc.wantKey, got, tc.wantVal)
			}
			if cfg.Engine.Args.Raw != tc.raw {
				t.Errorf("Args.Raw = %q, want %q", cfg.Engine.Args.Raw, tc.raw)
			}
		})
	}
}

func TestLoad_MissingMODEL_MODE(t *testing.T) {
	env := minimalEnv(nil)
	delete(env, "MODEL_MODE")
	_, err := Load(mapEnv(env))
	if err == nil || !strings.Contains(err.Error(), "MODEL_MODE") {
		t.Fatalf("err = %v, want MODEL_MODE required error", err)
	}
}

func TestLoad_InvalidMODEL_MODE(t *testing.T) {
	_, err := Load(mapEnv(minimalEnv(map[string]string{"MODEL_MODE": "bogus"})))
	if err == nil || !strings.Contains(err.Error(), "MODEL_MODE") {
		t.Fatalf("err = %v, want MODEL_MODE enum error", err)
	}
}

func TestLoad_ModeEmbeddingDerivesType(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{"MODEL_MODE": "embedding"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Spec.Mode != "embedding" {
		t.Errorf("Spec.Mode = %q", cfg.Spec.Mode)
	}
	if cfg.Model.Type != ModelEmbedding {
		t.Errorf("Model.Type = %q, want embedding", cfg.Model.Type)
	}
}

func TestLoad_ModeOCRDerivesType(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{"MODEL_MODE": "ocr"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Spec.Mode != "ocr" {
		t.Errorf("Spec.Mode = %q", cfg.Spec.Mode)
	}
	if cfg.Model.Type != ModelOCR {
		t.Errorf("Model.Type = %q, want ocr", cfg.Model.Type)
	}
}

func TestLoad_ModeTranslateDerivesType(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{"MODEL_MODE": "translate"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Spec.Mode != "translate" {
		t.Errorf("Spec.Mode = %q", cfg.Spec.Mode)
	}
	if cfg.Model.Type != ModelTranslate {
		t.Errorf("Model.Type = %q, want translate", cfg.Model.Type)
	}
}

func TestLoad_ModeMusicGenerationDerivesType(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{"MODEL_MODE": "music_generation"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Spec.Mode != "music_generation" || cfg.Model.Type != ModelMusicGeneration {
		t.Errorf("Model.Type/Spec.Mode = %q / %q", cfg.Model.Type, cfg.Spec.Mode)
	}
}

func TestLoad_SupportsCSV(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"MODEL_SUPPORTS": "supports_reasoning, supports_vision",
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !cfg.Spec.Supports["supports_reasoning"] || !cfg.Spec.Supports["supports_vision"] {
		t.Errorf("Supports = %+v, want reasoning+vision true", cfg.Spec.Supports)
	}
	if _, present := cfg.Spec.Supports["supports_tool_choice"]; present {
		t.Errorf("unlisted key should be absent, got %+v", cfg.Spec.Supports)
	}
	if !cfg.Model.ThinkSupported {
		t.Error("ThinkSupported should be derived true from supports_reasoning")
	}
}

func TestLoad_SupportsEmptyDefaults(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Spec.Supports == nil || len(cfg.Spec.Supports) != 0 {
		t.Errorf("Supports = %+v, want empty map", cfg.Spec.Supports)
	}
	if cfg.Model.ThinkSupported {
		t.Error("ThinkSupported should default false with no MODEL_SUPPORTS")
	}
}

// An unknown env key boots and is tagged, the same treatment a card gets.
// The env used to reject it, which meant a published manifest's typo — or a
// key Router ratified first — kept the container down.
func TestLoad_SupportsUnknownTokenKeptAndTagged(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{"MODEL_SUPPORTS": "supports_bogus"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !cfg.Spec.Supports["supports_bogus"] {
		t.Errorf("Supports = %+v, want the unrecognized key kept", cfg.Spec.Supports)
	}
	if got := unknownFlag(t, cfg.Spec); len(got) != 1 || got[0] != "supports_bogus" {
		t.Errorf("_unknown_flag = %v, want [supports_bogus]", got)
	}
}

func TestLoad_SupportsTranslateTagged(t *testing.T) {
	key := "supports_" + "translate"
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"MODEL_SUPPORTS": key,
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := unknownFlag(t, cfg.Spec); len(got) != 1 || got[0] != key {
		t.Errorf("_unknown_flag = %v, want [%s]", got, key)
	}
}

// unknownFlag reads the tag validateModelSpec stamps on a spec carrying keys
// this mirror does not recognize.
func unknownFlag(t *testing.T, spec ModelSpec) []string {
	t.Helper()
	raw, ok := spec.Extensions["_unknown_flag"]
	if !ok {
		t.Fatalf("Extensions = %v, want _unknown_flag", spec.Extensions)
	}
	keys, ok := raw.([]string)
	if !ok {
		t.Fatalf("_unknown_flag is %T, want []string", raw)
	}
	return keys
}

func TestLoad_TranslateLanguagesSeedsExtension(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"MODEL_MODE":          "translate",
		"TRANSLATE_LANGUAGES": "en,zh-CN,ja",
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	raw, ok := cfg.Spec.Extensions["translate"]
	if !ok {
		t.Fatalf("Extensions=%v, want translate", cfg.Spec.Extensions)
	}
	ext, ok := raw.(translate.Extension)
	if !ok {
		t.Fatalf("type %T, want translate.Extension", raw)
	}
	if len(ext.Languages) != 3 || ext.Languages[1] != "zh-Hans" {
		t.Fatalf("languages=%v", ext.Languages)
	}
}

func TestLoad_ReasoningEffortSeedsRule(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"MODEL_SUPPORTS":                 "supports_reasoning,supports_reasoning_effort",
		"MODEL_REASONING_EFFORT":         "xhigh, LOW ,medium,low",
		"MODEL_REASONING_EFFORT_DEFAULT": "xhigh",
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Spec.ParameterRules) != 1 {
		t.Fatalf("ParameterRules = %+v, want one rule", cfg.Spec.ParameterRules)
	}
	rule := cfg.Spec.ParameterRules[0]
	if rule.Name != "reasoning_effort" || rule.Type != "string" {
		t.Errorf("rule = %+v, want a string reasoning_effort rule", rule)
	}
	// Ladder order regardless of how they were typed, deduplicated,
	// case- and space-insensitive.
	want := []string{"low", "medium", "xhigh"}
	if len(rule.Options) != len(want) {
		t.Fatalf("options = %v, want %v", rule.Options, want)
	}
	for i := range want {
		if rule.Options[i] != want[i] {
			t.Fatalf("options = %v, want %v", rule.Options, want)
		}
	}
	if rule.Default != "xhigh" {
		t.Errorf("default = %v, want xhigh", rule.Default)
	}
}

func TestLoad_NoReasoningEffortEnvLeavesNoRule(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Spec.ParameterRules) != 0 {
		t.Errorf("ParameterRules = %+v, want none declared", cfg.Spec.ParameterRules)
	}
}

// A level the engine's chat template does not know is a request that
// errors out for whoever trusted the card, so it fails at boot instead.
func TestLoad_ReasoningEffortUnknownLevelFailFast(t *testing.T) {
	for _, level := range []string{"turbo", "thinking"} {
		_, err := Load(mapEnv(minimalEnv(map[string]string{
			"MODEL_REASONING_EFFORT": "low," + level,
		})))
		if err == nil {
			t.Fatalf("level %q: expected fail-fast", level)
		}
		for _, want := range []string{"MODEL_REASONING_EFFORT", level} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("level %q: error = %v, want %q", level, err, want)
			}
		}
	}
}

func TestLoad_ReasoningEffortDefaultMustBeDeclared(t *testing.T) {
	_, err := Load(mapEnv(minimalEnv(map[string]string{
		"MODEL_REASONING_EFFORT":         "low,medium",
		"MODEL_REASONING_EFFORT_DEFAULT": "xhigh",
	})))
	if err == nil || !strings.Contains(err.Error(), "MODEL_REASONING_EFFORT_DEFAULT") {
		t.Fatalf("err = %v, want the default rejected for not being on the ladder", err)
	}
}

func TestLoad_ReasoningEffortDefaultWithoutLadderRejected(t *testing.T) {
	_, err := Load(mapEnv(minimalEnv(map[string]string{
		"MODEL_REASONING_EFFORT_DEFAULT": "medium",
	})))
	if err == nil || !strings.Contains(err.Error(), "MODEL_REASONING_EFFORT") {
		t.Fatalf("err = %v, want a default-without-ladder error", err)
	}
}

// TestLoad_SupportsAudioKeys: audio keys are supports_* like every other key.
func TestLoad_SupportsAudioKeys(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"MODEL_MODE":     "audio",
		"MODEL_SUPPORTS": "supports_stt, supports_vad",
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Spec.Mode != "audio" {
		t.Errorf("Spec.Mode = %q, want audio", cfg.Spec.Mode)
	}
	if !cfg.Spec.Supports["supports_stt"] || !cfg.Spec.Supports["supports_vad"] {
		t.Errorf("Supports = %+v, want supports_stt+supports_vad true", cfg.Spec.Supports)
	}
	if cfg.Model.Type != ModelAudio {
		t.Errorf("Model.Type = %q, want audio", cfg.Model.Type)
	}
}

// tts is its own mode but not its own engine: ENGINE_KIND stays audio, and
// the four synthesis keys come out of the same supports vocabulary. A
// deployment that got either half wrong would be discovered by Router as
// something it cannot route.
func TestLoad_TTSModeKeepsTheAudioEngine(t *testing.T) {
	var keys []string
	for _, k := range allAudioSupports {
		if strings.HasPrefix(k, "supports_tts") {
			keys = append(keys, k)
		}
	}
	if len(keys) != 5 {
		t.Fatalf("synthesis keys = %v, want the five in allAudioSupports", keys)
	}
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"ENGINE_KIND":    "audio",
		"MODEL_MODE":     "tts",
		"MODEL_NAME":     "FireRedTeam/FireRedTTS3",
		"MODEL_SOURCE":   "hf://FireRedTeam/FireRedTTS3 --exclude fireredtts3_base/*",
		"MODEL_SUPPORTS": strings.Join(keys, ", "),
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Model.Type != ModelTTS || cfg.Spec.Mode != "tts" {
		t.Errorf("Model.Type = %q / Spec.Mode = %q, want tts / tts", cfg.Model.Type, cfg.Spec.Mode)
	}
	if cfg.Engine.Kind != EngineAudio {
		t.Errorf("Engine.Kind = %q, want audio", cfg.Engine.Kind)
	}
	for _, k := range keys {
		if !cfg.Spec.Supports[k] {
			t.Errorf("Supports[%q] = false, want true", k)
		}
	}
	if _, unknown := cfg.Spec.Extensions["_unknown_flag"]; unknown {
		t.Errorf("design/custom were tagged unknown: %v", cfg.Spec.Extensions["_unknown_flag"])
	}
}

func TestLoad_SeedsTTSSpeedRange(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"ENGINE_KIND":       "audio",
		"MODEL_MODE":        "tts",
		"MODEL_SOURCE":      "hf://FireRedTeam/FireRedTTS3",
		"MODEL_SUPPORTS":    "supports_tts",
		"TTS_SPEED_MIN":     "0.25",
		"TTS_SPEED_DEFAULT": "1",
		"TTS_SPEED_MAX":     "4",
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tts := cfg.Spec.Extensions["tts"].(map[string]any)
	settings := tts["voice_settings"].(map[string]any)
	speed := settings["speed"].(map[string]any)
	if speed["min"] != 0.25 || speed["default"] != float64(1) || speed["max"] != float64(4) {
		t.Fatalf("speed = %#v", speed)
	}
}

func TestLoad_RejectsPartialTTSSpeedRange(t *testing.T) {
	_, err := Load(mapEnv(minimalEnv(map[string]string{"TTS_SPEED_MIN": "0.25"})))
	if err == nil || !strings.Contains(err.Error(), "must be set together") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoad_MissingMODEL_NAME(t *testing.T) {
	env := minimalEnv(nil)
	delete(env, "MODEL_NAME")
	_, err := Load(mapEnv(env))
	if err == nil || !strings.Contains(err.Error(), "MODEL_NAME") {
		t.Fatalf("err = %v, want MODEL_NAME error", err)
	}
}

func TestLoad_MissingMODEL_SOURCE(t *testing.T) {
	env := minimalEnv(nil)
	delete(env, "MODEL_SOURCE")
	_, err := Load(mapEnv(env))
	if err == nil || !strings.Contains(err.Error(), "MODEL_SOURCE") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoad_HFTokenSetReflected(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "m",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://o/r",
		"HF_TOKEN":     "hf_super_secret",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !cfg.HFTokenSet {
		t.Error("HFTokenSet should be true")
	}
}

func TestLoad_HFEndpointMirror(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "m",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://o/r",
		"HF_ENDPOINT":  "https://hf-mirror.com",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.HFEndpoint != "https://hf-mirror.com" {
		t.Errorf("HFEndpoint = %q", cfg.HFEndpoint)
	}
}

func TestLoad_HFDeployIgnoredWithoutHFSource(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "ollama",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "ollama://qwen2.5:7b",
		"HF_ENDPOINT":  "not-a-valid-url",
		"HF_TOKEN":     "hf_super_secret",
	}))
	if err != nil {
		t.Fatalf("Load: %v (HF envs must be ignored for a non-hf source)", err)
	}
	if cfg.HFEndpoint != "" {
		t.Errorf("HFEndpoint = %q, want empty (no hf:// source)", cfg.HFEndpoint)
	}
	if cfg.HFTokenSet || cfg.HFToken != "" {
		t.Errorf("HFToken=%q HFTokenSet=%v, want unset (no hf:// source)", cfg.HFToken, cfg.HFTokenSet)
	}
}

func TestLoad_DownloadOnlyEmptyEngine(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "",
		"MODEL_NAME":   "qwen-q4",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct",
		"ENGINE_ARGS":  "--ignored 1",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Engine.Kind != "" {
		t.Errorf("Engine.Kind = %q, want empty (download-only)", cfg.Engine.Kind)
	}
	if cfg.Engine.URL != "" {
		t.Errorf("Engine.URL = %q, want empty (download-only)", cfg.Engine.URL)
	}
	if cfg.Engine.Args.Raw != "" {
		t.Errorf("Engine.Args.Raw = %q, want empty (ENGINE_ARGS skipped without engine)", cfg.Engine.Args.Raw)
	}
	if len(cfg.Sources) != 1 || cfg.Sources[0].Kind != KindHF {
		t.Fatalf("Sources = %+v", cfg.Sources)
	}
}

func TestLoad_DownloadOnlyRejectsOllamaSource(t *testing.T) {
	for _, src := range []string{
		"ollama://qwen2.5:7b",
		"ollama://https://huggingface.co/owner/repo/resolve/main/q4.gguf",
	} {
		_, err := Load(mapEnv(map[string]string{
			"ENGINE_KIND":  "",
			"MODEL_NAME":   "m",
			"MODEL_MODE":   "chat",
			"MODEL_SOURCE": src,
		}))
		if err == nil {
			t.Fatalf("expected fail-fast for download-only + %q", src)
		}
		if !strings.Contains(err.Error(), "download-only") {
			t.Errorf("error should mention download-only mode: %v", err)
		}
	}
}

func TestLoad_CrossField_VLLMRejectsOllamaTag(t *testing.T) {
	_, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "m",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "ollama://qwen3:0.6b",
	}))
	if err == nil || !strings.Contains(err.Error(), "does not accept ollama:// Library tag") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoad_CrossField_LlamacppHFNeedsInclude(t *testing.T) {
	_, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "llamacpp",
		"MODEL_NAME":   "m",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://o/r",
	}))
	if err == nil || !strings.Contains(err.Error(), "requires exactly one --include") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoad_RuntimeDefaults(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.Port != defaultPort {
		t.Errorf("Port = %d", cfg.Runtime.Port)
	}
	if cfg.Runtime.MaxConcurrentDownloads != defaultMaxConcurrentDLs {
		t.Errorf("MaxConcurrentDownloads = %d", cfg.Runtime.MaxConcurrentDownloads)
	}
	if cfg.Runtime.UpstreamResponseHeaderTimeout != DefaultUpstreamResponseHeaderTimeout {
		t.Errorf("UpstreamResponseHeaderTimeout = %v, want %v",
			cfg.Runtime.UpstreamResponseHeaderTimeout, DefaultUpstreamResponseHeaderTimeout)
	}
	if cfg.Log.Level != defaultLogLevel {
		t.Errorf("Log.Level = %q", cfg.Log.Level)
	}
	if cfg.Runtime.EnableXet {
		t.Errorf("EnableXet default = true, want false (stable LFS path)")
	}
}

func TestLoad_HFEnableXet(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want bool
	}{
		{"true", true},
		{"1", true},
		{"false", false},
		{"0", false},
	} {
		cfg, err := Load(mapEnv(minimalEnv(map[string]string{"HF_ENABLE_XET": tc.raw})))
		if err != nil {
			t.Fatalf("HF_ENABLE_XET=%q: Load: %v", tc.raw, err)
		}
		if cfg.Runtime.EnableXet != tc.want {
			t.Errorf("HF_ENABLE_XET=%q: EnableXet = %v, want %v", tc.raw, cfg.Runtime.EnableXet, tc.want)
		}
	}

	if _, err := Load(mapEnv(minimalEnv(map[string]string{"HF_ENABLE_XET": "yesplease"}))); err == nil ||
		!strings.Contains(err.Error(), "HF_ENABLE_XET") {
		t.Fatalf("invalid HF_ENABLE_XET err = %v", err)
	}
}

// VERIFY_LEVEL is the standing depth of the local check. `remote` is
// rejected here on purpose: it reaches the upstream, and a standing
// default that does so trades away the ability to boot offline. It is
// reachable per pass through POST /api/retry?level=remote.
func TestLoad_VerifyLevel(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.VerifyLevel != defaultVerifyLevel {
		t.Errorf("default VerifyLevel = %q, want %q", cfg.Runtime.VerifyLevel, defaultVerifyLevel)
	}

	for _, level := range []string{"size", "sha256"} {
		cfg, err := Load(mapEnv(minimalEnv(map[string]string{"VERIFY_LEVEL": level})))
		if err != nil {
			t.Fatalf("VERIFY_LEVEL=%q: Load: %v", level, err)
		}
		if cfg.Runtime.VerifyLevel != level {
			t.Errorf("VERIFY_LEVEL=%q: VerifyLevel = %q", level, cfg.Runtime.VerifyLevel)
		}
	}

	for _, bad := range []string{"remote", "md5", "yes"} {
		if _, err := Load(mapEnv(minimalEnv(map[string]string{"VERIFY_LEVEL": bad}))); err == nil ||
			!strings.Contains(err.Error(), "VERIFY_LEVEL") {
			t.Errorf("VERIFY_LEVEL=%q: err = %v, want it rejected by name", bad, err)
		}
	}
}

// VERIFY_ON_DRIFT decides what a remote check does when the upstream has
// moved. Reporting is the default because following would swap the
// weights under a model that is currently serving.
func TestLoad_VerifyOnDrift(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.VerifyOnDrift != VerifyOnDriftReport {
		t.Errorf("default VerifyOnDrift = %q, want %q", cfg.Runtime.VerifyOnDrift, VerifyOnDriftReport)
	}

	cfg, err = Load(mapEnv(minimalEnv(map[string]string{"VERIFY_ON_DRIFT": VerifyOnDriftFollow})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.VerifyOnDrift != VerifyOnDriftFollow {
		t.Errorf("VerifyOnDrift = %q, want %q", cfg.Runtime.VerifyOnDrift, VerifyOnDriftFollow)
	}

	if _, err := Load(mapEnv(minimalEnv(map[string]string{"VERIFY_ON_DRIFT": "delete"}))); err == nil ||
		!strings.Contains(err.Error(), "VERIFY_ON_DRIFT") {
		t.Fatalf("invalid VERIFY_ON_DRIFT err = %v", err)
	}
}

func TestLoad_ModelSpecPathDefaultAndOverride(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(nil)))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.ModelSpecPath != defaultRunDir+"/model-spec.json" {
		t.Errorf("ModelSpecPath default = %q", cfg.Runtime.ModelSpecPath)
	}

	cfg, err = Load(mapEnv(minimalEnv(map[string]string{"MODEL_SPEC_PATH": "/etc/llm-init/model-spec.json"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.ModelSpecPath != "/etc/llm-init/model-spec.json" {
		t.Errorf("ModelSpecPath override = %q", cfg.Runtime.ModelSpecPath)
	}
}

func TestLoad_PORTOverrides(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{"PORT": "9090"})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.Port != 9090 {
		t.Errorf("Port = %d", cfg.Runtime.Port)
	}
}

func TestRedacted_JSONShape(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "qwen-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision " + strings.Repeat("a", 40),
		"HF_TOKEN":     "hf_super_secret",
		"ENGINE_ARGS":  "--max-model-len 8192",
		"APP_URL":      "https://models.example.test",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	r := cfg.Redacted()
	if r.HFTokenSet != true {
		t.Error("HFTokenSet should be true in Redacted output")
	}
	if r.HFEndpoint != "https://huggingface.co" {
		t.Errorf("HFEndpoint = %q", r.HFEndpoint)
	}
	if len(r.Sources) != 1 || r.Sources[0].Kind != KindHF {
		t.Fatalf("Sources = %+v", r.Sources)
	}
	if r.Engine.Args.Known["max_model_len"] != "8192" {
		t.Errorf("Engine.Args.Known = %v", r.Engine.Args.Known)
	}
	if r.ModelSpecFile != defaultRunDir+"/model-spec.json" {
		t.Errorf("ModelSpecFile = %q", r.ModelSpecFile)
	}
	// JSON marshal smoke test - ensure no secrets leak in raw JSON.
	out, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	js := string(out)
	if strings.Contains(js, "hf_super_secret") {
		t.Errorf("HF_TOKEN leaked into Redacted JSON: %s", js)
	}
	if !strings.Contains(js, `"hf_token_set":true`) {
		t.Errorf("hf_token_set missing from JSON: %s", js)
	}
	// runtime / log are nested structs; without json tags they marshal as
	// Go field names, which is what silently broke the dashboard's
	// APP_URL read (it looked for c.Runtime.PublicURL against a
	// `json:"runtime"` parent).
	for _, want := range []string{
		`"run_dir":`,
		`"public_url":"https://models.example.test"`,
		`"upstream_trace":`,
		`"upstream_response_header_timeout":`,
	} {
		if !strings.Contains(js, want) {
			t.Errorf("%s missing from JSON: %s", want, js)
		}
	}
	for _, unwanted := range []string{`"RunDir"`, `"PublicURL"`, `"UpstreamTrace"`} {
		if strings.Contains(js, unwanted) {
			t.Errorf("%s should be snake_case in JSON: %s", unwanted, js)
		}
	}
}

func TestRedacted_URLUserinfoMasked(t *testing.T) {
	cfg, err := Load(mapEnv(map[string]string{
		"ENGINE_KIND":        "llamacpp",
		"MODEL_NAME":         "m",
		"MODEL_MODE":         "chat",
		"MODEL_SOURCE":       "https://user:pass@example.com/q4.gguf",
		"MODEL_SOURCE_LOCAL": "/cache/m.gguf",
	}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	r := cfg.Redacted()
	if !strings.Contains(r.Sources[0].RedactedSource, "REDACTED") {
		t.Errorf("RedactedSource = %q", r.Sources[0].RedactedSource)
	}
	if strings.Contains(r.Sources[0].RedactedSource, "user:pass") {
		t.Errorf("userinfo leaked in redacted source: %q", r.Sources[0].RedactedSource)
	}
}

func TestLoad_UpstreamResponseHeaderTimeout(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"UPSTREAM_RESPONSE_HEADER_TIMEOUT": "90s",
	})))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Runtime.UpstreamResponseHeaderTimeout != 90*time.Second {
		t.Errorf("got %v, want 90s", cfg.Runtime.UpstreamResponseHeaderTimeout)
	}

	for _, raw := range []string{"4s", "31m", "120", "abc"} {
		if _, err := Load(mapEnv(minimalEnv(map[string]string{
			"UPSTREAM_RESPONSE_HEADER_TIMEOUT": raw,
		}))); err == nil {
			t.Errorf("UPSTREAM_RESPONSE_HEADER_TIMEOUT=%q: expected error", raw)
		}
	}
}

func TestRuntime_ResponseHeaderTimeoutFallback(t *testing.T) {
	if got := (Runtime{}).ResponseHeaderTimeout(); got != DefaultUpstreamResponseHeaderTimeout {
		t.Errorf("zero Runtime = %v, want default %v", got, DefaultUpstreamResponseHeaderTimeout)
	}
	want := 90 * time.Second
	if got := (Runtime{UpstreamResponseHeaderTimeout: want}).ResponseHeaderTimeout(); got != want {
		t.Errorf("explicit = %v, want %v", got, want)
	}
}
