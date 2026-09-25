package config

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseEngineArgs_OllamaKVList(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineOllama,
		"OLLAMA_NUM_CTX=8192 OLLAMA_KEEP_ALIVE=30m OLLAMA_NUM_PARALLEL=2")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	wantKnown := map[string]string{
		"num_ctx":      "8192",
		"keep_alive":   "30m",
		"num_parallel": "2",
	}
	if !reflect.DeepEqual(got.Known, wantKnown) {
		t.Errorf("Known = %v, want %v", got.Known, wantKnown)
	}
	if len(got.Unknown) != 0 {
		t.Errorf("Unknown = %v, want []", got.Unknown)
	}
}

func TestParseEngineArgs_OllamaQuotedValue(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineOllama,
		`OLLAMA_NUM_CTX=8192 OLLAMA_DEBUG="info verbose" OLLAMA_KEEP_ALIVE=30m`)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.Known["debug"] != "info verbose" {
		t.Errorf("debug = %q, want %q", got.Known["debug"], "info verbose")
	}
	if got.Known["num_ctx"] != "8192" || got.Known["keep_alive"] != "30m" {
		t.Errorf("Known = %v", got.Known)
	}
}

func TestParseEngineArgs_OllamaUnknownPassthrough(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineOllama,
		"OLLAMA_NUM_CTX=8192 NOT_OLLAMA=foo bare-token")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.Known["num_ctx"] != "8192" {
		t.Errorf("num_ctx = %q", got.Known["num_ctx"])
	}
	wantUnknown := []string{"NOT_OLLAMA=foo", "bare-token"}
	if !reflect.DeepEqual(got.Unknown, wantUnknown) {
		t.Errorf("Unknown = %v, want %v", got.Unknown, wantUnknown)
	}
}

func TestParseEngineArgs_VLLMCmdline(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineVLLM,
		"--max-model-len 8192 --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 --enforce-eager")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	wantKnown := map[string]string{
		"max_model_len":          "8192",
		"gpu_memory_utilization": "0.9",
		"kv_cache_dtype":         "fp8",
		"enforce_eager":          "true",
	}
	if !reflect.DeepEqual(got.Known, wantKnown) {
		t.Errorf("Known = %v, want %v", got.Known, wantKnown)
	}
}

func TestParseEngineArgs_VLLMEqForm(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineVLLM,
		"--max-model-len=8192 --enforce-eager --dtype=bfloat16")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.Known["max_model_len"] != "8192" {
		t.Errorf("max_model_len = %q", got.Known["max_model_len"])
	}
	if got.Known["enforce_eager"] != "true" {
		t.Errorf("enforce_eager = %q", got.Known["enforce_eager"])
	}
	if got.Known["dtype"] != "bfloat16" {
		t.Errorf("dtype = %q", got.Known["dtype"])
	}
}

func TestParseEngineArgs_VLLMUnknownFlagPreservesValue(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineVLLM, "--max-model-len 8192 --not-a-known-flag value42")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.Known["max_model_len"] != "8192" {
		t.Errorf("max_model_len = %q", got.Known["max_model_len"])
	}
	wantUnknown := []string{"--not-a-known-flag", "value42"}
	if !reflect.DeepEqual(got.Unknown, wantUnknown) {
		t.Errorf("Unknown = %v, want %v", got.Unknown, wantUnknown)
	}
}

func TestParseEngineArgs_LlamacppCmdline(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineLlamaCpp,
		"-c 8192 -ngl all -ctk q8_0 -ctv q8_0 -fa on")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	want := map[string]string{
		"ctx_size":     "8192",
		"n_gpu_layers": "all",
		"cache_type_k": "q8_0",
		"cache_type_v": "q8_0",
		"flash_attn":   "on",
	}
	if !reflect.DeepEqual(got.Known, want) {
		t.Errorf("Known = %v, want %v", got.Known, want)
	}
}

func TestParseEngineArgs_LlamacppEnvForm(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineLlamaCpp,
		"LLAMA_ARG_CTX_SIZE=8192 LLAMA_ARG_N_GPU_LAYERS=all LLAMA_ARG_FLASH_ATTN=on")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	want := map[string]string{
		"ctx_size":     "8192",
		"n_gpu_layers": "all",
		"flash_attn":   "on",
	}
	if !reflect.DeepEqual(got.Known, want) {
		t.Errorf("Known = %v, want %v", got.Known, want)
	}
}

func TestParseEngineArgs_LlamacppMixedForms(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineLlamaCpp,
		"-c 8192 LLAMA_ARG_FLASH_ATTN=on")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.Known["ctx_size"] != "8192" || got.Known["flash_attn"] != "on" {
		t.Errorf("Known = %v", got.Known)
	}
}

func TestParseEngineArgs_SGLangCmdline(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineSGLang,
		"--mem-fraction-static 0.8 --max-running-requests 256 --tp 1 --enable-torch-compile")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	want := map[string]string{
		"mem_fraction_static":  "0.8",
		"max_running_requests": "256",
		"tensor_parallel_size": "1",
		"enable_torch_compile": "true",
	}
	if !reflect.DeepEqual(got.Known, want) {
		t.Errorf("Known = %v, want %v", got.Known, want)
	}
}

func TestParseEngineArgs_EmptyRaw(t *testing.T) {
	t.Parallel()
	for _, kind := range []EngineKind{EngineOllama, EngineVLLM, EngineLlamaCpp, EngineSGLang, EngineEmbed, EngineClipEmbed, EngineOCR, EngineRerank} {
		got, err := ParseEngineArgs(kind, "")
		if err != nil {
			t.Errorf("kind=%s: %v", kind, err)
			continue
		}
		if got.Raw != "" || len(got.Known) != 0 || len(got.Unknown) != 0 {
			t.Errorf("kind=%s: expected zero EngineArgs, got %+v", kind, got)
		}
	}
}

func TestParseEngineArgs_UnbalancedQuoteFailFast(t *testing.T) {
	t.Parallel()
	for _, kind := range []EngineKind{EngineOllama, EngineVLLM, EngineLlamaCpp, EngineSGLang} {
		_, err := ParseEngineArgs(kind, `OLLAMA_NUM_CTX="8192`)
		if err == nil {
			t.Errorf("kind=%s: expected unbalanced-quote error", kind)
			continue
		}
		if !strings.Contains(err.Error(), "unbalanced quote") {
			t.Errorf("kind=%s: err = %v, want unbalanced quote", kind, err)
		}
	}
}

func TestParseEngineArgs_EmbedEmpty(t *testing.T) {
	t.Parallel()
	for _, kind := range []EngineKind{EngineEmbed, EngineClipEmbed, EngineOCR, EngineRerank, EngineMusic, EngineSystemOne} {
		got, err := ParseEngineArgs(kind, "")
		if err != nil {
			t.Fatalf("kind=%s err: %v", kind, err)
		}
		if got.Raw != "" || len(got.Known) != 0 {
			t.Errorf("kind=%s got %+v, want zero EngineArgs", kind, got)
		}
	}
}

func TestParseEngineArgs_EmbedRejectsNonEmpty(t *testing.T) {
	t.Parallel()
	for _, kind := range []EngineKind{EngineEmbed, EngineClipEmbed, EngineOCR, EngineRerank, EngineMusic, EngineSystemOne} {
		_, err := ParseEngineArgs(kind, "EMBED_DEVICE=cpu")
		if err == nil {
			t.Fatalf("kind=%s: expected error when ENGINE_ARGS set", kind)
		}
	}
}

func TestParseEngineArgs_UnsupportedKind(t *testing.T) {
	t.Parallel()
	if _, err := ParseEngineArgs("unknown", "x"); err == nil {
		t.Fatal("expected error for unsupported kind")
	}
}

func TestEngineArgs_Getters(t *testing.T) {
	t.Parallel()
	a := EngineArgs{Known: map[string]string{
		"num_ctx":                "8192",
		"gpu_memory_utilization": "0.9",
		"keep_alive":             "30m",
	}}
	if v, ok := a.GetString("keep_alive"); !ok || v != "30m" {
		t.Errorf("GetString(keep_alive) = (%q, %v)", v, ok)
	}
	if _, ok := a.GetString("missing"); ok {
		t.Error("GetString(missing) should be false")
	}
	if v, ok := a.GetInt("num_ctx"); !ok || v != 8192 {
		t.Errorf("GetInt(num_ctx) = (%d, %v)", v, ok)
	}
	if _, ok := a.GetInt("keep_alive"); ok {
		t.Error("GetInt(keep_alive) should be false (not parseable as int)")
	}
	if v, ok := a.GetFloat("gpu_memory_utilization"); !ok || v != 0.9 {
		t.Errorf("GetFloat(gpu_memory_utilization) = (%v, %v)", v, ok)
	}
	// Nil Known map should be safe.
	var b EngineArgs
	if _, ok := b.GetString("any"); ok {
		t.Error("nil Known: GetString should return false")
	}
}

func TestParseEngineArgs_VLLMUnknownThenKnown(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineVLLM,
		"--strange-flag --max-model-len 8192")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.Known["max_model_len"] != "8192" {
		t.Errorf("max_model_len = %q", got.Known["max_model_len"])
	}
	if !reflect.DeepEqual(got.Unknown, []string{"--strange-flag"}) {
		t.Errorf("Unknown = %v, want [--strange-flag]", got.Unknown)
	}
}

func TestShlex_Whitespace(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw  string
		want []string
	}{
		{"", nil},
		{"a b c", []string{"a", "b", "c"}},
		{"  a   b  ", []string{"a", "b"}},
		{`a "b c" d`, []string{"a", "b c", "d"}},
		{`a 'b c' d`, []string{"a", "b c", "d"}},
		{`a\ b`, []string{"a b"}},
		{`KEY="x y" OTHER=z`, []string{"KEY=x y", "OTHER=z"}},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			t.Parallel()
			got, err := shlex(tc.raw)
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestShlex_Errors(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw     string
		wantErr string
	}{
		{`"unbalanced`, "unbalanced quote"},
		{`'unbalanced`, "unbalanced quote"},
		{`a\`, "dangling backslash"},
	}
	for _, tc := range cases {
		t.Run(tc.raw, func(t *testing.T) {
			t.Parallel()
			_, err := shlex(tc.raw)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want %q", err, tc.wantErr)
			}
		})
	}
}
