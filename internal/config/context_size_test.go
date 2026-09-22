package config

import "testing"

func TestDeriveContextSize_PerEngineFlag(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		kind EngineKind
		raw  string
		want int
		ok   bool
	}{
		{"llamacpp short flag", EngineLlamaCpp, "-c 104448 -ngl all -np 1", 104448, true},
		{"llamacpp long flag", EngineLlamaCpp, "--ctx-size 8192", 8192, true},
		{"llamacpp env form", EngineLlamaCpp, "LLAMA_ARG_CTX_SIZE=8192", 8192, true},
		{"vllm", EngineVLLM, "--max-model-len 8192 --dtype bfloat16", 8192, true},
		{"vllm eq form", EngineVLLM, "--max-model-len=131072", 131072, true},
		{"sglang", EngineSGLang, "--context-length 16384 --tp 1", 16384, true},
		{"ollama", EngineOllama, "OLLAMA_NUM_CTX=4096 OLLAMA_KEEP_ALIVE=30m", 4096, true},

		{"llamacpp no ctx flag", EngineLlamaCpp, "-ngl all -fa on", 0, false},
		{"vllm no len flag", EngineVLLM, "--dtype bfloat16", 0, false},
		{"sglang max-total-tokens is not the window", EngineSGLang, "--max-total-tokens 40960", 0, false},
		{"empty args", EngineLlamaCpp, "", 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			args, err := ParseEngineArgs(tc.kind, tc.raw)
			if err != nil {
				t.Fatalf("ParseEngineArgs: %v", err)
			}
			got, ok := DeriveContextSize(tc.kind, args)
			if got != tc.want || ok != tc.ok {
				t.Errorf("DeriveContextSize = (%d, %v), want (%d, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

// llama.cpp splits -c across the parallel slots, so a conversation gets
// n_ctx/n_parallel. Reporting the undivided total would promise a window
// no single request can use.
func TestDeriveContextSize_LlamacppDividesByParallelSlots(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw  string
		want int
	}{
		{"-c 104448 -np 1", 104448},
		{"-c 65536 -np 2", 32768},
		{"-c 65536 --parallel 4", 16384},
		// A share that is not a multiple of 256 is padded up, and the pool
		// grows to match (src/llama-context.cpp:293-301).
		{"-c 12288 -np 5", 2560},
	}
	for _, tc := range cases {
		args, err := ParseEngineArgs(EngineLlamaCpp, tc.raw)
		if err != nil {
			t.Fatalf("ParseEngineArgs(%q): %v", tc.raw, err)
		}
		got, ok := DeriveContextSize(EngineLlamaCpp, args)
		if !ok || got != tc.want {
			t.Errorf("%q: got (%d, %v), want (%d, true)", tc.raw, got, ok, tc.want)
		}
	}
}

// With one shared pool there is no division: each slot may ask for the
// whole thing. Dividing anyway would under-report the window by the slot
// count, telling callers to truncate prompts the engine would have served.
func TestDeriveContextSize_LlamacppUnifiedPoolIsNotDivided(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		raw  string
		want int
	}{
		{"explicit pool with explicit slots", "-c 12288 -np 2 -kvu", 12288},
		// The two shapes that reach unified mode without saying so: an
		// absent slot count, and an opt-out that the auto branch overrules.
		{"absent slot count", "-c 131072 -ngl all -fa on", 131072},
		{"slot count written without a value", "-c 65536 -np", 65536},
		{"opt-out overruled by the auto branch", "-c 8192 -no-kvu", 8192},

		// --kv-unified-per-slot is the one thing that narrows a shared
		// pool, and n_ctx_slot() applies it as a min, never as a raise.
		{"per-slot ceiling binds", "-c 12288 -np 2 -kvu --kv-unified-per-slot 4096", 4096},
		{"per-slot ceiling above the pool cannot bind", "-c 8192 -np 2 -kvu --kv-unified-per-slot 16384", 8192},
		// Without -c the pool is sized to n_parallel * per_slot, so the
		// per-slot value is the window (tools/server/server.cpp:162-170).
		{"no pool flag, sized from the per-slot ceiling", "-np 4 -kvu --kv-unified-per-slot 8192", 8192},

		{"pool padded up to a multiple of 256", "-c 12000 -kvu", 12032},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			args, err := ParseEngineArgs(EngineLlamaCpp, tc.raw)
			if err != nil {
				t.Fatalf("ParseEngineArgs(%q): %v", tc.raw, err)
			}
			got, ok := DeriveContextSize(EngineLlamaCpp, args)
			if !ok || got != tc.want {
				t.Errorf("%q: got (%d, %v), want (%d, true)", tc.raw, got, ok, tc.want)
			}
		})
	}
}

// `-c 0` tells llama.cpp to read the training context out of the model
// file, which is not a number these flags carry.
func TestDeriveContextSize_LlamacppZeroMeansFromModel(t *testing.T) {
	t.Parallel()
	args, err := ParseEngineArgs(EngineLlamaCpp, "-c 0 -ngl all")
	if err != nil {
		t.Fatalf("ParseEngineArgs: %v", err)
	}
	if got, ok := DeriveContextSize(EngineLlamaCpp, args); ok {
		t.Errorf("got (%d, true), want no derivation", got)
	}
}

// The engines whose ENGINE_ARGS belong to a sibling container carry no
// launch flags here, so nothing can be derived for them.
func TestDeriveContextSize_NonLLMEnginesDeriveNothing(t *testing.T) {
	t.Parallel()
	for _, kind := range []EngineKind{EngineEmbed, EngineClipEmbed, EngineAudio, EngineOCR, EngineRerank, ""} {
		args, err := ParseEngineArgs(kind, "")
		if err != nil && kind != "" {
			t.Fatalf("kind=%s: %v", kind, err)
		}
		if got, ok := DeriveContextSize(kind, args); ok {
			t.Errorf("kind=%s: got (%d, true), want no derivation", kind, got)
		}
	}
}

// A card claiming a window the engine will not serve is what sends a
// caller into a truncated response, so the flags overwrite it.
func TestSyncContextSizeFromEngineArgs_FlagsOverwriteStoredValue(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Engine: Engine{Kind: EngineLlamaCpp},
		Spec:   ModelSpec{Name: "m", Mode: "chat", EngineArgs: "-c 104448 -np 1", ContextSize: 65536},
	}
	if err := ApplyEngineArgsFromSpec(cfg); err != nil {
		t.Fatalf("ApplyEngineArgsFromSpec: %v", err)
	}
	if !syncContextSizeFromEngineArgs(cfg) {
		t.Fatal("expected the card to change")
	}
	if cfg.Spec.ContextSize != 104448 {
		t.Errorf("ContextSize = %d, want 104448", cfg.Spec.ContextSize)
	}
	if syncContextSizeFromEngineArgs(cfg) {
		t.Error("second call should report no change")
	}
}

// Without a flag to read there is nothing better than what the operator
// wrote, so an existing value must not be zeroed.
func TestSyncContextSizeFromEngineArgs_KeepsValueWhenFlagsSaySilent(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Engine: Engine{Kind: EngineLlamaCpp},
		Spec:   ModelSpec{Name: "m", Mode: "chat", EngineArgs: "-ngl all", ContextSize: 32768},
	}
	if err := ApplyEngineArgsFromSpec(cfg); err != nil {
		t.Fatalf("ApplyEngineArgsFromSpec: %v", err)
	}
	if syncContextSizeFromEngineArgs(cfg) {
		t.Error("expected no change")
	}
	if cfg.Spec.ContextSize != 32768 {
		t.Errorf("ContextSize = %d, want the stored 32768", cfg.Spec.ContextSize)
	}
}
