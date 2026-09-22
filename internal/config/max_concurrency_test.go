package config

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestDeriveMaxConcurrency_PerEngineFlag(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		kind EngineKind
		raw  string
		want int
		ok   bool
	}{
		{"llamacpp short flag", EngineLlamaCpp, "-c 104448 -ngl all -np 1", 1, true},
		{"llamacpp long flag", EngineLlamaCpp, "--parallel 4", 4, true},
		{"llamacpp env form", EngineLlamaCpp, "LLAMA_ARG_PARALLEL=8", 8, true},
		{"vllm", EngineVLLM, "--max-num-seqs 64 --dtype bfloat16", 64, true},
		{"vllm eq form", EngineVLLM, "--max-num-seqs=256", 256, true},
		{"sglang", EngineSGLang, "--max-running-requests 32 --tp 1", 32, true},
		{"ollama", EngineOllama, "OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=30m", 2, true},

		// Silence is the engine's own default, and for these three that
		// default is unknowable from here: two read the hardware and one
		// recomputes over the operator's flag. Guessing would be worse
		// than saying nothing, because the point of the field is to be
		// trustworthy.
		{"vllm no seqs flag", EngineVLLM, "--max-model-len 8192", 0, false},
		{"sglang no running-requests flag", EngineSGLang, "--context-length 8192", 0, false},
		{"ollama no parallel flag", EngineOllama, "OLLAMA_NUM_CTX=8192", 0, false},
		// The queue is not the batch. max_queue bounds how many may
		// wait, which says nothing about how many are being served.
		{"ollama max_queue is not the batch", EngineOllama, "OLLAMA_MAX_QUEUE=512", 0, false},

		// llamacpp is the one engine whose silence is still a number: the
		// server assigns four slots to an absent -np, on any hardware.
		// Reporting nothing here is what makes four served callers look
		// like one slow model.
		{"llamacpp no parallel flag is four slots", EngineLlamaCpp, "-c 8192 -ngl all", 4, true},
		{"llamacpp empty args", EngineLlamaCpp, "", 4, true},
		{"llamacpp explicit auto", EngineLlamaCpp, "-c 8192 -np -1", 4, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			args, err := ParseEngineArgs(tc.kind, tc.raw)
			if err != nil {
				t.Fatalf("ParseEngineArgs: %v", err)
			}
			got, ok := DeriveMaxConcurrency(tc.kind, args)
			if got != tc.want || ok != tc.ok {
				t.Errorf("DeriveMaxConcurrency = (%d, %v), want (%d, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

// The engines whose ENGINE_ARGS belong to a sibling container carry no
// launch flags here, so nothing can be derived for them.
func TestDeriveMaxConcurrency_NonLLMEnginesDeriveNothing(t *testing.T) {
	t.Parallel()
	for _, kind := range []EngineKind{EngineEmbed, EngineClipEmbed, EngineAudio, EngineOCR, EngineRerank, ""} {
		args, err := ParseEngineArgs(kind, "")
		if err != nil && kind != "" {
			t.Fatalf("kind=%s: %v", kind, err)
		}
		if got, ok := DeriveMaxConcurrency(kind, args); ok {
			t.Errorf("kind=%s: got (%d, true), want no derivation", kind, got)
		}
	}
}

// The width is derived, never stored: a card carrying a field an older
// build does not know is a boot failure for that build, and the card on
// the shared cache PVC outlives the app that wrote it. Normalising a
// card whose flags state a width must therefore leave the card's own
// bytes alone.
func TestNormalizeSpecForEngine_DoesNotStoreConcurrencyOnTheCard(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Engine: Engine{Kind: EngineLlamaCpp},
		Spec:   ModelSpec{Name: "m", Mode: "chat", EngineArgs: "-c 65536 -np 2"},
	}
	if _, err := NormalizeSpecForEngine(cfg); err != nil {
		t.Fatalf("NormalizeSpecForEngine: %v", err)
	}
	if cfg.Spec.ContextSize != 32768 {
		t.Errorf("ContextSize = %d, want 32768 (65536 split across 2 slots)", cfg.Spec.ContextSize)
	}
	raw, err := json.Marshal(cfg.Spec)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(raw, []byte("max_concurrency")) {
		t.Errorf("card carries max_concurrency: %s", raw)
	}
	if got, ok := DeriveMaxConcurrency(cfg.Engine.Kind, cfg.Engine.Args); !ok || got != 2 {
		t.Errorf("DeriveMaxConcurrency = (%d, %v), want (2, true)", got, ok)
	}
}
