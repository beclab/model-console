package config

import "testing"

// TestDeriveEngineURL locks the sibling-engine URL contract: the derived
// host:port must match deploy/compose/<kind>.yml service names + the port
// each engine wrapper actually binds. llamacpp listens on 8081 (8080 is
// llm-init's own control plane), so a regression to :8080 would make the
// proxy adapter probe the wrong port and never reach phase=ready.
func TestDeriveEngineURL(t *testing.T) {
	cases := map[EngineKind]string{
		EngineOllama:    "http://ollama:11434",
		EngineVLLM:      "http://vllm:8000",
		EngineLlamaCpp:  "http://llamacpp:8081",
		EngineSGLang:    "http://sglang:30000",
		EngineEmbed:     "http://embed:8080",
		EngineClipEmbed: "http://clipembed:8080",
		EngineOCR:       "http://ocradapter:8080",
		EngineRerank:    "http://rerank:8080",
		EngineMusic:     "http://music-engine:8001",
	}
	for kind, want := range cases {
		if got := deriveEngineURL(kind); got != want {
			t.Errorf("deriveEngineURL(%q) = %q, want %q", kind, got, want)
		}
	}
	if got := deriveEngineURL("nonexistent"); got != "" {
		t.Errorf("deriveEngineURL(unknown) = %q, want empty", got)
	}
}
