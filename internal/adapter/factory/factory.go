// Package factory wires concrete Adapter implementations behind a single
// New() function. It lives in a sub-package because Go forbids
// internal/adapter from importing its own sub-packages: ollama and proxy
// already import adapter.ReadyState, so the factory must sit one level
// below to avoid an import cycle.
//
// Callers (cmd/llm-init/main.go and lifecycle.New) depend only on this
// package, never on the per-engine sub-packages directly.
package factory

import (
	"fmt"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/adapter/nullengine"
	"github.com/llm-init/llm-init/internal/adapter/ollama"
	"github.com/llm-init/llm-init/internal/adapter/proxy"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/obs"
)

// New returns the Adapter for cfg.Engine.Kind. metrics may be nil for
// callers (mostly tests) that don't have a metrics registry to plumb;
// the adapters degrade gracefully (counters / histograms become
// no-ops) when nil is passed.
//
// Ollama gets its bespoke OpenAI<->Ollama translator; the three OpenAI-
// native engines (vLLM, llama.cpp, SGLang) all share the proxy.Adapter
// implementation parameterised by Kind. An empty Kind selects the
// download-only nullengine adapter (no engine to serve). Anything else
// is rejected here rather than at first request so misconfiguration
// fails fast at boot.
func New(cfg config.Config, metrics *obs.Metrics) (adapter.Adapter, error) {
	switch cfg.Engine.Kind {
	case "":
		return nullengine.New(), nil
	case config.EngineOllama:
		return ollama.NewAdapter(cfg), nil
	case config.EngineVLLM, config.EngineLlamaCpp, config.EngineSGLang, config.EngineEmbed, config.EngineClipEmbed, config.EngineAudio, config.EngineOCR, config.EngineRerank, config.EngineMusic, config.EngineSystemOne:
		a, err := proxy.NewAdapter(cfg, cfg.Engine.Kind)
		if err != nil {
			return nil, err
		}
		a.Metrics = metrics
		return a, nil
	default:
		return nil, fmt.Errorf("adapter/factory: unknown engine kind %q", cfg.Engine.Kind)
	}
}
