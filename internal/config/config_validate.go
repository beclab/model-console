package config

import "fmt"

// validateCrossField enforces engine x source-kind compatibility for
// the v1.1 source kinds.
func (c Config) validateCrossField() error {
	var errs []error

	main := mainSource(c.Sources)
	if main == nil {
		return nil
	}

	// Download-only mode (empty ENGINE_KIND): there is no engine to
	// serve, so only file-download sources make sense. ollama:// (a
	// Library tag or URL overload) needs a running ollama daemon to
	// pull into and cannot work here.
	if c.Engine.Kind == "" {
		for i := range c.Sources {
			if c.Sources[i].ViaOllama() {
				errs = append(errs, fmt.Errorf(
					"download-only mode (empty ENGINE_KIND) cannot use an ollama:// source (no ollama daemon to pull into); set ENGINE_KIND=ollama or switch to hf:// / https://"))
				break
			}
		}
		if len(errs) > 0 {
			return joinErrors(errs)
		}
		return nil
	}

	switch main.Kind {
	case KindOllama:
		if c.Engine.Kind != EngineOllama {
			errs = append(errs, fmt.Errorf(
				"engine %q does not accept ollama:// Library tag; switch ENGINE_KIND=ollama or use hf:// / https?://",
				c.Engine.Kind))
		}
	case KindOllamaURL:
		if c.Engine.Kind != EngineOllama {
			errs = append(errs, fmt.Errorf(
				"engine %q does not accept ollama:// URL overload; switch ENGINE_KIND=ollama or use https?:// directly",
				c.Engine.Kind))
		}
	case KindHF:
		// llama.cpp consumes a single GGUF file: --include is
		// required (exactly one). Ollama + hf:// is also single-file
		// (we synthesise FROM @sha256 in batch 3) so we apply the
		// same constraint there.
		if (c.Engine.Kind == EngineLlamaCpp || c.Engine.Kind == EngineOllama) &&
			len(main.HFInclude) != 1 {
			errs = append(errs, fmt.Errorf(
				"engine %q + hf:// requires exactly one --include <file.gguf> on MODEL_SOURCE; got %d",
				c.Engine.Kind, len(main.HFInclude)))
		}
	}

	if len(errs) > 0 {
		return joinErrors(errs)
	}
	return nil
}
