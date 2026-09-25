package config

import (
	"fmt"
	"strings"
)

// loadEngine parses ENGINE_KIND and derives Engine.URL. v1.1 has no
// user-facing ENGINE_URL env; the URL defaults to deploy/compose's
// service-name + standard engine port. Tests can override the derived
// value via the hidden LLM_INIT_TEST_ENGINE_URL env so they can wire
// httptest fakes without flipping the legacy env name back on (which
// would otherwise hit the v1.0 fail-fast list).
//
// ENGINE_KIND is optional. An empty/unset value selects download-only
// mode: no engine is started or proxied (loadEngineArgs skips, no data
// plane is mounted), and llm-init degrades to a model downloader that
// drops files into the shared cache for a sibling app (e.g. ComfyUI /
// audio) to consume. Cross-field validation rejects ollama:// sources
// in this mode (no daemon to pull into).
func (l *loader) loadEngine() {
	v := strings.TrimSpace(l.g("ENGINE_KIND"))
	if v == "" {
		return
	}
	kind, err := parseEnum("ENGINE_KIND", v, AllEngineKinds)
	if err != nil {
		l.collect(err)
		return
	}
	l.c.Engine.Kind = EngineKind(kind)
	l.c.Engine.URL = deriveEngineURL(l.c.Engine.Kind)
	if raw := strings.TrimSpace(l.g("ENGINE_MAX_CONCURRENCY")); raw != "" {
		v, err := parseInt("ENGINE_MAX_CONCURRENCY", raw, 0)
		switch {
		case err != nil:
			l.collect(err)
		case v <= 0:
			l.collect(fmt.Errorf("ENGINE_MAX_CONCURRENCY: %d must be a positive integer", v))
		default:
			l.c.Engine.MaxConcurrency = v
		}
	}
	if override := strings.TrimSpace(l.g("LLM_INIT_TEST_ENGINE_URL")); override != "" {
		if u, err := parseHTTPURL("LLM_INIT_TEST_ENGINE_URL", override); err == nil {
			l.c.Engine.URL = u
		} else {
			l.collect(err)
		}
	}
}

// Conventional sibling-engine URLs derived from ENGINE_KIND. They match
// the deploy/compose/<kind>.yml service names and upstream default ports.
const (
	defaultOllamaURL    = "http://ollama:11434"
	defaultVLLMURL      = "http://vllm:8000"
	defaultLlamaCppURL  = "http://llamacpp:8081"
	defaultSGLangURL    = "http://sglang:30000"
	defaultEmbedURL     = "http://embed:8080"
	defaultClipEmbedURL = "http://clipembed:8080"
	defaultAudioURL     = "http://audio-engine:8000"
	defaultMusicURL     = "http://music-engine:8001"
	// OCRAdapter (not llamacpp): proxy Ready probes GET /v1/models here.
	defaultOCRAdapterURL = "http://ocradapter:8080"
	defaultRerankURL     = "http://rerank:8080"
	defaultSystemOneURL  = "http://systemone:8000"
)

// deriveEngineURL returns the conventional sibling-engine URL for a
// given Kind. Matches deploy/compose/<kind>.yml service names and
// upstream default ports.
func deriveEngineURL(kind EngineKind) string {
	switch kind {
	case EngineOllama:
		return defaultOllamaURL
	case EngineVLLM:
		return defaultVLLMURL
	case EngineLlamaCpp:
		return defaultLlamaCppURL
	case EngineSGLang:
		return defaultSGLangURL
	case EngineEmbed:
		return defaultEmbedURL
	case EngineClipEmbed:
		return defaultClipEmbedURL
	case EngineAudio:
		return defaultAudioURL
	case EngineOCR:
		return defaultOCRAdapterURL
	case EngineRerank:
		return defaultRerankURL
	case EngineMusic:
		return defaultMusicURL
	case EngineSystemOne:
		return defaultSystemOneURL
	}
	return ""
}

// loadEngineArgs is a no-op retained for Load() step ordering comments.
// ENGINE_ARGS is seeded onto Spec.EngineArgs in loadSpec and parsed into
// Engine.Args via ApplyEngineArgsFromSpec (again after disk reconcile).
// Early fail-fast for illegal non-empty args on embed/audio/ocr still
// happens when loadSpec calls ApplyEngineArgsFromSpec.
func (l *loader) loadEngineArgs() {}
