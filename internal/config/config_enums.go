package config

import "time"

// EngineKind enumerates the supported inference engines.
type EngineKind string

// EngineKind values. The set is closed; AllEngineKinds is the canonical
// slice form used for env validation.
const (
	EngineOllama    EngineKind = "ollama"
	EngineVLLM      EngineKind = "vllm"
	EngineLlamaCpp  EngineKind = "llamacpp"
	EngineSGLang    EngineKind = "sglang"
	EngineEmbed     EngineKind = "embed"
	EngineClipEmbed EngineKind = "clipembed"
	EngineAudio     EngineKind = "audio"
	EngineOCR       EngineKind = "ocr"
	EngineRerank    EngineKind = "rerank"
	EngineMusic     EngineKind = "music"
	EngineSystemOne EngineKind = "systemone"
)

// AllEngineKinds lists every supported engine; used for env validation.
var AllEngineKinds = []string{
	string(EngineOllama),
	string(EngineVLLM),
	string(EngineLlamaCpp),
	string(EngineSGLang),
	string(EngineEmbed),
	string(EngineClipEmbed),
	string(EngineAudio),
	string(EngineOCR),
	string(EngineRerank),
	string(EngineMusic),
	string(EngineSystemOne),
}

// SourceKind enumerates how llm-init obtains the model bytes. The four
// values come from modelsource.go (KindHF / KindOllama / KindOllamaURL /
// KindURL) and match the public closed enum
// ModelSource.kind. The pre-v1.1.0 SourceOllamaName / SourceHF /
// SourceURL constants were retired alongside the Source / Inference /
// GGUF compat shims.
type SourceKind string

// ModelType is the MODEL_MODE / model-spec mode value.
//
// Which of the /v1 paths a mode actually serves lives in
// internal/moderoutes, and both GET /api/endpoints and
// dataplane.ModeGuard read it from there; a mode with no declared set
// forwards everything. This enum is the vocabulary, not the route map, so
// the per-mode rules are not restated here — one of them was, and it went
// on claiming /v1/* was an untrimmed catch-all for every mode long after
// the guard shipped.
type ModelType string

// ModelType values, with the endpoint family each one is about:
// ModelChat -> /v1/chat/completions + /v1/completions;
// ModelEmbedding -> /v1/embeddings; ModelRerank -> /v1/rerank;
// ModelTranslate -> MTran-compatible root
// routes; ModelAudio / ModelTTS / ModelOCR gate engine-owned or proxied
// routes.
//
// ModelTTS is speech synthesis and ModelAudio is everything else the audio
// engine does. They are separate because a synthesis application answers a
// voice surface no recognition engine has — ElevenLabs-shaped /v1/voices,
// /v1/text-to-voice and /v1/text-to-speech/{voice_id} alongside the OpenAI
// /v1/audio/speech — and Router routes on mode. Both still run
// ENGINE_KIND=audio: the sibling container is the same audio-engine image,
// and the capability is carried by MODEL_SUPPORTS.
const (
	ModelChat            ModelType = "chat"
	ModelEmbedding       ModelType = "embedding"
	ModelAudio           ModelType = "audio"
	ModelTTS             ModelType = "tts"
	ModelOCR             ModelType = "ocr"
	ModelRerank          ModelType = "rerank"
	ModelTranslate       ModelType = "translate"
	ModelMusicGeneration ModelType = "music_generation"
	ModelSystemOne       ModelType = "system_one"
)

// Defaults applied when the corresponding env is unset.
const (
	defaultModelType  = string(ModelChat)
	defaultHFEndpoint = "https://huggingface.co"
	// defaultHFHomeBase is the shared root mount for HF downloads. Each
	// model lands under "<HomeBase>/<repo>/" so two charts referencing
	// the same repo+revision share the same on-disk directory. Chart
	// templates can override via HF_HOME_BASE; the value must be an
	// absolute path.
	defaultHFHomeBase       = "/Huggingface"
	defaultPort             = 8080
	defaultRunDir           = "/run/llm-init"
	defaultVerifyLevel      = "size"
	defaultVerifyOnDrift    = VerifyOnDriftReport
	defaultMaxConcurrentDLs = 4
	defaultRetryRateLimit   = 1.0
	// DefaultUpstreamResponseHeaderTimeout is how long the data-plane
	// proxy waits for the engine's first HTTP response header (TTFT /
	// prefill) when UPSTREAM_RESPONSE_HEADER_TIMEOUT is unset. 5m
	// covers large-GGUF first-token latency (e.g. Qwen3-27B) that
	// used to miss the previous 60s hard cap.
	DefaultUpstreamResponseHeaderTimeout = 5 * time.Minute
	upstreamHeaderTimeoutMin             = 5 * time.Second
	upstreamHeaderTimeoutMax             = 30 * time.Minute
	// defaultLogLevel is intentionally "debug" while llm-init is still
	// stabilising on the huggingface_hub / sentinel pipeline: deploy
	// failures on Olares nodes are easier to triage from a single Pod
	// log dump than from on-the-fly LOG_LEVEL overrides. Charts that
	// want quieter logs can set LOG_LEVEL explicitly to "info"/"warn".
	defaultLogLevel      = "debug"
	defaultLogFormat     = "json"
	defaultUpstreamTrace = "off"
)

// LOG_LEVEL enum values, mirroring slog.Level. Pulled out of the
// allowedLogLevel slice literal so production + tests can reference
// the same names (rather than re-stating the strings inline) and so
// the goconst v2 sweep stays clean. Exported because cmd/llm-init's
// newLogger references LogFormatText to pick the slog handler shape;
// keeping the constant set close to allowedLogLevel / allowedLogFormat
// is more discoverable than duplicating it cross-package.
const (
	LogLevelDebug = "debug"
	LogLevelInfo  = "info"
	LogLevelWarn  = "warn"
	LogLevelError = "error"

	LogFormatJSON = "json"
	LogFormatText = "text"

	// UPSTREAM_TRACE_LEVEL enum values. See Log.UpstreamTrace for
	// behavioural contract. Kept in this file alongside LogLevel /
	// LogFormat constants so the env-vocabulary surface lives in one
	// place; the ollama adapter mirrors these as TraceLevel string
	// constants to avoid a config package import inside the data
	// plane.
	LogUpstreamTraceOff   = "off"
	LogUpstreamTraceInfo  = "info"
	LogUpstreamTraceDebug = "debug"

	// VERIFY_ON_DRIFT enum values, consulted only when a pass was asked
	// to check the upstream (`POST /api/retry?level=remote`) and found
	// that the bytes it serves today differ from the ones on disk.
	//
	// The default reports rather than acts because drift is usually a
	// moving branch, and re-downloading it would swap the weights under
	// a model that is serving requests. An operator who does want to
	// track upstream sets follow.
	VerifyOnDriftReport = "report"
	VerifyOnDriftFollow = "follow"
)

// Allowed enumerations for env validation. The string literals match
// the named constants above so a one-line edit to the constant
// rotates the entire enum.
var (
	allowedModelType     = []string{string(ModelChat), string(ModelEmbedding), string(ModelAudio), string(ModelTTS), string(ModelOCR), string(ModelRerank), string(ModelTranslate), string(ModelMusicGeneration), string(ModelSystemOne)}
	allowedVerifyLevel   = []string{"size", "sha256"}
	allowedVerifyOnDrift = []string{VerifyOnDriftReport, VerifyOnDriftFollow}
	allowedLogLevel      = []string{LogLevelDebug, LogLevelInfo, LogLevelWarn, LogLevelError}
	allowedLogFormat     = []string{LogFormatJSON, LogFormatText}
	allowedUpstreamTrace = []string{LogUpstreamTraceOff, LogUpstreamTraceInfo, LogUpstreamTraceDebug}
)
