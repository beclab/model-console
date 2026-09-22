// Package config loads, validates, and exposes the env-driven configuration
// for llm-init.
//
// The package is split across several files for readability; they all
// share package config:
//   - config.go            Config struct + Load() orchestration + loader.
//   - config_enums.go       EngineKind / SourceKind / ModelType enums,
//     defaults, and the allowed-value validation slices.
//   - config_load_engine.go ENGINE_KIND + ENGINE_ARGS loaders.
//   - config_load_model.go  MODEL_NAME / MODEL_SOURCE / spec loaders +
//     the Primary* source accessors.
//   - config_load_runtime.go PORT / RUN_DIR / VERIFY_* / LOG_* loaders.
//   - config_validate.go     cross-field (engine x source kind) checks.
//   - config_error.go        ConfigError/FieldError structured aggregate.
//   - config_redacted.go     the /api/config redaction view.
package config

import (
	"os"
	"time"
)

// Config is the complete, validated runtime configuration.
//
// As of v1.1.0 the user-facing env surface is:
//   - ENGINE_KIND + ENGINE_ARGS (single env, parsed per Kind)
//   - MODEL_NAME
//   - MODEL_MODE (chat | embedding | audio | tts | translate | ocr | rerank) + MODEL_SUPPORTS (CSV of enabled
//     supports_* keys) — seed the model-spec identity
//   - MODEL_SOURCE
//   - MODEL_SOURCE_LOCAL (only for https?://)
//   - HF_ENDPOINT / HF_TOKEN (deployment-level, not per-source)
//   - PORT / RUN_DIR / VERIFY_* / LOG_* / MODEL_SPEC_PATH /
//     UPSTREAM_RESPONSE_HEADER_TIMEOUT / etc.
//   - model-spec.json (persisted at MODEL_SPEC_PATH; disk wins after
//     first boot, editable via PUT /api/model-spec)
//
// The v1.0 llm-init-specific envs (HF_REPO, HF_FILE, MODEL_URL,
// OLLAMA_MODEL, MODEL_DIR, ENGINE_URL, CONTEXT_LENGTH, GGUF_*,
// MODEL_TYPE, MODEL_THINK_SUPPORTED, MODEL_SPEC_JSON, ENGINE_LLAMACPP_*,
// ...) are REMOVED in v1.1.0; setting any one is fail-fast with a
// migration hint (see loadRemovedEnvs in config_load_legacy.go and the
// CHANGELOG migration matrix). Standard third-party envs llm-init
// shares its environment with (HF_HUB_CACHE, OLLAMA_KEEP_ALIVE, ...)
// are NOT claimed and are simply ignored; any other unrecognised env is
// ignored too.
type Config struct {
	Engine Engine
	Model  Model
	// Sources is the model-source list (1+ entries). Index 0 is the
	// deployment's main source; later entries are extras.
	Sources []ModelSource

	Runtime Runtime
	Log     Log

	// ModelName is MODEL_NAME (deployment identity / OpenAI alias). It
	// seeds ModelSpec.Name in the env-synthesised spec; once disk wins
	// (ReconcileModelSpecFile), the on-disk spec.name is authoritative.
	ModelName string
	// HFEndpoint mirrors HF_ENDPOINT (deployment-level HF mirror).
	// Default https://huggingface.co.
	HFEndpoint string
	// HFToken is the deployment-level HF_TOKEN value (may be empty).
	// Internal-only; never serialised to /api/config — HFTokenSet
	// below is what operators see.
	HFToken string
	// HFTokenSet is true iff HF_TOKEN was non-empty at Load time.
	// Exposed via /api/config for operators to confirm token wiring;
	// the token itself is never serialised.
	HFTokenSet bool

	// Spec carries the v2 ProviderModelSpec view loaded from
	// /etc/llm-init/model-spec.json (Required; missing/malformed file
	// is fail-fast). Served verbatim by GET /api/model-spec.
	Spec ModelSpec
}

// Engine identifies the sibling inference engine.
//
// As of v1.1.0:
//   - URL is derived from Kind (no ENGINE_URL env). The default is
//     http://<service-name>:<engine-port> matching deploy/compose
//     conventions; tests override directly on the struct.
//   - Args is the parsed view of the single ENGINE_ARGS env, per
//     ParseEngineArgs. Diagnostics read engine
//     flags (e.g. n_gpu_layers) straight from Args; there are no
//     separate GPU-mirror fields (the v1.0 ENGINE_* mirror envs were
//     removed in v1.1.0).
type Engine struct {
	Kind EngineKind
	URL  string
	Args EngineArgs
	// MaxConcurrency is an optional declaration from
	// ENGINE_MAX_CONCURRENCY for sibling engines which cannot publish
	// GET /api/engine-capacity themselves. Zero means unspecified.
	MaxConcurrency int
}

// Model holds the externally-visible model identity.
//
// ThinkSupported flags whether the deployed model honours the
// reasoning/thinking switch (Qwen-3, DeepSeek-R1, GPT-OSS, ...). It is
// derived from the model card's supports_reasoning rather than guessed
// from the model name, because engines silently 400 when the flag is
// sent to a non-reasoning model.
type Model struct {
	Name           string
	Type           ModelType
	Dir            string
	ThinkSupported bool
}

// Runtime holds process-level options.
//
// The json tags matter: this struct is embedded verbatim in
// ConfigRedacted and therefore serialised by GET /api/config, whose
// other fields are snake_case.
// ConfigRedacted.runtime.
type Runtime struct {
	Port              int    `json:"port"`
	RunDir            string `json:"run_dir"`
	ProgressStatePath string `json:"progress_state_path"`
	// ModelSpecPath is where the v2 model-spec.json is persisted and
	// reloaded. MODEL_SPEC_PATH overrides; default ${RunDir}/model-spec.json.
	// On boot the env-synthesised spec is written here when absent, and
	// an existing file wins (disk is the source of truth after first boot).
	// PUT /api/model-spec rewrites it atomically.
	ModelSpecPath string `json:"model_spec_path"`
	// VerifyLevel is how hard each ensure pass tries to confirm the
	// bytes already on disk before deciding it can skip the download
	// (VERIFY_LEVEL: size or sha256). `remote` is deliberately not
	// accepted here: it reaches the network, and a standing default
	// that does so would give up booting offline.
	VerifyLevel string `json:"verify_level"`
	// VerifyOnDrift is what to do when a remote check finds the
	// upstream now serves different bytes (VERIFY_ON_DRIFT).
	VerifyOnDrift string `json:"verify_on_drift"`
	// MaxConcurrentDownloads bounds the parallel fan-out for HF tree
	// mode (see lifecycle.downloadAll). Default 4 is a safe middle
	// ground for HF Hub rate limits and operator NICs; tune up to 16
	// on big LAN-attached storage, down to 1 on slow shared S3-FUSE.
	// Range [1, 16] is enforced by Load(); 0 means "use default".
	MaxConcurrentDownloads int `json:"max_concurrent_downloads"`

	// EnableXet opts INTO the hf_xet chunked-transfer download backend
	// (HF_ENABLE_XET). Default false = the stable LFS path. hf_xet is faster
	// on large sharded repos but buffers byte-range chunks in RAM and has
	// OOM-killed the download Pod (huggingface_hub#3300), so LFS is the
	// default and xet is opt-in. Read straight from cfg by lifecycle.runHF
	// (like HFToken/HFEndpoint) and plumbed onto hfwrap.Config.EnableXet.
	EnableXet bool `json:"enable_xet"`

	// MaxDownloadBytes is a hard per-URL-download byte cap from
	// MAX_DOWNLOAD_BYTES. 0 (default) means unlimited. Bounds the
	// cache volume when an upstream reports no usable Content-Length;
	// plumbed into fetch.RangeOptions.MaxBytes by lifecycle.
	MaxDownloadBytes int64 `json:"max_download_bytes"`

	// RetryRateLimit is the steady-state token refill rate for POST
	// /api/retry, in tokens per second. Burst is fixed at 5 in code.
	// Default 1.0 (one retry per second steady, 5 burst) — generous
	// enough that operators clicking "Retry" several times during a
	// flap are unaffected, tight enough that a runaway script trying
	// to reset the lifecycle every 10 ms is throttled. 0 disables
	// the limiter entirely (always allow). Pre-v1.0.5 the endpoint
	// had no protection; an attacker spamming POST /api/retry could
	// re-trigger ensure (HF tree walk + manifest.Diff filesystem walk)
	// at line speed.
	RetryRateLimit float64 `json:"retry_rate_limit"`

	// UpstreamResponseHeaderTimeout is how long the data-plane proxy
	// waits for the sibling engine to emit the first HTTP response
	// header (TTFT / prompt prefill). From
	// UPSTREAM_RESPONSE_HEADER_TIMEOUT; unset → 5m. Range [5s, 30m]
	// is enforced by Load(). The body stream itself is unbounded.
	// Marshals as integer nanoseconds (time.Duration's native JSON).
	UpstreamResponseHeaderTimeout time.Duration `json:"upstream_response_header_timeout"`

	// PublicURL is the externally-reachable URL of this llm-init
	// instance, used by the dashboard's Overview tab to render an
	// "API URL" panel after the model is ready. On Olares the Helm
	// chart sets this via APP_URL=https://{{ .Values.domain.<entrance> }}
	// (see ollamaqwen359bv2/.../api.yaml + vllmqwen3508b/.../llm-init.yaml);
	// in plain local-dev runs the env is unset and PublicURL stays
	// empty, in which case the front-end falls back to
	// window.location.origin so the panel still works for someone
	// hitting :8090 directly.
	//
	// No validation beyond "if set, must parse as an http(s) URL with
	// a host" — the value is informational (display only) and not
	// dispatched anywhere internally, so a malformed APP_URL must not
	// take down the entire control plane. Invalid values are logged
	// at WARN by Load() and silently dropped.
	PublicURL string `json:"public_url"`
}

// ResponseHeaderTimeout returns the wait for the engine's first HTTP
// response header. A zero value (tests constructing Config by hand,
// or a field the loader never filled) falls back to the 5m default
// rather than Go's "0 means no timeout".
func (r Runtime) ResponseHeaderTimeout() time.Duration {
	if r.UpstreamResponseHeaderTimeout > 0 {
		return r.UpstreamResponseHeaderTimeout
	}
	return DefaultUpstreamResponseHeaderTimeout
}

// Log configures structured logging. Serialised by GET /api/config via
// ConfigRedacted, hence the snake_case json tags.
type Log struct {
	Level  string `json:"level"`
	Format string `json:"format"`
	// UpstreamTrace controls per-request audit logging of HTTP calls
	// llm-init makes to its upstream engine (currently used by the
	// Ollama adapter). Three values:
	//   - "off"   (default): success silent; failures still ERROR
	//     with minimal context (path + total_ms + err). Zero
	//     httptrace overhead on the hot path.
	//   - "info" : on every call, one INFO line at exit carrying
	//     status, total_ms, connection-reuse info, TTFB. Useful for
	//     diagnosing first-request-stalls without going full debug.
	//   - "debug": entry + exit at DEBUG with all httptrace phase
	//     timings (got_conn, wrote_request_ms, ttfb_ms, idle_time_ms,
	//     remote addr). Verbose; expect ~2 lines per HTTP call.
	// Errors are ALWAYS surfaced at ERROR regardless of this setting.
	UpstreamTrace string `json:"upstream_trace"`
}

// LoadFromEnv loads configuration from the process environment.
func LoadFromEnv() (Config, error) {
	return Load(os.Getenv)
}

// Load builds and validates a Config using the supplied env lookup function.
//
// v1.1.0 ordering (loadRemovedEnvs runs first to surface migration
// hints for retired v1.0 envs before anything else):
//  1. loadEngine - ENGINE_KIND + derived URL.
//  2. loadEngineArgs - reserved (engine args now seed on the model card).
//  3. loadModel - MODEL_NAME (Type / ThinkSupported come from spec).
//  4. loadSources - MODEL_SOURCE into Sources.
//  5. loadHFDeploy - HF_ENDPOINT + HF_TOKEN (deployment-level; only
//     read when an hf:// source is present).
//  6. loadRuntime - PORT / RUN_DIR / verify / rate-limit knobs.
//  7. loadLog - LOG_LEVEL / LOG_FORMAT / UPSTREAM_TRACE_LEVEL.
//  8. loadSpec - env-synthesised spec (MODEL_MODE / MODEL_SUPPORTS +
//     MODEL_NAME + ENGINE_ARGS → Spec.EngineArgs), validated and parsed
//     into Engine.Args; disk reconcile happens post-Load in main.
//
// Unrecognised envs are ignored. Cross-field validation (engine x source
// kind compatibility) runs after step 8, only when no loader collected an
// error.
func Load(g Getenv) (Config, error) {
	l := &loader{g: g}
	l.loadRemovedEnvs()
	l.loadEngine()
	l.loadEngineArgs()
	l.loadModel()
	l.loadSources()
	l.loadHFDeploy()
	l.loadRuntime()
	l.loadLog()
	l.loadSpec()

	if len(l.errs) == 0 {
		if err := l.c.validateCrossField(); err != nil {
			l.errs = append(l.errs, err)
		}
	}

	if len(l.errs) > 0 {
		return Config{}, newConfigError(l.errs)
	}
	return l.c, nil
}

// loader is a transient builder for a single Load() call. Holding the env
// getter, the partially-built Config, and the error accumulator on one
// receiver lets each loadX method be a short, single-section parser instead
// of one giant function -- which both reads better and keeps gocyclo on
// the per-section methods (each <10) instead of on Load itself (49 in
// the pre-split form, see commit 40153f1).
type loader struct {
	g    Getenv
	c    Config
	errs []error
}

func (l *loader) collect(err error) {
	if err != nil {
		l.errs = append(l.errs, err)
	}
}
