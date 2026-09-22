package config

import (
	"fmt"
	"strconv"
	"strings"
)

func (l *loader) loadRuntime() {
	if v, err := parseIntInRange("PORT",
		stringDefault(l.g, "PORT", ""), defaultPort, 1, 65535); err != nil {
		l.collect(err)
	} else {
		l.c.Runtime.Port = v
	}
	if v, err := parseAbsPath("RUN_DIR",
		stringDefault(l.g, "RUN_DIR", defaultRunDir)); err != nil {
		l.collect(err)
	} else {
		l.c.Runtime.RunDir = v
	}
	// PROGRESS_STATE_PATH default lives under RUN_DIR rather than
	// MODEL_DIR. Pre-v1.0.10 we wrote ${MODEL_DIR}/.progress_state.json
	// which had two operator-visible smells:
	//
	//   1. The state file mixed in with model artefacts. ``ls`` of the
	//      HF repo dir showed both *.safetensors and a hidden runtime
	//      file owned by llm-init, which confused operators sharing
	//      the model dir with other tooling (e.g. inspecting the
	//      download with ``du``).
	//   2. With the new HF_HOME_BASE layout where one repo is shared
	//      across multiple charts (vllm + sglang both mount
	//      /Huggingface/Qwen/Qwen3.5-0.8B), two llm-init Pods raced
	//      on the same MODEL_DIR/.progress_state.json. RUN_DIR is
	//      per-app by convention (chart sets it to
	//      /Huggingface/.llm-init/<chart-name>) so each Pod keeps its
	//      own audit trail.
	//
	// Operators can still override with PROGRESS_STATE_PATH if they
	// have a specific persistence policy in mind. RUN_DIR is
	// guaranteed populated above, so the default is always writable
	// (modulo permissions, which is operator-territory).
	if v := strings.TrimSpace(l.g("PROGRESS_STATE_PATH")); v != "" {
		if v, err := parseAbsPath("PROGRESS_STATE_PATH", v); err != nil {
			l.collect(err)
		} else {
			l.c.Runtime.ProgressStatePath = v
		}
	} else if l.c.Runtime.RunDir != "" {
		l.c.Runtime.ProgressStatePath = l.c.Runtime.RunDir + "/progress_state.json"
	}
	// MODEL_SPEC_PATH: where the v2 model-spec.json is persisted and
	// reloaded. Default lives under RUN_DIR (writable, per-app) so the
	// env-synthesised spec can be seeded on first boot and the dashboard
	// editor can rewrite it. Operators can point it at a pre-mounted file.
	if v := strings.TrimSpace(l.g("MODEL_SPEC_PATH")); v != "" {
		if v, err := parseAbsPath("MODEL_SPEC_PATH", v); err != nil {
			l.collect(err)
		} else {
			l.c.Runtime.ModelSpecPath = v
		}
	} else if l.c.Runtime.RunDir != "" {
		l.c.Runtime.ModelSpecPath = l.c.Runtime.RunDir + "/model-spec.json"
	}
	if v, err := parseEnum("VERIFY_LEVEL",
		stringDefault(l.g, "VERIFY_LEVEL", defaultVerifyLevel),
		allowedVerifyLevel); err != nil {
		l.collect(err)
	} else {
		l.c.Runtime.VerifyLevel = v
	}
	if v, err := parseEnum("VERIFY_ON_DRIFT",
		stringDefault(l.g, "VERIFY_ON_DRIFT", defaultVerifyOnDrift),
		allowedVerifyOnDrift); err != nil {
		l.collect(err)
	} else {
		l.c.Runtime.VerifyOnDrift = v
	}
	// APP_URL: optional externally-reachable URL for the dashboard's
	// API panel. Sourced by Olares charts from .Values.domain.<entrance>
	// (see vllmqwen3508b/.../llm-init.yaml). Validation is best-effort:
	// invalid values are dropped (PublicURL left empty), not collected
	// as a hard error, because this is purely a UI affordance — the
	// control plane must keep starting even if the operator set
	// APP_URL=garbage. The dashboard already has a fallback to
	// window.location.origin when PublicURL is empty.
	if v := strings.TrimSpace(l.g("APP_URL")); v != "" {
		if parsed, err := parseHTTPURL("APP_URL", v); err == nil {
			l.c.Runtime.PublicURL = parsed
		}
	}
	l.loadRuntimeLimits()
}

// loadRuntimeLimits handles the numeric guardrails: download
// concurrency and retry rate. (Pre-v1.1.0 it also covered the SSE
// subscriber cap; SSE was retired in favour of dashboard polling.)
func (l *loader) loadRuntimeLimits() {
	// MAX_CONCURRENT_DOWNLOADS: clamp to [1, 16]. Empty / 0 -> default
	// 4. Range upper bound is intentionally loose; the realistic
	// failure mode is HF Hub rate-limiting at 8-16 parallel connections,
	// so values past 16 are almost always counterproductive. The
	// `lifecycle.Options.MaxConcurrentDownloads` field is wired from
	// this in cmd/llm-init/main.go.
	if v, err := parseIntInRange("MAX_CONCURRENT_DOWNLOADS",
		strings.TrimSpace(l.g("MAX_CONCURRENT_DOWNLOADS")),
		defaultMaxConcurrentDLs, 1, 16); err != nil {
		l.collect(err)
	} else {
		l.c.Runtime.MaxConcurrentDownloads = v
	}
	// HF_ENABLE_XET opts INTO the hf_xet chunked-transfer backend. Empty /
	// unset -> false: the stable LFS download path (see
	// hfwrap.Config.EnableXet). hf_xet is faster on large sharded repos but
	// buffers byte-range chunks in RAM and has OOM-killed the download Pod
	// (huggingface_hub#3300), so LFS is the default and xet is opt-in.
	if raw := strings.TrimSpace(l.g("HF_ENABLE_XET")); raw != "" {
		if b, err := strconv.ParseBool(raw); err != nil {
			l.collect(fmt.Errorf("HF_ENABLE_XET: %q must be a boolean (true/false/1/0)", raw))
		} else {
			l.c.Runtime.EnableXet = b
		}
	}
	// RETRY_RATE_LIMIT: float tokens/sec for the POST /api/retry token
	// bucket. 0 disables, negatives reject. Burst is fixed (5) in the
	// limiter; only the steady-state rate is tunable. Default 1.0.
	// Ceiling 1000 is sanity-only -- nobody should be running 1k retries
	// per second through this endpoint, and an unbounded float here
	// would risk operator typos like "10000" silently passing.
	if v, err := parseFloat("RETRY_RATE_LIMIT",
		strings.TrimSpace(l.g("RETRY_RATE_LIMIT")),
		defaultRetryRateLimit); err != nil {
		l.collect(err)
	} else if v < 0 || v > 1000 {
		l.collect(fmt.Errorf("RETRY_RATE_LIMIT: %v out of range [0, 1000]", v))
	} else {
		l.c.Runtime.RetryRateLimit = v
	}
	// MAX_DOWNLOAD_BYTES: hard per-URL-download byte cap. Empty / 0 ->
	// unlimited (default). Bounds cache-volume usage when an upstream
	// reports no usable Content-Length; wired into
	// fetch.RangeOptions.MaxBytes by lifecycle.
	if raw := strings.TrimSpace(l.g("MAX_DOWNLOAD_BYTES")); raw != "" {
		if n, err := strconv.ParseInt(raw, 10, 64); err != nil || n < 0 {
			l.collect(fmt.Errorf("MAX_DOWNLOAD_BYTES: %q must be a non-negative integer in bytes (0 = unlimited)", raw))
		} else {
			l.c.Runtime.MaxDownloadBytes = n
		}
	}
	// UPSTREAM_RESPONSE_HEADER_TIMEOUT: wait for the engine's first
	// HTTP response header on proxied chat/messages. Empty → 5m.
	// Range [5s, 30m] so a missing unit ("120") or a day-long typo
	// fail-fast instead of hanging a request or bouncing large GGUFs.
	if v, err := parseDuration("UPSTREAM_RESPONSE_HEADER_TIMEOUT",
		strings.TrimSpace(l.g("UPSTREAM_RESPONSE_HEADER_TIMEOUT")),
		DefaultUpstreamResponseHeaderTimeout); err != nil {
		l.collect(err)
	} else if v < upstreamHeaderTimeoutMin || v > upstreamHeaderTimeoutMax {
		l.collect(fmt.Errorf("UPSTREAM_RESPONSE_HEADER_TIMEOUT: %v out of range [%v, %v]",
			v, upstreamHeaderTimeoutMin, upstreamHeaderTimeoutMax))
	} else {
		l.c.Runtime.UpstreamResponseHeaderTimeout = v
	}
}

func (l *loader) loadLog() {
	if v, err := parseEnum("LOG_LEVEL",
		stringDefault(l.g, "LOG_LEVEL", defaultLogLevel),
		allowedLogLevel); err != nil {
		l.collect(err)
	} else {
		l.c.Log.Level = v
	}
	if v, err := parseEnum("LOG_FORMAT",
		stringDefault(l.g, "LOG_FORMAT", defaultLogFormat),
		allowedLogFormat); err != nil {
		l.collect(err)
	} else {
		l.c.Log.Format = v
	}
	if v, err := parseEnum("UPSTREAM_TRACE_LEVEL",
		stringDefault(l.g, "UPSTREAM_TRACE_LEVEL", defaultUpstreamTrace),
		allowedUpstreamTrace); err != nil {
		l.collect(err)
	} else {
		l.c.Log.UpstreamTrace = v
	}
}
