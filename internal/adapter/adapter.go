// Package adapter is the boundary between llm-init and the sibling
// inference engine. Each engine kind (Ollama / vLLM / llama.cpp / SGLang)
// supplies an Adapter implementation; lifecycle drives them through a
// fixed surface — wait for the engine to come up, optionally pull or
// register the model, expose a /v1/* HTTP handler — and the data plane
// mounts that handler.
//
// The interface is intentionally narrow: anything engine-specific (Ollama
// HTTP translation, reverse-proxy model rewrite) lives in subpackages,
// never leaks into lifecycle.
package adapter

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

// ReadyState is what one probe of the engine learned.
type ReadyState struct {
	Alive       bool
	ModelExists bool
	// ModelBytes is the total on-disk size of the model as reported
	// by the engine, used to backfill the download gauge on a cache
	// hit (the daemon owns the bytes, so nothing streamed). 0 means
	// unknown; only the Ollama adapter populates it (from /api/tags
	// size). File-based sources size their paths directly instead.
	ModelBytes int64
}

// GPUResidencyMode is the closed enum returned by EngineNativeStats's
// GPU.Mode field. Any string outside this set is a programming error.
const (
	GPUModeFull    = "full"     // model is 100% on GPU
	GPUModePartial = "partial"  // CPU+GPU split (cpu_offload_gb > 0 or partial -ngl)
	GPUModeCPUOnly = "cpu_only" // model is wholly resident on CPU/RAM
	GPUModeUnknown = "unknown"  // engine cannot tell us (fallback)
)

// GPUResidencyHints is the engine-native answer to "is the model
// 100% on GPU?". Mode is always populated; the optional pointer
// fields surface the auxiliary numbers the engine exposed (so diag
// can render "65% on GPU" or "32 of 80 layers" when available).
type GPUResidencyHints struct {
	// Mode is one of GPUModeFull / GPUModePartial / GPUModeCPUOnly
	// / GPUModeUnknown.
	Mode string
	// VRAMBytes is the bytes of model weight currently resident on
	// the device. Populated by Ollama's /api/ps (size_vram).
	VRAMBytes *int64
	// ModelBytes is the total model weight size. Populated by
	// Ollama's /api/ps (size).
	ModelBytes *int64
	// GPULayers / TotalLayers come from llama.cpp -ngl + GGUF
	// metadata when both are known. Currently nil for every engine
	// because llama.cpp's runtime endpoints do not expose either
	// field (verified against /props, /slots, /metrics; see plan
	// §9). Reserved for a future llama.cpp upstream that adds them.
	GPULayers   *int
	TotalLayers *int
	// CPUOffloadGB comes from the engine's own introspection:
	// vLLM /metrics (cache_config_info{cpu_offload_gb}) or SGLang
	// /server_info. 0 means "no CPU offload"; >0 means partial GPU.
	CPUOffloadGB *int
	// KVCacheUsagePerc is the fraction (0..1) of GPU KV-cache blocks
	// currently in use, read from the engine's Prometheus /metrics:
	// vllm:kv_cache_usage_perc, sglang:kv_cache_usage_perc,
	// llamacpp:kv_cache_usage_ratio. nil when /metrics is
	// unavailable or the metric is absent. This is an activity
	// signal (0 when idle even if the model is resident), not a
	// residency measurement.
	KVCacheUsagePerc *float64
	// GPUMemoryUtilization is the fraction (0..1) of GPU memory the
	// engine was configured to reserve: vLLM
	// cache_config_info{gpu_memory_utilization} or SGLang
	// mem_fraction_static. nil for Ollama / llama.cpp.
	GPUMemoryUtilization *float64
}

// NativeStats is the engine-native snapshot /api/diag/gpu reads.
//
// Returning (NativeStats{GPU: {Mode: "unknown"}}, nil) is a valid
// answer when the engine cannot tell us the GPU residency: callers
// translate it into a Warning + gpu.mode="unknown" without surfacing
// it as an error to the operator.
//
// Returning a non-nil error means "I tried to introspect the engine
// and the engine itself was unreachable / returned a bad response";
// callers translate this into 503 + a structured error envelope so
// the dashboard can distinguish "engine down" from "engine up but
// cannot answer" (the unknown path).
type NativeStats struct {
	// Source identifies provenance of the GPU+Payload data, e.g.
	// "ollama_ps", "sglang_server_info", "vllm_metrics",
	// "llamacpp_fit_off_alive", "unavailable".
	Source string
	// Payload is the raw engine response (or env-mirror dump) used
	// for /api/diag/gpu's engine_native passthrough field. May be
	// nil when no useful payload exists.
	Payload json.RawMessage
	// GPU is the translated GPU residency state.
	GPU GPUResidencyHints
	// Warnings are operator-actionable strings the diag handler
	// appends to its Warnings array (e.g. "compose contract
	// violated: ENGINE_LLAMACPP_FIT=on, expected off").
	Warnings []string
}

// Adapter is the surface lifecycle and the data plane depend on. All
// methods must be safe for concurrent use; lifecycle drives the engine
// from one goroutine while the data plane invokes OpenAIHandler-built
// http.Handlers from many.
//
// Installing a model is not part of it — see ModelInstaller, which one
// engine implements and the rest have no use for.
type Adapter interface {
	// Kind returns the engine flavour this adapter targets. Equals
	// cfg.Engine.Kind for any non-test wiring.
	Kind() config.EngineKind

	// WaitAlive blocks until the engine accepts requests. Returns
	// ctx.Err() on cancellation. Lifecycle calls this once per Run
	// — for ollama (AliveBeforeBoot()==true) before the first ensure,
	// for proxy adapters (AliveBeforeBoot()==false) after the first
	// successful ensure has written the sentinel and the engine's
	// wrapper has had a chance to exec the engine binary.
	WaitAlive(ctx context.Context) error

	// AliveBeforeBoot reports whether the engine accepts requests
	// independent of lifecycle work. Returning true means WaitAlive
	// is meaningful BEFORE the first ensure pass: the engine (e.g.
	// an Ollama daemon) is already up because docker-compose
	// depends_on ordering brought it up first. Returning false means
	// the engine binary is gated on a sentinel that lifecycle.ensure
	// writes; lifecycle therefore runs ensure first and calls
	// WaitAlive only after the wrapper has had a chance to exec the
	// engine binary.
	//
	// ollama: true. vllm/llamacpp/sglang: false.
	//
	// The lifecycle has two
	// orders and how they differ when the engine never comes up.
	AliveBeforeBoot() bool

	// Ready reports the latest health snapshot. Cheap (single HTTP call)
	// so lifecycle.engineHealthLoop can poll it on a 10s tick without
	// thrashing the engine.
	Ready(ctx context.Context) (ReadyState, error)

	// OpenAIHandler returns the http.Handler that data plane mounts at
	// /v1/*. The Adapter owns the routing inside that handler (Ollama
	// translates field-by-field; proxy adapters are reverse proxies).
	OpenAIHandler(cfg config.Config) http.Handler

	// AnthropicHandler returns the http.Handler that data plane mounts
	// at POST /v1/messages (and POST /v1/messages/count_tokens once
	// Stage 5 lands). Every backend (Ollama / vLLM / llama.cpp /
	// SGLang) must implement this regardless of whether the upstream
	// engine speaks the Anthropic Messages API natively — the
	// adapter does the wire-shape translation under the hood so a
	// single client surface works across every engine.
	//
	// Translation is per-backend (not one shared Anthropic→OpenAI
	// transcoder) so each adapter keeps full control over field
	// mapping and streaming. Ollama
	// emits NDJSON, the proxy engines emit OpenAI SSE; each path
	// has its own framer to lift those into Anthropic's event
	// sequence (message_start / content_block_* / message_delta /
	// message_stop).
	AnthropicHandler(cfg config.Config) http.Handler

	// EngineNativeStats returns the snapshot of engine state
	// /api/diag/gpu reads. Cheap and idempotent: must NOT trigger
	// model load or generation. Two flavours based on what the engine
	// exposes:
	//
	//   - HTTP-introspection engines (Ollama, SGLang) call out to
	//     the engine and translate the response.
	//       * Ollama:  GET {OLLAMA_URL}/api/ps -- compares
	//         size_vram to size to derive mode
	//         (full/partial/cpu_only).
	//       * SGLang:  GET {ENGINE_URL}/server_info (with
	//         /get_server_info as deprecated fallback) reads
	//         cpu_offload_gb to derive mode.
	//
	//   - Env-mirror + abort-on-shortage engines (vLLM, llama.cpp)
	//     cannot be introspected over HTTP for GPU residency. The
	//     runtime endpoints (vLLM /metrics, /v1/models; llama.cpp
	//     /props, /slots, /metrics) do not expose
	//     --cpu-offload-gb or --n-gpu-layers (verified upstream).
	//     Instead we rely on:
	//       (1) the engine aborting on insufficient VRAM, so a
	//           running container implies the requested GPU
	//           placement succeeded; and
	//       (2) the n_gpu_layers flag parsed from ENGINE_ARGS
	//           (cfg.Engine.Args) confirming full GPU placement
	//           ("all" or a positive integer) was requested.
	//
	//     If n_gpu_layers is absent/partial, GPU.Mode falls back to
	//     "unknown" + a warning rather than lying.
	//
	// Returning (NativeStats{GPU: {Mode: GPUModeUnknown}}, nil) is
	// a valid answer; non-nil error means the engine was
	// unreachable.
	EngineNativeStats(ctx context.Context) (NativeStats, error)
}

// ModelInstaller is the part of an engine that has to be told about a
// model before it can serve it.
//
// Only the Ollama daemon does. It keeps its own store, so bytes sitting
// on disk are nothing to it until they have been pushed through
// /api/blobs and named with /api/create, and a model it already has is
// fetched by name rather than downloaded. Every other engine is launched
// with a path and reads it; for those, installation is the download
// having finished, and there is nothing to call.
//
// So this is optional, and it is one interface rather than the three
// methods it replaces on Adapter. Those made every adapter implement two
// no-ops to satisfy a surface only one of them uses, and left the
// digest-hint variant as a second optional interface whose only reason
// for being separate — that widening Register would add a parameter
// three adapters ignore — stops applying once the whole thing is
// Ollama's alone.
type ModelInstaller interface {
	// Pull asks the daemon to fetch a model by name.
	Pull(ctx context.Context, ref string, sink progress.Sink) error

	// Register makes downloaded files visible to the engine.
	//
	// known carries digests the caller already has, keyed by the
	// absolute path of the file each one describes; an entry is used as
	// given and a path missing from the map is hashed as usual. It
	// exists because hashing is a full read of the model, and on the
	// path that supplies it that read was just done: an ollama:// URL
	// with a #sha256= fragment has had exactly those bytes verified by
	// the downloader. nil is normal.
	Register(ctx context.Context, files []string, known map[string]string, sink progress.Sink) error
}

// InstallerFor resolves the installer once, here, so no caller has to
// ask an Adapter what else it might be. An engine with nothing to
// install gets one that does nothing, rather than a nil to check at
// every call site.
func InstallerFor(a Adapter) ModelInstaller {
	if mi, ok := a.(ModelInstaller); ok {
		return mi
	}
	return noInstall{}
}

// noInstall is what an engine with its own store's absence looks like.
// Register is reached and has nothing to do; Pull is not, because
// config validation rejects an ollama:// source on any other engine.
type noInstall struct{}

func (noInstall) Pull(context.Context, string, progress.Sink) error { return nil }
func (noInstall) Register(context.Context, []string, map[string]string, progress.Sink) error {
	return nil
}
