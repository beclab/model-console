// Package diag implements the runtime diagnostic surface
// (GET /api/diag/gpu).
//
// GPUReport (gpu.go) is the cheap, idempotent answer to "is the model
// 100% on GPU?" — millisecond-class, no model side-effects, safe to
// poll. Its primitives — EngineKind, ModelMeta, GPUResidency — are
// defined in this file. Adding fields is allowed while SchemaVersion
// stays at 1; renaming or repurposing fields requires bumping
// SchemaVersion and a CHANGELOG entry.
package diag

import (
	"encoding/json"
	"time"
)

// SchemaVersion is the structural version of GPUReport. Bump when
// removing a field or repurposing it; adding new optional fields keeps
// the version at 1.
const SchemaVersion = 1

// EngineKind echoes config.EngineKind (we don't import config here
// to keep diag's public surface free of loop-creating imports).
type EngineKind string

// ModelMeta describes the deployed model. Bytes / Format / Quantize
// are best-effort: not every adapter populates them.
type ModelMeta struct {
	Name     string `json:"name"`
	Format   string `json:"format,omitempty"`   // "gguf" | "safetensors" | ""
	Bytes    *int64 `json:"bytes,omitempty"`    // total weight bytes if known
	Quantize string `json:"quantize,omitempty"` // "Q4_K_M", etc.
}

// GPUResidency is the wire representation of GPUResidencyHints
// (internal/adapter). Source is the provenance string the diag layer
// stamps after consulting EngineNativeStats; it is what dashboards
// surface so operators can tell "engine introspection" apart from
// "env-mirror with compose contract".
//
// v1.1.0 BREAKING: the inferred `mode` field (full|partial|cpu_only|
// unknown) was removed from the wire. The four-state enum was a
// best-effort summarisation that hid the underlying numbers behind a
// label that drifted across engines; dashboards now consume
// VRAMBytes / ModelBytes / GPULayers / CPUOffloadGB directly and let
// the human classify. The internal adapter.GPUResidencyHints.Mode
// remains for adapter-side branching (e.g. "is partial → emit
// warning").
type GPUResidency struct {
	// VRAMBytes / ModelBytes (Ollama only): how much of the model
	// weight is currently on the device. nil when the engine doesn't
	// expose the numbers.
	VRAMBytes  *int64 `json:"vram_bytes,omitempty"`
	ModelBytes *int64 `json:"model_bytes,omitempty"`
	// GPULayers / TotalLayers: reserved for a future llama.cpp upstream.
	GPULayers   *int `json:"gpu_layers,omitempty"`
	TotalLayers *int `json:"total_layers,omitempty"`
	// CPUOffloadGB is --cpu-offload-gb when known, read from the
	// engine itself (vLLM /metrics cache_config_info or SGLang
	// /server_info). Always nil for Ollama / llama.cpp.
	CPUOffloadGB *int `json:"cpu_offload_gb,omitempty"`
	// KVCacheUsagePerc is the fraction (0..1) of GPU KV-cache blocks
	// in use right now, from the engine's Prometheus /metrics
	// (vLLM/SGLang/llama.cpp). nil for Ollama and when /metrics is
	// unavailable. Activity signal, not a residency measurement.
	KVCacheUsagePerc *float64 `json:"kv_cache_usage_perc,omitempty"`
	// GPUMemoryUtilization is the configured fraction (0..1) of GPU
	// memory the engine reserves: vLLM gpu_memory_utilization or
	// SGLang mem_fraction_static. nil for Ollama / llama.cpp.
	GPUMemoryUtilization *float64 `json:"gpu_memory_utilization,omitempty"`
	// Source identifies the data lineage of the optional numbers.
	// One of: "ollama_ps" | "sglang_server_info" |
	// "sglang_get_server_info" | "vllm_metrics" |
	// "llamacpp_fit_off_alive" | "unavailable".
	Source string `json:"source"`
}

// GPUReport is the response body for GET /api/diag/gpu.
//
// Cheap to produce (single Adapter.EngineNativeStats call + env
// translation, ~milliseconds). Idempotent: calling it twice in a
// row returns the same answer modulo GeneratedAt (the engine state
// it samples is itself stable across milliseconds).
//
// Warnings is an array of operator-actionable strings. Examples:
//
//   - "ollama: model not currently resident in /api/ps" (KEEP_ALIVE
//     expired, the residency numbers reflect cold state)
//   - "llamacpp: compose contract violated; ENGINE_LLAMACPP_FIT="on""
//   - "sglang: cpu_offload_gb missing from /server_info"
//
// Each warning is independently resolvable; clients should display
// them all rather than coalescing into a single banner.
type GPUReport struct {
	SchemaVersion int          `json:"schema_version"`
	GeneratedAt   time.Time    `json:"generated_at"`
	EngineKind    EngineKind   `json:"engine_kind"`
	Model         ModelMeta    `json:"model"`
	GPU           GPUResidency `json:"gpu"`
	// EngineNative is the verbatim upstream introspection payload. The
	// GET /api/diag/gpu handler omits it unless the caller passes
	// ?include_engine_native=true, so it is absent from routine polls.
	EngineNative json.RawMessage `json:"engine_native,omitempty"`
	Warnings     []string        `json:"warnings,omitempty"`
}
