package diag

import (
	"context"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
)

// sourceUnavailable is the GPUResidency.Source value stamped when the
// adapter could not attribute the numbers to any introspection path.
// It is part of the /api/diag/gpu wire contract.
const sourceUnavailable = "unavailable"

// BuildGPUReport produces a GPUReport for GET /api/diag/gpu.
//
// Pure function: does not write metrics, does not lock. The caller
// (handler.go) wraps it with its own metrics story.
//
// The function consults Adapter.EngineNativeStats (cheap, single
// HTTP call for Ollama/SGLang; pure env-mirror for vLLM/llama.cpp).
// Errors from EngineNativeStats are surfaced verbatim — they mean
// the engine is unreachable, which is a 503 condition, not "give us
// unknown". The "unknown" path is reserved for the engine-up,
// can't-tell-us-the-mode case (e.g. llama.cpp compose contract
// violation).
func BuildGPUReport(ctx context.Context, ad adapter.Adapter, cfg config.Config, now func() time.Time) (GPUReport, error) {
	stats, err := ad.EngineNativeStats(ctx)
	if err != nil {
		return GPUReport{}, err
	}

	model := buildModelMeta(cfg)
	gpu := translateResidency(stats)

	return GPUReport{
		SchemaVersion: SchemaVersion,
		GeneratedAt:   now().UTC(),
		EngineKind:    EngineKind(cfg.Engine.Kind),
		Model:         model,
		GPU:           gpu,
		EngineNative:  stats.Payload,
		Warnings:      stats.Warnings,
	}, nil
}

// buildModelMeta extracts the externally-visible identity. Bytes /
// Format / Quantize stay nil because llm-init doesn't track them
// today; future M5 work could populate them from manifest reads.
func buildModelMeta(cfg config.Config) ModelMeta {
	return ModelMeta{Name: cfg.Model.Name}
}

// translateResidency maps adapter.GPUResidencyHints into the wire
// shape. v1.1.0 dropped the `mode` field; only the underlying
// numbers + source provenance survive. Source comes from
// NativeStats.Source (the adapter's choice) rather than being
// recomputed here, keeping the source-of-truth in one place.
func translateResidency(s adapter.NativeStats) GPUResidency {
	out := GPUResidency{
		Source:               s.Source,
		VRAMBytes:            s.GPU.VRAMBytes,
		ModelBytes:           s.GPU.ModelBytes,
		GPULayers:            s.GPU.GPULayers,
		TotalLayers:          s.GPU.TotalLayers,
		CPUOffloadGB:         s.GPU.CPUOffloadGB,
		KVCacheUsagePerc:     s.GPU.KVCacheUsagePerc,
		GPUMemoryUtilization: s.GPU.GPUMemoryUtilization,
	}
	if out.Source == "" {
		out.Source = sourceUnavailable
	}
	return out
}
