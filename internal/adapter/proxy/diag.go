package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	dto "github.com/prometheus/client_model/go"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/enginemetrics"
)

// diag.go houses the per-engine EngineNativeStats branches. Split
// out of adapter.go because the three branches have non-trivial
// per-engine quirks (SGLang HTTP fallback, vLLM env-only, llamacpp
// compose-contract validation) and inlining them would push
// adapter.go past its cyclomatic budget.
//
// Upstream documentation references for every claim in this file
// live in the repo plan §9. Don't change the signal-mapping
// without updating that table.

// Wire-level Source / env-name constants shared by adapter.go and
// the per-engine branches below. Lifting these to named constants
// is what keeps goconst quiet (they each appear 3+ times across
// the package + test fixtures), but more importantly it pins the
// strings that operators see in /api/diag/gpu's JSON Source field
// and engine_native.env keys -- changing them is a wire break.
const (
	sourceUnavailable        = "unavailable"
	sourceVLLMMetrics        = "vllm_metrics"
	sourceLlamacppFitOffAliv = "llamacpp_fit_off_alive"
	sourceSGLangServerInfo   = "sglang_server_info"
	sourceSGLangGetSrvInfo   = "sglang_get_server_info"

	// argLlamacppNGL is the canonical Args.Known key for llama.cpp's
	// -ngl / --n-gpu-layers / LLAMA_ARG_N_GPU_LAYERS (see
	// engine_args_known.go). defaultLlamacppNGL mirrors the value the
	// shipped deploy/compose/llamacpp.yml expects operators to pass.
	argLlamacppNGL     = "n_gpu_layers"
	defaultLlamacppNGL = "all"
	llamacppNGLAll     = "all"

	// Prometheus metric names per engine. vLLM exposes /metrics by
	// default; SGLang/llama.cpp need --enable-metrics / --metrics,
	// which the deploy wrappers bake in.
	metricVLLMCacheConfig = "vllm:cache_config_info"
	metricVLLMKVUsage     = "vllm:kv_cache_usage_perc"
	metricSGLangKVUsage   = "sglang:kv_cache_usage_perc"
	metricLlamacppKVUsage = "llamacpp:kv_cache_usage_ratio"

	labelCPUOffloadGB = "cpu_offload_gb"
	labelGPUMemUtil   = "gpu_memory_utilization"

	// Generic gpu_* gauges a proxy engine's /metrics may expose.
	sourceEngineMetrics = "engine_metrics"
	metricGPUPresent    = "gpu_present"
	metricGPUMemUsed    = "gpu_mem_used_bytes"
	metricGPUMemTotal   = "gpu_mem_total_bytes"
	metricGPUUtil       = "gpu_util_ratio"
)

// sglangNativeStats calls SGLang's /server_info (canonical) with a
// fallback to /get_server_info (deprecated alias, see issue #21054).
// The response is the full ServerArgs dump; we extract cpu_offload_gb
// to decide gpu.mode and pass everything through to engine_native.
func (a *Adapter) sglangNativeStats(ctx context.Context) (adapter.NativeStats, error) {
	body, source, err := a.sglangFetchServerInfo(ctx)
	if err != nil {
		return adapter.NativeStats{}, err
	}

	// Decode just enough to read cpu_offload_gb and
	// mem_fraction_static. The full body still flows through Payload
	// so dashboards see dtype, tp_size, etc.
	var fields struct {
		CPUOffloadGB      *int     `json:"cpu_offload_gb"`
		MemFractionStatic *float64 `json:"mem_fraction_static"`
	}
	if err := json.Unmarshal(body, &fields); err != nil {
		return adapter.NativeStats{}, fmt.Errorf(
			"sglang: decode %s: %w", source, err)
	}

	hints := adapter.GPUResidencyHints{GPUMemoryUtilization: fields.MemFractionStatic}
	var warnings []string
	switch {
	case fields.CPUOffloadGB == nil:
		// Older SGLang or unusual build that doesn't expose the
		// field. Fall back to unknown but pass payload so the
		// dashboard can render whatever the engine does send.
		hints.Mode = adapter.GPUModeUnknown
		warnings = append(warnings,
			"sglang: cpu_offload_gb missing from /server_info; cannot determine gpu residency")
	case *fields.CPUOffloadGB == 0:
		hints.Mode = adapter.GPUModeFull
	default:
		hints.Mode = adapter.GPUModePartial
		v := *fields.CPUOffloadGB
		hints.CPUOffloadGB = &v
	}

	// Best-effort: layer live KV-cache occupancy on top of the
	// /server_info config. /server_info already proved the engine is
	// reachable, so a /metrics miss degrades to a warning rather than
	// a 503 (unlike vLLM where /metrics is the primary signal).
	if fams, mErr := a.fetchMetricFamilies(ctx); mErr == nil {
		hints.KVCacheUsagePerc = enginemetrics.Gauge(fams, metricSGLangKVUsage)
	} else {
		warnings = append(warnings,
			"sglang: /metrics unavailable ("+mErr.Error()+"); kv_cache_usage_perc omitted (need --enable-metrics)")
	}

	return adapter.NativeStats{
		Source:   source,
		Payload:  body,
		GPU:      hints,
		Warnings: warnings,
	}, nil
}

// sglangFetchServerInfo tries /server_info first; on 404 falls back
// to /get_server_info. Returns (body, source-tag, error). The
// source tag is "sglang_server_info" for the canonical endpoint and
// "sglang_get_server_info" for the deprecated alias so dashboards
// can surface "you're hitting an EOL endpoint" without reading logs.
func (a *Adapter) sglangFetchServerInfo(ctx context.Context) ([]byte, string, error) {
	body, status, err := a.getJSON(ctx, "/server_info")
	if err != nil {
		return nil, "", err
	}
	if status == http.StatusOK {
		return body, sourceSGLangServerInfo, nil
	}
	if status != http.StatusNotFound {
		return nil, "", fmt.Errorf(
			"sglang: GET /server_info status %d", status)
	}
	// Fallback path. Treat any error here as fatal because we have
	// no third option.
	body, status, err = a.getJSON(ctx, "/get_server_info")
	if err != nil {
		return nil, "", err
	}
	if status != http.StatusOK {
		return nil, "", fmt.Errorf(
			"sglang: GET /get_server_info status %d (also tried /server_info -> 404)",
			status)
	}
	return body, sourceSGLangGetSrvInfo, nil
}

// getJSON is a tiny GET helper that returns (body, status, error).
// httpClient already has a 5s timeout (see NewAdapter); the ctx
// here is the caller's deadline.
func (a *Adapter) getJSON(ctx context.Context, path string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx,
		http.MethodGet, joinPath(a.cfg.Engine.URL, path), nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err := readAllLimit(resp.Body, 1<<20) // 1 MiB cap on engine self-reports
	if err != nil {
		return nil, resp.StatusCode, err
	}
	return body, resp.StatusCode, nil
}

// vllmNativeStats reads vLLM's Prometheus /metrics (always on for the
// OpenAI server). It pulls cache_config_info (cpu_offload_gb,
// gpu_memory_utilization config echo) and kv_cache_usage_perc (live
// KV-cache occupancy). /metrics is vLLM's primary GPU signal here, so
// a transport error / non-200 / parse failure returns an error which
// the diag handler renders as 503 engine_unreachable.
//
// vLLM exposes no per-model VRAM bytes, so VRAMBytes/ModelBytes stay
// nil; mode is derived from cpu_offload_gb (>0 -> partial, else full).
func (a *Adapter) vllmNativeStats(ctx context.Context) (adapter.NativeStats, error) {
	fams, err := a.fetchMetricFamilies(ctx)
	if err != nil {
		return adapter.NativeStats{}, fmt.Errorf("vllm: %w", err)
	}
	return a.vllmStatsFromFamilies(fams), nil
}

// vllmStatsFromFamilies maps parsed /metrics families into NativeStats.
func (a *Adapter) vllmStatsFromFamilies(fams map[string]*dto.MetricFamily) adapter.NativeStats {
	cpuOff := enginemetrics.IntLabel(fams, metricVLLMCacheConfig, labelCPUOffloadGB)
	gpuMem := enginemetrics.FloatLabel(fams, metricVLLMCacheConfig, labelGPUMemUtil)
	kvUsage := enginemetrics.Gauge(fams, metricVLLMKVUsage)

	hints := adapter.GPUResidencyHints{
		GPUMemoryUtilization: gpuMem,
		KVCacheUsagePerc:     kvUsage,
	}
	// Surface cpu_offload_gb only when it indicates partial GPU; 0
	// offload is "full" and leaves the field nil (omitempty wire).
	if cpuOff != nil && *cpuOff > 0 {
		hints.Mode = adapter.GPUModePartial
		hints.CPUOffloadGB = cpuOff
	} else {
		hints.Mode = adapter.GPUModeFull
	}

	var warnings []string
	if cpuOff == nil && gpuMem == nil {
		warnings = append(warnings,
			"vllm: cache_config_info absent from /metrics; cannot read cpu_offload_gb / gpu_memory_utilization")
	}

	dump := map[string]any{
		labelCPUOffloadGB:     derefInt(cpuOff),
		labelGPUMemUtil:       derefFloat(gpuMem),
		"kv_cache_usage_perc": derefFloat(kvUsage),
	}
	payload, _ := json.Marshal(dump)

	return adapter.NativeStats{
		Source:   sourceVLLMMetrics,
		Payload:  payload,
		GPU:      hints,
		Warnings: warnings,
	}
}

// proxyMetricsStats reads generic gpu_* gauges, else vLLM cache_config_info.
func (a *Adapter) proxyMetricsStats(ctx context.Context) (adapter.NativeStats, error) {
	fams, err := a.fetchMetricFamilies(ctx)
	if err != nil {
		return adapter.NativeStats{
			Source: sourceUnavailable,
			GPU:    adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown},
			Warnings: []string{
				"engine: /metrics unavailable (" + err.Error() + "); gpu residency unknown"},
		}, nil
	}

	used := enginemetrics.Gauge(fams, metricGPUMemUsed)
	total := enginemetrics.Gauge(fams, metricGPUMemTotal)
	util := enginemetrics.Gauge(fams, metricGPUUtil)
	present := enginemetrics.Gauge(fams, metricGPUPresent)

	if used != nil || present != nil {
		hints := adapter.GPUResidencyHints{}
		switch {
		case present != nil && *present == 0:
			hints.Mode = adapter.GPUModeCPUOnly
		case used != nil && *used > 0:
			hints.Mode = adapter.GPUModeFull
			u := int64(*used)
			hints.VRAMBytes = &u
		default:
			hints.Mode = adapter.GPUModeUnknown
		}
		if used != nil && total != nil && *total > 0 {
			r := *used / *total
			hints.GPUMemoryUtilization = &r
		}
		dump := map[string]any{
			metricGPUMemUsed:  derefFloat(used),
			metricGPUMemTotal: derefFloat(total),
			metricGPUUtil:     derefFloat(util),
		}
		payload, _ := json.Marshal(dump)
		return adapter.NativeStats{
			Source:  sourceEngineMetrics,
			Payload: payload,
			GPU:     hints,
		}, nil
	}

	// Fallback: an engine that re-exports vLLM's own metrics.
	if fams[metricVLLMCacheConfig] != nil {
		return a.vllmStatsFromFamilies(fams), nil
	}

	return adapter.NativeStats{
		Source: sourceUnavailable,
		GPU:    adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown},
		Warnings: []string{
			"engine: /metrics present but carries neither gpu_* nor vllm:cache_config_info gauges"},
	}, nil
}

// derefInt / derefFloat render optional numbers for the engine_native
// JSON payload (nil -> null).
func derefInt(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

func derefFloat(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

// llamacppNativeStats validates the GPU-placement contract from
// ENGINE_ARGS:
//
//   - n_gpu_layers (-ngl / --n-gpu-layers / LLAMA_ARG_N_GPU_LAYERS)
//     must be "all" or a positive integer, so we know full GPU
//     placement was requested. Absent means the shipped-compose
//     default ("all").
//
// When it holds, gpu.mode = full and source = "llamacpp_fit_off_alive"
// (a stable wire tag; the name predates the v1.0 ENGINE_LLAMACPP_FIT
// mirror removal). When it fails, gpu.mode = unknown with a warning.
// We don't try to compute "partial" for llama.cpp because the runtime
// endpoints lack the data we'd need (see plan §9 verification).
func (a *Adapter) llamacppNativeStats(ctx context.Context) adapter.NativeStats {
	ngl, ok := a.cfg.Engine.Args.GetString(argLlamacppNGL)
	if !ok {
		ngl = defaultLlamacppNGL
	}

	dump := map[string]any{argLlamacppNGL: ngl}
	payload, _ := json.Marshal(dump)

	// Contract warnings flip residency to unknown/unavailable; the
	// /metrics warning below is soft and does not.
	var warnings []string
	contractOK := true
	if ngl != llamacppNGLAll && !llamacppNGLIsPositiveInt(ngl) {
		contractOK = false
		warnings = append(warnings, fmt.Sprintf(
			"llamacpp: GPU-placement contract violated; n_gpu_layers=%q is neither %q nor a positive integer (expected %q to mean \"all layers on GPU\"); set it in ENGINE_ARGS",
			ngl, llamacppNGLAll, llamacppNGLAll))
	}

	// Residency mode comes from the env contract (llama.cpp's runtime
	// endpoints expose no offload-layer count). KV-cache occupancy is
	// a best-effort add-on from /metrics (needs --metrics); a miss is
	// a soft warning, not a hard failure and not a mode change.
	hints := adapter.GPUResidencyHints{}
	if fams, mErr := a.fetchMetricFamilies(ctx); mErr == nil {
		hints.KVCacheUsagePerc = enginemetrics.Gauge(fams, metricLlamacppKVUsage)
	} else {
		warnings = append(warnings,
			"llamacpp: /metrics unavailable ("+mErr.Error()+"); kv_cache_usage_perc omitted (need --metrics)")
	}

	source := sourceLlamacppFitOffAliv
	hints.Mode = adapter.GPUModeFull
	if !contractOK {
		source = sourceUnavailable
		hints.Mode = adapter.GPUModeUnknown
	}

	return adapter.NativeStats{
		Source:   source,
		Payload:  payload,
		GPU:      hints,
		Warnings: warnings,
	}
}

// llamacppNGLIsPositiveInt reports whether ngl parses as an integer
// >= 1. -ngl 0 is technically allowed by upstream llama.cpp but
// means "no GPU layers", which is functionally cpu_only; our
// contract-validating path treats that as a violation because the
// shipped compose says "all".
func llamacppNGLIsPositiveInt(ngl string) bool {
	n, err := strconv.Atoi(ngl)
	if err != nil {
		return false
	}
	return n >= 1
}

// readAllLimit drains r up to max bytes; anything beyond is
// truncated. Used by getJSON to stop a misbehaving engine from
// blowing the diag handler's memory.
func readAllLimit(r io.Reader, max int64) ([]byte, error) {
	return io.ReadAll(io.LimitReader(r, max))
}
