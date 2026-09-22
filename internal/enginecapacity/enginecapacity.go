// Package enginecapacity asks a running engine how much it can actually
// hold, and falls back to the launch flags when it will not say.
//
// The flags are a request, not an answer. Every one of these four engines
// resolves them at startup against the hardware it found, and three of
// them will quietly resolve them downwards: SGLang recomputes
// max_running_requests from the token pool it managed to allocate and
// applies that over an explicit flag, Ollama lowers both num_ctx and
// num_parallel when memory is short, and vLLM decides its block count by
// profiling the model. llama.cpp does not lower anything, but it does
// assign -- an absent -np becomes four slots sharing one pool, which is
// the configuration that looks least like concurrency and behaves most
// like it.
//
// So a number derived from engine_args is a declaration and a number read
// off the engine is a measurement, and a consumer deciding whether to
// admit a request needs to know which one it is holding. Capacity keeps
// them apart: Source names the probe that answered, FromArgs names the
// fields it did not.
//
// Nothing here returns an error. A probe that cannot be made leaves the
// declared numbers standing with a warning, because the alternative --
// reporting no capacity because the engine was briefly unreachable -- is
// worse than reporting a number that may be generous.
package enginecapacity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/enginemetrics"
)

// Source names where a Capacity's numbers came from. These strings reach
// the model card's extensions.capacity and from there the Router, so they
// are wire values: renaming one is a break.
type Source string

// The known sources. SourceEngineArgs means no probe answered and every
// number is derived from the launch flags; the rest name the endpoint
// that did. The two SGLang spellings are kept apart deliberately, as
// /server_info's deprecated alias is worth seeing without reading logs.
const (
	SourceEngineArgs       Source = "engine_args"
	SourceEngineCapacity   Source = "engine_capacity"
	SourceEngineConfig     Source = "engine_config"
	SourceLlamacppProps    Source = "llamacpp_props"
	SourceSGLangServerInfo Source = "sglang_server_info"
	SourceSGLangGetSrvInfo Source = "sglang_get_server_info"
	SourceOllamaPS         Source = "ollama_ps"
	SourceVLLMMetrics      Source = "vllm_metrics"
)

// Wire field names, used both as JSON keys and as the vocabulary of
// FromArgs, so a reader does not need a second one to work out which
// number a fallback refers to.
const (
	FieldContextSize    = "context_size"
	FieldMaxConcurrency = "max_concurrency"
	FieldPoolTokens     = "pool_tokens"
)

// Capacity is what the engine can hold, in three numbers that do not
// derive from each other.
type Capacity struct {
	// ContextSize is how many tokens one request may occupy.
	ContextSize int `json:"context_size,omitempty"`

	// MaxConcurrency is how many requests the scheduler runs at once.
	// Past it callers wait rather than being refused, which is why the
	// number has to travel: a queue that nobody can see reads as a slow
	// model, and the fix for a slow model is the opposite of the fix for
	// a queue.
	MaxConcurrency int `json:"max_concurrency,omitempty"`

	// PoolTokens is the whole KV cache. It is not ContextSize *
	// MaxConcurrency: an engine is free to let its slots promise more
	// than the pool holds, and llama.cpp in unified mode does exactly
	// that.
	PoolTokens int `json:"pool_tokens,omitempty"`

	// Source is the probe that answered, or SourceEngineArgs when none
	// did.
	Source Source `json:"source"`

	// ReportedAt is when the reading was taken. A consumer that cannot
	// see this cannot tell a live measurement from one that describes an
	// engine which has since been restarted with different flags.
	ReportedAt time.Time `json:"reported_at"`

	// FromArgs names the fields above whose value is a declaration from
	// engine_args rather than something the engine said. A field that
	// neither source could supply is zero and is named nowhere: absent
	// and unmeasured are the same thing to a reader, and both mean "do
	// not decide anything on this".
	FromArgs []string `json:"from_args,omitempty"`

	// Warnings are for an operator, not for control flow. A probe that
	// failed and a probe whose answer contradicts the flags both land
	// here; the second is the more interesting one, because it means
	// this repository's arithmetic about that engine is wrong.
	Warnings []string `json:"warnings,omitempty"`
}

// Options are what Probe needs to reach an engine.
type Options struct {
	Kind      config.EngineKind
	Args      config.EngineArgs
	EngineURL string
	// ConfiguredMaxConcurrency is the optional
	// ENGINE_MAX_CONCURRENCY fallback for non-LLM sibling engines.
	ConfiguredMaxConcurrency int

	// Model is the name to look for in Ollama's /api/ps, which reports
	// one entry per resident model. Ignored by the other engines.
	Model string

	// HTTPClient defaults to a client with probeTimeout.
	HTTPClient *http.Client

	// Now defaults to time.Now.
	Now func() time.Time
}

// probeTimeout is per-probe, and short on purpose: this runs on the
// lifecycle's path to ready, and an engine that needs longer than this to
// describe itself is one whose description is not worth blocking on.
const probeTimeout = 5 * time.Second

// probeBodyLimit caps what a self-report may spend of this process'
// memory. SGLang's /server_info is the large one -- it dumps every
// resolved server argument -- and is still far below this.
const probeBodyLimit = 1 << 20

// Probe reads the engine's own account of its capacity, seeded by the
// launch flags. It always returns a usable Capacity.
func Probe(ctx context.Context, opts Options) Capacity {
	now := time.Now
	if opts.Now != nil {
		now = opts.Now
	}

	out := Capacity{Source: SourceEngineArgs, ReportedAt: now()}
	if !isLLMEngine(opts.Kind) && opts.ConfiguredMaxConcurrency > 0 {
		out.Source = SourceEngineConfig
	}
	declared := declaredCapacity(opts.Kind, opts.Args)
	if !isLLMEngine(opts.Kind) {
		declared.maxConcurrency = opts.ConfiguredMaxConcurrency
	}

	got, warnings := probeEngine(ctx, opts)
	out.Warnings = warnings

	if got.source != "" {
		out.Source = got.source
	}
	out.ContextSize, out.FromArgs = pick(
		got.contextSize, declared.contextSize, FieldContextSize, out.FromArgs)
	switch {
	case isLLMEngine(opts.Kind):
		out.MaxConcurrency, out.FromArgs = pick(
			got.maxConcurrency, declared.maxConcurrency, FieldMaxConcurrency, out.FromArgs)
	case got.maxConcurrency > 0:
		out.MaxConcurrency = got.maxConcurrency
	case declared.maxConcurrency > 0:
		out.MaxConcurrency = declared.maxConcurrency
	}
	out.PoolTokens, out.FromArgs = pick(
		got.poolTokens, declared.poolTokens, FieldPoolTokens, out.FromArgs)
	return out
}

// fellBack renders the one warning every unanswered probe produces. The
// second clause is the part an operator needs: the numbers are still
// there, so the symptom is not a missing capacity but a stale one.
func fellBack(engine, reason string) []string {
	return []string{engine + ": " + reason + "; capacity falls back to engine_args"}
}

// reported is one engine's answer. A zero field means the engine does not
// publish that number, which is not the same as publishing a zero -- the
// per-engine readers reject a non-positive value before it gets here.
type reported struct {
	contextSize    int
	maxConcurrency int
	poolTokens     int
	source         Source
}

// pick prefers what the engine said, records a fallback, and leaves a
// field that neither source could supply out of both.
func pick(measured, declared int, field string, fromArgs []string) (int, []string) {
	if measured > 0 {
		return measured, fromArgs
	}
	if declared > 0 {
		return declared, append(fromArgs, field)
	}
	return 0, fromArgs
}

// declaredCapacity is the seed: the three numbers as the launch flags
// state them.
func declaredCapacity(kind config.EngineKind, args config.EngineArgs) reported {
	var r reported
	r.contextSize, _ = config.DeriveContextSize(kind, args)
	r.maxConcurrency, _ = config.DeriveMaxConcurrency(kind, args)
	r.poolTokens, _ = config.DerivePoolTokens(kind, args)
	return r
}

// probeEngine dispatches to the one endpoint that engine answers with.
func probeEngine(ctx context.Context, opts Options) (reported, []string) {
	if opts.EngineURL == "" {
		return reported{}, nil
	}
	c := &client{base: opts.EngineURL, http: opts.HTTPClient}
	switch opts.Kind {
	case config.EngineLlamaCpp:
		return probeLlamacpp(ctx, c, opts.Args)
	case config.EngineSGLang:
		return probeSGLang(ctx, c)
	case config.EngineOllama:
		return probeOllama(ctx, c, opts.Model, opts.Args)
	case config.EngineVLLM:
		return probeVLLM(ctx, c)
	default:
		return probeSiblingCapacity(ctx, c)
	}
}

func isLLMEngine(kind config.EngineKind) bool {
	switch kind {
	case config.EngineLlamaCpp, config.EngineSGLang, config.EngineOllama, config.EngineVLLM:
		return true
	default:
		return false
	}
}

// probeSiblingCapacity reads the small cross-engine capacity contract.
// It intentionally contains only the maximum number of requests that can
// execute at once; queue depth and current load are different quantities.
func probeSiblingCapacity(ctx context.Context, c *client) (reported, []string) {
	var capacity struct {
		MaxConcurrency int `json:"max_concurrency"`
	}
	if err := c.getJSON(ctx, "/api/engine-capacity", &capacity); err != nil {
		return reported{}, []string{"engine capacity: " + err.Error() +
			"; capacity falls back to ENGINE_MAX_CONCURRENCY when configured"}
	}
	if capacity.MaxConcurrency <= 0 {
		return reported{}, []string{fmt.Sprintf(
			"engine capacity: GET /api/engine-capacity returned invalid max_concurrency=%d; "+
				"capacity falls back to ENGINE_MAX_CONCURRENCY when configured",
			capacity.MaxConcurrency)}
	}
	return reported{
		maxConcurrency: capacity.MaxConcurrency,
		source:         SourceEngineCapacity,
	}, nil
}

// probeLlamacpp reads GET /props, which needs no flag: --props gates only
// the POST form (common/common.h endpoint_props), and the GET answers even
// while the server is sleeping.
//
// It is also the regression guard for kvunified.go. The two numbers below
// are the resolved n_ctx_slot and n_parallel, which is exactly what
// deriveLlamacppContextSize and LlamacppSlots try to predict from the
// flags -- so a mismatch means this repository's model of the auto-slot
// and unified-KV rules has drifted from upstream's, and the warning is the
// only place that would ever be visible.
func probeLlamacpp(ctx context.Context, c *client, args config.EngineArgs) (reported, []string) {
	var props struct {
		TotalSlots                int `json:"total_slots"`
		DefaultGenerationSettings struct {
			NCtx int `json:"n_ctx"`
		} `json:"default_generation_settings"`
	}
	if err := c.getJSON(ctx, "/props", &props); err != nil {
		return reported{}, fellBack("llamacpp", err.Error())
	}

	r := reported{source: SourceLlamacppProps}
	if props.DefaultGenerationSettings.NCtx > 0 {
		r.contextSize = props.DefaultGenerationSettings.NCtx
	}
	if props.TotalSlots > 0 {
		r.maxConcurrency = props.TotalSlots
	}
	// /props publishes no pool total, and there is no honest way to
	// reconstruct one: in unified mode n_ctx_slot equals the pool only
	// when the model's training context does not cap it first, and that
	// value is in the GGUF. -c is left to say what the pool is.
	return r, llamacppDriftWarnings(args, r)
}

// llamacppDriftWarnings compares the engine's resolved numbers against the
// ones the flags were read to mean.
func llamacppDriftWarnings(args config.EngineArgs, got reported) []string {
	var warnings []string
	if want, ok := config.DeriveContextSize(config.EngineLlamaCpp, args); ok &&
		got.contextSize > 0 && want != got.contextSize {
		warnings = append(warnings, fmt.Sprintf(
			"llamacpp: /props reports n_ctx_slot=%d but engine_args derive %d; "+
				"the model's training context may be lower than -c, or this build's "+
				"unified-KV arithmetic is wrong",
			got.contextSize, want))
	}
	if want, ok := config.DeriveMaxConcurrency(config.EngineLlamaCpp, args); ok &&
		got.maxConcurrency > 0 && want != got.maxConcurrency {
		warnings = append(warnings, fmt.Sprintf(
			"llamacpp: /props reports total_slots=%d but engine_args derive %d",
			got.maxConcurrency, want))
	}
	return warnings
}

// probeSGLang reads /server_info, falling back to the deprecated
// /get_server_info alias. The body is the resolved server arguments plus
// the scheduler's own handshake state, which is where the number worth
// having lives: max_total_num_tokens is the pool SGLang managed to
// allocate under --mem-fraction-static, and it is what the engine derives
// its running width from.
func probeSGLang(ctx context.Context, c *client) (reported, []string) {
	var info struct {
		MaxTotalNumTokens  int `json:"max_total_num_tokens"`
		MaxRunningRequests int `json:"max_running_requests"`
		ContextLength      int `json:"context_length"`
		DPSize             int `json:"dp_size"`
		InternalStates     []struct {
			EffectiveMaxRunningRequestsPerDP int `json:"effective_max_running_requests_per_dp"`
		} `json:"internal_states"`
	}
	source := SourceSGLangServerInfo
	err := c.getJSON(ctx, "/server_info", &info)
	if isNotFound(err) {
		source = SourceSGLangGetSrvInfo
		err = c.getJSON(ctx, "/get_server_info", &info)
	}
	if err != nil {
		return reported{}, fellBack("sglang", err.Error())
	}

	r := reported{source: source}
	if info.ContextLength > 0 {
		r.contextSize = info.ContextLength
	}
	if info.MaxRunningRequests > 0 {
		r.maxConcurrency = info.MaxRunningRequests
	} else if info.DPSize > 0 && len(info.InternalStates) > 0 &&
		info.InternalStates[0].EffectiveMaxRunningRequestsPerDP > 0 {
		// SGLang 0.5.18 stopped resolving max_running_requests at the
		// top level. Its benchmark tooling defines the total running-batch
		// cap as the per-DP effective width multiplied by dp_size.
		r.maxConcurrency =
			info.InternalStates[0].EffectiveMaxRunningRequestsPerDP * info.DPSize
	}
	if info.MaxTotalNumTokens > 0 {
		r.poolTokens = info.MaxTotalNumTokens
	}
	return r, nil
}

// probeOllama reads GET /api/ps, whose context_length is the per-request
// window the runner was actually launched with -- Ollama writes the
// runner's answer back over the requested num_ctx before recording it
// (server/sched.go), so this is the value after any memory-driven
// reduction.
//
// The width is not there to be read. /api/ps describes models, not the
// scheduler, and OLLAMA_NUM_PARALLEL is exactly the knob Ollama overrides
// in silence, so the declared width stands with a warning rather than
// being reconstructed from a product this endpoint cannot confirm.
func probeOllama(ctx context.Context, c *client, model string, args config.EngineArgs) (reported, []string) {
	var ps struct {
		Models []struct {
			Name          string `json:"name"`
			Model         string `json:"model"`
			ContextLength int    `json:"context_length"`
		} `json:"models"`
	}
	if err := c.getJSON(ctx, "/api/ps", &ps); err != nil {
		return reported{}, fellBack("ollama", err.Error())
	}
	if len(ps.Models) == 0 {
		return reported{}, fellBack("ollama", "/api/ps lists no resident model")
	}

	entry := -1
	for i, m := range ps.Models {
		if m.Name == model || m.Model == model {
			entry = i
			break
		}
	}
	if entry < 0 {
		if len(ps.Models) > 1 {
			return reported{}, fellBack("ollama", fmt.Sprintf(
				"/api/ps lists %d resident models, none named %q",
				len(ps.Models), model))
		}
		// A single resident model on a daemon this process launched for
		// one model is that model, whatever tag rewriting sits between
		// the name on the card and the name Ollama echoes back.
		entry = 0
	}

	r := reported{source: SourceOllamaPS}
	if n := ps.Models[entry].ContextLength; n > 0 {
		r.contextSize = n
	}

	var warnings []string
	if want, ok := args.GetInt("num_ctx"); ok && r.contextSize > 0 && want != r.contextSize {
		warnings = append(warnings, fmt.Sprintf(
			"ollama: running with context_length=%d, not the %d that OLLAMA_NUM_CTX asked for; "+
				"the daemon lowered it to fit memory",
			r.contextSize, want))
	}
	return r, warnings
}

// probeVLLM reads the pool out of /metrics.
//
// vllm:cache_config_info is an info-style gauge whose labels are the whole
// CacheConfig, published once the engine is initialised -- which is after
// the profiling run that decides num_gpu_blocks. So the block count there
// is the measured one, and multiplied by block_size it is the token
// capacity that no flag states.
//
// The other two numbers are not on /metrics. max_model_len and
// max_num_seqs belong to configs that publish no info gauge, and vLLM's
// resolved max_concurrency appears only in a startup log line, so the
// declared values stand.
func probeVLLM(ctx context.Context, c *client) (reported, []string) {
	body, status, err := c.get(ctx, "/metrics")
	if err != nil {
		return reported{}, fellBack("vllm", err.Error())
	}
	if status != http.StatusOK {
		return reported{}, fellBack("vllm",
			fmt.Sprintf("GET /metrics status %d", status))
	}
	fams, err := enginemetrics.Parse(body)
	if err != nil {
		return reported{}, fellBack("vllm", err.Error())
	}

	blocks := enginemetrics.IntLabel(fams, metricVLLMCacheConfig, labelNumGPUBlocks)
	blockSize := enginemetrics.IntLabel(fams, metricVLLMCacheConfig, labelBlockSize)
	if blocks == nil || blockSize == nil || *blocks <= 0 || *blockSize <= 0 {
		return reported{}, fellBack("vllm", "/metrics carries no usable "+
			metricVLLMCacheConfig+"{"+labelNumGPUBlocks+","+labelBlockSize+"}")
	}
	return reported{
		poolTokens: *blocks * *blockSize,
		source:     SourceVLLMMetrics,
	}, nil
}

const (
	metricVLLMCacheConfig = "vllm:cache_config_info"
	labelNumGPUBlocks     = "num_gpu_blocks"
	labelBlockSize        = "block_size"
)

// client is a GET-only HTTP helper over one engine's base URL.
type client struct {
	base string
	http *http.Client
}

// notFoundErr lets probeSGLang tell "this endpoint is the deprecated one"
// apart from every other failure without inspecting a status code the
// getJSON contract has already folded into an error.
type notFoundErr struct{ path string }

func (e notFoundErr) Error() string { return "GET " + e.path + " status 404" }

func isNotFound(err error) bool {
	var nf notFoundErr
	return errors.As(err, &nf)
}

func (c *client) get(ctx context.Context, path string) ([]byte, int, error) {
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		strings.TrimRight(c.base, "/")+path, http.NoBody)
	if err != nil {
		return nil, 0, err
	}
	hc := c.http
	if hc == nil {
		hc = &http.Client{Timeout: probeTimeout}
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("GET %s: %w", path, err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, probeBodyLimit))
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read GET %s: %w", path, err)
	}
	return body, resp.StatusCode, nil
}

func (c *client) getJSON(ctx context.Context, path string, into any) error {
	body, status, err := c.get(ctx, path)
	if err != nil {
		return err
	}
	if status == http.StatusNotFound {
		return notFoundErr{path: path}
	}
	if status != http.StatusOK {
		return fmt.Errorf("GET %s status %d", path, status)
	}
	if err := json.Unmarshal(body, into); err != nil {
		return fmt.Errorf("decode GET %s: %w", path, err)
	}
	return nil
}
