package enginecapacity

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

// parseArgs builds the Known table the way the model card does, so a test
// case can be written as the flag string an operator wrote.
func parseArgs(t *testing.T, kind config.EngineKind, raw string) config.EngineArgs {
	t.Helper()
	args, err := config.ParseEngineArgs(kind, raw)
	if err != nil {
		t.Fatalf("ParseEngineArgs(%q): %v", raw, err)
	}
	return args
}

func TestProbeLlamacpp_PropsWins(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/props" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{
			"total_slots": 4,
			"default_generation_settings": {"n_ctx": 8192}
		}`))
	}))
	defer srv.Close()

	// -c 32768 with four auto slots and a unified pool derives 32768 as
	// the window; the engine reports 8192 because the model's training
	// context caps it. The measurement has to win, and the disagreement
	// has to be said out loud.
	got := Probe(context.Background(), Options{
		Kind:      config.EngineLlamaCpp,
		Args:      parseArgs(t, config.EngineLlamaCpp, "-c 32768"),
		EngineURL: srv.URL,
	})

	if got.Source != SourceLlamacppProps {
		t.Errorf("Source = %q, want %q", got.Source, SourceLlamacppProps)
	}
	if got.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want 8192", got.ContextSize)
	}
	if got.MaxConcurrency != 4 {
		t.Errorf("MaxConcurrency = %d, want 4", got.MaxConcurrency)
	}
	// /props publishes no pool total, so -c stands and says so.
	if got.PoolTokens != 32768 {
		t.Errorf("PoolTokens = %d, want 32768", got.PoolTokens)
	}
	if !slices.Contains(got.FromArgs, FieldPoolTokens) {
		t.Errorf("FromArgs = %v, want it to name %q", got.FromArgs, FieldPoolTokens)
	}
	if slices.Contains(got.FromArgs, FieldContextSize) {
		t.Errorf("FromArgs = %v, must not name a measured field", got.FromArgs)
	}
	if !hasWarning(got.Warnings, "n_ctx_slot=8192") {
		t.Errorf("Warnings = %v, want the n_ctx_slot drift named", got.Warnings)
	}
}

// The slot count is the half of kvunified.go with no other check on it:
// an absent -np is read here as four, and only the engine can confirm it.
func TestProbeLlamacpp_SlotDriftIsWarned(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{
			"total_slots": 1,
			"default_generation_settings": {"n_ctx": 4096}
		}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineLlamaCpp,
		Args:      parseArgs(t, config.EngineLlamaCpp, "-c 4096"),
		EngineURL: srv.URL,
	})
	if got.MaxConcurrency != 1 {
		t.Errorf("MaxConcurrency = %d, want 1", got.MaxConcurrency)
	}
	if !hasWarning(got.Warnings, "total_slots=1 but engine_args derive 4") {
		t.Errorf("Warnings = %v, want the slot drift named", got.Warnings)
	}
}

func TestProbeLlamacpp_UnreachableFallsBackToArgs(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineLlamaCpp,
		Args:      parseArgs(t, config.EngineLlamaCpp, "-c 12288 -np 2 -no-kvu"),
		EngineURL: srv.URL,
	})
	if got.Source != SourceEngineArgs {
		t.Errorf("Source = %q, want %q", got.Source, SourceEngineArgs)
	}
	// Split mode: 12288 / 2 slots.
	if got.ContextSize != 6144 {
		t.Errorf("ContextSize = %d, want 6144", got.ContextSize)
	}
	if got.MaxConcurrency != 2 {
		t.Errorf("MaxConcurrency = %d, want 2", got.MaxConcurrency)
	}
	if got.PoolTokens != 12288 {
		t.Errorf("PoolTokens = %d, want 12288", got.PoolTokens)
	}
	want := []string{FieldContextSize, FieldMaxConcurrency, FieldPoolTokens}
	if !slices.Equal(got.FromArgs, want) {
		t.Errorf("FromArgs = %v, want %v", got.FromArgs, want)
	}
	if len(got.Warnings) != 1 || !strings.Contains(got.Warnings[0], "falls back to engine_args") {
		t.Errorf("Warnings = %v, want one naming the fallback", got.Warnings)
	}
}

func TestProbeSGLang_ServerInfo(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/server_info" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(`{
			"context_length": 32768,
			"max_running_requests": 33,
			"max_total_num_tokens": 1048576,
			"mem_fraction_static": 0.87
		}`))
	}))
	defer srv.Close()

	// The declared width is 8; SGLang recomputes it from the pool it
	// managed to allocate and applies that over the flag, which is the
	// whole reason this probe exists.
	got := Probe(context.Background(), Options{
		Kind: config.EngineSGLang,
		Args: parseArgs(t, config.EngineSGLang,
			"--context-length 65536 --max-running-requests 8"),
		EngineURL: srv.URL,
	})
	if got.Source != SourceSGLangServerInfo {
		t.Errorf("Source = %q, want %q", got.Source, SourceSGLangServerInfo)
	}
	if got.ContextSize != 32768 || got.MaxConcurrency != 33 || got.PoolTokens != 1048576 {
		t.Errorf("got (%d, %d, %d), want (32768, 33, 1048576)",
			got.ContextSize, got.MaxConcurrency, got.PoolTokens)
	}
	if len(got.FromArgs) != 0 {
		t.Errorf("FromArgs = %v, want empty: every field was measured", got.FromArgs)
	}
}

func TestProbeSGLang_EffectiveWidthFromInternalState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/server_info" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(`{
			"context_length": 65536,
			"max_running_requests": null,
			"max_total_num_tokens": 192988,
			"dp_size": 2,
			"internal_states": [
				{"effective_max_running_requests_per_dp": 21}
			]
		}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineSGLang,
		Args:      parseArgs(t, config.EngineSGLang, "--context-length 65536"),
		EngineURL: srv.URL,
	})

	if got.Source != SourceSGLangServerInfo {
		t.Errorf("Source = %q, want %q", got.Source, SourceSGLangServerInfo)
	}
	if got.ContextSize != 65536 || got.MaxConcurrency != 42 || got.PoolTokens != 192988 {
		t.Errorf("got (%d, %d, %d), want (65536, 42, 192988)",
			got.ContextSize, got.MaxConcurrency, got.PoolTokens)
	}
	if len(got.FromArgs) != 0 {
		t.Errorf("FromArgs = %v, want empty: every field was measured", got.FromArgs)
	}
}

func TestProbeSGLang_FallsBackToDeprecatedAlias(t *testing.T) {
	var paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		if r.URL.Path == "/server_info" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_, _ = w.Write([]byte(`{"max_total_num_tokens": 4096}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineSGLang,
		Args:      parseArgs(t, config.EngineSGLang, ""),
		EngineURL: srv.URL,
	})
	if got.Source != SourceSGLangGetSrvInfo {
		t.Errorf("Source = %q, want %q", got.Source, SourceSGLangGetSrvInfo)
	}
	if got.PoolTokens != 4096 {
		t.Errorf("PoolTokens = %d, want 4096", got.PoolTokens)
	}
	// Nothing declared the window or the width, and nothing measured
	// them. Both stay absent rather than being named as a fallback.
	if got.ContextSize != 0 || got.MaxConcurrency != 0 {
		t.Errorf("got (%d, %d), want both zero", got.ContextSize, got.MaxConcurrency)
	}
	if len(got.FromArgs) != 0 {
		t.Errorf("FromArgs = %v, want empty", got.FromArgs)
	}
	want := []string{"/server_info", "/get_server_info"}
	if !slices.Equal(paths, want) {
		t.Errorf("paths = %v, want %v", paths, want)
	}
}

func TestProbeOllama_ContextLengthAfterReduction(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ps" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"models": [
			{"name": "other:latest", "model": "other:latest", "context_length": 2048},
			{"name": "qwen3:8b", "model": "qwen3:8b", "context_length": 8192}
		]}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineOllama,
		Args:      config.EngineArgs{Known: map[string]string{"num_ctx": "40960", "num_parallel": "4"}},
		EngineURL: srv.URL,
		Model:     "qwen3:8b",
	})
	if got.Source != SourceOllamaPS {
		t.Errorf("Source = %q, want %q", got.Source, SourceOllamaPS)
	}
	if got.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want 8192", got.ContextSize)
	}
	// The width and the pool are not on /api/ps; the declared product
	// stands, and is marked as declared.
	if got.MaxConcurrency != 4 || got.PoolTokens != 40960*4 {
		t.Errorf("got (%d, %d), want (4, %d)", got.MaxConcurrency, got.PoolTokens, 40960*4)
	}
	if !slices.Contains(got.FromArgs, FieldMaxConcurrency) ||
		!slices.Contains(got.FromArgs, FieldPoolTokens) {
		t.Errorf("FromArgs = %v, want the width and the pool named", got.FromArgs)
	}
	if !hasWarning(got.Warnings, "lowered it to fit memory") {
		t.Errorf("Warnings = %v, want the silent reduction named", got.Warnings)
	}
}

// A daemon this process launched for one model holds one model, whatever
// tag rewriting sits between the card and the name Ollama echoes back.
func TestProbeOllama_SoleResidentModelIsTheModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(
			`{"models": [{"name": "sha256-abc:latest", "context_length": 4096}]}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineOllama,
		EngineURL: srv.URL,
		Model:     "qwen3:8b",
	})
	if got.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096", got.ContextSize)
	}
}

func TestProbeOllama_AmbiguousMultiModelFallsBack(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"models": [
			{"name": "a:latest", "context_length": 2048},
			{"name": "b:latest", "context_length": 4096}
		]}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineOllama,
		Args:      config.EngineArgs{Known: map[string]string{"num_ctx": "8192"}},
		EngineURL: srv.URL,
		Model:     "qwen3:8b",
	})
	if got.Source != SourceEngineArgs {
		t.Errorf("Source = %q, want %q", got.Source, SourceEngineArgs)
	}
	if got.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want the declared 8192", got.ContextSize)
	}
	if !hasWarning(got.Warnings, "none named") {
		t.Errorf("Warnings = %v, want the ambiguity named", got.Warnings)
	}
}

func TestProbeVLLM_PoolFromBlockCount(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",num_gpu_blocks="4096",gpu_memory_utilization="0.9"} 1.0
`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind: config.EngineVLLM,
		Args: parseArgs(t, config.EngineVLLM,
			"--max-model-len 32768 --max-num-seqs 64"),
		EngineURL: srv.URL,
	})
	if got.Source != SourceVLLMMetrics {
		t.Errorf("Source = %q, want %q", got.Source, SourceVLLMMetrics)
	}
	// The one number no vLLM flag states: blocks counted after the
	// profiling run, times the block size.
	if got.PoolTokens != 4096*16 {
		t.Errorf("PoolTokens = %d, want %d", got.PoolTokens, 4096*16)
	}
	if got.ContextSize != 32768 || got.MaxConcurrency != 64 {
		t.Errorf("got (%d, %d), want (32768, 64)", got.ContextSize, got.MaxConcurrency)
	}
	want := []string{FieldContextSize, FieldMaxConcurrency}
	if !slices.Equal(got.FromArgs, want) {
		t.Errorf("FromArgs = %v, want %v", got.FromArgs, want)
	}
}

func TestProbeVLLM_NoCacheConfigGauge(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0\n"))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineVLLM,
		Args:      parseArgs(t, config.EngineVLLM, "--max-model-len 4096"),
		EngineURL: srv.URL,
	})
	if got.Source != SourceEngineArgs {
		t.Errorf("Source = %q, want %q", got.Source, SourceEngineArgs)
	}
	if got.PoolTokens != 0 {
		t.Errorf("PoolTokens = %d, want 0: neither source knows it", got.PoolTokens)
	}
	if slices.Contains(got.FromArgs, FieldPoolTokens) {
		t.Errorf("FromArgs = %v, must not name an absent field", got.FromArgs)
	}
	if !hasWarning(got.Warnings, "cache_config_info") {
		t.Errorf("Warnings = %v, want the missing gauge named", got.Warnings)
	}
}

func TestProbe_NonLLMEngineCapacityWinsOverConfig(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/engine-capacity" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		_, _ = w.Write([]byte(`{"max_concurrency": 4}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:                     config.EngineAudio,
		EngineURL:                srv.URL,
		ConfiguredMaxConcurrency: 2,
	})
	if got.Source != SourceEngineCapacity || got.MaxConcurrency != 4 {
		t.Fatalf("got source=%q max_concurrency=%d, want engine_capacity/4",
			got.Source, got.MaxConcurrency)
	}
}

func TestProbe_NonLLMInvalidCapacityFallsBackToConfig(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
	}{
		{name: "zero", status: http.StatusOK, body: `{"max_concurrency": 0}`},
		{name: "negative", status: http.StatusOK, body: `{"max_concurrency": -2}`},
		{name: "non integer", status: http.StatusOK, body: `{"max_concurrency": 1.5}`},
		{name: "not found", status: http.StatusNotFound},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			got := Probe(context.Background(), Options{
				Kind:                     config.EngineOCR,
				EngineURL:                srv.URL,
				ConfiguredMaxConcurrency: 3,
			})
			if got.Source != SourceEngineConfig || got.MaxConcurrency != 3 {
				t.Fatalf("got source=%q max_concurrency=%d, want engine_config/3",
					got.Source, got.MaxConcurrency)
			}
			if len(got.Warnings) == 0 {
				t.Fatal("expected a probe warning")
			}
		})
	}
}

func TestProbe_NonLLMTimeoutDoesNotInventCapacity(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(50 * time.Millisecond)
		_, _ = w.Write([]byte(`{"max_concurrency": 9}`))
	}))
	defer srv.Close()

	got := Probe(context.Background(), Options{
		Kind:      config.EngineEmbed,
		EngineURL: srv.URL,
		HTTPClient: &http.Client{
			Timeout: time.Millisecond,
		},
	})
	if got.MaxConcurrency != 0 {
		t.Fatalf("MaxConcurrency = %d, want unknown", got.MaxConcurrency)
	}
	if got.Source == SourceEngineCapacity {
		t.Fatalf("Source = %q, must not report a measurement", got.Source)
	}
}

// No engine URL is the shape a test harness and an unbooted process both
// arrive in. It must not dial anything, and must still answer.
func TestProbe_NoEngineURLStillDeclares(t *testing.T) {
	got := Probe(context.Background(), Options{
		Kind: config.EngineLlamaCpp,
		Args: parseArgs(t, config.EngineLlamaCpp, "-c 8192 -np 1"),
	})
	if got.Source != SourceEngineArgs {
		t.Errorf("Source = %q, want %q", got.Source, SourceEngineArgs)
	}
	if got.ContextSize != 8192 {
		t.Errorf("ContextSize = %d, want 8192", got.ContextSize)
	}
	if len(got.Warnings) != 0 {
		t.Errorf("Warnings = %v, want none: nothing was attempted", got.Warnings)
	}
}

func hasWarning(warnings []string, substr string) bool {
	for _, w := range warnings {
		if strings.Contains(w, substr) {
			return true
		}
	}
	return false
}
