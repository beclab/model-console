package controlplane

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

const llamacppMetricsBody = `# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# HELP llamacpp:requests_deferred Number of requests deferred.
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 19
`

// metricsEngine stands in for a llama.cpp server and counts how many
// times /metrics was actually read, which is what the TTL test needs.
func metricsEngine(t *testing.T, body string, status int) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		hits.Add(1)
		if status != http.StatusOK {
			w.WriteHeader(status)
			return
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func decodeEngineLoad(t *testing.T, body []byte) engineLoad {
	t.Helper()
	var got engineLoad
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v\nbody=%s", err, body)
	}
	return got
}

func errCode(t *testing.T, body []byte) string {
	t.Helper()
	var got errorBody
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v\nbody=%s", err, body)
	}
	return got.Error.Code
}

func TestEngineLoad_ReportsQueueDepth(t *testing.T) {
	t.Parallel()
	engine, _ := metricsEngine(t, llamacppMetricsBody, http.StatusOK)
	s := endpointsFixture(t, config.EngineLlamaCpp, func(o *Options) {
		withCfg(o, func(c *config.Config) {
			c.Engine.URL = engine.URL
			c.Spec.EngineArgs = "-c 8192 -np 1"
			if err := config.ApplyEngineArgsFromSpec(c); err != nil {
				t.Fatalf("ApplyEngineArgsFromSpec: %v", err)
			}
		})
	})
	r := get(t, s, pathEngineLoad)
	if r.Status != http.StatusOK {
		t.Fatalf("status = %d, body=%s", r.Status, r.Body)
	}
	got := decodeEngineLoad(t, r.Body)
	if got.Processing != 1 || got.Deferred != 19 {
		t.Errorf("processing=%d deferred=%d, want 1 / 19", got.Processing, got.Deferred)
	}
	// Slots is the declaration, not a gauge: the engine publishes no
	// such metric, so it is derived from the launch flags.
	if got.Slots != 1 {
		t.Errorf("slots = %d, want the -np the engine was launched with (1)", got.Slots)
	}
	if got.EngineKind != string(config.EngineLlamaCpp) {
		t.Errorf("engine_kind = %q", got.EngineKind)
	}
	if got.ObservedAt.IsZero() {
		t.Error("observed_at unset; a cached reading has to say when it was taken")
	}
}

// An absent -np is not an absent width. The server assigns four slots to
// it, so reporting nothing here would leave four callers being served at
// once looking like one slow model — the exact reading this endpoint
// exists to correct. Omission is still the answer for a width nobody can
// derive; it is just not reachable on llama.cpp, which is the only engine
// this endpoint answers for.
func TestEngineLoad_AbsentSlotCountIsFourNotSilence(t *testing.T) {
	t.Parallel()
	engine, _ := metricsEngine(t, llamacppMetricsBody, http.StatusOK)
	s := endpointsFixture(t, config.EngineLlamaCpp, func(o *Options) {
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
	})
	r := get(t, s, pathEngineLoad)
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(r.Body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := raw["slots"]; !ok {
		t.Fatalf("slots omitted; an unset -np resolves to four: %s", r.Body)
	}
	if got := decodeEngineLoad(t, r.Body); got.Slots != 4 {
		t.Errorf("slots = %d, want 4", got.Slots)
	}
}

func TestEngineLoad_NonLlamacppSaysSoRatherThanZero(t *testing.T) {
	t.Parallel()
	for _, kind := range []config.EngineKind{config.EngineVLLM, config.EngineSGLang, config.EngineOllama} {
		s := endpointsFixture(t, kind, nil)
		r := get(t, s, pathEngineLoad)
		if r.Status != http.StatusNotImplemented {
			t.Errorf("%s: status = %d, want 501; body=%s", kind, r.Status, r.Body)
			continue
		}
		if code := errCode(t, r.Body); code != codeEngineLoadUnsupported {
			t.Errorf("%s: code = %q, want %q", kind, code, codeEngineLoadUnsupported)
		}
	}
}

// llama.cpp serves /metrics only when launched with --metrics. That is
// the flag being absent, not the engine being down, and an operator
// fixes the two differently.
func TestEngineLoad_MetricsOffIsNotUnreachable(t *testing.T) {
	t.Parallel()
	engine, _ := metricsEngine(t, "", http.StatusNotFound)
	s := endpointsFixture(t, config.EngineLlamaCpp, func(o *Options) {
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
	})
	r := get(t, s, pathEngineLoad)
	if r.Status != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body=%s", r.Status, r.Body)
	}
	if code := errCode(t, r.Body); code != codeEngineMetricsOff {
		t.Errorf("code = %q, want %q", code, codeEngineMetricsOff)
	}
	if r.Header.Get("Retry-After") == "" {
		t.Error("no Retry-After; a client with no interval hammers a struggling engine")
	}
}

// A /metrics that answers but carries neither gauge is the same operator
// problem as one that 404s, and must not be answered with an idle queue.
func TestEngineLoad_MissingGaugesAreNotAnIdleQueue(t *testing.T) {
	t.Parallel()
	engine, _ := metricsEngine(t, "# HELP other something\n# TYPE other gauge\nother 3\n", http.StatusOK)
	s := endpointsFixture(t, config.EngineLlamaCpp, func(o *Options) {
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
	})
	r := get(t, s, pathEngineLoad)
	if r.Status != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503; body=%s", r.Status, r.Body)
	}
	if code := errCode(t, r.Body); code != codeEngineMetricsOff {
		t.Errorf("code = %q, want %q", code, codeEngineMetricsOff)
	}
}

func TestEngineLoad_CachesWithinTTL(t *testing.T) {
	t.Parallel()
	engine, hits := metricsEngine(t, llamacppMetricsBody, http.StatusOK)
	now := time.Now()
	s := endpointsFixture(t, config.EngineLlamaCpp, func(o *Options) {
		o.NowFunc = func() time.Time { return now }
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
	})
	for i := 0; i < 3; i++ {
		if r := get(t, s, pathEngineLoad); r.Status != http.StatusOK {
			t.Fatalf("call %d: status = %d, body=%s", i, r.Status, r.Body)
		}
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("engine /metrics read %d times, want 1 within the TTL", got)
	}
	now = now.Add(engineLoadTTL + time.Millisecond)
	if r := get(t, s, pathEngineLoad); r.Status != http.StatusOK {
		t.Fatalf("after TTL: status = %d, body=%s", r.Status, r.Body)
	}
	if got := hits.Load(); got != 2 {
		t.Errorf("engine /metrics read %d times after the TTL expired, want 2", got)
	}
}

// A failing engine is memoised too. Without that, a dashboard polling
// every second turns into a second load generator against the process
// that is already in trouble.
func TestEngineLoad_CachesFailures(t *testing.T) {
	t.Parallel()
	engine, hits := metricsEngine(t, "", http.StatusNotFound)
	now := time.Now()
	s := endpointsFixture(t, config.EngineLlamaCpp, func(o *Options) {
		o.NowFunc = func() time.Time { return now }
		withCfg(o, func(c *config.Config) { c.Engine.URL = engine.URL })
	})
	for i := 0; i < 3; i++ {
		r := get(t, s, pathEngineLoad)
		if r.Status != http.StatusServiceUnavailable {
			t.Fatalf("call %d: status = %d, body=%s", i, r.Status, r.Body)
		}
		if code := errCode(t, r.Body); code != codeEngineMetricsOff {
			t.Fatalf("call %d: code = %q, want the first failure's code", i, code)
		}
	}
	if got := hits.Load(); got != 1 {
		t.Errorf("engine /metrics read %d times, want 1 within the TTL", got)
	}
}
