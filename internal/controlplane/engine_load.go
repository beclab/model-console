package controlplane

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/enginemetrics"
)

// engine_load.go answers "is anybody waiting?".
//
// A single-slot llama.cpp queues rather than refuses: twenty concurrent
// requests all completed, the last of them after two minutes, and
// nothing in this process' own surface said so. /readyz was green,
// /api/progress said ready, and the only place the queue existed was two
// gauges on the engine's /metrics that nobody read. This endpoint reads
// them.

// pathEngineLoad reports how many requests the engine is working on and
// how many are behind them.
const pathEngineLoad = "/api/engine/load"

// engineLoadTTL keeps a polling dashboard off the engine's back while
// staying short enough to be worth reading: a queue turns over in
// seconds, and a value averaged over longer would describe a moment
// that has passed.
const engineLoadTTL = 2 * time.Second

const engineLoadTimeout = 2 * time.Second

// The two gauges llama.cpp's server publishes with --metrics. Their
// names are the engine's, colons and all.
const (
	metricRequestsProcessing = "llamacpp:requests_processing"
	metricRequestsDeferred   = "llamacpp:requests_deferred"
)

const (
	codeEngineLoadUnsupported = "engine_load_unsupported"
	codeEngineMetricsOff      = "engine_metrics_unavailable"
	codeEngineUnreachable     = "engine_unreachable"
)

// engineLoad is the wire shape of GET /api/engine/load.
type engineLoad struct {
	EngineKind string `json:"engine_kind"`
	// Processing is how many requests the engine has in flight;
	// Deferred is how many are queued behind them.
	Processing int `json:"processing"`
	Deferred   int `json:"deferred"`
	// Slots is how many the engine was launched to serve at once,
	// derived from engine_args by the model card rather than measured.
	// Omitted only when the width is not derivable, which on llama.cpp
	// -- the one engine this endpoint answers for -- does not happen: an
	// unset -np resolves to four rather than to silence.
	Slots int `json:"slots,omitempty"`
	// ObservedAt is when the gauges were read, which is up to
	// engineLoadTTL before the request that returns them.
	ObservedAt time.Time `json:"observed_at"`
}

type engineLoadCache struct {
	mu       sync.Mutex
	load     engineLoad
	fetched  time.Time
	lastErr  string
	lastCode string
}

// handleEngineLoad serves the queue depth, or says why it cannot.
//
// Every failure is named rather than answered with zeros. A dashboard
// reading `deferred: 0` off an engine that publishes no such gauge would
// report an idle queue on a machine with twenty requests waiting, which
// is the exact wrong answer at the exact wrong moment.
func (s *Server) handleEngineLoad(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if cfg.Engine.Kind != config.EngineLlamaCpp {
		writeError(w, http.StatusNotImplemented, codeEngineLoadUnsupported,
			"queue depth is only reported for ENGINE_KIND=llamacpp; this engine is "+
				string(cfg.Engine.Kind))
		return
	}
	if cfg.Engine.URL == "" {
		writeError(w, http.StatusServiceUnavailable, codeEngineUnreachable,
			"no engine URL configured")
		return
	}
	load, code, err := s.engineLoad(r.Context(), cfg)
	if err != nil {
		w.Header().Set("Retry-After", "2")
		writeError(w, http.StatusServiceUnavailable, code, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, load)
}

// engineLoad returns the memoised reading, refreshing it when stale. A
// failure is memoised too: an engine that is down is down for the next
// caller a moment later as well, and re-dialling it once per request
// turns a dashboard into a load generator against a struggling process.
func (s *Server) engineLoad(ctx context.Context, cfg config.Config) (engineLoad, string, error) {
	s.engineLoadCache.mu.Lock()
	defer s.engineLoadCache.mu.Unlock()
	now := s.opts.NowFunc()
	if !s.engineLoadCache.fetched.IsZero() && now.Sub(s.engineLoadCache.fetched) < engineLoadTTL {
		if s.engineLoadCache.lastErr != "" {
			return engineLoad{}, s.engineLoadCache.lastCode, errors.New(s.engineLoadCache.lastErr)
		}
		return s.engineLoadCache.load, "", nil
	}
	load, code, err := fetchEngineLoad(ctx, cfg.Engine.URL)
	s.engineLoadCache.fetched = now
	if err != nil {
		s.engineLoadCache.lastErr = err.Error()
		s.engineLoadCache.lastCode = code
		return engineLoad{}, code, err
	}
	load.EngineKind = string(cfg.Engine.Kind)
	// Derived per reading rather than read off the card: the width is
	// deliberately not a card field, because an older llm-init refuses a
	// top-level field it does not know and the card outlives the app.
	// Zero stays zero, which `omitempty` drops.
	load.Slots, _ = config.DeriveMaxConcurrency(cfg.Engine.Kind, cfg.Engine.Args)
	load.ObservedAt = now
	s.engineLoadCache.load = load
	s.engineLoadCache.lastErr = ""
	s.engineLoadCache.lastCode = ""
	return load, "", nil
}

// fetchEngineLoad reads the two gauges off the engine's /metrics.
//
// The two failures it distinguishes are the two an operator acts on
// differently: a transport error means the engine is not answering, and
// a /metrics that answers without the gauges means llama.cpp was
// launched without --metrics — the engine is fine and the flag is
// missing.
func fetchEngineLoad(ctx context.Context, engineURL string) (engineLoad, string, error) {
	ctx, cancel := context.WithTimeout(ctx, engineLoadTimeout)
	defer cancel()
	url := strings.TrimRight(engineURL, "/") + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return engineLoad{}, codeEngineUnreachable, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return engineLoad{}, codeEngineUnreachable,
			fmt.Errorf("engine /metrics unreachable: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return engineLoad{}, codeEngineMetricsOff,
			fmt.Errorf("engine /metrics returned %s; llama.cpp publishes it only when launched with --metrics",
				resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return engineLoad{}, codeEngineUnreachable,
			fmt.Errorf("read engine /metrics: %w", err)
	}
	fams, err := enginemetrics.Parse(body)
	if err != nil {
		return engineLoad{}, codeEngineMetricsOff, err
	}
	processing := enginemetrics.Gauge(fams, metricRequestsProcessing)
	deferredG := enginemetrics.Gauge(fams, metricRequestsDeferred)
	if processing == nil || deferredG == nil {
		return engineLoad{}, codeEngineMetricsOff,
			fmt.Errorf("engine /metrics carries no %s / %s gauge",
				metricRequestsProcessing, metricRequestsDeferred)
	}
	return engineLoad{
		Processing: int(*processing),
		Deferred:   int(*deferredG),
	}, "", nil
}
