// Package controlplane exposes the HTTP control surface: liveness,
// readiness, progress, retry, config view, Prometheus metrics, and the
// static dashboard.
//
// v1.1.0 dropped the SSE `/api/events` stream in favour of dashboard 1s
// polling against `/api/progress`; `progress.Manager.Subscribe()` is
// retained for `internal/obs/subscriber.go` (Prometheus mirror).
//
// The server is a thin layer over http.ServeMux; all stateful logic lives
// in the injected dependencies (progress.Manager, obs.Metrics,
// config.Config). When the DataPlane registrar is supplied, /v1/* is
// mounted via dataplane.Mount; otherwise the server installs a 503 stub
// so probes back off instead of hammering an unwired engine.
package controlplane

import (
	"context"
	"net/http"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/handoff"
	"github.com/llm-init/llm-init/internal/obs"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/runtimecfg"
	"github.com/llm-init/llm-init/internal/version"
)

// RetryFunc is the lifecycle entry point invoked by POST /api/retry.
// May be nil in tests; the handler then returns 503 lifecycle_not_wired.
type RetryFunc func(ctx context.Context, opts RetryOptions) error

// RetryOptions are parsed from the /api/retry query string.
type RetryOptions struct {
	Force bool
	Level string // "size" | "sha256" | "remote" | "" (use VERIFY_LEVEL)
}

// ReadinessProbe reports the readiness verdict, computed by whoever owns
// both of its inputs — lifecycle.Manager.Readiness in production. A
// function because the answer changes under a running process, and a
// struct rather than a bool because /readyz has to say what is missing
// and /healthz has to show the parts it is made of.
//
// nil is fine for a test that exercises the control surface alone; the
// phase is still knowable from Manager and nothing has claimed the
// engine is up.
type ReadinessProbe func() Readiness

// Readiness mirrors lifecycle.Readiness. They are separate types for the
// same reason RetryOptions is duplicated: lifecycle must not import
// controlplane, and main translates between them.
type Readiness struct {
	Phase       progress.Phase
	EngineAlive bool
	ModelExists bool
	Ready       bool
	Reason      string
}

// Options configures Server. Only Manager and Metrics are mandatory; the
// rest fall back to safe stubs so the package can be exercised in tests
// without needing a fully wired lifecycle.
type Options struct {
	Manager progress.Manager
	Metrics *obs.Metrics

	// Config is the runtime configuration, held by the one thing allowed
	// to change it. Handlers read Server.config() rather than this field
	// so nobody accumulates a stale copy of the model card.
	Config *runtimecfg.Store

	Version version.Info

	// Readiness is the single source for /readyz and /healthz. Both used
	// to rebuild the verdict from a phase and an engine probe of their
	// own, which is how they came to disagree.
	Readiness ReadinessProbe

	Retry   RetryFunc // nil → /api/retry returns 503 lifecycle_not_wired
	NowFunc func() time.Time
	// Diag, when non-nil, is called from buildMux with the server's
	// *http.ServeMux so the diag package can install /api/diag/gpu on
	// the SAME mux as the rest of the control plane. Mirrors the
	// DataPlane registrar pattern; we use a function rather than
	// importing the diag package here because diag depends on
	// progress / config / obs which would otherwise import
	// controlplane indirectly.
	Diag DiagRegistrar

	// DataPlane is the hook through which dataplane.Mount installs the
	// /v1/* and /api/chat/completions handlers on the same ServeMux that
	// owns the control plane. When non-nil it is invoked once during
	// NewServer and the controlplane's own /v1/ stub is suppressed,
	// preventing pattern collisions on the shared ServeMux.
	DataPlane DataPlaneRegistrar

	// OllamaNative, when non-nil, installs Ollama-compatible metadata and
	// native inference routes on the same ServeMux.
	OllamaNative OllamaNativeRegistrar

	// RetryRateLimit is the steady-state token refill rate (tokens
	// per second) for POST /api/retry. Burst capacity is fixed at 5
	// in code; only the steady-state rate is operator-tunable. <= 0
	// disables the limiter (always Allow). Production callers pass
	// cfg.Runtime.RetryRateLimit; nil/zero on construction means
	// "use the package default" (1.0).
	RetryRateLimit float64
}

// DataPlaneRegistrar is the callback shape dataplane.Mount fits into.
// It receives the controlplane's mux so it can install /v1/* without
// the controlplane needing to import the dataplane package — that
// import would create a cycle, since dataplane imports adapter and the
// production wiring is controlplane → lifecycle → adapter.
type DataPlaneRegistrar func(mux *http.ServeMux)

// DiagRegistrar is the callback shape diag.Mount fits into. Same
// pattern as DataPlaneRegistrar (see godoc above) — keeps the diag
// package out of controlplane's import graph.
type DiagRegistrar func(mux *http.ServeMux)

// OllamaNativeRegistrar is the callback shape ollamanative.Mount.Apply
// fits into. Same import-cycle-avoidance rationale as the Diag and
// DataPlane registrars; main.go assembles the concrete *Mount and
// passes its Apply method-value into Options.OllamaNative.
type OllamaNativeRegistrar func(mux *http.ServeMux)

// Server is the assembled control plane. Use NewServer to construct.
type Server struct {
	opts Options
	mux  *http.ServeMux

	// retryLimiter rate-limits POST /api/retry. nil means the limiter
	// is fully disabled (RETRY_RATE_LIMIT=0). See ratelimit.go for the
	// token-bucket details.
	retryLimiter *tokenBucket

	// engineSpec memoises the engine's own endpoint report; see engine_spec.go.
	engineSpec engineSpecCache

	// engineLoadCache memoises the queue-depth reading; see engine_load.go.
	engineLoadCache engineLoadCache
}

// config is the current runtime configuration. Every handler reads it
// per request: the model card changes under a running process, and a
// copy taken once at construction would serve the boot value forever.
func (s *Server) config() config.Config { return s.opts.Config.Snapshot() }

// handoff is the RUN_DIR channel to the engine wrapper.
func (s *Server) handoff() *handoff.Tracker { return s.opts.Config.Handoff() }

// readiness is the verdict every readiness answer is derived from.
func (s *Server) readiness() Readiness {
	if s.opts.Readiness != nil {
		return s.opts.Readiness()
	}
	snap := s.opts.Manager.Snapshot()
	r := Readiness{Phase: snap.Phase, Reason: snap.LastError}
	if r.Reason == "" {
		r.Reason = "engine_not_alive"
	}
	return r
}

// NewServer wires every control-plane endpoint.
func NewServer(opts Options) *Server {
	if opts.NowFunc == nil {
		opts.NowFunc = time.Now
	}
	if opts.Config == nil {
		opts.Config = runtimecfg.New(config.Config{}, nil)
	}
	s := &Server{opts: opts}
	// Build the /api/retry limiter. Burst is fixed at 5; refill rate
	// comes from opts.RetryRateLimit (config knob RETRY_RATE_LIMIT,
	// default 1.0/s applied by config.Load). A non-positive rate
	// (operator opt-out via RETRY_RATE_LIMIT=0, documented in
	// .env.example and OpenAPI) leaves retryLimiter
	// nil; (*tokenBucket)(nil).Allow() returns true so the request
	// path remains hot without a branch. Pre-fix v1.0.5 silently
	// remapped 0 to 1.0 here, defeating the documented disable
	// semantics; tests caught it only by passing -1, which
	// config.Load itself rejects -- so the contract gap shipped.
	if rate := opts.RetryRateLimit; rate > 0 {
		s.retryLimiter = newTokenBucket(5, rate, opts.NowFunc())
	}
	s.mux = s.buildMux()
	s.publishInitialMetrics()
	return s
}

// AllPhases lists every phase string used by metrics.SetPhase. Kept here
// (rather than in package progress) because it is a controlplane concern;
// progress consumers care about transitions, not the closed-world set.
var AllPhases = []string{
	string(progress.PhaseInit),
	string(progress.PhaseDownload),
	string(progress.PhaseLoading),
	string(progress.PhaseReady),
	string(progress.PhaseDegraded),
	string(progress.PhaseFailed),
}

func (s *Server) publishInitialMetrics() {
	if s.opts.Metrics == nil {
		return
	}
	s.opts.Metrics.SetBuildInfo(s.opts.Version)
	s.opts.Metrics.SetPhase(string(s.opts.Manager.Snapshot().Phase), AllPhases)
}

// Handler returns the underlying http.Handler. ServeHTTP delegates to it.
func (s *Server) Handler() http.Handler { return s.mux }

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *Server) buildMux() *http.ServeMux {
	m := http.NewServeMux()

	// Control plane.
	m.HandleFunc("GET /livez", s.handleLivez)
	m.HandleFunc("GET /readyz", s.handleReadyz)
	m.HandleFunc("GET /healthz", s.handleHealthz)
	m.HandleFunc("GET /api/progress", s.handleProgress)
	m.HandleFunc("POST /api/retry", s.handleRetry)
	m.HandleFunc("GET /api/config", s.handleConfig)
	// /api/model-spec returns the v2 ProviderModelSpec (Router wire
	// shape) populated from MODEL_SPEC_JSON or synthesised from the
	// scalar Model envs. Always 200 once config.Load succeeds — the
	// spec is static config, not gated on lifecycle readiness, so
	// Router scraping during boot can stage its upsert before
	// /readyz flips.
	m.HandleFunc("GET /api/model-spec", s.handleModelSpec)
	// PUT replaces the spec wholesale: validate, persist to disk, and
	// serve the new value immediately. Powers the dashboard editor.
	m.HandleFunc("PUT /api/model-spec", s.handlePutModelSpec)
	m.HandleFunc("POST /api/engine/restart", s.handleEngineRestart)
	// GET reports what became of the last restart request. Confirmation
	// is asynchronous by necessity — the wrapper polls, then drains the
	// engine — so the POST cannot answer it and this is where the answer
	// lands.
	m.HandleFunc("GET /api/engine/restart", s.handleEngineRestartStatus)
	// /api/engine/load reports how many requests the engine is working
	// on and how many are waiting behind them. Only llama.cpp publishes
	// the gauges; every other kind is told so rather than given zeros.
	m.HandleFunc("GET /api/engine/load", s.handleEngineLoad)
	// /api/build-info returns llm-init's own build metadata as JSON.
	// Named with -info suffix to avoid colliding with the Ollama-native
	// /api/version route (which mirrors the upstream daemon). The
	// dashboard's Overview tab reads this to render the build banner.
	m.HandleFunc("GET /api/build-info", s.handleBuildInfo)
	// /api/endpoints catalogs every wire-callable route on this build,
	// stamped with availability based on the concrete Options. Powers
	// the dashboard's API tab and lets operators discover the surface
	// without combing through OpenAPI YAML.
	m.HandleFunc("GET /api/endpoints", s.handleEndpoints)

	// Metrics.
	if s.opts.Metrics != nil {
		m.Handle("GET /metrics", s.opts.Metrics.Handler())
	}

	// Static UI (registered by ui.go via attachUI).
	s.attachUI(m)

	// Diag endpoints (v1.0.7 Track P.1). Registered before
	// dataplane so its routes never collide with /v1/* matching;
	// the mux Pattern uses HTTP method + path so they wouldn't
	// anyway, but registering early documents the intent.
	if s.opts.Diag != nil {
		s.opts.Diag(m)
	}

	// Ollama-native metadata routes (v1.0.9 Track Q.1). Only wired
	// when cfg.Engine.Kind == EngineOllama (main.go enforces that).
	// Registered before DataPlane for the same documentation reason
	// as Diag: the routes are exact-method-path patterns and cannot
	// collide with /v1/*, but the ordering signals intent.
	if s.opts.OllamaNative != nil {
		s.opts.OllamaNative(m)
	}

	// Data plane: prefer the dataplane.Mount registrar when present.
	// In tests that only exercise the control surface (no /v1/*),
	// install a fallback stub that returns 503 + Retry-After so probes
	// back off instead of hammering an unwired engine.
	if s.opts.DataPlane != nil {
		s.opts.DataPlane(m)
	} else {
		m.HandleFunc("/v1/", s.handleDataPlaneStub)
	}

	return m
}

// handleDataPlaneStub stands in when no DataPlane registrar is wired —
// returns 503 + Retry-After while booting, or 501 once the manager
// reports ready (caller should have wired the dataplane by then).
//
// This is the one readiness-shaped condition that is deliberately not
// the shared verdict, because it answers a different question: whether
// the absence of a data plane is a wiring fact or still a timing one.
// That turns on the phase alone. The engine parts of the verdict say
// nothing about it — download-only mode has no engine to be alive, and
// gating on them would answer "not ready, retry" forever to a client
// asking about a surface that is never going to appear.
//
// Error code "not_ready" matches dataplane.writeNotReady (the post-
// registration guard) verbatim. Pre-v1.0.5 this stub returned
// "model_not_ready" while the registered guard returned "not_ready",
// so a client tied to one code missed the same operator state on the
// other end of the registration boundary. Both paths now agree, and
// the OpenAPI NotReady example documents the single canonical code.
func (s *Server) handleDataPlaneStub(w http.ResponseWriter, r *http.Request) {
	if s.opts.Manager.Snapshot().Phase == progress.PhaseReady {
		writeError(w, http.StatusNotImplemented, "dataplane_not_wired",
			"data plane registrar is not configured")
		return
	}
	w.Header().Set("Retry-After", "5")
	writeError(w, http.StatusServiceUnavailable, statusNotReady,
		"phase != ready; data plane disabled")
}
