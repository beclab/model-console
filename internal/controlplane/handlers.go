package controlplane

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/llm-init/llm-init/internal/progress"
)

// Content-Type and JSON status-key literals shared across handlers and
// tests. Single-source-of-truth so a wire-shape rename is a one-line
// change and goconst picks no fight with the ~15 call sites that key
// on "application/json" / "status" across the controlplane package.
const (
	contentTypeJSON = "application/json"
	contentTypeHTML = "text/html"

	// JSON envelope key names used by /livez, /readyz, /healthz, and
	// /api/retry response bodies.
	jsonKeyStatus = "status"
	jsonKeyPhase  = "phase"
	jsonKeyReady  = "ready"
	jsonKeyModel  = "model"
	jsonKeyReason = "reason"

	// status / code values emitted by readiness handlers. Both tests
	// (server_test.go) and dataplane.writeNotReady key on these
	// strings; dataplane has its own copy of "not_ready" because the
	// two packages don't share a constant module.
	statusReady    = "ready"
	statusNotReady = "not_ready"
	boolTrue       = "true"
)

// errorBody is the unified error envelope.
type errorBody struct {
	Error errorPayload `json:"error"`
}

type errorPayload struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", contentTypeJSON)
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, errorBody{Error: errorPayload{Code: code, Message: msg}})
}

// handleLivez always returns 200; the goroutine is alive if it can serve.
func (s *Server) handleLivez(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{jsonKeyStatus: "ok"})
}

// handleReadyz answers 200 only when the shared verdict says ready.
// Loaded is intentionally not required (Q-5: keep readyz a "light"
// probe).
func (s *Server) handleReadyz(w http.ResponseWriter, _ *http.Request) {
	rd := s.readiness()
	if rd.Ready {
		cfg := s.config()
		writeJSON(w, http.StatusOK, map[string]string{
			jsonKeyStatus: statusReady,
			jsonKeyModel:  cfg.Model.Name,
			"engine":      string(cfg.Engine.Kind),
		})
		return
	}
	writeJSON(w, http.StatusServiceUnavailable, map[string]string{
		jsonKeyStatus: statusNotReady,
		jsonKeyPhase:  string(rd.Phase),
		jsonKeyReason: rd.Reason,
	})
}

// handleHealthz returns the composite document described in
// the /healthz response.
//
// status is derived from the same verdict as ready, so the two cannot
// contradict each other. They used to: status only asked whether the
// engine was alive, so an engine that was up without the model it had
// been given read as "ok" beside "ready": false, for as long as the
// health loop's grace window — while every /v1 request was refused.
func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	rd := s.readiness()

	healthStatus := "ok"
	switch {
	case rd.Phase == progress.PhaseFailed:
		healthStatus = "down"
	case !rd.Ready:
		healthStatus = "degraded"
	}

	// last_verify_at is RFC3339 string when populated, else nil so the
	// JSON shape is "ISO timestamp or null". The
	// pointer-bool last_verify_ok serialises as null when nil, true/false
	// when set.
	snap := s.opts.Manager.Snapshot()
	var lastVerify any
	if !snap.LastVerifyAt.IsZero() {
		lastVerify = snap.LastVerifyAt.UTC().Format(time.RFC3339)
	}
	body := map[string]any{
		jsonKeyStatus:    healthStatus,
		jsonKeyReady:     rd.Ready,
		"engine_alive":   rd.EngineAlive,
		"model_exists":   rd.ModelExists,
		jsonKeyPhase:     rd.Phase,
		"last_verify":    lastVerify,
		"last_verify_ok": snap.LastVerifyOK, // *bool encodes as null when nil
	}
	writeJSON(w, http.StatusOK, body)
}

// handleProgress returns the latest snapshot.
func (s *Server) handleProgress(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.opts.Manager.Snapshot())
}

// handleConfig returns the redacted Config, ready to dump in support tickets.
func (s *Server) handleConfig(w http.ResponseWriter, _ *http.Request) {
	cfg := s.config()
	writeJSON(w, http.StatusOK, cfg.Redacted())
}

// handleBuildInfo emits llm-init's build metadata. Distinct from the
// Ollama-native /api/version (which proxies the upstream daemon) —
// this one always returns this binary's build metadata so dashboards
// can show llm-init's own version irrespective of engine kind. Pre-
// v1.0.9 this information was only reachable through the /metrics
// build_info gauge, which is noisy for HTML dashboards; v1.0.9
// exposes it as JSON for the Overview tab.
func (s *Server) handleBuildInfo(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.opts.Version)
}

// handleRetry parses ?force / ?level and dispatches to RetryFunc with a
// fresh context. The lifecycle's RetryFunc is non-blocking (it nudges
// the ensure backoff via channel and returns immediately), so this
// handler stays synchronous and reports outcomes via /api/progress.
//
// v1.0.5 throttling: pre-fix /api/retry had no rate limit. The handler
// is cheap on its own (just channel nudge) but every accepted call can
// trigger an ensure pass that re-runs source.Resolve and (for HF
// sources post v1.1.0) spawns the Python wrapper subprocess. A caller
// spamming retry at line speed could starve real ensures and bleed CPU.
// A single token bucket (default 1/s burst 5, RETRY_RATE_LIMIT env, 0
// disables) protects the path; overflow returns 429 + Retry-After: 1 +
// code "rate_limited" and is counted in llm_init_retry_throttled_total
// for operator visibility.
func (s *Server) handleRetry(w http.ResponseWriter, r *http.Request) {
	if s.opts.Retry == nil {
		writeError(w, http.StatusServiceUnavailable, "lifecycle_not_wired",
			"retry endpoint requires a wired lifecycle")
		return
	}
	if !s.retryLimiter.Allow() {
		if s.opts.Metrics != nil {
			s.opts.Metrics.RetryThrottledTotal.Inc()
		}
		w.Header().Set("Retry-After", "1")
		// Generic wording: a previous version hard-coded "1/s burst 5
		// by default" which silently lied whenever the operator
		// customised RETRY_RATE_LIMIT. The configured rate is
		// reflected in llm_init_retry_throttled_total + Retry-After
		// already; the body just needs to tell the client it is
		// throttled and they should back off.
		writeError(w, http.StatusTooManyRequests, "rate_limited",
			"too many retry calls; back off and respect Retry-After")
		return
	}
	q := r.URL.Query()
	// ?level= is forwarded verbatim; the lifecycle decides what it
	// means (lifecycle.passLevel). The handler deliberately does not
	// enforce the enum: an unknown value falls back to VERIFY_LEVEL
	// there, and rejecting it here would 400 dashboards still sending
	// the pre-v1.1.0 query.
	opts := RetryOptions{
		Force: q.Get("force") == boolTrue,
		Level: q.Get("level"),
	}
	previous := s.opts.Manager.Snapshot().Phase

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	if err := s.opts.Retry(ctx, opts); err != nil {
		writeError(w, http.StatusInternalServerError, "retry_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{
		jsonKeyStatus:    "retry_triggered",
		"previous_phase": previous,
	})
}
