// Package dataplane mounts the OpenAI-compatible /v1/* surface onto the
// control plane's http.ServeMux. It owns no state of its own — readiness
// comes from a func() bool passed at Mount time, the engine handler
// comes from adapter.Adapter.OpenAIHandler — but it is the single
// authoritative place where both are stitched together.
//
// The response uses the shared API error envelope.
package dataplane

import (
	"encoding/json"
	"net/http"

	"github.com/llm-init/llm-init/internal/progress"
)

// NotReadyGuard wraps next so that requests are rejected with 503 +
// Retry-After while ready() returns false. ready is a callback rather
// than a boolean snapshot because lifecycle's PhaseReady transition can
// flip between requests; reading it once at Mount time would cache a
// stale answer.
//
// phase is read from mgr only to populate the error envelope's "phase"
// field — clients use it to render meaningful
// "Downloading" / "Verifying" UI.
//
// Exported in Track Q.1 so internal/ollamanative can reuse the same
// gate semantics (and 503 envelope shape) for /api/tags, /api/ps,
// /api/show; the original lowercase `notReadyGuard` was renamed to
// avoid two parallel implementations drifting on the wire shape.
func NotReadyGuard(ready func() bool, mgr progress.Manager, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if ready() {
			next.ServeHTTP(w, r)
			return
		}
		var phase progress.Phase
		if mgr != nil {
			phase = mgr.Snapshot().Phase
		}
		writeNotReady(w, phase)
	})
}

// codeNotReady is the operator-visible error code emitted on the 503
// envelope when the data plane is gated by !ready. controlplane has
// its own copy as statusNotReady; clients pin on the literal string
// across both packages.
const codeNotReady = "not_ready"

// writeNotReady emits the unified 503 envelope.
func writeNotReady(w http.ResponseWriter, phase progress.Phase) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", "5")
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"code":    codeNotReady,
			"message": "model not ready; data plane disabled",
			"phase":   string(phase),
		},
	})
}
