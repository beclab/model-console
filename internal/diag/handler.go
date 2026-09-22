package diag

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/obs"
)

// Handler holds the per-server state needed by the diag endpoint. It
// is constructed by Mount() and registered onto the caller's
// *http.ServeMux.
//
// GET /api/diag/gpu takes no locks and is fully concurrent.
//
// Adapter / Config / Metrics / Now are dependency-injected so tests
// can drive the handler with deterministic engines and clocks.
type Handler struct {
	Adapter adapter.Adapter
	Config  config.Config
	Metrics *obs.Metrics
	// Now is the clock used for GeneratedAt; defaults to time.Now in
	// Mount().
	Now func() time.Time
}

// errorBody mirrors controlplane.errorBody so the diag endpoints emit
// the same envelope as the rest of the control plane. Duplicated
// here rather than imported because that would create a
// controlplane->diag->controlplane import cycle once we mount on the
// controlplane mux.
type errorBody struct {
	Error errorPayload `json:"error"`
}
type errorPayload struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Mount wires the diag endpoint onto mux. Pass-through deps
// (Adapter / Metrics / Config) are stored on the returned Handler so
// the same handle is observable from test code.
//
// Mount is the only public entry point of the package; everything
// else is exported only for testability.
func Mount(mux *http.ServeMux, h *Handler) *Handler {
	if h.Now == nil {
		h.Now = time.Now
	}
	mux.HandleFunc("GET /api/diag/gpu", h.handleDiagGPU)
	return h
}

// handleDiagGPU is the cheap, lock-free GPU residency endpoint.
// Single Adapter.EngineNativeStats call (~milliseconds for env-mirror
// engines, single HTTP probe for Ollama/SGLang).
func (h *Handler) handleDiagGPU(w http.ResponseWriter, r *http.Request) {
	report, err := BuildGPUReport(r.Context(), h.Adapter, h.Config, h.Now)
	if err != nil {
		h.observeGPU("error")
		writeError(w, http.StatusServiceUnavailable, "engine_unreachable",
			"engine introspection failed: "+err.Error())
		return
	}
	// engine_native is the verbatim upstream introspection dump (up to
	// ~1 MiB of engine internals). Omit it from the default response so
	// routine dashboard polling doesn't leak engine state; callers that
	// want it opt in with ?include_engine_native=true.
	if !queryTrue(r, "include_engine_native") {
		report.EngineNative = nil
	}
	h.observeGPU("ok")
	writeJSON(w, http.StatusOK, report)
}

// queryTrue reports whether a query parameter is present with a truthy
// value (1/t/true/y/yes/on, case-insensitive). A bare ?k (no value) and
// any unparseable value are treated as false.
func queryTrue(r *http.Request, key string) bool {
	switch strings.ToLower(strings.TrimSpace(r.URL.Query().Get(key))) {
	case "1", "t", "true", "y", "yes", "on":
		return true
	default:
		return false
	}
}

func (h *Handler) observeGPU(result string) {
	if h.Metrics == nil {
		return
	}
	h.Metrics.DiagGPUCallsTotal.WithLabelValues(result).Inc()
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, errorBody{Error: errorPayload{Code: code, Message: msg}})
}
