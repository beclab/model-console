package controlplane

import (
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/handoff"
	"github.com/llm-init/llm-init/internal/runtimecfg"
)

// maxModelSpecBytes caps the PUT body. A model-spec.json is a small
// capability descriptor; anything past this is either a mistake or abuse.
const maxModelSpecBytes = 256 << 10 // 256 KiB

// handleModelSpec serves the live v2 ProviderModelSpec describing the
// deployed model — the wire shape Router consumes for
// `provider_models.model_spec`. Seeded from config.Load at boot and
// replaceable at runtime via PUT, so it comes from the store rather than
// from a copy taken at construction.
//
// Not gated on lifecycle readiness: the persisted spec answers during boot.
// Once the sibling engine is alive, its valid v1 engine spec overlays dynamic
// extension namespaces such as creative operations. Model Console retains
// ownership of extensions.capacity.
func (s *Server) handleModelSpec(w http.ResponseWriter, _ *http.Request) {
	spec := s.config().Spec
	report := s.engineEndpoints()
	if report.authoritative && report.modelMatches && len(report.extensions) > 0 {
		extensions := make(map[string]any, len(spec.Extensions)+len(report.extensions))
		for key, value := range spec.Extensions {
			extensions[key] = value
		}
		for key, value := range report.extensions {
			extensions[key] = value
		}
		spec.Extensions = extensions
	}
	writeJSON(w, http.StatusOK, spec)
}

// handlePutModelSpec replaces the served spec wholesale. The body is the
// full model-spec.json; it is validated with the same rules as the boot
// loader (config.ParseModelSpecBytes: DisallowUnknownFields + mode /
// pricing / parameter_rules checks). Everything after parsing belongs to
// runtimecfg.Store, which normalises the card against the engine,
// persists it, publishes the launch flags and swaps memory as one
// serialised step; this handler only turns the outcome into HTTP.
//
// Some fields are stored and served without taking effect: mode chose
// which routes were mounted, supports_reasoning the thinking gate, and
// extensions.translate the language catalogue, all when the process
// started. Those come back in headerPendingAppRestart rather than being
// rejected — Router's copy of the card should match what the operator
// wrote — and the model name is the exception the store refuses outright,
// because a half-applied rename is an outage rather than a stale field.
func (s *Server) handlePutModelSpec(w http.ResponseWriter, r *http.Request) {
	data, err := io.ReadAll(io.LimitReader(r.Body, maxModelSpecBytes+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", "read body: "+err.Error())
		return
	}
	if len(data) > maxModelSpecBytes {
		writeError(w, http.StatusRequestEntityTooLarge, "payload_too_large",
			"model-spec body exceeds 256 KiB")
		return
	}

	spec, err := config.ParseModelSpecBytes(data, "PUT /api/model-spec")
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_model_spec", err.Error())
		return
	}

	applied, err := s.opts.Config.ApplySpec(spec)
	switch {
	case errors.Is(err, runtimecfg.ErrInvalidSpec):
		writeError(w, http.StatusBadRequest, "invalid_model_spec", err.Error())
		return
	case err != nil:
		writeError(w, http.StatusInternalServerError, "persist_failed", err.Error())
		return
	}

	h := s.handoff()
	outcome := restartNotNeeded
	receipt := handoff.Receipt{Supervision: h.Supervision()}
	if applied.EngineArgsChanged && s.config().Engine.Kind != "" {
		if h.RunDir() == "" {
			// Same contract as POST /api/engine/restart: cannot signal.
			writeError(w, http.StatusServiceUnavailable, "run_dir_unset",
				"RUN_DIR is empty; model-spec saved but engine restart was not signaled")
			return
		}
		if receipt, err = h.Request(); err != nil {
			writeError(w, http.StatusInternalServerError, "restart_failed", err.Error())
			return
		}
		outcome = restartSignaled
	}

	writeRestartHeaders(w, outcome, receipt)
	if len(applied.PendingAppRestart) > 0 {
		w.Header().Set(headerPendingAppRestart, strings.Join(applied.PendingAppRestart, ","))
	}
	writeJSON(w, http.StatusOK, applied.Spec)
}
