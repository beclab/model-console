package controlplane

import (
	"net/http"

	"github.com/llm-init/llm-init/internal/handoff"
)

// Restart receipt headers. They ride on PUT /api/model-spec, whose body
// is the card itself and cannot carry anything else: Router stores that
// body verbatim and PUTs it back on the next edit, and llm-init rejects
// unknown fields — so a receipt field in the body would come back as a
// validation failure on the following write.
const (
	// headerRestart is the synchronous fact: whether a signal went out.
	//   not-needed  the card changed nothing the engine launches with
	//   signaled    the generation file was written
	//   unavailable there is no run dir to write it to
	headerRestart = "X-Engine-Restart"

	// headerRestartGeneration identifies the request, so a caller can
	// tie a later GET /api/engine/restart to the signal it sent.
	headerRestartGeneration = "X-Engine-Restart-Generation"

	// headerRestartSupervision carries prior evidence, not a claim about
	// this request: "confirmed" means a restart signal has been observed
	// to stop the engine before, "unknown" means it never has.
	headerRestartSupervision = "X-Engine-Restart-Supervision"

	// headerRestarted predates the receipt and reads as a completed
	// restart, which no synchronous answer can support: the wrapper polls
	// on an interval and then lets the engine drain. It is now set only
	// when a supervisor has been observed acting on an earlier signal, so
	// a client that reads nothing else is told "true" where a relaunch is
	// established and never where none has been seen. Prefer the three
	// headers above.
	headerRestarted = "X-Engine-Restarted"

	// headerPendingAppRestart lists card fields that were stored and are
	// now served but that the running process cannot act on, because it
	// derived something from them when it started: mode decided which
	// routes exist, supports_reasoning the thinking gate,
	// extensions.translate the language catalogue. Comma-separated,
	// absent when there are none.
	//
	// This is a different restart from the three headers above. Those
	// concern the engine sidecar, which a wrapper relaunches in place.
	// This one needs the application itself to start again, which
	// nothing inside the container can do — it is the operator's move,
	// and the point of saying so is that the card and the behaviour
	// disagree until they make it.
	headerPendingAppRestart = "X-Model-Spec-Pending-App-Restart"
)

const (
	restartNotNeeded   = "not-needed"
	restartSignaled    = "signaled"
	restartUnavailable = "unavailable"
)

// writeRestartHeaders stamps a receipt onto a response whose body belongs
// to something else.
func writeRestartHeaders(w http.ResponseWriter, outcome string, r handoff.Receipt) {
	h := w.Header()
	h.Set(headerRestart, outcome)
	h.Set(headerRestartSupervision, string(r.Supervision))
	if r.Generation != "" {
		h.Set(headerRestartGeneration, r.Generation)
	}
	if outcome == restartSignaled && r.Supervision == handoff.SupervisionConfirmed {
		h.Set(headerRestarted, "true")
	}
}

// handleEngineRestart bumps the generation file so a supervising wrapper
// stops its child and relaunches on the current engine_args handoff.
//
// 200 means the signal is on disk. It does not mean the engine restarted,
// and the response says so: `restart.supervision` reports whether a
// supervisor has ever been observed acting on this signal, and the
// observation of this particular request lands on
// GET /api/engine/restart once it concludes. Apps whose engine runs under
// a plain exec — no supervise loop — settle at "no-supervisor" there,
// which is the only way to learn that from outside the container.
func (s *Server) handleEngineRestart(w http.ResponseWriter, _ *http.Request) {
	if s.config().Engine.Kind == "" {
		writeError(w, http.StatusBadRequest, "no_engine",
			"ENGINE_KIND is empty; nothing to restart")
		return
	}
	h := s.handoff()
	runDir := h.RunDir()
	if runDir == "" {
		writeError(w, http.StatusServiceUnavailable, "run_dir_unset",
			"RUN_DIR is empty; cannot signal the engine wrapper")
		return
	}
	receipt, err := h.Request()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "restart_failed", err.Error())
		return
	}
	writeRestartHeaders(w, restartSignaled, receipt)
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":      true,
		"run_dir": runDir,
		"restart": receipt,
	})
}

// handleEngineRestartStatus reports what became of the last restart
// request, including the part that is only knowable after the requesting
// call returned.
func (s *Server) handleEngineRestartStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.handoff().Status())
}
