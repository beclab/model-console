package lifecycle

import "github.com/llm-init/llm-init/internal/progress"

// Readiness is the one answer to "may this process serve a request", and
// the parts that answer is made of.
//
// It exists because that answer used to be spelled out separately at
// every place that needed it — the /v1 gate, /readyz, /healthz's `ready`
// field and /healthz's `status` — and the spellings were not the same.
// `status` left ModelExists out, so an Ollama daemon that came back
// without the model it had been given was reported `ok` for a whole
// grace window while every /v1 request was refused with 503.
//
// Both inputs are read once, here, so everything derived from one
// Readiness describes one instant. Reading the phase and the probe cache
// separately at each call site let /readyz and /healthz disagree about
// the same moment for no reason a reader could see.
type Readiness struct {
	// Phase is the lifecycle's own state, and the only readiness word
	// with a published vocabulary (Router stores it verbatim).
	Phase progress.Phase

	// EngineAlive and ModelExists are the last answer the engine gave.
	// They are a cache: a probe that could not be made leaves the
	// previous answer standing, which is what the health loop's grace
	// window is built on.
	EngineAlive bool
	ModelExists bool

	// Ready is the verdict. Nothing else may recompute it.
	Ready bool

	// Reason names what is missing, and is empty when Ready. The
	// lifecycle's own recorded error wins when there is one: "hf
	// download failed: …" tells an operator what to do, where
	// "engine_not_alive" only describes the consequence.
	Reason string
}

// Readiness reports the current verdict. Cheap enough to call per
// request: one snapshot read and one atomic load.
func (m *Manager) Readiness() Readiness {
	snap := m.opts.Manager.Snapshot()
	r := Readiness{Phase: snap.Phase}
	if rs := m.lastReady.Load(); rs != nil {
		r.EngineAlive = rs.Alive
		r.ModelExists = rs.ModelExists
	}
	r.Ready = r.Phase == progress.PhaseReady && r.EngineAlive && r.ModelExists
	if r.Ready {
		return r
	}
	r.Reason = snap.LastError
	if r.Reason == "" {
		switch {
		case !r.EngineAlive:
			r.Reason = "engine_not_alive"
		case !r.ModelExists:
			r.Reason = "model_missing"
		default:
			r.Reason = "phase != ready"
		}
	}
	return r
}

// Ready is the boolean form dataplane.Mount and ollamanative gate on.
func (m *Manager) Ready() bool { return m.Readiness().Ready }
