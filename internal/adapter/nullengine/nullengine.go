// Package nullengine is the no-op Adapter used in download-only mode
// (empty ENGINE_KIND). llm-init still runs the download phases, but
// there is no engine to wait for, register into, or proxy requests to —
// the model bytes land in the shared cache for a sibling app (ComfyUI,
// audio, etc.) to consume.
//
// It lives alongside the other adapter implementations so the factory
// can hand it back without a special case leaking into lifecycle: the
// state machine drives it exactly like a proxy engine, except every
// engine interaction is a no-op. The data plane is not mounted in this
// mode (main.go passes a nil DataPlane registrar), so OpenAIHandler /
// AnthropicHandler are unreachable in production; they answer 501
// defensively.
package nullengine

import (
	"context"
	"net/http"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
)

// Adapter implements adapter.Adapter with no engine behind it.
type Adapter struct{}

var _ adapter.Adapter = (*Adapter)(nil)

// New returns the download-only Adapter.
func New() *Adapter { return &Adapter{} }

// Kind is the empty engine kind, mirroring cfg.Engine.Kind.
func (*Adapter) Kind() config.EngineKind { return "" }

// WaitAlive returns immediately: there is no engine to wait for.
func (*Adapter) WaitAlive(context.Context) error { return nil }

// AliveBeforeBoot is false so lifecycle runs the download (ensure)
// first, like the proxy engines, then "comes alive" with no wait.
func (*Adapter) AliveBeforeBoot() bool { return false }

// Ready reports the model as present and the (absent) engine as alive
// so /readyz flips once lifecycle reaches PhaseReady. The phase gate in
// lifecycle.Manager.Ready keeps this from reading ready before the
// download finishes.
func (*Adapter) Ready(context.Context) (adapter.ReadyState, error) {
	return adapter.ReadyState{Alive: true, ModelExists: true}, nil
}

// This adapter does not implement adapter.ModelInstaller: there is no
// engine to make files visible to, and ollama:// sources are rejected in
// download-only mode, so nothing here was ever reachable.

// OpenAIHandler answers 501; the data plane is not mounted in this mode.
func (*Adapter) OpenAIHandler(config.Config) http.Handler { return notImplemented() }

// AnthropicHandler answers 501; the data plane is not mounted in this mode.
func (*Adapter) AnthropicHandler(config.Config) http.Handler { return notImplemented() }

// EngineNativeStats has nothing to report without an engine.
func (*Adapter) EngineNativeStats(context.Context) (adapter.NativeStats, error) {
	return adapter.NativeStats{
		Source: "unavailable",
		GPU:    adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown},
	}, nil
}

func notImplemented() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "engine not configured (download-only mode)", http.StatusNotImplemented)
	})
}
