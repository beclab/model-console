package dataplane

import (
	"net/http"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/dataplane/kvbudget"
	"github.com/llm-init/llm-init/internal/obs"
	"github.com/llm-init/llm-init/internal/progress"
)

// Options bundles every dataplane.Mount parameter so the caller can
// build it field-by-field without remembering the positional argument
// order.
type Options struct {
	// Ready is consulted on every inbound /v1/* request. It must return
	// true only when the model is actually serving (lifecycle phase=Ready
	// AND adapter.Ready reports Alive && ModelExists).
	Ready func() bool

	// Manager exposes the current phase so the 503 error envelope
	// includes meaningful client UI hints. May be nil in tests.
	Manager progress.Manager

	// Adapter supplies OpenAIHandler — the actual /v1/* router.
	Adapter adapter.Adapter

	// Config is forwarded to Adapter.OpenAIHandler. Some adapters use it
	// to dispatch behaviour by ModelType.
	Config config.Config

	// Metrics, when non-nil, is wrapped around the adapter handler so
	// every served /v1/* request is counted in
	// `llm_init_dataplane_requests_total` and timed in
	// `llm_init_dataplane_request_duration_seconds`. May be nil — the
	// middleware degrades to a passthrough.
	Metrics *obs.Metrics

	// KVBudget, when non-nil, is stitched in between NotReadyGuard and the
	// adapter handler. Build it with KVBudgetFor, which returns nil for
	// every deployment that does not need it.
	KVBudget *kvbudget.Guard

	// AllowUnsafeNilReady opts in to the legacy "nil Ready means
	// always-ready" behavior. Pre-v1.0.4 Mount silently degraded a
	// missing Ready callback to `func() bool { return true }`, so a
	// production wiring bug that forgot to pass `lc.Ready` would
	// route every /v1/* request to the engine before the lifecycle
	// reached PhaseReady — exactly the failure mode notReadyGuard
	// exists to prevent.
	//
	// Set true only in tests that genuinely don't care about
	// readiness gating (e.g. they hand-mount their own muxes for a
	// single-shot adapter handler test). Production callers must
	// supply a real Ready and leave this false; Mount panics when
	// Ready is nil and this flag is false.
	AllowUnsafeNilReady bool
}

// Mount registers the OpenAI-compatible routes on mux. Two paths are
// registered:
//
//   - "/v1/" — covers chat/completions, completions, embeddings, models,
//     responses, and any other /v1/* the adapter knows about. ServeMux pattern
//     matching means the adapter handler is reached for every nested
//     path.
//   - "/api/chat/completions" — the alias OpenWebUI posts to.
//
// Each handler is wrapped in notReadyGuard so requests during boot
// receive 503 + Retry-After:5 rather than spamming the engine, and in
// ModeGuard so a path this mode does not serve is refused here instead
// of being forwarded to an engine that has no such route.
func Mount(mux *http.ServeMux, opts Options) {
	if opts.Ready == nil {
		if !opts.AllowUnsafeNilReady {
			panic("dataplane.Mount: Options.Ready is nil; production wiring must " +
				"pass lifecycle.Manager.Ready (set AllowUnsafeNilReady: true in tests " +
				"that genuinely don't gate on readiness)")
		}
		// Test opt-in: nil Ready means "always ready". Pre-v1.0.4 this
		// was the silent default for everyone — see
		// AllowUnsafeNilReady on Options for why we now require an
		// explicit opt-in.
		opts.Ready = func() bool { return true }
	}

	// The mode is read once, here, like everything else main assembled
	// from the card at boot — which is why a card edit that changes it
	// comes back on X-Model-Spec-Pending-App-Restart rather than taking
	// effect.
	mode := opts.Config.Model.Type

	// Anthropic Messages API (Track R.1, v1.0.9). Wired BEFORE the
	// /v1/ catch-all so Go 1.22+ method-routing picks the
	// engine-agnostic /v1/messages over the OpenAI fallthrough.
	// Backend-specific translation lives in
	// Adapter.AnthropicHandler — every adapter implements it,
	// regardless of whether the upstream speaks Anthropic natively.
	// Guarded + instrumented identically to /v1/chat/completions.
	//
	// Two paths register here: /v1/messages and
	// /v1/messages/count_tokens (Stage 5 / Track R.5). They share
	// the same AnthropicHandler tree — the inner ServeMux inside
	// the adapter dispatches on the exact path. Registering both
	// at the dataplane level ensures the /v1/ catch-all doesn't
	// claim count_tokens before AnthropicHandler sees it.
	// The KV budget sits inside NotReadyGuard and outside the adapter: an
	// engine that is not ready should not be asked to tokenize anything,
	// and a request refused for want of pool room must never reach it. It
	// is left off the Anthropic pair, whose bodies are a different shape
	// than the OpenAI one this gate knows how to size; count_tokens in
	// particular carries no completion to reserve for.
	budget := func(h http.Handler) http.Handler { return h }
	if opts.KVBudget != nil {
		budget = opts.KVBudget.Wrap
	}

	ah := opts.Adapter.AnthropicHandler(opts.Config)
	anthropicGuarded := ModeGuard(mode, NotReadyGuard(opts.Ready, opts.Manager, ah))
	anthropicInstrumented := observe(opts.Metrics, anthropicGuarded)
	mux.Handle("POST /v1/messages", anthropicInstrumented)
	mux.Handle("POST /v1/messages/count_tokens", anthropicInstrumented)

	h := opts.Adapter.OpenAIHandler(opts.Config)
	guarded := ModeGuard(mode, NotReadyGuard(opts.Ready, opts.Manager, budget(h)))
	// Order matters: observe() wraps the *result* of notReadyGuard so a
	// 503 from the guard still counts as a dataplane request. If we put
	// observe() inside the guard a not-ready bounce would be invisible
	// in the metrics — exactly the case dashboards most want to see.
	instrumented := observe(opts.Metrics, guarded)
	mux.Handle("/v1/", instrumented)
	mux.Handle(pathAPIChatCompletions, instrumented)
}
