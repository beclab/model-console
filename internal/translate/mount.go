// Package translate is an optional MTranServer-compatible side surface.
//
// It is deliberately isolated from controlplane / dataplane packages:
// main wires it only when MODEL_MODE=translate, and passes
// an optional Wrap (typically dataplane.NotReadyGuard) so this package
// does not import lifecycle plumbing.
//
// Routes:
//
//	POST /translate            {from,to,text}     → {result}
//	POST /translate/batch      {from,to,texts}    → {results}
//	POST /translate/transcript {from,to,segments,…} → {segments}
//	GET  /languages                           → {languages,pairs}
//	POST /detect           {text,minConfidence?} → {language[,confidence]}
//
// The first two are MTranServer-compatible and translate each string on its
// own. The third is not: it translates a stretch of dialogue together and
// guarantees one answer back per turn, which is what a transcript needs and
// what a page translator does not. See transcript.go.
//
// GET /languages is never wrapped: language metadata does not need the
// engine. POST routes use Wrap for readiness / shared middleware.
//
// Internally each call builds a user prompt (DefaultPrompt / DetectPrompt,
// or an injected PromptBuilder) and invokes the OpenAI chat handler
// (non-streaming). The underlying weights can be any chat-capable
// translation model.
package translate

import "net/http"

// Options configures Mount.
type Options struct {
	// Completer.Handler should be Adapter.OpenAIHandler (raw, not the
	// dataplane-guarded wrapper). Wrap below supplies readiness for
	// POST routes only.
	Completer Completer

	// Wrap, when non-nil, wraps POST routes (e.g. readiness 503).
	// Nil means passthrough — useful in unit tests.
	// GET /languages is never wrapped.
	Wrap func(http.Handler) http.Handler

	// Catalog overrides GET /languages and request validation.
	// Zero value (empty Languages+Pairs) means ResolveCatalog(Extensions, Getenv).
	Catalog Catalog

	// Extensions is model-spec Extensions (extensions.translate).
	// Used when Catalog is zero.
	Extensions map[string]any

	// Getenv reads TRANSLATE_* when Catalog is zero and Extensions
	// has no translate override. Nil uses os.Getenv.
	Getenv func(string) string

	// ContextSize is what the engine serves one conversation at a time,
	// from the model card (config.Spec.ContextSize). Only the transcript
	// route reads it, to split a group that would not fit rather than
	// having its reply cut off. Zero means the card does not pin one down
	// and the engine decides.
	//
	// Read off the card rather than probed from the engine: the card
	// derives it from the launch flags for all four engines, already
	// divides llama.cpp's -c by -np, and is the runtime source of truth
	// that a stale ENGINE_ARGS env does not override. An engine probe
	// would be llama.cpp-only and would answer the same question later.
	ContextSize int
}

// Handler holds the request handlers after Mount.
type Handler struct {
	Completer   Completer
	prompt      PromptBuilder
	catalog     Catalog
	contextSize int
}

// Mount registers the MTran-compatible routes on mux.
func Mount(mux *http.ServeMux, opts Options) *Handler {
	prompt := opts.Completer.BuildPrompt
	if prompt == nil {
		prompt = DefaultPrompt
	}
	catalog := opts.Catalog
	if len(catalog.Languages) == 0 && len(catalog.Pairs) == 0 {
		catalog = ResolveCatalog(opts.Extensions, opts.Getenv)
	}
	h := &Handler{
		Completer:   opts.Completer,
		prompt:      prompt,
		catalog:     catalog,
		contextSize: opts.ContextSize,
	}

	wrap := opts.Wrap
	if wrap == nil {
		wrap = func(next http.Handler) http.Handler { return next }
	}

	mux.HandleFunc("GET /languages", h.handleLanguages)
	mux.Handle("POST /translate", wrap(http.HandlerFunc(h.handleTranslate)))
	mux.Handle("POST /translate/batch", wrap(http.HandlerFunc(h.handleBatch)))
	mux.Handle("POST /translate/transcript", wrap(http.HandlerFunc(h.handleTranscript)))
	mux.Handle("POST /detect", wrap(http.HandlerFunc(h.handleDetect)))
	return h
}
