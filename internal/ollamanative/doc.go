// Package ollamanative mounts Ollama-compatible metadata routes on
// llm-init's HTTP surface — /api/version, /api/tags, /api/ps,
// /api/show — so Ollama-native clients (OpenWebUI, Chatbox, LobeChat,
// `ollama` CLI, monitoring scripts) succeed at their startup handshake
// without modification. Without these routes those clients probe
// /api/version and /api/tags before any /v1/* request and mark the
// endpoint "unavailable" when the probes 404.
//
// The package is wire-shape compatible with the upstream Ollama
// daemon: response field names, nesting, and types match what the
// daemon would return verbatim. Filtering and renaming are applied so
// llm-init still presents itself as a single-model abstraction:
//
//   - /api/version  — byte-for-byte passthrough of the upstream
//     daemon's /api/version response, including the upstream status
//     code. The route is NOT gated on lifecycle readiness: clients
//     handshake on it before /readyz turns green and we want them to
//     learn the engine flavour as early as possible.
//   - /api/tags     — calls upstream, filters to the one model entry
//     whose name matches Source.OllamaModel (the daemon tag), rewrites
//     that entry's `name` field to Model.Name (the OpenAI alias), and
//     returns the upstream JSON shape. If no entry matches the result
//     is {"models":[]} — still a valid Ollama-native shape.
//   - /api/ps       — same filter/rewrite rules as /api/tags applied
//     to the PSModel shape (which carries both `name` and `model`
//     fields; both are rewritten). Size / size_vram are NOT modified
//     because front-ends use them for VRAM accounting.
//   - /api/show     — validates the request body's `name` (or `model`)
//     field is either Model.Name (alias) or Source.OllamaModel
//     (upstream tag); returns 404 + an Ollama-native error envelope
//     (`{"error":"model '<name>' not found"}`) when neither matches.
//     On a valid name the body is rewritten so the upstream name is
//     forwarded, then the upstream response is byte-copied to the
//     caller (no rewrite — the user opted into true-passthrough so
//     introspection tools see exactly what the daemon returned).
//
// /api/tags, /api/ps, /api/show are wrapped in dataplane.NotReadyGuard
// so they return the same 503 envelope as /v1/* until the lifecycle
// reaches PhaseReady. /api/version is intentionally NOT gated: clients
// poll it during boot to choose a transport profile, and answering 503
// would defeat that handshake.
//
// Mount is invoked from cmd/llm-init/main.go ONLY when
// cfg.Engine.Kind == config.EngineOllama. For vLLM / llama.cpp /
// SGLang backends these routes are not registered on the mux at all —
// requests return Go's default 404 from the unmatched ServeMux. The
// engine-conditional mount lives in main, not here, because this
// package has no dependency on factory.New and stays trivially
// importable in tests.
//
// The handlers below define the supported Ollama-native schema
// reference.
package ollamanative
