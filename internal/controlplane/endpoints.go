package controlplane

import (
	"net/http"
	"sort"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/moderoutes"
)

// EndpointInfo is one row in the `GET /api/endpoints` response.
//
// It tells dashboards (and curious operators) what HTTP surfaces this
// build actually exposes given the current wiring + ENGINE_KIND. The
// "available" flag distinguishes "the binary supports this path" from
// "the path is reachable in this process" — e.g. /api/diag/* is
// compiled in but only wired when Options.Diag is non-nil; the four
// Ollama-native passthroughs only register when ENGINE_KIND=ollama.
type EndpointInfo struct {
	Method           string               `json:"method"`
	Path             string               `json:"path"`
	Category         string               `json:"category"`
	Group            string               `json:"group,omitempty"`
	Description      string               `json:"description"`
	Available        bool                 `json:"available"`
	Reasons          []string             `json:"reasons,omitempty"`
	CurlHint         string               `json:"curl_hint,omitempty"`
	AsyncSupported   *bool                `json:"async_supported,omitempty"`
	OperationID      string               `json:"operation_id,omitempty"`
	Protocol         string               `json:"protocol,omitempty"`
	Transport        string               `json:"transport,omitempty"`
	SyncSupported    *bool                `json:"sync_supported,omitempty"`
	Streaming        bool                 `json:"streaming,omitempty"`
	RequiredSupports []string             `json:"required_supports,omitempty"`
	InputModalities  []string             `json:"input_modalities,omitempty"`
	OutputModalities []string             `json:"output_modalities,omitempty"`
	OutputFormats    []string             `json:"output_formats,omitempty"`
	SampleRates      []int                `json:"sample_rates,omitempty"`
	Parameters       []OperationParameter `json:"parameters,omitempty"`
	Limits           map[string]any       `json:"limits,omitempty"`
	ResourceScope    string               `json:"resource_scope,omitempty"`
	Extensions       map[string]any       `json:"extensions,omitempty"`
}

// OperationParameter is the deliberately small, UI-safe parameter vocabulary
// engines may advertise. Protocol adapters still own request serialization.
type OperationParameter struct {
	Name     string   `json:"name"`
	Type     string   `json:"type"`
	Required bool     `json:"required,omitempty"`
	Default  any      `json:"default,omitempty"`
	Minimum  *float64 `json:"minimum,omitempty"`
	Maximum  *float64 `json:"maximum,omitempty"`
	Enum     []string `json:"enum,omitempty"`
	Unit     string   `json:"unit,omitempty"`
}

// EndpointList is the wire shape of GET /api/endpoints.
type EndpointList struct {
	SchemaVersion int            `json:"schema_version"`
	EngineKind    string         `json:"engine_kind"`
	Endpoints     []EndpointInfo `json:"endpoints"`
}

// Endpoint categories. Stable strings: dashboards group rows by this
// field, so renaming is a wire-shape break.
const (
	categoryControl   = "control"
	categoryHealth    = "health"
	categoryOpenAI    = "data-plane-openai"
	categoryAnthropic = "anthropic"
	categoryOllama    = "ollama-native"
	categoryTranslate = "translate"
	categoryDiag      = "diag"
	categoryMetrics   = "metrics"
	categoryUI        = "ui"
)

const (
	groupTranslate = "Translate"
	groupMusic     = "Music"
)

// HTTP method literals — pulled into package constants so goconst stops
// flagging them at every catalog row. We don't try to use net/http's
// MethodGet / MethodPost because those are full Go identifiers
// ("GET", "POST" as exported constants) and would clutter the table
// literal without saving any bytes; the local two-letter names are
// equivalent and read better at the call site.
const (
	mGET    = "GET"
	mPOST   = "POST"
	mPUT    = "PUT"
	mDELETE = "DELETE"

	// Methods the catalog itself never serves but an engine spec may
	// declare — see engineSpecMethods.
	mPATCH   = "PATCH"
	mHEAD    = "HEAD"
	mOPTIONS = "OPTIONS"
	mWS      = "WS"
)

// Repeated path/group literals. Same reason: the goconst threshold is
// 3, and we already hit it on several strings shared across catalog
// rows (the same path can appear in description and curl_hint, the
// same group label across multiple rows).
const (
	pathChatCompletions = "/v1/chat/completions"
	pathResponses       = "/v1/responses"
	pathMessages        = "/v1/messages"
	pathDiagGPU         = "/api/diag/gpu"
	// The mode-scoped paths are spelled in internal/moderoutes, which
	// both this catalogue and the /v1 gate match against. A second
	// spelling here would let a row describe a path nothing serves.
	pathModels           = moderoutes.PathModels
	pathEmbeddings       = moderoutes.PathEmbeddings
	pathRerank           = moderoutes.PathRerank
	pathOCR              = moderoutes.PathOCR
	pathOCRQueue         = moderoutes.PathOCRQueue
	pathOCRJob           = moderoutes.PathOCRJob
	pathTasks            = moderoutes.PathTasks
	pathTask             = moderoutes.PathTask
	pathTaskResult       = moderoutes.PathTaskResult
	pathMusicGenerations = moderoutes.PathMusicGenerations
	pathMusicGeneration  = moderoutes.PathMusicGeneration
	pathMusicContent     = moderoutes.PathMusicContent
	pathMusicFormats     = moderoutes.PathMusicFormats
	pathMusicFormat      = moderoutes.PathMusicFormat
	pathMusicDrafts      = moderoutes.PathMusicDrafts
	pathMusicDraft       = moderoutes.PathMusicDraft
	pathTranslate        = "/translate"
	pathTranslateBatch   = "/translate/batch"
	pathTranslateTrans   = "/translate/transcript"
	pathLanguages        = "/languages"
	pathDetect           = "/detect"
	pathProgress         = "/api/progress"
	pathModelSpec        = "/api/model-spec"
	pathEndpoints        = "/api/endpoints"
	pathHealthz          = "/healthz"
	pathMetrics          = "/metrics"

	groupOpenAI = "OpenAI"
	groupOllama = "Ollama-native"
	groupOCR    = "OCR"
	groupTasks  = "Async tasks"

	defaultModelPlaceholder = "MODEL_NAME"
	dataPlaneRegistrarOff   = "data plane registrar not wired"
)

// handleEndpoints renders the static endpoint catalog filtered by the
// concrete wiring on this Server. The catalog is built per-request so a
// future hot-reload of Options would reflect immediately; cost is one
// allocation of ~25 structs which is negligible against the request
// itself.
func (s *Server) handleEndpoints(w http.ResponseWriter, _ *http.Request) {
	kind := string(s.config().Engine.Kind)
	out := EndpointList{
		SchemaVersion: 2,
		EngineKind:    kind,
		Endpoints:     s.buildEndpointCatalog(),
	}
	writeJSON(w, http.StatusOK, out)
}

// buildEndpointCatalog enumerates every route this binary can serve,
// stamping Available based on Options.{Diag,DataPlane,OllamaNative,Metrics}
// and Config.Engine.Kind. Pure function of Options snapshot — no
// hidden globals — so tests can construct a Server with any subset of
// registrars and read the predicted availability back out.
func (s *Server) buildEndpointCatalog() []EndpointInfo {
	cfg := s.config()
	kind := cfg.Engine.Kind
	mode := cfg.Model.Type

	dataPlaneOn := s.opts.DataPlane != nil
	dataPlaneReason := []string{dataPlaneRegistrarOff}
	diagOn := s.opts.Diag != nil
	diagReason := []string{"diag registrar not wired"}
	metricsOn := s.opts.Metrics != nil
	metricsReason := []string{"metrics collector not wired"}
	ollamaOn := s.opts.OllamaNative != nil
	ollamaReason := []string{"ollama-native routes only register when ENGINE_KIND=ollama"}
	translateOn := dataPlaneOn && mode == config.ModelTranslate
	translateReason := translateUnavailableReasons(dataPlaneOn, mode)
	// OCR catalog rows stay green only for MODEL_MODE=ocr (soft catalog;
	// chat/embedding must not show them as available).
	ocrOn := dataPlaneOn && mode == config.ModelOCR
	ocrOffReasons := []string{"not shown unless MODEL_MODE=ocr (catalog only)"}
	if !dataPlaneOn {
		ocrOffReasons = dataPlaneReason
	}
	musicOn := dataPlaneOn && mode == config.ModelMusicGeneration
	musicOffReasons := []string{"not shown unless MODEL_MODE=music_generation (catalog only)"}
	if !dataPlaneOn {
		musicOffReasons = dataPlaneReason
	}
	// The async task contract belongs to whichever engine runs
	// work a request cannot wait out — ocr, audio and tts today, image next. llm-init proxies it
	// and knows nothing about the documents.
	tasksOn := dataPlaneOn && (mode == config.ModelOCR || mode == config.ModelAudio || mode == config.ModelTTS)
	tasksOffReasons := []string{"not shown unless MODEL_MODE=ocr|audio|tts (catalog only)"}
	if !dataPlaneOn {
		tasksOffReasons = dataPlaneReason
	}

	// Queue depth is two gauges only llama.cpp publishes. Another engine
	// answering 0 would report an idle queue on a machine with requests
	// waiting, so the row says unavailable instead.
	loadOn := kind == config.EngineLlamaCpp
	loadReason := []string{"queue depth is only reported for ENGINE_KIND=llamacpp"}

	always := []string{}

	endpoints := []EndpointInfo{
		// ─── control plane (always on) ───
		{Method: mGET, Path: pathProgress, Category: categoryControl,
			Description: "Lifecycle / download progress snapshot. Dashboard polls every 1s (SSE was removed in v1.1.0).",
			Available:   true, Reasons: always,
			CurlHint: "curl -sS http://HOST:PORT/api/progress | jq ."},
		{Method: mPOST, Path: "/api/retry", Category: categoryControl,
			Description: "Trigger a fresh ensure pass; ?force=true / ?level=sha256 supported.",
			Available:   true,
			CurlHint:    "curl -sS -X POST 'http://HOST:PORT/api/retry?force=true'"},
		{Method: mGET, Path: "/api/config", Category: categoryControl,
			Description: "Redacted runtime config.",
			Available:   true,
			CurlHint:    "curl -sS http://HOST:PORT/api/config | jq ."},
		{Method: mGET, Path: pathModelSpec, Category: categoryControl,
			Description: "v2 ProviderModelSpec (Router upsert shape).",
			Available:   true,
			CurlHint:    "curl -sS http://HOST:PORT/api/model-spec | jq ."},
		{Method: mPUT, Path: pathModelSpec, Category: categoryControl,
			Description: "Replace the model spec (full JSON); validated, persisted to disk, served immediately. Changing engine_args also bumps the engine_restart handoff; the X-Engine-Restart* headers report what that achieved.",
			Available:   true,
			CurlHint:    "curl -sS -X PUT http://HOST:PORT/api/model-spec -d @model-spec.json"},
		{Method: mPOST, Path: "/api/engine/restart", Category: categoryControl,
			Description: "Bump engine_restart handoff so a supervising wrapper relaunches with current model-card engine_args. 200 means signaled, not restarted.",
			Available:   true,
			CurlHint:    "curl -sS -X POST http://HOST:PORT/api/engine/restart"},
		{Method: mGET, Path: "/api/engine/restart", Category: categoryControl,
			Description: "Outcome of the last restart signal: confirmed (the engine went down), no-supervisor (it never did), watching, or unverified.",
			Available:   true,
			CurlHint:    "curl -sS http://HOST:PORT/api/engine/restart | jq ."},
		{Method: mGET, Path: pathEngineLoad, Category: categoryControl,
			Description: "How many requests the engine is working on and how many are queued behind them. llama.cpp only, and only when it was launched with --metrics.",
			Available:   loadOn, Reasons: unlessOn(loadOn, loadReason),
			CurlHint: "curl -sS http://HOST:PORT/api/engine/load | jq ."},
		{Method: mGET, Path: "/api/build-info", Category: categoryControl,
			Description: "llm-init build metadata (version, commit, build time). Distinct from /api/version (which proxies the upstream Ollama daemon).",
			Available:   true,
			CurlHint:    "curl -sS http://HOST:PORT/api/build-info | jq ."},
		{Method: mGET, Path: pathEndpoints, Category: categoryControl,
			Description: "This endpoint catalog (self-reference).",
			Available:   true,
			CurlHint:    "curl -sS http://HOST:PORT/api/endpoints | jq ."},

		// ─── health probes (always on) ───
		{Method: mGET, Path: "/livez", Category: categoryHealth,
			Description: "Liveness probe; always 200 when the process is up.",
			Available:   true,
			CurlHint:    "curl -fsS http://HOST:PORT/livez"},
		{Method: mGET, Path: "/readyz", Category: categoryHealth,
			Description: "Readiness probe (engine reachable + model present).",
			Available:   true,
			CurlHint:    "curl -fsS http://HOST:PORT/readyz"},
		{Method: mGET, Path: pathHealthz, Category: categoryHealth,
			Description: "Composite health (readyz + last verify + model loaded).",
			Available:   true,
			CurlHint:    "curl -sS http://HOST:PORT/healthz | jq ."},

		// ─── UI ───
		{Method: mGET, Path: "/", Category: categoryUI,
			Description: "Dashboard (HTML); Accept: application/json returns a status stub.",
			Available:   true,
			CurlHint:    "open http://HOST:PORT/"},
		{Method: mGET, Path: "/static/*", Category: categoryUI,
			Description: "Static assets backing the dashboard (CSS / JS / vendor).",
			Available:   true},

		// ─── OpenAI data plane ───
		{Method: mPOST, Path: pathChatCompletions, Category: categoryOpenAI,
			Group:       groupOpenAI,
			Description: "Chat completion; translated (Ollama) or proxied (vLLM/SGLang/llama.cpp).",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason),
			CurlHint: openAIChatCurl(cfg)},
		{Method: mPOST, Path: "/v1/completions", Category: categoryOpenAI,
			Group:       groupOpenAI,
			Description: "Legacy text completion.",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason)},
		{Method: mPOST, Path: pathEmbeddings, Category: categoryOpenAI,
			Group:       groupOpenAI,
			Description: "Embeddings (catalog highlight when MODEL_MODE=embedding).",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason)},
		{Method: mPOST, Path: pathRerank, Category: categoryOpenAI,
			Group:       groupOpenAI,
			Description: "Rerank documents (catalog highlight when MODEL_MODE=rerank).",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason),
			CurlHint: `curl -sS -X POST http://HOST:PORT/v1/rerank -H 'content-type: application/json' -d '{"query":"what is panda?","documents":["hi","panda is a mammal"]}'`},
		{Method: mGET, Path: pathModels, Category: categoryOpenAI,
			Group:       groupOpenAI,
			Description: "Lists this instance's MODEL_NAME (single-entry list).",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason)},
		{Method: mPOST, Path: pathOCR, Category: categoryOpenAI,
			Group:       groupOCR,
			Description: "Submit OCR job (multipart); returns job_id + task (proxied to OCRAdapter).",
			Available:   ocrOn, Reasons: unlessOn(ocrOn, ocrOffReasons),
			CurlHint: "curl -sS -X POST http://HOST:PORT/v1/ocr -F 'file=@scan.png' -F 'format=text'"},
		{Method: mGET, Path: pathOCRQueue, Category: categoryOpenAI,
			Group:       groupOCR,
			Description: "OCR queue capacity (proxied to OCRAdapter). Deprecated: read capacity from GET /v1/tasks.",
			Available:   ocrOn, Reasons: unlessOn(ocrOn, ocrOffReasons),
			CurlHint: "curl -sS http://HOST:PORT/v1/ocr/queue"},
		{Method: mGET, Path: pathOCRJob, Category: categoryOpenAI,
			Group:       groupOCR,
			Description: "Poll OCR job status/result (proxied to OCRAdapter). Deprecated: use GET /v1/tasks/{id}.",
			Available:   ocrOn, Reasons: unlessOn(ocrOn, ocrOffReasons),
			CurlHint: "curl -sS http://HOST:PORT/v1/ocr/jobs/JOB_ID"},
		{Method: mDELETE, Path: pathOCRJob, Category: categoryOpenAI,
			Group:       groupOCR,
			Description: "Cancel queued or abort running OCR job (proxied to OCRAdapter). Deprecated: use DELETE /v1/tasks/{id}.",
			Available:   ocrOn, Reasons: unlessOn(ocrOn, ocrOffReasons),
			CurlHint: "curl -sS -X DELETE http://HOST:PORT/v1/ocr/jobs/JOB_ID"},
		{Method: mPOST, Path: pathMusicGenerations, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Create an asynchronous music generation; returns 202 and a generation ID.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -sS -X POST http://HOST:PORT/v1/music/generations -H 'Content-Type: application/json' -d '{\"model\":\"MODEL_NAME\",\"prompt\":\"warm indie pop\",\"duration_seconds\":240}'"},
		{Method: mGET, Path: pathMusicGeneration, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Poll one music generation and its normalized outputs.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -sS http://HOST:PORT/v1/music/generations/GENERATION_ID | jq ."},
		{Method: mGET, Path: pathMusicContent, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Stream one completed music output by generation and output ID.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -fLo song.wav 'http://HOST:PORT/v1/music/generations/GENERATION_ID/content?output_id=OUTPUT_ID'"},
		{Method: mDELETE, Path: pathMusicGeneration, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Request cancellation; an engine without reliable interruption returns 422 cancellation_unsupported.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -sS -X DELETE http://HOST:PORT/v1/music/generations/GENERATION_ID"},
		{Method: mPOST, Path: pathMusicFormats, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Create an asynchronous ACE-oriented caption and lyrics formatting task.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -sS -X POST http://HOST:PORT/v1/music/formats -H 'Content-Type: application/json' -d '{\"model\":\"MODEL_NAME\",\"prompt\":\"warm indie pop\",\"lyrics\":\"[Verse 1]...\",\"vocal_language\":\"en\",\"duration_seconds\":240}'"},
		{Method: mGET, Path: pathMusicFormat, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Poll one music caption and lyrics formatting task.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -sS http://HOST:PORT/v1/music/formats/FORMAT_ID | jq ."},
		{Method: mPOST, Path: pathMusicDrafts, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Create an asynchronous song draft; the engine's own LM writes the caption, lyrics and style metadata from one brief.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -sS -X POST http://HOST:PORT/v1/music/drafts -H 'Content-Type: application/json' -d '{\"model\":\"MODEL_NAME\",\"brief\":\"walking home alone after working late\",\"vocal_language\":\"en\"}'"},
		{Method: mGET, Path: pathMusicDraft, Category: categoryOpenAI,
			Group:       groupMusic,
			Description: "Poll one song draft task.",
			Available:   musicOn, Reasons: unlessOn(musicOn, musicOffReasons),
			CurlHint: "curl -sS http://HOST:PORT/v1/music/drafts/DRAFT_ID | jq ."},

		// ─── Async tasks (one contract for every engine) ───
		{Method: mGET, Path: pathTasks, Category: categoryOpenAI,
			Group:       groupTasks,
			Description: "List async tasks + queue capacity; ?status= and ?limit= (default 100) narrow it (proxied to the engine).",
			Available:   tasksOn, Reasons: unlessOn(tasksOn, tasksOffReasons),
			CurlHint: "curl -sS 'http://HOST:PORT/v1/tasks?status=running' | jq ."},
		{Method: mGET, Path: pathTask, Category: categoryOpenAI,
			Group:       groupTasks,
			Description: "Poll one task: status / progress / JSON result (proxied to the engine).",
			Available:   tasksOn, Reasons: unlessOn(tasksOn, tasksOffReasons),
			CurlHint: "curl -sS http://HOST:PORT/v1/tasks/TASK_ID | jq ."},
		{Method: mGET, Path: pathTaskResult, Category: categoryOpenAI,
			Group:       groupTasks,
			Description: "Fetch a task's result body; 409 until it succeeded, 410 once reclaimed.",
			Available:   tasksOn, Reasons: unlessOn(tasksOn, tasksOffReasons),
			CurlHint: "curl -sS http://HOST:PORT/v1/tasks/TASK_ID/result"},
		{Method: mDELETE, Path: pathTask, Category: categoryOpenAI,
			Group:       groupTasks,
			Description: "Cancel a running task, or drop a finished one's result.",
			Available:   tasksOn, Reasons: unlessOn(tasksOn, tasksOffReasons),
			CurlHint: "curl -sS -X DELETE http://HOST:PORT/v1/tasks/TASK_ID"},
		{Method: mPOST, Path: pathResponses, Category: categoryOpenAI,
			Group:       groupOpenAI,
			Description: responsesDescription(kind),
			Available:   dataPlaneOn,
			Reasons:     responsesReasons(kind, dataPlaneOn, dataPlaneReason),
			CurlHint:    openAIResponsesCurl(cfg)},
		{Method: mPOST, Path: "/api/chat/completions", Category: categoryOpenAI,
			Group:       "OpenAI (OpenWebUI alias)",
			Description: "Alias of /v1/chat/completions (same wire shape).",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason)},

		// ─── Translate (MTran-compatible; root paths, not under /v1) ───
		{Method: mPOST, Path: pathTranslate, Category: categoryTranslate,
			Group:       groupTranslate,
			Description: "Single-text translation → {result}. Base URL is the host root (no /v1).",
			Available:   translateOn, Reasons: unlessOn(translateOn, translateReason),
			CurlHint: translateCurl()},
		{Method: mPOST, Path: pathTranslateBatch, Category: categoryTranslate,
			Group:       groupTranslate,
			Description: "Batch translation → {results}. Sequential LLM calls; max 64 texts.",
			Available:   translateOn, Reasons: unlessOn(translateOn, translateReason),
			CurlHint: translateBatchCurl()},
		{Method: mPOST, Path: pathTranslateTrans, Category: categoryTranslate,
			Group:       groupTranslate,
			Description: "Dialogue translation → one result per turn, in order. Not MTran-compatible: turns are translated together with context, a glossary and speaker labels; max 64 turns.",
			Available:   translateOn, Reasons: unlessOn(translateOn, translateReason),
			CurlHint: translateTranscriptCurl()},
		{Method: mGET, Path: pathLanguages, Category: categoryTranslate,
			Group:       groupTranslate,
			Description: "Supported language codes and pairs (available even before engine ready).",
			Available:   translateOn, Reasons: unlessOn(translateOn, translateReason),
			CurlHint: "curl -sS http://HOST:PORT/languages | jq ."},
		{Method: mPOST, Path: pathDetect, Category: categoryTranslate,
			Group:       groupTranslate,
			Description: "Language detection → {language[,confidence]}.",
			Available:   translateOn, Reasons: unlessOn(translateOn, translateReason),
			CurlHint: translateDetectCurl()},

		// ─── Anthropic Messages ───
		{Method: mPOST, Path: pathMessages, Category: categoryAnthropic,
			Group:       "Anthropic",
			Description: "Anthropic Messages API (SSE / tools / vision base64).",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason),
			CurlHint: anthropicMessagesCurl(cfg)},
		{Method: mPOST, Path: "/v1/messages/count_tokens", Category: categoryAnthropic,
			Group:       "Anthropic",
			Description: "Heuristic token estimate for /v1/messages payloads.",
			Available:   dataPlaneOn, Reasons: unlessOn(dataPlaneOn, dataPlaneReason)},

		// ─── Diag (P.1) ───
		{Method: mGET, Path: pathDiagGPU, Category: categoryDiag,
			Description: "Engine GPU residency + warnings.",
			Available:   diagOn, Reasons: unlessOn(diagOn, diagReason),
			CurlHint: "curl -sS http://HOST:PORT/api/diag/gpu | jq ."},

		// ─── Ollama-native passthroughs (Track Q.1) ───
		{Method: mGET, Path: "/api/version", Category: categoryOllama,
			Group:       groupOllama,
			Description: "Mirrors Ollama daemon /api/version (engine_kind=ollama only).",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},
		{Method: mGET, Path: "/api/tags", Category: categoryOllama,
			Group:       groupOllama,
			Description: "List models in the upstream Ollama library.",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},
		{Method: mGET, Path: "/api/ps", Category: categoryOllama,
			Group:       groupOllama,
			Description: "Currently-resident models (memory residency).",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},
		{Method: mPOST, Path: "/api/show", Category: categoryOllama,
			Group:       groupOllama,
			Description: "Model details (template, parameters, modelfile).",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},
		{Method: mPOST, Path: "/api/chat", Category: categoryOllama,
			Group:       groupOllama,
			Description: "Native Ollama chat (reverse-proxied; model rewritten to upstream tag).",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},
		{Method: mPOST, Path: "/api/generate", Category: categoryOllama,
			Group:       groupOllama,
			Description: "Native Ollama generate (reverse-proxied).",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},
		{Method: mPOST, Path: "/api/embed", Category: categoryOllama,
			Group:       groupOllama,
			Description: "Native Ollama embeddings (reverse-proxied).",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},
		{Method: mPOST, Path: "/api/embeddings", Category: categoryOllama,
			Group:       groupOllama,
			Description: "Legacy Ollama embeddings API (reverse-proxied).",
			Available:   ollamaOn, Reasons: unlessOn(ollamaOn, ollamaReason)},

		// ─── Metrics ───
		{Method: mGET, Path: pathMetrics, Category: categoryMetrics,
			Description: "Prometheus exposition.",
			Available:   metricsOn, Reasons: unlessOn(metricsOn, metricsReason),
			CurlHint: "curl -sS http://HOST:PORT/metrics"},
	}

	reported := s.engineEndpoints()
	mergeEngineReportState(endpoints, reported.rows, reported.authoritative, reported.diagnostics)
	if extra := newEngineRows(endpoints, reported.rows); len(extra) > 0 {
		endpoints = append(endpoints, extra...)
	}
	endpoints = dropUndeclaredProxyRows(endpoints, reported.rows, reported.authoritative)

	// Soft-filter data-plane catalog rows for modes with a declared route table.
	// chat / audio keep the legacy Available values computed above.
	for i := range endpoints {
		e := &endpoints[i]
		switch e.Category {
		case categoryOpenAI, categoryAnthropic:
			e.Available, e.Reasons = applyDataPlaneCatalog(
				mode, dataPlaneOn, e.Method, e.Path, e.Available, e.Reasons)
		}
	}

	if reported.state == engineSpecUnknown {
		markUnconfirmedProxyRows(mode, endpoints)
	}

	// Stable sort: category first, then method, then path. Dashboards
	// rely on the deterministic order to render section headers without
	// re-grouping client-side.
	sort.SliceStable(endpoints, func(i, j int) bool {
		if endpoints[i].Category != endpoints[j].Category {
			return endpoints[i].Category < endpoints[j].Category
		}
		if endpoints[i].Path != endpoints[j].Path {
			return endpoints[i].Path < endpoints[j].Path
		}
		return endpoints[i].Method < endpoints[j].Method
	})
	return endpoints
}

func mergeEngineReportState(
	own, reported []EndpointInfo,
	authoritative bool,
	diagnostics []string,
) {
	declared := make(map[string]EndpointInfo, len(reported))
	for _, e := range reported {
		declared[e.Method+" "+e.Path] = e
	}
	for i := range own {
		if !proxiedCategories[own[i].Category] {
			continue
		}
		if engine, ok := declared[own[i].Method+" "+own[i].Path]; authoritative && ok {
			own[i].Available = engine.Available
			own[i].Reasons = engine.Reasons
			own[i].AsyncSupported = engine.AsyncSupported
		} else if ok && len(diagnostics) > 0 {
			own[i].Reasons = append(own[i].Reasons, diagnostics...)
		}
	}
}

const reasonSpecPending = "engine has not reported its capability set yet"

// dataPlaneOwners names, for the static proxied rows, the mode whose model
// is the reason that row exists. A row listed here on any other mode is
// there on the engine's word alone — llm-init would forward it, but nothing
// about the model says it will be answered.
//
// Only the rows of one modality that another modality could plausibly be
// asked for. /v1/models and the async task routes are deliberately absent:
// every mode serves them.
var dataPlaneOwners = map[string]config.ModelType{
	pathChatCompletions: config.ModelChat,
	pathResponses:       config.ModelChat,
	pathMessages:        config.ModelChat,
	pathEmbeddings:      config.ModelEmbedding,
	pathRerank:          config.ModelRerank,
	pathOCR:             config.ModelOCR,
	pathOCRQueue:        config.ModelOCR,
	pathOCRJob:          config.ModelOCR,
}

// markUnconfirmedProxyRows withholds another modality's rows while the
// engine that would have to answer them has not said what it serves.
//
// applyDataPlaneCatalog already refuses these for the modes with a declared
// route table. The modes without one — chat, audio, tts, translate — keep
// the whole OpenAI surface, because their real surface is whatever the
// engine declares at runtime. Until it does, an audio application therefore
// advertises chat and embeddings, and a dashboard rendered in that window,
// or a Router model sync landing in it, reads capabilities that will
// disappear the moment the first spec arrives.
//
// Withheld with a reason rather than dropped: "not yet" is the fact, and a
// missing row cannot say it.
func markUnconfirmedProxyRows(mode config.ModelType, endpoints []EndpointInfo) {
	for i := range endpoints {
		e := &endpoints[i]
		if !proxiedCategories[e.Category] || !e.Available {
			continue
		}
		if owner, claimed := dataPlaneOwners[e.Path]; !claimed || owner == mode {
			continue
		}
		e.Available = false
		e.Reasons = append(e.Reasons, reasonSpecPending)
	}
}

// A valid v1 engine spec is authoritative about its own data plane, so rows we would only proxy — and it never declared — are dropped.
func dropUndeclaredProxyRows(in, reported []EndpointInfo, authoritative bool) []EndpointInfo {
	if !authoritative {
		return in
	}
	// Keyed by method too, like newEngineRows: GET and DELETE share /v1/tasks/{id}, so a path
	// alone would keep a method the engine never declared.
	declared := make(map[string]bool, len(reported))
	for _, e := range reported {
		declared[e.Method+" "+e.Path] = true
	}
	out := make([]EndpointInfo, 0, len(in))
	for _, e := range in {
		if proxiedCategories[e.Category] && !declared[e.Method+" "+e.Path] {
			continue
		}
		out = append(out, e)
	}
	return out
}

// proxiedCategories are the rows llm-init merely forwards to the engine; everything else (control, health, UI, translate, metrics) it serves itself.
var proxiedCategories = map[string]bool{
	categoryOpenAI:    true,
	categoryAnthropic: true,
}

// unlessOn returns reasons only when on is false; the helper keeps the
// catalog literal terse (every row would otherwise need an inline
// if-expression for Reasons).
func unlessOn(on bool, reasons []string) []string {
	if on {
		return nil
	}
	return reasons
}

// applyDataPlaneCatalog filters OpenAI and Anthropic rows down to the
// routes this mode declares. Modes with no declared set (chat, audio,
// translate) keep the availability computed above, unchanged.
//
// The set comes from internal/moderoutes, which dataplane.ModeGuard also
// reads: a row this marks unavailable is a path the port refuses. It used
// to be a switch of its own here, and the two answers were different —
// the catalogue said chat was not available on an embedding application
// and the catch-all mount forwarded it anyway.
func applyDataPlaneCatalog(
	mode config.ModelType,
	dataPlaneOn bool,
	method, path string,
	legacyAvail bool,
	legacyReasons []string,
) (bool, []string) {
	if !moderoutes.Restricted(mode) {
		return legacyAvail, legacyReasons
	}
	if !dataPlaneOn {
		return false, []string{dataPlaneRegistrarOff}
	}
	if moderoutes.Declares(mode, method, path) {
		return legacyAvail, legacyReasons
	}
	return false, []string{
		"MODEL_MODE=" + string(mode) + " does not serve this path; requests to it are refused",
	}
}

// responsesDescription customises the /v1/responses description per
// engine kind so the dashboard reflects the partial-compliance caveat
// without forcing operators to read CHANGELOG.
func responsesDescription(kind config.EngineKind) string {
	switch kind {
	case config.EngineOllama:
		return "OpenAI Responses API; reverse-proxied to the Ollama daemon (stateless only — no previous_response_id / conversation)."
	case config.EngineSGLang:
		return "OpenAI Responses API; pass-through. SGLang v0.5.x is partial: structured input arrays, custom function tools, and stream=true may return upstream 4xx/5xx — prefer /v1/chat/completions for complex payloads."
	case config.EngineVLLM:
		return "OpenAI Responses API; pass-through to vLLM v0.20+. Stateless only."
	case config.EngineLlamaCpp:
		return "OpenAI Responses API; pass-through to llama.cpp server (upstream converts to chat completions internally). Stateless only."
	default:
		return "OpenAI Responses API; pass-through."
	}
}

// responsesReasons concatenates the data-plane-wiring reason (if any)
// with the SGLang-only caveat so the UI can show both in one row.
func responsesReasons(kind config.EngineKind, dataPlaneOn bool, dataPlaneReason []string) []string {
	var out []string
	if !dataPlaneOn {
		out = append(out, dataPlaneReason...)
	}
	if kind == config.EngineSGLang {
		out = append(out, "SGLang v0.5.x upstream is partial; complex payloads may receive 4xx from upstream")
	}
	return out
}

// modelName returns the configured model name or the placeholder
// constant when unset. Centralised here so every curl-hint helper
// shares the same fallback string (goconst's eye stays clean and we
// avoid drift across snippets).
func modelName(cfg config.Config) string {
	if cfg.Model.Name == "" {
		return defaultModelPlaceholder
	}
	return cfg.Model.Name
}

// openAIChatCurl renders a minimal-but-runnable curl for chat. We use
// MODEL_NAME from cfg so the snippet is copy-paste accurate for this
// deployment; falls back to a literal placeholder when unset.
func openAIChatCurl(cfg config.Config) string {
	return `curl -sS -X POST http://HOST:PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"` + modelName(cfg) + `","messages":[{"role":"user","content":"hi"}]}'`
}

func openAIResponsesCurl(cfg config.Config) string {
	return `curl -sS -X POST http://HOST:PORT/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"` + modelName(cfg) + `","input":"hi"}'`
}

func anthropicMessagesCurl(cfg config.Config) string {
	return `curl -sS -X POST http://HOST:PORT/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"` + modelName(cfg) + `","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'`
}

func translateUnavailableReasons(dataPlaneOn bool, mode config.ModelType) []string {
	if !dataPlaneOn {
		return []string{dataPlaneRegistrarOff}
	}
	if mode != config.ModelTranslate {
		return []string{"not shown unless MODEL_MODE=translate"}
	}
	return nil
}

func translateCurl() string {
	return `curl -sS -X POST http://HOST:PORT/translate \
  -H 'Content-Type: application/json' \
  -d '{"from":"zh-Hans","to":"en","text":"你好"}'`
}

func translateBatchCurl() string {
	return `curl -sS -X POST http://HOST:PORT/translate/batch \
  -H 'Content-Type: application/json' \
  -d '{"from":"en","to":"zh-Hans","texts":["Hello","Thanks"]}'`
}

func translateTranscriptCurl() string {
	return `curl -sS -X POST http://HOST:PORT/translate/transcript \
  -H 'Content-Type: application/json' \
  -d '{"from":"en","to":"zh-Hans","segments":[
        {"id":"1","speaker":"Alice","text":"Did you read it?"},
        {"id":"2","speaker":"Bob","text":"Yes"}]}'`
}

func translateDetectCurl() string {
	return `curl -sS -X POST http://HOST:PORT/detect \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天天气真好"}'`
}
