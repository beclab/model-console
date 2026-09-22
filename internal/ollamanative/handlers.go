package ollamanative

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/llm-init/llm-init/internal/adapter/ollama"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/dataplane"
	"github.com/llm-init/llm-init/internal/progress"
)

// showBodyCap bounds the /api/show request body size. The endpoint
// accepts a small JSON envelope ({name|model, verbose?}); 64 KiB is
// orders of magnitude above any sane caller and well below the limits
// large LLM-bound payloads (chat completions) might justify, so a
// dedicated low cap protects the upstream from accidental abuse via
// metadata probes.
const showBodyCap = 64 * 1024

// Mount carries the dependencies the four handlers need. Constructed
// once at process start in cmd/llm-init/main.go (only when
// cfg.Engine.Kind == config.EngineOllama) and reused across requests.
//
// Holds a *config.Config value (not pointer) so the handlers cannot
// observe a partially-mutated config snapshot mid-request; config is
// frozen at boot for the life of the process.
type Mount struct {
	cfg      config.Config
	client   *ollama.Client
	ready    func() bool
	progress progress.Manager
}

// New builds a Mount. Required: cfg, client. ready may be nil only in
// tests that don't care about gate semantics — production callers must
// pass lifecycle.Manager.Ready (the same callback the data plane uses)
// to keep the gate contract uniform across both surfaces.
//
// progressMgr is consulted by the underlying dataplane.NotReadyGuard
// to populate the 503 envelope's `phase` field. nil is tolerated; the
// envelope's phase becomes an empty string and clients fall back to
// timing-based retry, identical to dataplane.Mount's behaviour with a
// nil manager.
func New(cfg config.Config, client *ollama.Client, ready func() bool, progressMgr progress.Manager) *Mount {
	return &Mount{
		cfg:      cfg,
		client:   client,
		ready:    ready,
		progress: progressMgr,
	}
}

// Apply registers Ollama-native routes on mux. Signature is
// shaped to fit controlplane.Options.OllamaNative (a func(*http.ServeMux))
// so main.go can pass it as a method value without a wrapper closure.
//
// Metadata: /api/version (ungated), /api/tags, /api/ps, /api/show.
// Inference (P.3): POST /api/chat, /api/generate, /api/embed, /api/embeddings
// reverse-proxy to the daemon with model rewrite and NotReadyGuard.
func (m *Mount) Apply(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/version", m.handleVersion)
	mux.Handle("GET /api/tags",
		dataplane.NotReadyGuard(m.ready, m.progress, http.HandlerFunc(m.handleTags)))
	mux.Handle("GET /api/ps",
		dataplane.NotReadyGuard(m.ready, m.progress, http.HandlerFunc(m.handlePS)))
	mux.Handle("POST /api/show",
		dataplane.NotReadyGuard(m.ready, m.progress, http.HandlerFunc(m.handleShow)))
	m.registerNativeInference(mux)
}

// handleVersion is a byte-level passthrough of GET /api/version. The
// upstream daemon returns a tiny JSON envelope ({"version":"0.5.4"})
// that has no model-specific fields, so no filter/rewrite is needed —
// we simply mirror status, Content-Type, and body bytes.
//
// On transport failure we surface 502 + Ollama-native error envelope
// (`{"error":"upstream unavailable: ..."}`) so OpenWebUI / Chatbox can
// distinguish "engine not reachable" from "engine answered with an
// error". The 502 status is consistent with how a reverse proxy would
// describe a dead upstream.
func (m *Mount) handleVersion(w http.ResponseWriter, r *http.Request) {
	resp, err := m.client.Version(r.Context())
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	defer resp.Body.Close()
	copyContentType(w, resp)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// handleTags fetches /api/tags from the upstream daemon, filters out
// every entry whose name doesn't match Source.OllamaModel (the daemon
// tag llm-init manages), and rewrites the surviving entry's name to
// Model.Name (the OpenAI alias). The outer schema (`{"models":[...]}`)
// matches the upstream verbatim so Ollama-native clients see the same
// shape they expect from the daemon.
//
// Returns {"models":[]} when no upstream entry matches (e.g. the
// daemon has been restarted and the pull / create hasn't run yet, or
// the daemon serves a completely different inventory). An empty
// models array is still a valid /api/tags response shape — clients
// interpret it as "no models available" rather than "endpoint
// broken", which is the correct UX during boot.
func (m *Mount) handleTags(w http.ResponseWriter, r *http.Request) {
	tags, err := m.client.Tags(r.Context())
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	upstream := ollama.UpstreamModel(m.cfg)
	alias := m.cfg.Model.Name

	out := struct {
		Models []ollama.Model `json:"models"`
	}{Models: []ollama.Model{}}

	for _, t := range tags.Models {
		if ollama.MatchesModelName(t.Name, upstream) ||
			ollama.MatchesModelName(t.Model, upstream) {
			t.Name = alias
			t.Model = alias
			out.Models = append(out.Models, t)
			// Single-model deployment: stop at the first match even
			// if the upstream reported duplicates. Prevents a
			// misbehaving daemon from leaking inventory through us.
			break
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// handlePS mirrors handleTags for /api/ps (currently-resident models).
// PSModel has both `name` (display) and `model` (canonical storage tag)
// fields; both are rewritten to the alias because front-ends use either
// one for identity rendering and a mixed result would surface the
// upstream tag through whichever field the client picks.
//
// `size` and `size_vram` are intentionally NOT modified: front-ends use
// them for VRAM accounting and changing the bytes would corrupt the UI
// (Ollama-native dashboards typically render a "GPU loaded: X of Y MB"
// strip from these two fields).
func (m *Mount) handlePS(w http.ResponseWriter, r *http.Request) {
	ps, err := m.client.PS(r.Context())
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	upstream := ollama.UpstreamModel(m.cfg)
	alias := m.cfg.Model.Name

	out := struct {
		Models []ollama.PSModel `json:"models"`
	}{Models: []ollama.PSModel{}}

	for _, p := range ps.Models {
		if ollama.MatchesModelName(p.Name, upstream) ||
			ollama.MatchesModelName(p.Model, upstream) {
			p.Name = alias
			p.Model = alias
			out.Models = append(out.Models, p)
			break
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// handleShow validates the request body's name/model against the
// llm-init-served identity and, on success, forwards the request with
// the upstream tag substituted. The response body is byte-copied from
// the daemon without rewriting (true passthrough) so introspection tools
// see exactly the metadata the daemon emitted.
//
// Validation rules (matches the user's spec exactly):
//   - Body parses as JSON with a `name` and/or `model` field.
//   - The asked name (preferring `name` over `model` when both are
//     present, mirroring Ollama daemon behaviour) is matched against
//     {Model.Name, Source.OllamaModel} using MatchesModelName.
//   - On mismatch: 404 + `{"error":"model '<asked>' not found"}`
//     (Ollama-native error shape, NOT llm-init's structured envelope —
//     the user chose this for max client compatibility).
func (m *Mount) handleShow(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(io.LimitReader(r.Body, showBodyCap))
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	asked, ok := parseShowName(raw)
	if !ok || asked == "" {
		writeOllamaError(w, http.StatusBadRequest, "missing model name")
		return
	}

	alias := m.cfg.Model.Name
	upstream := ollama.UpstreamModel(m.cfg)
	if !ollama.MatchesModelName(asked, alias) && !ollama.MatchesModelName(asked, upstream) {
		writeOllamaError(w, http.StatusNotFound, fmt.Sprintf("model %q not found", asked))
		return
	}

	patched, err := patchShowBody(raw, upstream)
	if err != nil {
		// Should be unreachable: parseShowName succeeded so raw is
		// at least valid JSON with a string-typed name/model. Surface
		// as a 400 rather than masking with a 500 so a future schema
		// change is visible in client logs.
		writeOllamaError(w, http.StatusBadRequest, "invalid request body")
		return
	}

	resp, err := m.client.Show(r.Context(), patched)
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	defer resp.Body.Close()
	copyContentType(w, resp)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

// parseShowName extracts the asked-for model name from /api/show body.
// Mirrors Ollama daemon behaviour: prefer `name` when present; fall
// back to `model` if `name` is absent or empty. Returns (asked, true)
// when at least one field carried a non-empty string, (zero, false)
// when neither was usable.
//
// The function is unmarshal-once + lookup so a malformed JSON body
// short-circuits to ok=false rather than crashing the handler.
func parseShowName(raw []byte) (string, bool) {
	var req struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return "", false
	}
	if n := strings.TrimSpace(req.Name); n != "" {
		return n, true
	}
	if n := strings.TrimSpace(req.Model); n != "" {
		return n, true
	}
	return "", false
}

// patchShowBody rewrites `name` and `model` in raw to upstream while
// preserving every other field in the body (verbose, system, template,
// modelfile, etc. — Ollama daemons accept a growing set, and we
// shouldn't strip unknown ones).
//
// Implementation: decode into map[string]json.RawMessage, overwrite
// the two known fields, re-encode. JSON field order is not preserved
// across map round-trips, but upstream Ollama parses by field name so
// order doesn't matter on the wire.
func patchShowBody(raw []byte, upstream string) ([]byte, error) {
	m := map[string]json.RawMessage{}
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	tag, err := json.Marshal(upstream)
	if err != nil {
		return nil, err
	}
	m["name"] = tag
	m["model"] = tag
	return json.Marshal(m)
}

// writeJSON emits v as application/json with the given status. Body is
// always terminated with a newline so curl pipes / log readers don't
// merge two consecutive responses on the screen.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeOllamaError emits the Ollama-native error envelope shape
// `{"error":"<msg>"}` (single string-valued `error` field). Picked over
// llm-init's structured `{"error":{"code":...,"message":...}}` shape so
// existing Ollama clients (which assume the upstream daemon's wire) do
// not need transport-level adaptation to consume llm-init's responses
// on these four routes.
func writeOllamaError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

// writeUpstreamError surfaces a transport-level failure as 502 + the
// Ollama-native error envelope. 502 (rather than 503) is the
// conventional shape for "I reached you, but the upstream I'm
// brokering for is unreachable"; 503 is reserved for our own
// lifecycle gate (handled by NotReadyGuard).
func writeUpstreamError(w http.ResponseWriter, err error) {
	writeOllamaError(w, http.StatusBadGateway, "upstream unavailable: "+err.Error())
}

// copyContentType propagates upstream's Content-Type to the response
// writer. Falls back to "application/json" if the upstream didn't set
// one (some old Ollama builds omit it on /api/show); always-JSON is
// the safe default for these four routes' real payloads.
func copyContentType(w http.ResponseWriter, resp *http.Response) {
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
		return
	}
	w.Header().Set("Content-Type", "application/json")
}

// compile-time guard that Apply matches the controlplane.Options.OllamaNative
// callback shape (func(*http.ServeMux)). Catches a signature drift on
// either side at build time rather than at test time.
var _ func(*http.ServeMux) = (*Mount)(nil).Apply
