package ollama

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

// Adapter is the Ollama-specific implementation of adapter.Adapter.
//
// It owns a *Client (for daemon RPCs) and a snapshot of cfg (for the
// data-plane translator paths). The Adapter is created once at process
// start and shared across goroutines: lifecycle drives Pull/Register
// from a single goroutine, while OpenAIHandler-built handlers serve
// many concurrent /v1/* requests.
type Adapter struct {
	cfg    config.Config
	client *Client
}

// NewAdapter constructs a fresh Adapter. The HTTP client is configured by
// NewClient with multi-minute timeouts so /api/pull and /api/create
// streams do not time out on slow networks.
//
// cfg.Log.UpstreamTrace plumbs through to client.TraceLevel so the
// audit logging gated by UPSTREAM_TRACE_LEVEL fires on every Ollama
// HTTP call this adapter makes (chat / embed / completion / show / ...).
// Empty string maps to TraceOff, matching pre-tracing behaviour.
func NewAdapter(cfg config.Config) *Adapter {
	cli := NewClient(cfg.Engine.URL)
	cli.TraceLevel = TraceLevel(cfg.Log.UpstreamTrace)
	cli.SetResponseHeaderTimeout(cfg.Runtime.ResponseHeaderTimeout())
	return &Adapter{cfg: cfg, client: cli}
}

// Kind returns config.EngineOllama.
func (a *Adapter) Kind() config.EngineKind { return config.EngineOllama }

// upstreamModel returns the name used when talking to the Ollama daemon
// (pull tag, registered model name, /api/chat upstream "model" field).
// It is distinct from cfg.Model.Name, which is the alias OpenAI clients
// see in the "model" field. The resolution rule is delegated to
// config.Config.PrimaryOllamaTag (Sources entry with Kind=ollama wins;
// otherwise ModelName covers ollama-url + the standalone HF / URL
// paths).
func (a *Adapter) upstreamModel() string {
	return UpstreamModel(a.cfg)
}

// UpstreamModel is the package-exported form of (*Adapter).upstreamModel
// for callers (notably internal/ollamanative) that need to resolve the
// daemon tag from a config.Config without owning an Adapter instance.
// The free function is the single source of truth; the method above is
// a thin wrapper kept for the dense call sites inside this package.
func UpstreamModel(cfg config.Config) string {
	return cfg.PrimaryOllamaTag()
}

// WaitAlive polls /api/tags every 2s until it returns 200 or ctx is done.
// 2s is a sweet spot: short enough to give a tight ready signal in dev
// (Ollama starts in <5s on most machines) but slow enough not to spam
// the daemon's first-boot handler with thousands of probes.
func (a *Adapter) WaitAlive(ctx context.Context) error {
	const interval = 2 * time.Second
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		_, err := a.client.Tags(probeCtx)
		cancel()
		if err == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
		}
	}
}

// AliveBeforeBoot reports true: the daemon is brought up by
// docker-compose before llm-init starts (deploy/compose/ollama.yml has
// `llm-init depends_on: ollama (service_healthy)`), so /api/tags is
// reachable from llm-init's very first instruction. lifecycle.Run can
// therefore call WaitAlive before the first ensure pass.
func (a *Adapter) AliveBeforeBoot() bool { return true }

// Ready snapshots /api/tags: Alive, ModelExists, and the matched tag's
// on-disk size (ModelBytes) used to backfill the download gauge.
//
// Tag matching uses upstreamModel() (i.e. OLLAMA_MODEL, falling back to
// MODEL_NAME) because that is the name the daemon stores. The OpenAI
// alias the client sees lives in cfg.Model.Name and never appears in
// /api/tags unless it happens to equal the upstream tag.
func (a *Adapter) Ready(ctx context.Context) (adapter.ReadyState, error) {
	tags, err := a.client.Tags(ctx)
	if err != nil {
		return adapter.ReadyState{}, err
	}
	want := a.upstreamModel()
	exists := false
	var modelBytes int64
	for _, m := range tags.Models {
		if m.Name == want || matchesModelName(m.Name, want) {
			exists = true
			modelBytes = m.Size
			break
		}
	}
	return adapter.ReadyState{Alive: true, ModelExists: exists, ModelBytes: modelBytes}, nil
}

// Pull drives /api/pull and reports each NDJSON frame to sink. The frame
// schema (digest / total / completed) is mapped onto OnFileStart /
// OnBytes / OnFileDone so progress.Manager renders the same UI whether
// bytes flow from HF or from Ollama's library mirror.
func (a *Adapter) Pull(ctx context.Context, ref string, sink progress.Sink) error {
	if sink == nil {
		sink = progress.NopSink{}
	}
	// Track the (digest, last completed) tuple per file so we can emit
	// monotonically-increasing OnBytes deltas; Ollama frames echo total
	// completed bytes per layer rather than per-frame deltas.
	lastCompleted := make(map[string]int64)
	startedFiles := make(map[string]bool)

	err := a.client.Pull(ctx, ref, func(p PullResponse) {
		if p.Status != "" && p.Digest == "" && p.Total == 0 {
			return // status-only frame ("pulling manifest", "verifying sha256 digest")
		}
		if p.Digest == "" {
			return
		}
		if !startedFiles[p.Digest] && p.Total > 0 {
			sink.OnFileStart(p.Digest, p.Total)
			startedFiles[p.Digest] = true
		}
		if delta := p.Completed - lastCompleted[p.Digest]; delta > 0 {
			sink.OnBytes(delta)
			lastCompleted[p.Digest] = p.Completed
		}
		if p.Total > 0 && p.Completed == p.Total {
			sink.OnFileDone(p.Digest)
		}
	})
	// ErrPullNoSuccess: stream ended early. Per the reference's
	// behavior (olares-ollama 380–409), give the daemon a second to
	// finish writing then re-check via /api/tags before declaring
	// failure.
	if errors.Is(err, ErrPullNoSuccess) {
		probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		exists, _ := a.client.ModelExists(probeCtx, ref)
		cancel()
		if exists {
			return nil
		}
		sink.OnError(err)
		return err
	}
	if err != nil {
		sink.OnError(err)
		return err
	}
	return nil
}

// Register pushes each file as a blob and creates the named model. The
// flow is:
//
//  1. compute sha256 of every file (streaming);
//  2. HEAD /api/blobs/{digest}; if missing, POST /api/blobs/{digest};
//  3. POST /api/create with files map + template / system / parameters;
//  4. when GGUF_TEMPLATE is supplied (raw, not the named enum), use the
//     two-step base+final pattern so the explicit Go template overrides
//     whatever Ollama auto-detects from GGUF metadata.
//
// Step 1 is a full read of the model, which is why known exists: on the
// path that supplies it, that read was just done — an ollama:// URL
// source with a #sha256= fragment has had those exact bytes hashed by
// the downloader and the result written to its download record, and
// repeating it costs a multi-gigabyte read on every boot to arrive at
// the same string.
//
// A hint is used as given. Confirming it would be the read it exists to
// avoid, so a wrong one produces a blob Ollama rejects rather than a
// silently mismatched model — it checks the digest of what it is sent.
func (a *Adapter) Register(ctx context.Context, files []string, known map[string]string, sink progress.Sink) error {
	if sink == nil {
		sink = progress.NopSink{}
	}
	if len(files) == 0 {
		return errors.New("ollama: Register: no files supplied")
	}
	digestByName := make(map[string]string, len(files))
	for _, p := range files {
		if err := ctx.Err(); err != nil {
			return err
		}
		info, statErr := os.Stat(p)
		if statErr != nil {
			return fmt.Errorf("ollama: stat %s: %w", p, statErr)
		}
		base := filepath.Base(p)
		sink.OnFileStart(p, info.Size())
		digest, ok := known[p]
		if !ok {
			var err error
			if digest, err = sha256OfFile(ctx, p); err != nil {
				sink.OnError(err)
				return err
			}
		}
		digestByName[base] = digest
		sink.OnBytes(info.Size())

		present, err := a.client.BlobExists(ctx, digest)
		if err != nil {
			sink.OnError(err)
			return err
		}
		if !present {
			if err := a.client.PushBlob(ctx, digest, p); err != nil {
				sink.OnError(err)
				return err
			}
		}
		sink.OnFileDone(p)
	}

	req := CreateRequest{
		Model:      a.upstreamModel(),
		Files:      digestByName,
		Parameters: buildCreateParams(a.cfg),
	}
	if err := a.client.Create(ctx, req); err != nil {
		sink.OnError(err)
		return err
	}
	return nil
}

// buildCreateParams fills in /api/create parameters from cfg. v1.1.0
// onwards the only create-time knob is num_ctx, sourced from
// ENGINE_ARGS=OLLAMA_NUM_CTX (normalised to "num_ctx"); modern GGUF
// self-describes the chat template / sampling defaults so the legacy
// GGUF_TEMPLATE / GGUF_SYSTEM / GGUF_PARAMS knobs were retired
// alongside their envs.
func buildCreateParams(cfg config.Config) map[string]interface{} {
	out := map[string]interface{}{}
	if v, ok := cfg.Engine.Args.GetInt("num_ctx"); ok && v > 0 {
		out["num_ctx"] = v
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// applyInferenceInjection mirrors the v1.0 Inference / KeepAlive sticky
// fields onto an /api/chat-style request map. The "options" sub-map and
// "keep_alive" top-level field are populated only when the operator set
// the corresponding ENGINE_ARGS=OLLAMA_<NAME> entry, matching the v1.0
// "no env set → daemon default applies" behaviour.
func applyInferenceInjection(out, options map[string]any, cfg config.Config) {
	if v, ok := cfg.Engine.Args.GetInt("num_ctx"); ok && v > 0 {
		options["num_ctx"] = v
	}
	if v, ok := cfg.Engine.Args.GetFloat("repeat_penalty"); ok && v > 0 {
		options["repeat_penalty"] = v
	}
	if v, ok := cfg.Engine.Args.GetInt("repeat_last_n"); ok && v > 0 {
		options["repeat_last_n"] = v
	}
	if v, ok := cfg.Engine.Args.GetString("keep_alive"); ok && v != "" {
		out["keep_alive"] = v
	}
}

// sha256OfFile streams path through sha256.Sum and returns the
// "sha256:<hex>" digest format Ollama expects.
func sha256OfFile(ctx context.Context, path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	buf := make([]byte, 1<<20)
	for {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		n, rerr := f.Read(buf)
		if n > 0 {
			_, _ = h.Write(buf[:n])
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return "", fmt.Errorf("ollama: sha256 read: %w", rerr)
		}
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

// matchesModelName mirrors the prefix logic in Client.ModelExists for
// /api/ps responses where the daemon may report "name:tag" while
// MODEL_NAME is unqualified.
func matchesModelName(have, want string) bool {
	return MatchesModelName(have, want)
}

// MatchesModelName is the package-exported form of matchesModelName.
// Same semantics: exact match, "name:latest" alias, or "name:tag"
// prefix when want is unqualified. Exported so callers in
// internal/ollamanative can reuse the same matching rules that
// /api/tags and /api/ps already use inside the adapter.
func MatchesModelName(have, want string) bool {
	if have == want {
		return true
	}
	if want != "" && (have == want+":latest" || hasColonPrefix(have, want)) {
		return true
	}
	return false
}

func hasColonPrefix(s, prefix string) bool {
	if len(s) <= len(prefix)+1 {
		return false
	}
	return s[:len(prefix)] == prefix && s[len(prefix)] == ':'
}

// OpenAIHandler wires the four OpenAI-compatible routes onto an
// http.ServeMux. The handler is constructed once and reused across all
// requests; the underlying *Client is shared.
func (a *Adapter) OpenAIHandler(_ config.Config) http.Handler {
	return a.buildHandler()
}

// AnthropicHandler returns the HTTP handler dataplane mounts at
// POST /v1/messages (and POST /v1/messages/count_tokens). The handler
// does Anthropic → Ollama /api/chat translation (translate_messages.go)
// and Ollama NDJSON → Anthropic SSE on the streaming path. One
// handler per Adapter; safe for concurrent use.
//
// Internal routing uses an http.ServeMux so /v1/messages and
// /v1/messages/count_tokens can dispatch to dedicated handlers
// instead of one giant function with a path branch.
func (a *Adapter) AnthropicHandler(_ config.Config) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", a.handleMessages)
	mux.HandleFunc("POST /v1/messages/count_tokens", a.handleCountTokens)
	return mux
}

// Wire-level Source values surfaced through NativeStats.Source.
// Lifted to named constants so goconst stays quiet on the 3-way
// "ollama_ps" / 2-way "unavailable" duplication and so the strings
// operators see in /api/diag/gpu's JSON Source field live in one
// place. Changing these is a wire break.
const (
	sourceOllamaPS    = "ollama_ps"
	sourceUnavailable = "unavailable"
)

// EngineNativeStats translates Ollama's GET /api/ps into the unified
// NativeStats shape. The endpoint exposes both `size` (total model
// bytes) and `size_vram` (bytes currently resident on GPU) per
// loaded model; we compare the two to derive gpu.mode.
//
// Caveats:
//   - /api/ps does not exist on Ollama versions <0.1.31 — those
//     return 404 and we surface NativeStats{Mode:unknown} + a
//     warning rather than an error so the diag dashboard can show
//     "old engine, upgrade to >=0.1.31".
//   - The matched entry uses the SAME `upstreamModel` resolution as
//     /api/tags so OLLAMA_MODEL = MODEL_NAME deployments pin to the
//     daemon's storage tag, not the OpenAI alias.
//   - When the daemon reports the model NOT in /api/ps at all the
//     model is unloaded; we report cpu_only with a warning saying
//     "model not currently resident; trigger a request to load".
func (a *Adapter) EngineNativeStats(ctx context.Context) (adapter.NativeStats, error) {
	ps, err := a.client.PS(ctx)
	if err != nil {
		// Distinguish "transport error" (engine unreachable, propagate)
		// from "daemon refused the endpoint" (older Ollama). The
		// client.PS() helper returns a descriptive HTTP status string
		// so we treat the unreachable case as a real error and the
		// 404/Method-Not-Allowed shape as "engine has no native
		// answer".
		if isOllamaPSUnsupported(err) {
			return adapter.NativeStats{
				Source: sourceUnavailable,
				GPU:    adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown},
				Warnings: []string{
					"ollama: /api/ps not supported by this engine version; upgrade to >=0.1.31 for GPU residency reporting",
				},
			}, nil
		}
		return adapter.NativeStats{}, err
	}

	want := a.upstreamModel()
	var match *PSModel
	for i := range ps.Models {
		m := &ps.Models[i]
		// /api/ps reports both `name` (display) and `model` (the
		// canonical model_name used by the daemon's storage). Match
		// either, mirroring the same logic /api/tags uses elsewhere
		// in this file.
		if m.Name == want || m.Model == want ||
			matchesModelName(m.Name, want) || matchesModelName(m.Model, want) {
			match = m
			break
		}
	}

	payload, _ := json.Marshal(ps)

	if match == nil {
		// Model is not currently resident. We treat this as cpu_only
		// + warning rather than "unknown" because /api/ps reliably
		// distinguishes the two: an empty match means the daemon
		// freed the VRAM (KEEP_ALIVE expired) or never loaded it,
		// not that the daemon cannot tell us.
		return adapter.NativeStats{
			Source:  sourceOllamaPS,
			Payload: payload,
			GPU:     adapter.GPUResidencyHints{Mode: adapter.GPUModeCPUOnly},
			Warnings: []string{fmt.Sprintf(
				"ollama: model %q is not currently resident in /api/ps; gpu.mode reflects the cold state, send a request to warm the cache",
				want)},
		}, nil
	}

	return adapter.NativeStats{
		Source:  sourceOllamaPS,
		Payload: payload,
		GPU:     ollamaGPUHintsFromPS(*match),
	}, nil
}

// ollamaGPUHintsFromPS maps a single /api/ps entry to GPUResidencyHints.
// Logic per plan §3.2.A:
//   - SizeVRAM == Size (and Size > 0) -> full
//   - 0 < SizeVRAM < Size              -> partial
//   - SizeVRAM == 0                    -> cpu_only
func ollamaGPUHintsFromPS(m PSModel) adapter.GPUResidencyHints {
	hints := adapter.GPUResidencyHints{}
	if m.Size > 0 {
		size := m.Size
		hints.ModelBytes = &size
	}
	if m.SizeVRAM > 0 {
		v := m.SizeVRAM
		hints.VRAMBytes = &v
	}
	switch {
	case m.Size <= 0:
		// Daemon did not populate size. Conservative fallback.
		hints.Mode = adapter.GPUModeUnknown
	case m.SizeVRAM == 0:
		hints.Mode = adapter.GPUModeCPUOnly
	case m.SizeVRAM >= m.Size:
		hints.Mode = adapter.GPUModeFull
	default:
		hints.Mode = adapter.GPUModePartial
	}
	return hints
}

// isOllamaPSUnsupported reports whether the error from Client.PS
// looks like "this Ollama version doesn't have /api/ps" rather than
// "transport failed". Older daemons return 404 or 405.
func isOllamaPSUnsupported(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	// Client.PS formats failures as "ollama: GET /api/ps status NNN".
	// Match on the documented status codes; transport errors return
	// a different prefix ("Get \"http://...\": ...").
	for _, s := range []string{"status 404", "status 405", "status 501"} {
		if containsString(msg, s) {
			return true
		}
	}
	return false
}

// containsString is a tiny fmt-free strings.Contains so we don't
// pull strings into adapter.go for one call site. Inlined for
// clarity over the import bloat trade.
func containsString(s, sub string) bool {
	if len(sub) == 0 {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
