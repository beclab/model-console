package proxy

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"runtime/debug"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/obs"
)

// Adapter is the reverse-proxy implementation shared by vLLM, llama.cpp
// and SGLang. The Kind is supplied at construction time so observability
// and the factory can tell the three apart.
//
// Metrics is optional: when nil the adapter still works (tests construct
// adapters bare) but proxy-panics counter increments are dropped. The
// production wiring in factory.New plumbs it from cmd/llm-init.
type Adapter struct {
	cfg        config.Config
	kind       config.EngineKind
	target     *url.URL
	proxy      *httputil.ReverseProxy
	httpClient *http.Client
	Metrics    *obs.Metrics
	// enums is the checkable part of the card's parameter_rules, read
	// once here rather than per request. Nil for the cards that declare
	// none, which is most of them.
	enums enumRules
}

// NewAdapter builds a proxy adapter for the given kind. cfg.Engine.URL
// must be a base URL like "http://vllm:8000"; it is parsed once at
// startup so the per-request hot path doesn't re-parse it.
func NewAdapter(cfg config.Config, kind config.EngineKind) (*Adapter, error) {
	target, err := url.Parse(cfg.Engine.URL)
	if err != nil {
		return nil, fmt.Errorf("proxy: parse engine URL %q: %w", cfg.Engine.URL, err)
	}
	if target.Scheme == "" || target.Host == "" {
		return nil, fmt.Errorf("proxy: engine URL must include scheme and host: %q", cfg.Engine.URL)
	}
	a := &Adapter{
		cfg:    cfg,
		kind:   kind,
		target: target,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		enums: buildEnumRules(cfg.Spec),
	}
	a.proxy = &httputil.ReverseProxy{
		Director: a.director,
		// Long timeouts in the transport — the upstream may stream SSE
		// for minutes for chat completions. Disable response header
		// compression buffering by leaving FlushInterval at the
		// default (-1 means flush immediately for every chunk).
		Transport: &http.Transport{
			Proxy:           http.ProxyFromEnvironment,
			IdleConnTimeout: 90 * time.Second,
			// UPSTREAM_RESPONSE_HEADER_TIMEOUT (default 5m). The body
			// stream itself is unbounded once headers arrive.
			ResponseHeaderTimeout: cfg.Runtime.ResponseHeaderTimeout(),
		},
		FlushInterval: -1,
		// Match the public API error envelope
		// components.schemas.Error) so OpenAI Python / LangChain /
		// OpenWebUI parse the failure cleanly instead of choking on
		// plain text. http.Error would have written
		// `text/plain; charset=utf-8` and a bare string body.
		ErrorHandler: writeUpstreamError,
	}
	return a, nil
}

// writeUpstreamError emits a JSON-shaped 502 the data plane's clients
// already know how to parse (same envelope as
// internal/dataplane/notready.go and internal/controlplane/handlers.go).
func writeUpstreamError(w http.ResponseWriter, _ *http.Request, err error) {
	w.Header().Set(headerContentType, contentTypeJSON)
	if engineRefusedDial(err) {
		writeEngineNotListening(w, err)
		return
	}
	w.WriteHeader(http.StatusBadGateway)
	_ = json.NewEncoder(w).Encode(map[string]any{
		jsonKeyError: map[string]string{
			jsonKeyCode:    "upstream_unreachable",
			jsonKeyMessage: "upstream engine unreachable: " + err.Error(),
		},
	})
}

// codeNotReady is the data plane's not-ready code; dataplane and
// controlplane each hold the same literal, and clients pin on it.
const codeNotReady = "not_ready"

// engineRefusedDial reports whether the request never reached the engine
// because nothing was listening. That is the window a card edit opens: the
// wrapper stops the engine, and until the next health probe notices, the
// data plane still reads ready. Nothing was sent, so nothing is lost by
// asking the caller to come back.
func engineRefusedDial(err error) bool {
	var op *net.OpError
	return errors.As(err, &op) && op.Op == "dial"
}

// writeEngineNotListening answers a refused dial the way the readiness gate
// answers a known-down engine, so a client sees one retryable 503 rather
// than a 502 that reads as a broken gateway.
func writeEngineNotListening(w http.ResponseWriter, err error) {
	w.Header().Set("Retry-After", "5")
	w.WriteHeader(http.StatusServiceUnavailable)
	_ = json.NewEncoder(w).Encode(map[string]any{
		jsonKeyError: map[string]string{
			jsonKeyCode:    codeNotReady,
			jsonKeyMessage: "engine is not accepting connections (restarting?): " + err.Error(),
		},
	})
}

// director runs in the reverse-proxy hot path: it rewrites Host /
// Scheme so the request is sent to cfg.Engine.URL, drops the
// Authorization header, and rewrites the JSON model field.
func (a *Adapter) director(req *http.Request) {
	req.URL.Scheme = a.target.Scheme
	req.URL.Host = a.target.Host
	// Preserve the full inbound path/query as the upstream path. Don't
	// prepend a.target.Path because the engines mount /v1 at root and
	// dataplane.Mount routes /v1 in.
	if a.target.Path != "" && a.target.Path != "/" {
		req.URL.Path = strings.TrimSuffix(a.target.Path, "/") + req.URL.Path
	}
	req.Host = a.target.Host
	stripAuthorization(req.Header)
	rewriteModelInBody(req, a.cfg.Model.Name, a.kind)
}

// Kind returns the configured engine kind (vllm / llamacpp / sglang).
func (a *Adapter) Kind() config.EngineKind { return a.kind }

// AliveBeforeBoot reports false: the engine binary is gated by
// deploy/wrappers/<engine>.sh's wait_for_sentinel, which only fires
// after llm-init writes /run/llm-init/model_download_finish inside
// ensure. lifecycle.Run therefore inverts the order for proxy engines
// (ensure → WaitAlive → healthLoop); see
// the engine boot order.
func (a *Adapter) AliveBeforeBoot() bool { return false }

// WaitAlive polls the upstream's /v1/models on a 2s ticker. Each probe
// has a 5s deadline; the function returns nil on first 200 or ctx.Err.
func (a *Adapter) WaitAlive(ctx context.Context) error {
	const interval = 2 * time.Second
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		req, err := http.NewRequestWithContext(probeCtx,
			http.MethodGet, joinPath(a.cfg.Engine.URL, pathModels), nil)
		if err != nil {
			cancel()
			return err
		}
		resp, doErr := a.httpClient.Do(req)
		cancel()
		if doErr == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
		}
	}
}

// Ready reports whether /v1/models is reachable and lists the configured
// model. Loaded is left nil because vLLM/llama.cpp/SGLang have no
// loaded-vs-cold-start distinction.
func (a *Adapter) Ready(ctx context.Context) (adapter.ReadyState, error) {
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(probeCtx,
		http.MethodGet, joinPath(a.cfg.Engine.URL, pathModels), nil)
	if err != nil {
		return adapter.ReadyState{}, err
	}
	resp, err := a.httpClient.Do(req)
	if err != nil {
		return adapter.ReadyState{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return adapter.ReadyState{}, fmt.Errorf("proxy: GET /v1/models status %d", resp.StatusCode)
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return adapter.ReadyState{}, fmt.Errorf("proxy: decode /v1/models: %w", err)
	}
	exists := false
	for _, m := range body.Data {
		if m.ID == a.cfg.Model.Name {
			exists = true
			break
		}
	}
	return adapter.ReadyState{Alive: true, ModelExists: exists}, nil
}

// These adapters install nothing, so they do not implement
// adapter.ModelInstaller. vLLM, llama.cpp and SGLang are launched with a
// path and read it: llm-init puts the files there and the engine's own
// startup args say where (see the wrapper scripts under
// deploy/wrappers/), so there is nothing to tell the engine afterwards.

// EngineNativeStats returns the GPU-residency snapshot for the
// reverse-proxied engine. Branches keyed on a.kind:
//
//   - SGLang: HTTP self-introspection via
//     /server_info -> /get_server_info fallback. Reads
//     cpu_offload_gb to derive mode.
//   - vLLM: env-mirror only. Trusts ENGINE_CPU_OFFLOAD_GB (vLLM
//     OOMs at boot when VRAM is insufficient, so a running engine
//     means the requested placement succeeded).
//   - llama.cpp: env-mirror + compose-contract validation. The
//     shipped deploy/compose/llamacpp.yml hardcodes
//     `--fit off -ngl all`; if the operator opts out of that
//     contract (env disagrees) we report unknown + warning.
//   - default (embed / clipembed / audio): the generic /metrics gpu_* relay.
func (a *Adapter) EngineNativeStats(ctx context.Context) (adapter.NativeStats, error) {
	switch a.kind {
	case config.EngineSGLang:
		return a.sglangNativeStats(ctx)
	case config.EngineVLLM:
		return a.vllmNativeStats(ctx)
	case config.EngineLlamaCpp:
		return a.llamacppNativeStats(ctx), nil
	default:
		return a.proxyMetricsStats(ctx)
	}
}

// OpenAIHandler returns the reverse proxy wrapped in a panic-recover
// middleware. The proxy already matches every path it cares about
// implicitly (any inbound path is forwarded to the upstream); routing
// concerns at /v1/* live in dataplane.Mount which restricts what
// callers can hit.
//
// Why recover() here:
//
// Pre-v1.0.5 a panic inside Director / ModifyResponse / a
// roundtripper goroutine would propagate up through the net/http
// server's default panicHandler, which logs the stack and tears down
// the in-flight connection but DOES NOT take the process down — for a
// process running multiple unrelated handlers that's the right
// behavior. However, in the dataplane.Mount path the proxy is also
// the engine sidecar's only request surface; an upstream-triggered
// nil-pointer (e.g. a malformed Content-Type header that breaks
// ModifyResponse, or a failed url.Parse in Director) would close the
// caller's connection mid-stream and leave net/http logging the
// stack to stdout — operators saw "EOF" with no useful root cause.
//
// The recover middleware:
//
//   - Catches any panic inside ServeHTTP.
//   - Bumps llm_init_proxy_panics_total{kind} so an alert can fire
//     when the engine is repeatedly tripping the proxy (a strong
//     signal of an upstream bug).
//   - Emits a slog "error" with the recovered value and a debug
//     stack so the panic is preserved in the operator log alongside
//     the request id, route, and engine kind.
//   - Returns a 502 + the same OpenAI-compatible Error envelope as
//     writeUpstreamError so clients (OpenAI Python, LangChain,
//     OpenWebUI) parse the failure cleanly instead of seeing a
//     truncated body.
//
// We deliberately do NOT recover() inside director / ModifyResponse
// individually -- net/http's server-side recover is the right last
// resort; we only need a deterministic error envelope and a metric
// surface, both of which the wrapper provides.
func (a *Adapter) OpenAIHandler(_ config.Config) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			rec := recover()
			if rec == nil {
				return
			}
			if a.Metrics != nil {
				a.Metrics.ProxyPanicsTotal.WithLabelValues(string(a.kind)).Inc()
			}
			slog.Error("proxy: recovered panic in ServeHTTP",
				"engine_kind", string(a.kind),
				"path", r.URL.Path,
				"panic", fmt.Sprintf("%v", rec),
				"stack", string(debug.Stack()))
			// If the response has not yet been written, emit the same
			// JSON envelope clients already parse for upstream errors.
			// If headers were already flushed (mid-stream panic) the
			// http.ResponseWriter will silently no-op the Header /
			// WriteHeader calls, which is the least-bad outcome — the
			// stream is already corrupt either way and the metric +
			// log capture the diagnostic surface.
			w.Header().Set(headerContentType, contentTypeJSON)
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(map[string]any{
				jsonKeyError: map[string]string{
					jsonKeyCode:    "proxy_panic",
					jsonKeyMessage: "internal proxy panic; the request was dropped before reaching the upstream",
				},
			})
		}()
		// Admission: reject bodies too large for model-rewrite BEFORE
		// the proxy hot path. Pre-v1.0.5 oversized bodies were passed
		// through verbatim, leaking the OpenAI alias `model` field
		// upstream and silently breaking the "consistent model
		// identity" contract. See rewrite.go:rewriteAdmission.
		body, reject := rewriteAdmission(w, r, a.kind)
		if reject {
			return
		}
		// After admission, because the body is in memory by then and
		// nothing has been written to w yet — the only point in this
		// handler where a 400 can still be the whole response.
		if checkEnumParams(w, a.enums, r.URL.Path, body) {
			return
		}
		a.serveLogged(w, r, body)
	})
}

// serveLogged forwards the request and, at debug, leaves behind one line
// saying what was asked for and how it went.
//
// A successful proxied request used to be invisible: the engines log neither
// the parameters they were handed nor the prompt they rendered from them, and
// on this side the whole adapter is a body rewrite plus a Prometheus counter.
// So "did the reasoning_effort this client sent survive the hops in front of
// us" had no answer once the request was over, which is the question that
// matters when a level is configurable and the engine silently defaults when
// it is missing.
//
// The line is per request and therefore debug: LOG_LEVEL is editable on the
// application, so an operator chasing one model's behaviour turns it on, and
// nobody pays for it the rest of the time. The digest is computed inside the
// same guard, so an info-level deployment does not walk request JSON at all.
func (a *Adapter) serveLogged(w http.ResponseWriter, r *http.Request, body []byte) {
	if len(body) == 0 || !slog.Default().Enabled(r.Context(), slog.LevelDebug) {
		a.proxy.ServeHTTP(w, r)
		return
	}
	params, modelIn := paramDigest(body), digestModel(body)
	started := time.Now()
	sw := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
	a.proxy.ServeHTTP(sw, r)
	slog.Debug("proxy: forwarded",
		slog.String("path", r.URL.Path),
		slog.String("model_in", modelIn),
		slog.String("model_out", a.cfg.Model.Name),
		slog.String("params", params),
		slog.Int("upstream_status", sw.status),
		slog.Int64("duration_ms", time.Since(started).Milliseconds()))
}

// statusRecorder captures the status without buffering the body, because
// http.ResponseWriter does not expose what WriteHeader was called with.
// Deliberately a second copy of internal/dataplane's: that one is unexported
// there, and the dataplane package is the one that mounts this adapter, so
// importing it back would close a cycle.
//
// Flush and Hijack have to forward or a chat completion stops streaming: the
// wrapper is in the path of every SSE chunk.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (s *statusRecorder) WriteHeader(code int) {
	if !s.wroteHeader {
		s.status = code
		s.wroteHeader = true
	}
	s.ResponseWriter.WriteHeader(code)
}

// Write triggers an implicit WriteHeader(200) per net/http convention; mirror
// that so a handler streaming bytes without an explicit WriteHeader is not
// logged as whatever the zero value happened to be.
func (s *statusRecorder) Write(b []byte) (int, error) {
	if !s.wroteHeader {
		s.status = http.StatusOK
		s.wroteHeader = true
	}
	return s.ResponseWriter.Write(b)
}

func (s *statusRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	hj, ok := s.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("proxy: ResponseWriter %T is not an http.Hijacker", s.ResponseWriter)
	}
	return hj.Hijack()
}

func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap exposes the underlying writer to http.ResponseController.
func (s *statusRecorder) Unwrap() http.ResponseWriter { return s.ResponseWriter }
