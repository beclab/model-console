// Package ollama implements the Adapter for Ollama-backed deployments.
//
// The package is split into a thin HTTP client (this file), the four
// OpenAI<->Ollama translators (translate_*.go), and the Adapter glue
// (adapter.go + handler.go). Templates that drive Modelfile-equivalent
// /api/create payloads live in templates.go.
//
// Spec sources:
//   - the ensure-loop blob and create flow
//   - the translate_*.go files here (OpenAI <-> Ollama field maps)
//   - olares-ollama/internal/ollama/client.go (NDJSON streaming behavior)
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptrace"
	"os"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

// TraceLevel selects how much per-request audit logging the Client
// emits around every HTTP call to the Ollama daemon. Mirrors the
// config.LogUpstreamTrace* constants (kept here as a plain string
// type so the adapter package does not import config). Values:
//
//   - TraceOff   (zero value / "off"): success silent; failures still
//     log one ERROR line with minimal context. Zero httptrace
//     overhead on the hot path.
//   - TraceInfo  ("info"): one INFO line per call exit carrying
//     status, total_ms, ttfb_ms, connection-reuse info.
//   - TraceDebug ("debug"): DEBUG entry + DEBUG exit, all httptrace
//     phase timings (got_conn, wrote_request, ttfb, idle_time).
//
// Plumbed into Client.TraceLevel by NewAdapter (Ollama engine) and
// the ollama-native registrar in cmd/llm-init/main.go.
type TraceLevel string

// TraceOff disables every per-call trace line; only ERROR-level
// failures from the upstream call still surface.
// TraceInfo emits one INFO line per upstream call carrying status,
// total_ms, and connection-reuse info.
// TraceDebug emits DEBUG entry/exit pairs with full httptrace phase
// timings (got_conn, wrote_request_ms, ttfb_ms, idle_time_ms, remote
// addr); expect ~2 lines per HTTP call.
const (
	TraceOff   TraceLevel = "off"
	TraceInfo  TraceLevel = "info"
	TraceDebug TraceLevel = "debug"
)

// Client is a thin Ollama HTTP wrapper. All methods are safe for
// concurrent use; the underlying http.Client may pool connections.
type Client struct {
	BaseURL string
	HTTP    *http.Client

	// Upload is used for blob pushes and create streams. Ollama can take
	// minutes (large GGUF) so a separate client with no overall timeout
	// avoids the 30-min cap on HTTP being mistaken for a fault.
	Upload *http.Client

	// TraceLevel gates per-request audit logging. Default zero value
	// is TraceOff which matches the pre-tracing behaviour exactly
	// (no log lines on success, just propagate the http error).
	// Set by NewAdapter from cfg.Log.UpstreamTrace; tests can mutate
	// directly after NewClient since the field is exported.
	TraceLevel TraceLevel
}

// NewClient builds a Client with sane defaults. baseURL is normalized to
// drop any trailing slash so we can string-concatenate /api/* without
// double slashes leaking into the URL.
func NewClient(baseURL string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		HTTP: &http.Client{
			Timeout: 30 * time.Minute, // long inference requests
			Transport: &http.Transport{
				Proxy:           http.ProxyFromEnvironment,
				IdleConnTimeout: 90 * time.Second,
				// UPSTREAM_RESPONSE_HEADER_TIMEOUT (default 5m).
				// Previously 60s, which still missed large-model
				// prefills and the long tail of HAMI vGPU lock
				// re-acquire after idle. The body stream itself is
				// unbounded (client Timeout is 30m).
				ResponseHeaderTimeout: config.DefaultUpstreamResponseHeaderTimeout,
				ExpectContinueTimeout: time.Second,
			},
		},
		Upload: &http.Client{
			Timeout: 0, // multi-GiB blob uploads are unbounded
			Transport: &http.Transport{
				Proxy:                 http.ProxyFromEnvironment,
				IdleConnTimeout:       10 * time.Minute,
				ResponseHeaderTimeout: 10 * time.Minute,
				ExpectContinueTimeout: 30 * time.Second,
			},
		},
	}
}

// SetResponseHeaderTimeout applies UPSTREAM_RESPONSE_HEADER_TIMEOUT to
// the inference HTTP client. Zero falls back to the 5m default. The
// Upload client keeps its own 10m cap (blob pushes are not TTFT-bound).
func (c *Client) SetResponseHeaderTimeout(d time.Duration) {
	if c == nil || c.HTTP == nil {
		return
	}
	if d <= 0 {
		d = config.DefaultUpstreamResponseHeaderTimeout
	}
	tr, ok := c.HTTP.Transport.(*http.Transport)
	if !ok {
		return
	}
	tr.ResponseHeaderTimeout = d
}

// ModelDetails mirrors the upstream Ollama /api/tags "details" object.
// Open WebUI and other native clients expect the full tags schema
// (model, digest, details), not just name/size.
type ModelDetails struct {
	ParentModel       string   `json:"parent_model,omitempty"`
	Format            string   `json:"format,omitempty"`
	Family            string   `json:"family,omitempty"`
	Families          []string `json:"families,omitempty"`
	ParameterSize     string   `json:"parameter_size,omitempty"`
	QuantizationLevel string   `json:"quantization_level,omitempty"`
}

// Model is one entry in /api/tags. Field set matches upstream Ollama
// (docs/api.md "List local models"); ollamanative rewrites name+model
// to the llm-init alias while preserving digest/details/size.
type Model struct {
	Name       string        `json:"name"`
	Model      string        `json:"model,omitempty"`
	ModifiedAt time.Time     `json:"modified_at"`
	Size       int64         `json:"size"`
	Digest     string        `json:"digest,omitempty"`
	Details    *ModelDetails `json:"details,omitempty"`
}

// TagsResponse is the /api/tags body.
type TagsResponse struct {
	Models []Model `json:"models"`
}

// PSResponse is the /api/ps body. Models in this list are currently
// resident; lifecycle uses len > 0 plus a name match to derive Loaded.
//
// Size / SizeVRAM are populated by Ollama >=0.1.31 (verified against
// upstream docs/api.md "List Running Models", May 2026): Size is the
// total model weight bytes, SizeVRAM is how much of that currently
// lives on the device. The diag layer uses the ratio to decide
// gpu.mode (full / partial / cpu_only); older Ollama versions return
// 0 for SizeVRAM in which case the adapter reports cpu_only and
// adds a warning so operators can tell "old engine" apart from
// "actually CPU-only".
type PSResponse struct {
	Models []PSModel `json:"models"`
}

// PSModel is one entry inside PSResponse.Models.
type PSModel struct {
	Name     string `json:"name"`
	Model    string `json:"model"`
	Size     int64  `json:"size"`
	SizeVRAM int64  `json:"size_vram"`
}

// Tags calls GET /api/tags. Errors include the HTTP status when the
// upstream returned non-200 so operators can see "model 502" vs
// "connection refused".
func (c *Client) Tags(ctx context.Context) (TagsResponse, error) {
	var out TagsResponse
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/api/tags", nil)
	if err != nil {
		return out, err
	}
	resp, err := c.do(c.HTTP, req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("ollama: GET /api/tags status %d", resp.StatusCode)
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return out, fmt.Errorf("ollama: decode tags: %w", err)
	}
	return out, nil
}

// PS calls GET /api/ps. Some Ollama versions return 404; the caller should
// treat error as "Loaded unknown" (Loaded=nil in ReadyState).
func (c *Client) PS(ctx context.Context) (PSResponse, error) {
	var out PSResponse
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/api/ps", nil)
	if err != nil {
		return out, err
	}
	resp, err := c.do(c.HTTP, req)
	if err != nil {
		return out, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return out, fmt.Errorf("ollama: GET /api/ps status %d", resp.StatusCode)
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return out, fmt.Errorf("ollama: decode ps: %w", err)
	}
	return out, nil
}

// Version calls GET /api/version. Unlike Tags/PS this returns the live
// *http.Response so the caller can byte-copy the body straight onto its
// own ResponseWriter (Ollama-native passthrough; see internal/ollamanative).
// The caller is responsible for closing resp.Body.
//
// On transport error returns (nil, err). On non-2xx the response is
// still returned (so the caller can mirror the upstream status to its
// client) plus a nil error -- the contract is "we made it to the
// daemon, here is what it said".
func (c *Client) Version(ctx context.Context) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/api/version", nil)
	if err != nil {
		return nil, err
	}
	return c.do(c.HTTP, req)
}

// Show calls POST /api/show with body forwarded verbatim. Caller MUST
// have already rewritten any `name`/`model` JSON fields to the upstream
// tag before invoking; this method does no parsing.
//
// Like Version, returns the live response for byte-level passthrough.
// Non-2xx responses are not treated as errors so callers can mirror the
// upstream status (404 on unknown model, 503 on engine restart, etc.).
func (c *Client) Show(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/api/show", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.do(c.HTTP, req)
}

// ModelExists reports whether modelName matches any /api/tags entry. The
// match logic mirrors olares-ollama/internal/ollama/client.go::ModelExists:
//
//   - exact match wins;
//   - "name:tag" → also accept any "name" or "name:..." in the list;
//   - "name"     → also accept "name:tag" suffix.
//
// This handles both `pull qwen2.5:7b` (which lists "qwen2.5:7b") and
// `MODEL_NAME=qwen2.5` lookups.
func (c *Client) ModelExists(ctx context.Context, modelName string) (bool, error) {
	t, err := c.Tags(ctx)
	if err != nil {
		return false, err
	}
	for _, m := range t.Models {
		if m.Name == modelName {
			return true, nil
		}
	}
	if strings.Contains(modelName, ":") {
		base := strings.SplitN(modelName, ":", 2)[0]
		for _, m := range t.Models {
			if m.Name == base || strings.HasPrefix(m.Name, base+":") {
				return true, nil
			}
		}
	}
	for _, m := range t.Models {
		if strings.HasPrefix(m.Name, modelName+":") {
			return true, nil
		}
	}
	return false, nil
}

// PullRequest is the /api/pull body.
type PullRequest struct {
	Name string `json:"name"`
}

// PullResponse is one NDJSON line emitted by /api/pull.
type PullResponse struct {
	Status    string `json:"status"`
	Digest    string `json:"digest,omitempty"`
	Total     int64  `json:"total,omitempty"`
	Completed int64  `json:"completed,omitempty"`
}

// PullCallback is invoked once per NDJSON frame. Implementations are
// expected to be cheap; the stream may emit dozens of frames per second
// for the duration of a multi-GiB pull.
type PullCallback func(PullResponse)

// ollamaStatusSuccess is the terminal `status` value Ollama emits on
// /api/pull and /api/create when a stream completes successfully.
// Centralised so /api/pull and /api/create stay in lock-step; if the
// upstream protocol ever renames it both call sites move together.
const ollamaStatusSuccess = "success"

// ErrPullNoSuccess is returned when /api/pull's stream ends without ever
// emitting status="success". The reference implementation distinguishes
// this from network-level errors so the caller can decide to verify via
// /api/tags before declaring failure (see olares-ollama 372–409).
var ErrPullNoSuccess = errors.New("ollama: pull stream ended without success")

// Pull streams /api/pull and reports each frame to cb. The stream is
// considered successful only when at least one frame had status="success";
// otherwise ErrPullNoSuccess is returned and the caller should fall back
// to ModelExists for confirmation.
func (c *Client) Pull(ctx context.Context, modelName string, cb PullCallback) error {
	body, err := json.Marshal(PullRequest{Name: modelName})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+"/api/pull", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.do(c.Upload, req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("ollama: POST /api/pull status %d: %s",
			resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	dec := json.NewDecoder(bufio.NewReaderSize(resp.Body, 256*1024))
	gotSuccess := false
	for {
		var fr PullResponse
		if err := dec.Decode(&fr); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			// Mid-stream decode failure (network blip, JSON corruption);
			// surface to caller verbatim.
			return fmt.Errorf("ollama: decode pull frame: %w", err)
		}
		if cb != nil {
			cb(fr)
		}
		if fr.Status == ollamaStatusSuccess {
			gotSuccess = true
		}
	}
	if !gotSuccess {
		return ErrPullNoSuccess
	}
	return nil
}

// BlobExists calls HEAD /api/blobs/{digest}. The digest is passed as the
// path segment verbatim (e.g. "sha256:abc..."); callers must include the
// "sha256:" prefix.
func (c *Client) BlobExists(ctx context.Context, digest string) (bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead,
		c.BaseURL+"/api/blobs/"+digest, nil)
	if err != nil {
		return false, err
	}
	resp, err := c.do(c.HTTP, req)
	if err != nil {
		return false, err
	}
	resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	default:
		return false, fmt.Errorf("ollama: HEAD /api/blobs status %d", resp.StatusCode)
	}
}

// PushBlob streams a local file to POST /api/blobs/{digest}. Ollama
// accepts both 200 and 201 (server-side dedup vs new write).
func (c *Client) PushBlob(ctx context.Context, digest, filePath string) error {
	f, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("ollama: open blob source: %w", err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return fmt.Errorf("ollama: stat blob source: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+"/api/blobs/"+digest, f)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	req.ContentLength = info.Size()
	resp, err := c.do(c.Upload, req)
	if err != nil {
		return fmt.Errorf("ollama: POST /api/blobs: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("ollama: POST /api/blobs status %d: %s",
			resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	return nil
}

// CreateRequest matches the new (Ollama >=0.5) /api/create JSON schema.
// Either Files (initial creation from local GGUF blobs) or From (clone an
// existing model and override metadata) drives the operation.
type CreateRequest struct {
	Model      string                 `json:"model"`
	From       string                 `json:"from,omitempty"`
	Files      map[string]string      `json:"files,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Template   string                 `json:"template,omitempty"`
	System     string                 `json:"system,omitempty"`
}

// CreateResponse is one NDJSON frame from /api/create.
type CreateResponse struct {
	Status string `json:"status"`
}

// ErrCreateNoSuccess mirrors ErrPullNoSuccess for /api/create streams.
var ErrCreateNoSuccess = errors.New("ollama: create stream ended without success")

// Create streams /api/create. Like Pull, it requires status="success" at
// least once before returning nil; bare EOF returns ErrCreateNoSuccess.
func (c *Client) Create(ctx context.Context, req CreateRequest) error {
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+"/api/create", bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.do(c.Upload, httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("ollama: POST /api/create status %d: %s",
			resp.StatusCode, strings.TrimSpace(string(snippet)))
	}
	dec := json.NewDecoder(bufio.NewReaderSize(resp.Body, 64*1024))
	gotSuccess := false
	for {
		var fr CreateResponse
		if err := dec.Decode(&fr); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return fmt.Errorf("ollama: decode create frame: %w", err)
		}
		if fr.Status == ollamaStatusSuccess {
			gotSuccess = true
		}
	}
	if !gotSuccess {
		return ErrCreateNoSuccess
	}
	return nil
}

// DeleteModel calls DELETE /api/delete. Errors are logged but should not
// be treated as fatal — the model may simply not exist yet.
func (c *Client) DeleteModel(ctx context.Context, modelName string) error {
	body, _ := json.Marshal(map[string]string{keyModel: modelName})
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete,
		c.BaseURL+"/api/delete", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.do(c.HTTP, req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNotFound {
		return fmt.Errorf("ollama: DELETE /api/delete status %d", resp.StatusCode)
	}
	return nil
}

// StatusError is the non-2xx result of a PostJSON call. The status code
// is carried as a field rather than only formatted into the message so a
// caller can branch on it: the chat translator retries a 400 once with a
// boolean `think`, because a daemon older than the thinking levels
// rejects `"think":"low"` with exactly that status.
type StatusError struct {
	Path   string
	Status int
	Body   string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("ollama: POST %s status %d: %s", e.Path, e.Status, e.Body)
}

// PostJSON is the building block translators use to forward already-built
// JSON bodies. It returns the live http.Response so callers can stream
// the body (chat / generate / embed).
//
// path must start with "/api/...". Non-2xx is reported as a *StatusError
// so callers can decide retry vs surface to client.
func (c *Client) PostJSON(ctx context.Context, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.do(c.HTTP, req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		resp.Body.Close()
		return nil, &StatusError{
			Path:   path,
			Status: resp.StatusCode,
			Body:   strings.TrimSpace(string(snippet)),
		}
	}
	return resp, nil
}

// do is the single funnel every HTTP call on this Client passes
// through. Centralising the call lets TraceLevel-gated httptrace
// instrumentation attach without each helper having to remember.
//
// Behaviour matrix (level on the left):
//
//   - off:   identical to client.Do(req) on the happy path; on
//     failure, logs one ERROR line carrying path, method, total
//     elapsed, and the unwrapped error. No httptrace registered,
//     so the hot path adds nothing beyond a time.Now() pair.
//   - info:  installs the minimal httptrace ClientTrace, then logs
//     one INFO line at exit (success or failure) with status,
//     total_ms, conn_reused, was_idle, idle_time_ms, remote,
//     ttfb_ms (and ttfb_observed=false when first byte never came).
//   - debug: same trace, plus a DEBUG "ollama upstream call" line
//     at entry. The exit line drops to DEBUG too so callers can
//     gate the full chatter with LOG_LEVEL=debug if desired.
//
// httpClient is passed explicitly (rather than always c.HTTP)
// because /api/blob and /api/create flow through c.Upload which
// has different timeout settings.
func (c *Client) do(httpClient *http.Client, req *http.Request) (*http.Response, error) {
	return doWithTrace(req, httpClient, c.TraceLevel, slog.Default())
}

// doWithTrace is the package-level workhorse split out from
// (*Client).do so unit tests can drive it without constructing a
// Client. Keeping the call site in one place also lets us add
// future trace fields (e.g. dns_ms, tls_ms) without touching every
// helper. logger is taken as a parameter rather than a global so
// tests can capture lines into a *bytes.Buffer.
func doWithTrace(req *http.Request, client *http.Client, level TraceLevel, logger *slog.Logger) (*http.Response, error) {
	path := req.URL.Path
	bodyBytes := -1
	if req.ContentLength > 0 {
		bodyBytes = int(req.ContentLength)
	}

	// Off path: cheapest possible. No httptrace registration, no
	// extra allocations on success. Failure still gets a one-line
	// audit because that's the case operators always want to see.
	if level == "" || level == TraceOff {
		start := time.Now()
		resp, err := client.Do(req)
		if err != nil {
			logger.Error("ollama upstream failed",
				slog.String("path", path),
				slog.String("method", req.Method),
				slog.Int("body_bytes", bodyBytes),
				slog.Int64("total_ms", time.Since(start).Milliseconds()),
				slog.String("err", err.Error()),
			)
		}
		return resp, err
	}

	// Info / debug path: register the trace, capture five times,
	// then emit one structured record per exit.
	var (
		wroteReq, firstByte time.Time
		reused, wasIdle     bool
		idleTime            time.Duration
		remote              string
	)
	trace := &httptrace.ClientTrace{
		GotConn: func(info httptrace.GotConnInfo) {
			reused = info.Reused
			wasIdle = info.WasIdle
			idleTime = info.IdleTime
			if info.Conn != nil {
				remote = info.Conn.RemoteAddr().String()
			}
		},
		WroteRequest:         func(httptrace.WroteRequestInfo) { wroteReq = time.Now() },
		GotFirstResponseByte: func() { firstByte = time.Now() },
	}
	req = req.WithContext(httptrace.WithClientTrace(req.Context(), trace))

	if level == TraceDebug {
		logger.Debug("ollama upstream call",
			slog.String("path", path),
			slog.String("method", req.Method),
			slog.Int("body_bytes", bodyBytes),
		)
	}

	start := time.Now()
	resp, err := client.Do(req)
	total := time.Since(start)

	fields := []any{
		slog.String("path", path),
		slog.String("method", req.Method),
		slog.Int("body_bytes", bodyBytes),
		slog.Int64("total_ms", total.Milliseconds()),
		slog.Bool("conn_reused", reused),
		slog.Bool("was_idle", wasIdle),
		slog.Int64("idle_time_ms", idleTime.Milliseconds()),
		slog.String("remote", remote),
		slog.Bool("ttfb_observed", !firstByte.IsZero()),
	}
	if !wroteReq.IsZero() {
		fields = append(fields, slog.Int64("wrote_request_ms", wroteReq.Sub(start).Milliseconds()))
	}
	if !firstByte.IsZero() {
		fields = append(fields, slog.Int64("ttfb_ms", firstByte.Sub(start).Milliseconds()))
	}

	if err != nil {
		// Prepend err so structured-log readers see the cause first.
		logger.Error("ollama upstream failed",
			append([]any{slog.String("err", err.Error())}, fields...)...,
		)
		return nil, err
	}

	fields = append(fields, slog.Int("status", resp.StatusCode))
	if level == TraceDebug {
		logger.Debug("ollama upstream done", fields...)
	} else {
		logger.Info("ollama upstream done", fields...)
	}
	return resp, nil
}
