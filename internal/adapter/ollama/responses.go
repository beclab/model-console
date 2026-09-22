package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
)

const maxResponsesBody = maxInferenceRewriteBody // 16 MiB

// handleResponses reverse-proxies POST /v1/responses to the Ollama
// daemon's OpenAI-compatible surface (v0.13.3+). The client-facing
// model alias is rewritten to the upstream daemon tag before forward;
// Authorization is stripped so llm-init's bearer never reaches the engine.
func (a *Adapter) handleResponses(w http.ResponseWriter, r *http.Request) {
	if responsesAdmission(w, r) {
		return
	}
	target, err := url.Parse(a.cfg.Engine.URL)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "internal_error",
			"parse engine URL: "+err.Error())
		return
	}
	if target.Scheme == "" || target.Host == "" {
		writeJSONError(w, http.StatusInternalServerError, "internal_error",
			"engine URL must include scheme and host")
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Transport = a.client.HTTP.Transport
	proxy.FlushInterval = -1
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		writeJSONError(w, http.StatusBadGateway, "adapter_unreachable",
			"ollama /v1/responses: "+err.Error())
	}
	upstreamModel := a.upstreamModel()
	origDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		origDirector(req)
		StripProxyAuth(req.Header)
		rewriteResponsesModel(req, upstreamModel)
	}
	proxy.ServeHTTP(w, r)
}

func responsesAdmission(w http.ResponseWriter, req *http.Request) bool {
	if req.ContentLength > maxResponsesBody {
		writeResponsesTooLarge(w, req.ContentLength)
		return true
	}
	if req.Body == nil {
		return false
	}
	buf, err := io.ReadAll(io.LimitReader(req.Body, maxResponsesBody+1))
	_ = req.Body.Close()
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "bad_request",
			"request body could not be read fully")
		return true
	}
	if len(buf) > maxResponsesBody {
		writeResponsesTooLarge(w, int64(len(buf)))
		return true
	}
	req.Body = io.NopCloser(bytes.NewReader(buf))
	req.ContentLength = int64(len(buf))
	req.Header.Set("Content-Length", fmt.Sprintf("%d", len(buf)))
	req.TransferEncoding = nil
	req.Header.Del("Transfer-Encoding")
	return false
}

func writeResponsesTooLarge(w http.ResponseWriter, observed int64) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Llm-Init-Max-Rewrite-Bytes", fmt.Sprintf("%d", maxResponsesBody))
	w.WriteHeader(http.StatusRequestEntityTooLarge)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"code":     "rewrite_body_too_large",
			keyMessage: "request body exceeds the model-rewrite cap (16 MiB); reduce payload size or address the engine directly",
		},
	})
	slog.Warn("ollama: rejected oversized /v1/responses body",
		slog.Int64("observed_bytes", observed),
		slog.Int("cap", maxResponsesBody))
}

func rewriteResponsesModel(req *http.Request, upstreamModel string) {
	if req.Body == nil {
		return
	}
	body, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	if err != nil {
		req.Body = io.NopCloser(bytes.NewReader(nil))
		req.ContentLength = 0
		return
	}
	rewritten, ok := RewriteJSONModelField(body, upstreamModel)
	if !ok {
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
		return
	}
	req.Body = io.NopCloser(bytes.NewReader(rewritten))
	req.ContentLength = int64(len(rewritten))
	req.Header.Set("Content-Length", fmt.Sprintf("%d", len(rewritten)))
}
