package ollamanative

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/llm-init/llm-init/internal/adapter/ollama"
	"github.com/llm-init/llm-init/internal/dataplane"
)

// nativeInferencePaths are Ollama daemon inference endpoints forwarded
// verbatim (Track P.3). Body JSON gets a top-level "model" rewrite to
// the upstream daemon tag; NDJSON streams are flushed without buffering
// the response.
var nativeInferencePaths = []string{
	"/api/chat",
	"/api/generate",
	"/api/embed",
	"/api/embeddings",
}

func (m *Mount) registerNativeInference(mux *http.ServeMux) {
	for _, path := range nativeInferencePaths {
		p := path
		mux.Handle("POST "+p,
			dataplane.NotReadyGuard(m.ready, m.progress, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				m.proxyNativeInference(w, r, p)
			})))
	}
}

func (m *Mount) proxyNativeInference(w http.ResponseWriter, r *http.Request, upstreamPath string) {
	if nativeInferenceAdmission(w, r) {
		return
	}
	target, err := url.Parse(m.client.BaseURL)
	if err != nil {
		writeUpstreamError(w, err)
		return
	}
	if target.Scheme == "" || target.Host == "" {
		writeOllamaError(w, http.StatusInternalServerError, "invalid engine URL")
		return
	}

	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Transport = m.client.HTTP.Transport
	proxy.FlushInterval = -1
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		writeUpstreamError(w, err)
	}
	upstreamModel := ollama.UpstreamModel(m.cfg)
	origDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		origDirector(req)
		ollama.StripProxyAuth(req.Header)
		rewriteNativeModel(req, upstreamModel)
	}
	r.URL.Path = upstreamPath
	proxy.ServeHTTP(w, r)
}

func nativeInferenceAdmission(w http.ResponseWriter, req *http.Request) bool {
	cap := int64(ollama.MaxInferenceRewriteBody)
	if req.ContentLength > cap {
		writeNativeBodyTooLarge(w, req.ContentLength)
		return true
	}
	if req.Body == nil {
		return false
	}
	buf, err := io.ReadAll(io.LimitReader(req.Body, cap+1))
	_ = req.Body.Close()
	if err != nil {
		writeOllamaError(w, http.StatusBadRequest, "invalid request body")
		return true
	}
	if int64(len(buf)) > cap {
		writeNativeBodyTooLarge(w, int64(len(buf)))
		return true
	}
	req.Body = io.NopCloser(bytes.NewReader(buf))
	req.ContentLength = int64(len(buf))
	req.Header.Set("Content-Length", fmt.Sprintf("%d", len(buf)))
	req.TransferEncoding = nil
	req.Header.Del("Transfer-Encoding")
	return false
}

func writeNativeBodyTooLarge(w http.ResponseWriter, _ int64) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Llm-Init-Max-Rewrite-Bytes", fmt.Sprintf("%d", ollama.MaxInferenceRewriteBody))
	w.WriteHeader(http.StatusRequestEntityTooLarge)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error": fmt.Sprintf("request body exceeds the model-rewrite cap (%d bytes)", ollama.MaxInferenceRewriteBody),
	})
}

func rewriteNativeModel(req *http.Request, upstreamModel string) {
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
	rewritten, ok := ollama.RewriteJSONModelField(body, upstreamModel)
	if !ok {
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
		return
	}
	req.Body = io.NopCloser(bytes.NewReader(rewritten))
	req.ContentLength = int64(len(rewritten))
	req.Header.Set("Content-Length", fmt.Sprintf("%d", len(rewritten)))
}
