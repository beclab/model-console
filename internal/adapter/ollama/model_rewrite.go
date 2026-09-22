package ollama

import (
	"bytes"
	"encoding/json"
	"net/http"
)

// maxInferenceRewriteBody caps JSON bodies whose top-level "model" field
// ollamanative and /v1/responses rewrite before forwarding upstream.
const maxInferenceRewriteBody = 16 << 20 // 16 MiB

// MaxInferenceRewriteBody is exported for ollamanative admission checks.
const MaxInferenceRewriteBody = maxInferenceRewriteBody

// StripProxyAuth removes client credentials before forwarding to the
// engine so llm-init's bearer never reaches the daemon.
func StripProxyAuth(h http.Header) {
	h.Del("Authorization")
	h.Del("Proxy-Authorization")
}

// RewriteJSONModelField replaces the top-level JSON "model" string when
// body is a JSON object. Non-object bodies are returned unchanged with
// ok=false so callers forward them verbatim.
func RewriteJSONModelField(body []byte, modelName string) ([]byte, bool) {
	trimmed := bytes.TrimLeft(body, " \t\r\n")
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return body, false
	}
	var obj map[string]any
	if err := json.Unmarshal(body, &obj); err != nil {
		return body, false
	}
	obj["model"] = modelName
	out, err := json.Marshal(obj)
	if err != nil {
		return body, false
	}
	return out, true
}
