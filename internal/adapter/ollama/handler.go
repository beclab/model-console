package ollama

import (
	"encoding/json"
	"net/http"
)

// buildHandler assembles the /v1/* router. OpenAI chat/completions/
// embeddings/models are translated to Ollama /api/*; /v1/responses is
// reverse-proxied to the daemon's native OpenAI-compatible endpoint.
func (a *Adapter) buildHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", a.handleChat)
	// OpenWebUI compatibility alias.
	mux.HandleFunc("POST /api/chat/completions", a.handleChat)
	mux.HandleFunc("POST /v1/completions", a.handleCompletion)
	mux.HandleFunc("POST /v1/embeddings", a.handleEmbed)
	mux.HandleFunc("GET /v1/models", a.handleModels)
	mux.HandleFunc("POST /v1/responses", a.handleResponses)
	return mux
}

// writeJSONError emits the unified error envelope.
func writeJSONError(w http.ResponseWriter, status int, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"code": code, keyMessage: msg},
	})
}
