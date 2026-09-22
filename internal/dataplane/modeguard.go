package dataplane

import (
	"encoding/json"
	"net/http"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/moderoutes"
)

// codeNotServed is emitted on the 404 envelope when a request names a
// path this application's mode does not serve.
const codeNotServed = "endpoint_not_served"

// ModeGuard refuses paths outside the mode's declared route set.
//
// /v1/ is registered as a catch-all, and the adapter forwards whatever
// arrives to the engine. On an embedding or OCR application that meant
// POST /v1/chat/completions was accepted and passed to an engine that
// has no such route, so the client received that engine's own 404 in
// that engine's own format — while GET /api/endpoints had already
// reported chat as unavailable. Same question, two answers; this makes
// the port agree with the catalogue, out of the same table.
//
// Modes with no declared set are passed through untouched, and this
// returns next unwrapped for them so there is nothing in their hot path.
//
// It sits outside NotReadyGuard on purpose. A path this application will
// never serve should say so while the model is still downloading too:
// 503 with a Retry-After invites a client to keep asking for something
// that is not going to appear.
func ModeGuard(mode config.ModelType, next http.Handler) http.Handler {
	if !moderoutes.Restricted(mode) {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if moderoutes.ServesPath(mode, r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		writeNotServed(w, mode, r.URL.Path)
	})
}

// writeNotServed emits the 404 envelope, shaped like every other error
// this package writes so the same client parser reads it.
//
// No Retry-After: this is not a timing answer.
func writeNotServed(w http.ResponseWriter, mode config.ModelType, path string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"code": codeNotServed,
			"message": "this application runs mode=" + string(mode) +
				" and does not serve " + path +
				"; GET /api/endpoints lists what it does serve",
			"mode": string(mode),
		},
	})
}
