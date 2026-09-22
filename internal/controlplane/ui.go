package controlplane

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
	"strings"
)

//go:embed static
var staticFS embed.FS

// attachUI registers `/` (dashboard) and `/static/*` (assets). The UI
// is a single-page dashboard that polls /api/progress every 1 s so
// operators can watch model download / engine warm-up live. (Pre-v1.1.0
// the dashboard also subscribed to /api/events; SSE was retired in
// favour of polling.)
func (s *Server) attachUI(m *http.ServeMux) {
	sub, err := fs.Sub(staticFS, "static")
	if err != nil {
		// Should be impossible at runtime: we control the embed root.
		return
	}
	fileServer := http.FileServer(http.FS(sub))

	m.Handle("/static/", http.StripPrefix("/static/", fileServer))
	m.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Only serve the dashboard for the literal root; everything else
		// (e.g. /unknown) returns 404 to keep telemetry meaningful.
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed",
				"only GET / HEAD are allowed on /")
			return
		}
		// Content negotiation: clients that explicitly ask for JSON (e.g.
		// scripted health probes that want a non-HTML body) get a tiny
		// status payload instead of the dashboard. We require BOTH that
		// application/json is in Accept AND that text/html is not, so
		// regular browsers (Accept: text/html,application/xhtml+xml,...)
		// still get the UI.
		if wantsJSON(r.Header.Get("Accept")) {
			w.Header().Set("Content-Type", contentTypeJSON)
			_ = json.NewEncoder(w).Encode(map[string]any{
				jsonKeyStatus: "ok",
				"ui":          "available at /",
			})
			return
		}
		http.ServeFileFS(w, r, sub, "index.html")
	})
}

// wantsJSON returns true when the Accept header lists application/json
// without listing text/html. Empty Accept header or a wildcard
// (*/*) → false (HTML wins).
func wantsJSON(accept string) bool {
	if accept == "" {
		return false
	}
	hasJSON := false
	hasHTML := false
	for _, part := range strings.Split(accept, ",") {
		mt := strings.TrimSpace(part)
		// Strip ;q=... weight markers; we don't honour weights, just
		// presence. Spec-strict negotiation is overkill for v1.0.
		if i := strings.Index(mt, ";"); i >= 0 {
			mt = strings.TrimSpace(mt[:i])
		}
		switch mt {
		case contentTypeJSON:
			hasJSON = true
		case contentTypeHTML:
			hasHTML = true
		}
	}
	return hasJSON && !hasHTML
}
