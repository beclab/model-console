// Package mockollama is a minimal, controllable stand-in for the Ollama
// daemon used by the offline download test suite and the manual runbook.
// It implements just the endpoints lifecycle's Ollama paths touch:
// /api/tags, /api/ps, /api/version, /api/pull, /api/blobs/{digest}, and
// /api/create. Faults (non-200 pull, missing success frame, blob push
// rejection) are configurable so the suite can assert that the frontend
// receives a correct, actionable message for invalid tags and
// permission failures.
package mockollama

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
)

// statusField is the NDJSON key the real daemon streams on /api/pull and
// /api/create progress frames.
const statusField = "status"

// Config controls the daemon's responses.
type Config struct {
	// ExistingModels pre-populates /api/tags (and is what a
	// "recover after no-success" pull falls back to).
	ExistingModels []string

	// PullStatus is the HTTP status for POST /api/pull (default 200).
	// Use 404 to simulate an invalid/unknown tag, 403 for a denied
	// pull.
	PullStatus int

	// PullBody is the body returned with a non-200 PullStatus
	// (default: a short message matching the status).
	PullBody string

	// PullFrames is how many progress NDJSON frames to emit before the
	// terminal frame (default 2).
	PullFrames int

	// PullOmitSuccess ends the /api/pull stream without a
	// status:"success" frame, exercising the ErrPullNoSuccess +
	// /api/tags re-check path.
	PullOmitSuccess bool

	// BlobPushStatus is the status for POST /api/blobs/{digest}
	// (default 201). Use 403 to simulate a rejected blob upload.
	BlobPushStatus int

	// CreateOmitSuccess ends the /api/create stream without
	// status:"success", exercising ErrCreateNoSuccess.
	CreateOmitSuccess bool
}

// Server wraps an httptest.Server and the live request log.
type Server struct {
	*httptest.Server

	cfg Config

	mu       sync.Mutex
	models   map[string]bool
	blobs    map[string]bool
	pullReqs int
	creates  int
}

// New starts a mock daemon.
func New(cfg Config) *Server {
	if cfg.PullStatus == 0 {
		cfg.PullStatus = http.StatusOK
	}
	if cfg.BlobPushStatus == 0 {
		cfg.BlobPushStatus = http.StatusCreated
	}
	if cfg.PullFrames == 0 {
		cfg.PullFrames = 2
	}
	s := &Server{
		cfg:    cfg,
		models: make(map[string]bool),
		blobs:  make(map[string]bool),
	}
	for _, m := range cfg.ExistingModels {
		s.models[m] = true
	}
	s.Server = httptest.NewServer(http.HandlerFunc(s.handle))
	return s
}

// PullCount / CreateCount expose live counters for assertions.
func (s *Server) PullCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pullReqs
}

// CreateCount returns the number of /api/create calls received.
func (s *Server) CreateCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.creates
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/api/tags":
		s.handleTags(w)
	case r.Method == http.MethodGet && r.URL.Path == "/api/ps":
		s.writeJSON(w, http.StatusOK, map[string]any{"models": []any{}})
	case r.Method == http.MethodGet && r.URL.Path == "/api/version":
		s.writeJSON(w, http.StatusOK, map[string]string{"version": "0.0.0-mock"})
	case r.Method == http.MethodPost && r.URL.Path == "/api/pull":
		s.handlePull(w, r)
	case strings.HasPrefix(r.URL.Path, "/api/blobs/"):
		s.handleBlob(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/api/create":
		s.handleCreate(w, r)
	default:
		http.Error(w, "mockollama: not found", http.StatusNotFound)
	}
}

func (s *Server) handleTags(w http.ResponseWriter) {
	s.mu.Lock()
	models := make([]map[string]any, 0, len(s.models))
	for name := range s.models {
		models = append(models, map[string]any{
			"name":   name,
			"model":  name,
			"size":   int64(1024),
			"digest": "sha256:0000",
		})
	}
	s.mu.Unlock()
	s.writeJSON(w, http.StatusOK, map[string]any{"models": models})
}

func (s *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name string `json:"name"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	s.mu.Lock()
	s.pullReqs++
	s.mu.Unlock()

	if s.cfg.PullStatus != http.StatusOK {
		body := s.cfg.PullBody
		if body == "" {
			body = fmt.Sprintf("mockollama pull rejected: %s", http.StatusText(s.cfg.PullStatus))
		}
		http.Error(w, body, s.cfg.PullStatus)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	enc := json.NewEncoder(w)
	digest := "sha256:abc123"
	total := int64(1 << 20)
	for i := 1; i <= s.cfg.PullFrames; i++ {
		_ = enc.Encode(map[string]any{
			statusField: "downloading",
			"digest":    digest,
			"total":     total,
			"completed": total * int64(i) / int64(s.cfg.PullFrames),
		})
		flush(w)
	}
	if !s.cfg.PullOmitSuccess {
		s.mu.Lock()
		s.models[req.Name] = true
		s.mu.Unlock()
		_ = enc.Encode(map[string]any{statusField: "success"})
	}
}

func (s *Server) handleBlob(w http.ResponseWriter, r *http.Request) {
	digest := strings.TrimPrefix(r.URL.Path, "/api/blobs/")
	switch r.Method {
	case http.MethodHead:
		s.mu.Lock()
		ok := s.blobs[digest]
		s.mu.Unlock()
		if ok {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	case http.MethodPost:
		_, _ = io.Copy(io.Discard, r.Body)
		if s.cfg.BlobPushStatus != http.StatusOK && s.cfg.BlobPushStatus != http.StatusCreated {
			http.Error(w, "mockollama: blob push denied", s.cfg.BlobPushStatus)
			return
		}
		s.mu.Lock()
		s.blobs[digest] = true
		s.mu.Unlock()
		w.WriteHeader(s.cfg.BlobPushStatus)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model string `json:"model"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)

	s.mu.Lock()
	s.creates++
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	enc := json.NewEncoder(w)
	_ = enc.Encode(map[string]any{statusField: "using existing layers"})
	flush(w)
	if !s.cfg.CreateOmitSuccess {
		s.mu.Lock()
		s.models[req.Model] = true
		s.mu.Unlock()
		_ = enc.Encode(map[string]any{statusField: "success"})
	}
}

func (s *Server) writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func flush(w http.ResponseWriter) {
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}
