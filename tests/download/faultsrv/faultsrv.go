// Package faultsrv is a controllable HTTP file server used by the
// offline download test suite (tests/download) and the manual runbook
// (tests/download/cmd/faultsrv). It serves a single in-memory payload
// and injects the abnormal conditions the fetch.RangeDownloader is
// expected to survive: HTTP Range / ETag handling, 5xx storms,
// permanent 4xx, 416, mid-stream connection resets, ETag rotation, and
// bandwidth throttling.
//
// The same struct backs both the Go tests (via New / NewTLS) and the
// standalone CLI, so a manually reproduced scenario is byte-for-byte
// the automated one.
package faultsrv

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// Config describes one payload plus the faults to inject. The zero
// value serves Body over a plain, range-capable 200/206 handler.
type Config struct {
	// Body is the payload served on GET. Generated deterministically
	// by GenerateBody when empty so callers always have a known
	// size + SHA-256.
	Body []byte

	// ETag is the strong validator returned on HEAD/GET. Defaults to
	// a content hash when empty.
	ETag string

	// SupportRange honours `Range: bytes=N-` requests with a 206. When
	// false the server ignores Range and always returns 200 (exercises
	// the downloader's "server ignored Range -> rewrite from 0" path).
	SupportRange bool

	// OmitContentLength drops the Content-Length header on HEAD/GET so
	// the downloader runs the unknown-size path.
	OmitContentLength bool

	// HeadStatus, when > 0, is returned verbatim on HEAD (e.g. 403 or
	// 405 to exercise the soft-fail "proceed without size/ETag" path).
	HeadStatus int

	// ForceStatus, when > 0, is returned on every GET (e.g. 403/404 for
	// a permanent failure, 429 for a retriable one).
	ForceStatus int

	// FailFirstN returns FailStatus for the first N GET attempts, then
	// serves normally — a recoverable "5xx storm" the inner retry loop
	// must ride out.
	FailFirstN int32

	// FailStatus is the status used for FailFirstN attempts. Defaults
	// to 503.
	FailStatus int

	// DropAfter, when > 0, writes that many body bytes on each GET then
	// hijacks and closes the connection, simulating a mid-stream reset.
	// Combine with FailFirstN to bound how many attempts drop.
	DropAfter int

	// DropFirstN bounds DropAfter to the first N GET attempts (0 means
	// "every attempt"). Lets a resume scenario eventually complete.
	DropFirstN int32

	// ETagRotateAfter, when > 0, changes the served ETag after that many
	// GET attempts. A resuming client that sent If-Range with the old
	// validator gets a fresh 200 and must restart from zero.
	ETagRotateAfter int32

	// Throttle sleeps for this duration after writing each 32 KiB chunk,
	// slowing the transfer so progress / cancellation can be observed.
	Throttle time.Duration
}

// Server wraps an httptest.Server and the live attempt counters.
type Server struct {
	*httptest.Server
	cfg       Config
	getCount  atomic.Int32
	headCount atomic.Int32
}

// GenerateBody returns deterministic pseudo-random bytes of length n so
// every consumer (tests, scripts) sees the same payload + digest.
func GenerateBody(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((i*31 + 7) % 251)
	}
	return b
}

// SHA256Hex returns the lowercase hex SHA-256 of b.
func SHA256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// New starts a plaintext fault server. Close it via Server.Close (or
// rely on httptest's cleanup when constructed inside a test).
func New(cfg Config) *Server {
	s := newServer(cfg)
	s.Server = httptest.NewServer(http.HandlerFunc(s.handle))
	return s
}

// NewTLS starts the same handler over HTTPS. Use Server.Client() for a
// client that trusts the test certificate.
func NewTLS(cfg Config) *Server {
	s := newServer(cfg)
	s.Server = httptest.NewTLSServer(http.HandlerFunc(s.handle))
	return s
}

// Handler returns the bare http.Handler (with its own live counters) so
// the standalone CLI can serve it on a fixed address. Tests use New /
// NewTLS instead.
func Handler(cfg Config) http.Handler {
	return http.HandlerFunc(newServer(cfg).handle)
}

func newServer(cfg Config) *Server {
	if len(cfg.Body) == 0 {
		cfg.Body = GenerateBody(64 * 1024)
	}
	if cfg.ETag == "" {
		cfg.ETag = `"` + SHA256Hex(cfg.Body)[:16] + `"`
	}
	if cfg.FailStatus == 0 {
		cfg.FailStatus = http.StatusServiceUnavailable
	}
	return &Server{cfg: cfg}
}

// BodySHA256 is the lowercase hex digest of the served payload, handy
// for building a `#sha256=` fragment in tests.
func (s *Server) BodySHA256() string { return SHA256Hex(s.cfg.Body) }

// BodySize is the served payload length.
func (s *Server) BodySize() int64 { return int64(len(s.cfg.Body)) }

// GetCount returns the live GET attempt counter.
func (s *Server) GetCount() int32 { return s.getCount.Load() }

// HeadCount returns the live HEAD attempt counter.
func (s *Server) HeadCount() int32 { return s.headCount.Load() }

// currentETag rotates the validator once ETagRotateAfter GETs have
// landed so a resuming client's If-Range stops matching.
func (s *Server) currentETag(getN int32) string {
	if s.cfg.ETagRotateAfter > 0 && getN > s.cfg.ETagRotateAfter {
		return `"rotated-` + strings.Trim(s.cfg.ETag, `"`) + `"`
	}
	return s.cfg.ETag
}

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodHead:
		s.handleHead(w)
	case http.MethodGet:
		s.handleGet(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleHead(w http.ResponseWriter) {
	s.headCount.Add(1)
	if s.cfg.HeadStatus > 0 {
		w.WriteHeader(s.cfg.HeadStatus)
		return
	}
	h := w.Header()
	h.Set("Accept-Ranges", "bytes")
	h.Set("ETag", s.cfg.ETag)
	if !s.cfg.OmitContentLength {
		h.Set("Content-Length", strconv.Itoa(len(s.cfg.Body)))
	}
	w.WriteHeader(http.StatusOK)
}

func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	n := s.getCount.Add(1)

	if s.cfg.ForceStatus > 0 {
		http.Error(w, http.StatusText(s.cfg.ForceStatus), s.cfg.ForceStatus)
		return
	}
	if s.cfg.FailFirstN > 0 && n <= s.cfg.FailFirstN {
		http.Error(w, http.StatusText(s.cfg.FailStatus), s.cfg.FailStatus)
		return
	}

	etag := s.currentETag(n)
	start, isRange := s.rangeStart(r, etag)
	body := s.cfg.Body

	if isRange {
		if start >= len(body) {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes */%d", len(body)))
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		w.Header().Set("ETag", etag)
		w.Header().Set("Content-Range",
			fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
		if !s.cfg.OmitContentLength {
			w.Header().Set("Content-Length", strconv.Itoa(len(body)-start))
		}
		w.WriteHeader(http.StatusPartialContent)
		s.writeBody(w, body[start:], n)
		return
	}

	w.Header().Set("ETag", etag)
	w.Header().Set("Accept-Ranges", "bytes")
	if !s.cfg.OmitContentLength {
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	}
	w.WriteHeader(http.StatusOK)
	s.writeBody(w, body, n)
}

// rangeStart parses `Range: bytes=N-` when SupportRange is on and the
// If-Range validator (if present) still matches the current ETag.
func (s *Server) rangeStart(r *http.Request, etag string) (start int, ok bool) {
	if !s.cfg.SupportRange {
		return 0, false
	}
	rng := r.Header.Get("Range")
	if !strings.HasPrefix(rng, "bytes=") {
		return 0, false
	}
	if ir := r.Header.Get("If-Range"); ir != "" && ir != etag {
		return 0, false // validator changed: serve full 200
	}
	spec := strings.TrimPrefix(rng, "bytes=")
	dash := strings.IndexByte(spec, '-')
	if dash < 0 {
		return 0, false
	}
	v, err := strconv.Atoi(spec[:dash])
	if err != nil || v < 0 {
		return 0, false
	}
	return v, true
}

// writeBody streams data, optionally throttling and optionally dropping
// the connection partway through to simulate a reset.
func (s *Server) writeBody(w http.ResponseWriter, data []byte, getN int32) {
	drop := s.cfg.DropAfter > 0 &&
		(s.cfg.DropFirstN == 0 || getN <= s.cfg.DropFirstN)
	if drop {
		limit := s.cfg.DropAfter
		if limit > len(data) {
			limit = len(data)
		}
		_, _ = w.Write(data[:limit])
		if hj, ok := w.(http.Hijacker); ok {
			if conn, _, err := hj.Hijack(); err == nil {
				_ = conn.Close()
			}
		}
		return
	}

	const chunk = 32 * 1024
	for off := 0; off < len(data); off += chunk {
		end := off + chunk
		if end > len(data) {
			end = len(data)
		}
		if _, err := w.Write(data[off:end]); err != nil {
			return
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		if s.cfg.Throttle > 0 {
			time.Sleep(s.cfg.Throttle)
		}
	}
}
