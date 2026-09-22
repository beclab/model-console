package ollama

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

// recordingSink captures progress events so tests can assert the right
// stream of OnFileStart / OnBytes / OnFileDone / OnError calls reached
// the sink even when frames are coalesced or fired asynchronously.
type recordingSink struct {
	starts  []string
	bytes   int64
	dones   []string
	errored bool
}

func (r *recordingSink) OnFileStart(name string, _ int64) { r.starts = append(r.starts, name) }
func (r *recordingSink) OnBytes(n int64)                  { atomic.AddInt64(&r.bytes, n) }
func (r *recordingSink) OnFileDone(name string)           { r.dones = append(r.dones, name) }
func (r *recordingSink) OnError(error)                    { r.errored = true }
func (r *recordingSink) OnPhase(progress.Phase)           {}
func (r *recordingSink) OnResolvedCommit(string)          {}
func (r *recordingSink) OnNote(string)                    {}
func (r *recordingSink) OnFilesTotal(int)                 {}

var _ progress.Sink = (*recordingSink)(nil)

func TestAdapter_KindAndOpenAIHandlerNotNil(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: "http://x"}, Model: config.Model{Name: "m"}})
	if a.Kind() != config.EngineOllama {
		t.Errorf("Kind = %v", a.Kind())
	}
	if a.OpenAIHandler(config.Config{}) == nil {
		t.Error("OpenAIHandler returned nil")
	}
}

func TestAdapter_WaitAliveSucceeds(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"models":[]}`))
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "m"}})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := a.WaitAlive(ctx); err != nil {
		t.Errorf("WaitAlive: %v", err)
	}
}

func TestAdapter_WaitAliveCancels(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: "http://127.0.0.1:1"}, Model: config.Model{Name: "m"}})
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := a.WaitAlive(ctx); err == nil {
		t.Error("expected ctx error")
	}
}

func TestAdapter_ReadyTags(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5:7b","size":4700000000}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "qwen2.5"}})
	rs, err := a.Ready(context.Background())
	if err != nil {
		t.Fatalf("Ready: %v", err)
	}
	if !rs.Alive || !rs.ModelExists {
		t.Errorf("rs=%+v", rs)
	}
	if rs.ModelBytes != 4700000000 {
		t.Errorf("ModelBytes = %d, want 4700000000 (from /api/tags size)", rs.ModelBytes)
	}
}

func TestAdapter_ReadyTagsError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "m"}})
	if _, err := a.Ready(context.Background()); err == nil {
		t.Error("expected error")
	}
}

func TestAdapter_PullSuccess(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[]}`))
		case "/api/pull":
			frames := []map[string]any{
				{"status": "pulling manifest"},
				{"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 25},
				{"status": "downloading", "digest": "sha256:abc", "total": 100, "completed": 100},
				{"status": "success"},
			}
			for _, fr := range frames {
				b, _ := json.Marshal(fr)
				_, _ = w.Write(b)
				_, _ = w.Write([]byte("\n"))
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "qwen"}})
	sink := &recordingSink{}
	if err := a.Pull(context.Background(), "qwen", sink); err != nil {
		t.Fatalf("Pull: %v", err)
	}
	if len(sink.starts) != 1 || sink.starts[0] != "sha256:abc" {
		t.Errorf("starts=%v", sink.starts)
	}
	if sink.bytes != 100 {
		t.Errorf("bytes=%d", sink.bytes)
	}
	if len(sink.dones) != 1 {
		t.Errorf("dones=%v", sink.dones)
	}
	if sink.errored {
		t.Error("unexpected error")
	}
}

func TestAdapter_PullEarlyEOFButTagsFound(t *testing.T) {
	t.Parallel()
	// First call /api/pull returns no "success", but /api/tags lists the
	// model — Pull should swallow the error in this case.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"qwen"}]}`))
		case "/api/pull":
			_, _ = w.Write([]byte(`{"status":"downloading","digest":"sha256:x","total":1,"completed":0}` + "\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "qwen"}})
	if err := a.Pull(context.Background(), "qwen", nil); err != nil {
		t.Errorf("Pull should swallow ErrPullNoSuccess when /api/tags has the model: %v", err)
	}
}

func TestAdapter_PullEarlyEOFAndAbsent(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[]}`))
		case "/api/pull":
			_, _ = w.Write([]byte(`{"status":"downloading","digest":"sha256:x","total":1,"completed":0}` + "\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: srv.URL}, Model: config.Model{Name: "qwen"}})
	sink := &recordingSink{}
	if err := a.Pull(context.Background(), "qwen", sink); err == nil {
		t.Error("expected error when stream ends without success and tags is empty")
	}
	if !sink.errored {
		t.Error("sink.OnError should fire")
	}
}

func TestAdapter_RegisterSingleStep(t *testing.T) {
	t.Parallel()
	tmp := filepath.Join(t.TempDir(), "model.gguf")
	body := []byte("fake gguf body for hashing")
	if err := os.WriteFile(tmp, body, 0o644); err != nil {
		t.Fatal(err)
	}
	wantDigest := "sha256:" + hex.EncodeToString(func() []byte { h := sha256.Sum256(body); return h[:] }())

	var headHits, postHits, createHits int32
	var gotCreate CreateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/blobs/"+wantDigest && r.Method == http.MethodHead:
			atomic.AddInt32(&headHits, 1)
			w.WriteHeader(http.StatusNotFound) // force push
		case r.URL.Path == "/api/blobs/"+wantDigest && r.Method == http.MethodPost:
			atomic.AddInt32(&postHits, 1)
			b, _ := io.ReadAll(r.Body)
			if string(b) != string(body) {
				t.Errorf("blob body mismatch")
			}
			w.WriteHeader(http.StatusCreated)
		case r.URL.Path == "/api/create":
			atomic.AddInt32(&createHits, 1)
			b, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(b, &gotCreate)
			_, _ = w.Write([]byte(`{"status":"success"}` + "\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	cfg := config.Config{
		Engine: config.Engine{
			URL:  srv.URL,
			Kind: config.EngineOllama,
			Args: config.EngineArgs{Known: map[string]string{"num_ctx": "8192"}},
		},
		Model: config.Model{Name: "qwen2.5"},
	}
	a := NewAdapter(cfg)
	sink := &recordingSink{}
	if err := a.Register(context.Background(), []string{tmp}, nil, sink); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if headHits != 1 || postHits != 1 || createHits != 1 {
		t.Errorf("hits head=%d post=%d create=%d", headHits, postHits, createHits)
	}
	if gotCreate.Model != "qwen2.5" {
		t.Errorf("model=%q", gotCreate.Model)
	}
	if gotCreate.Template != "" {
		t.Errorf("template should be empty (modern GGUF self-describes), got %q", gotCreate.Template)
	}
	if gotCreate.System != "" {
		t.Errorf("system should be empty in v1.1 (GGUF_SYSTEM retired), got %q", gotCreate.System)
	}
	if gotCreate.Parameters["num_ctx"] != float64(8192) {
		t.Errorf("num_ctx=%v", gotCreate.Parameters["num_ctx"])
	}
	if gotCreate.Files["model.gguf"] != wantDigest {
		t.Errorf("files=%v", gotCreate.Files)
	}
}

func TestAdapter_RegisterEmptyFiles(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: "http://x"}, Model: config.Model{Name: "m"}})
	if err := a.Register(context.Background(), nil, nil, nil); err == nil {
		t.Error("expected error for empty files")
	}
}

func TestAdapter_RegisterMissingFile(t *testing.T) {
	t.Parallel()
	a := NewAdapter(config.Config{Engine: config.Engine{URL: "http://x"}, Model: config.Model{Name: "m"}})
	if err := a.Register(context.Background(), []string{"/no/such/file"}, nil, nil); err == nil {
		t.Error("expected error")
	}
}

func TestAdapter_AdapterImplementsInterface(t *testing.T) {
	t.Parallel()
	var _ adapter.Adapter = (*Adapter)(nil)
	// The only engine with a store of its own, and so the only
	// implementation of ModelInstaller. Dropping it silently turns
	// every pull and registration into a no-op.
	var _ adapter.ModelInstaller = (*Adapter)(nil)
}

func TestSha256OfFile(t *testing.T) {
	t.Parallel()
	tmp := filepath.Join(t.TempDir(), "f")
	body := []byte("abc")
	_ = os.WriteFile(tmp, body, 0o644)
	want := "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
	got, err := sha256OfFile(context.Background(), tmp)
	if err != nil {
		t.Fatalf("hash: %v", err)
	}
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

func TestMatchesModelName(t *testing.T) {
	t.Parallel()
	if !matchesModelName("qwen2.5:7b", "qwen2.5") {
		t.Error("colon prefix")
	}
	if !matchesModelName("m:latest", "m") {
		t.Error("latest tag")
	}
	if !matchesModelName("m", "m") {
		t.Error("exact")
	}
	if matchesModelName("other", "m") {
		t.Error("unrelated")
	}
}
