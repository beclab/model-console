package comfyconsole

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func sum(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

// fixture serves one model file (Range + HEAD via ServeContent) and a
// ComfyUI /system_stats whose answer the test controls.
type fixture struct {
	srv     *httptest.Server
	payload []byte
	gets    atomic.Int32
	up      atomic.Bool
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{payload: bytes.Repeat([]byte("comfy-weights-"), 4096)}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/model.safetensors":
			if r.Method == http.MethodGet {
				f.gets.Add(1)
			}
			http.ServeContent(w, r, "model.safetensors", time.Unix(0, 0), bytes.NewReader(f.payload))
		case "/system_stats":
			if !f.up.Load() {
				http.Error(w, "starting", http.StatusServiceUnavailable)
				return
			}
			_, _ = w.Write([]byte(`{}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fixture) config(sha string) *AppConfig {
	return &AppConfig{Models: []Model{{
		Name: "model.safetensors", Dir: "diffusion_models",
		URL: f.srv.URL + "/model.safetensors", SHA256: sha, Size: int64(len(f.payload)),
	}}}
}

func (f *fixture) console(cfg *AppConfig, dir string) *Console {
	return New(Options{
		Config: cfg, ModelsDir: dir, ComfyURL: f.srv.URL,
		ProbeInterval: 10 * time.Millisecond, RecheckEvery: 20 * time.Millisecond,
		RetryBackoff: 20 * time.Millisecond, RetryBackoffMx: 20 * time.Millisecond,
	})
}

func waitFor(t *testing.T, c *Console, what string, ok func(Status) bool) Status {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		st := c.Status()
		if ok(st) {
			return st
		}
		if time.Now().After(deadline) {
			raw, _ := json.Marshal(st)
			t.Fatalf("timed out waiting for %s; last status %s", what, raw)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func run(t *testing.T, c *Console) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { c.Run(ctx); close(done) }()
	t.Cleanup(func() { cancel(); <-done })
}

func TestConsole_DownloadsVerifiesAndBecomesReady(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	dir := t.TempDir()
	c := f.console(f.config(sum(f.payload)), dir)
	run(t, c)

	st := waitFor(t, c, "models done", func(s Status) bool { return s.Models.State == StateDone })
	if st.Ready {
		t.Fatal("ready with ComfyUI down")
	}
	if st.Models.Progress != 1 || st.Models.Items[0].Downloaded != int64(len(f.payload)) {
		t.Fatalf("progress = %v downloaded = %d", st.Models.Progress, st.Models.Items[0].Downloaded)
	}
	got, err := os.ReadFile(filepath.Join(dir, "diffusion_models", "model.safetensors"))
	if err != nil || !bytes.Equal(got, f.payload) {
		t.Fatalf("model file not written correctly: %v", err)
	}

	f.up.Store(true)
	waitFor(t, c, "ready", func(s Status) bool { return s.Ready && s.Backend.State == BackendUp })
	f.up.Store(false)
	waitFor(t, c, "not ready after backend loss", func(s Status) bool { return !s.Ready })
}

func TestConsole_VerifiedModelIsNotFetchedAgain(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	dir := t.TempDir()
	cfg := f.config(sum(f.payload))

	first := f.console(cfg, dir)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { first.Run(ctx); close(done) }()
	waitFor(t, first, "first pass done", func(s Status) bool { return s.Models.State == StateDone })
	cancel()
	<-done
	gets := f.gets.Load()

	second := f.console(cfg, dir)
	run(t, second)
	waitFor(t, second, "second pass done", func(s Status) bool { return s.Models.State == StateDone })
	if f.gets.Load() != gets {
		t.Fatalf("restart re-downloaded a verified model: GETs %d -> %d", gets, f.gets.Load())
	}
}

func TestConsole_RedownloadsDeletedModel(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	dir := t.TempDir()
	c := f.console(f.config(sum(f.payload)), dir)
	run(t, c)
	waitFor(t, c, "done", func(s Status) bool { return s.Models.State == StateDone })

	dest := filepath.Join(dir, "diffusion_models", "model.safetensors")
	if err := os.Remove(dest); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(dest); err == nil && f.gets.Load() >= 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("deleted model was not downloaded again")
		}
		time.Sleep(10 * time.Millisecond)
	}
	waitFor(t, c, "done again", func(s Status) bool { return s.Models.State == StateDone })
}

func TestConsole_ShaMismatchReportsError(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	dir := t.TempDir()
	c := f.console(f.config(strings.Repeat("0", 64)), dir)
	run(t, c)

	st := waitFor(t, c, "error", func(s Status) bool { return s.Models.State == StateError })
	if !strings.Contains(st.Models.Items[0].Error, "sha256") || st.Ready {
		t.Fatalf("unexpected status: %+v", st.Models.Items[0])
	}
	if _, err := os.Stat(filepath.Join(dir, "diffusion_models", "model.safetensors")); !os.IsNotExist(err) {
		t.Fatalf("corrupt file kept on disk: %v", err)
	}
}

func TestHandler_StatusShape(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	c := f.console(f.config(sum(f.payload)), t.TempDir())
	rec := httptest.NewRecorder()
	c.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/status", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status code %d", rec.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"ready", "models", "backend"} {
		if _, ok := body[k]; !ok {
			t.Errorf("missing key %q in %s", k, rec.Body.String())
		}
	}
	rec = httptest.NewRecorder()
	c.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("readyz before ready = %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	c.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Model Console") {
		t.Fatalf("console page = %d", rec.Code)
	}
}

func TestLoadConfig_Validation(t *testing.T) {
	t.Parallel()
	good := `{"workflow":{"file":"w.json"},"models":[{"name":"a.safetensors","dir":"vae","url":"https://x/a","sha256":"` + strings.Repeat("a", 64) + `","size":1}]}`
	cases := map[string]string{
		"ok":         good,
		"empty":      `{"models":[]}`,
		"traversal":  strings.Replace(good, `"dir":"vae"`, `"dir":"../etc"`, 1),
		"slash name": strings.Replace(good, `"a.safetensors"`, `"x/a.safetensors"`, 1),
		"short sha":  strings.Replace(good, strings.Repeat("a", 64), "abc", 1),
		"no size":    strings.Replace(good, `"size":1`, `"size":0`, 1),
		"bad url":    strings.Replace(good, `https://x/a`, `ftp://x/a`, 1),
	}
	for name, body := range cases {
		file := filepath.Join(t.TempDir(), "c.json")
		if err := os.WriteFile(file, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := LoadConfig(file)
		if (err == nil) != (name == "ok") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}
