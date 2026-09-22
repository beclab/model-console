//go:build download

package download

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/lifecycle"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/tests/download/faultsrv"
)

// urlRig spins up a fault server + lifecycle Manager (llama.cpp engine,
// fakeAdapter) for one URL source and returns the rig, the dest path and
// the served body.
func urlRig(t *testing.T, fcfg faultsrv.Config, withSHA bool) (*rig, *faultsrv.Server, string, []byte, *fakeAdapter) {
	t.Helper()
	srv := faultsrv.New(fcfg)
	t.Cleanup(srv.Close)

	dir := t.TempDir()
	dest := filepath.Join(dir, "model.bin")
	sha := ""
	if withSHA {
		sha = srv.BodySHA256()
	}
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource(srv.URL+"/model.bin", sha, dest)}

	ad := &fakeAdapter{kind: config.EngineLlamaCpp}
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: ad})
	return r, srv, dest, fcfgBody(fcfg), ad
}

func fcfgBody(fcfg faultsrv.Config) []byte {
	if len(fcfg.Body) == 0 {
		return faultsrv.GenerateBody(64 * 1024)
	}
	return fcfg.Body
}

func TestURL_HappyPath_WithSHA(t *testing.T) {
	r, srv, dest, body, ad := urlRig(t, faultsrv.Config{SupportRange: true}, true)
	r.waitPhase(progress.PhaseReady, 10*time.Second)

	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read dest: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("dest content mismatch: got %d bytes want %d", len(got), len(body))
	}
	if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
		t.Errorf(".part should be cleaned up, stat err=%v", err)
	}
	if ad.regCalls.Load() != 1 {
		t.Errorf("Register calls = %d want 1", ad.regCalls.Load())
	}
	if reg := ad.lastRegister(); len(reg) != 1 || reg[0] != dest {
		t.Errorf("Register files = %v want [%s]", reg, dest)
	}
	_ = srv
}

func TestURL_ResumeAfterConnectionReset(t *testing.T) {
	// First GET writes ~20 KiB then drops the connection; the inner
	// retry loop must resume via Range and complete.
	r, srv, dest, body, _ := urlRig(t, faultsrv.Config{
		SupportRange: true,
		DropAfter:    20_000,
		DropFirstN:   1,
	}, true)
	r.waitPhase(progress.PhaseReady, 10*time.Second)

	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, body) {
		t.Fatalf("resumed file mismatch: got %d want %d", len(got), len(body))
	}
	if srv.GetCount() < 2 {
		t.Errorf("expected >=2 GET attempts (drop + resume), got %d", srv.GetCount())
	}
}

func TestURL_ServerIgnoresRange_RewritesFromZero(t *testing.T) {
	// Pre-seed a garbage .part; the server does NOT support Range so it
	// always returns 200 full and the downloader must rewrite from 0.
	srv := faultsrv.New(faultsrv.Config{SupportRange: false})
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.bin")
	if err := os.WriteFile(dest+".part", bytes.Repeat([]byte{0xFF}, 20_000), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource(srv.URL+"/model.bin", srv.BodySHA256(), dest)}
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: &fakeAdapter{kind: config.EngineLlamaCpp}})
	r.waitPhase(progress.PhaseReady, 10*time.Second)

	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, faultsrv.GenerateBody(64*1024)) {
		t.Fatalf("file not rewritten from zero: len=%d", len(got))
	}
}

func TestURL_ETagRotation_DiscardsStalePart(t *testing.T) {
	// Server supports Range but rotates its ETag from the first GET, so
	// the resuming client's If-Range no longer matches and it must do a
	// clean full download (not append onto the garbage prefix).
	srv := faultsrv.New(faultsrv.Config{SupportRange: true, ETagRotateAfter: 0})
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.bin")
	if err := os.WriteFile(dest+".part", bytes.Repeat([]byte{0xAB}, 20_000), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource(srv.URL+"/model.bin", srv.BodySHA256(), dest)}
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: &fakeAdapter{kind: config.EngineLlamaCpp}})
	r.waitPhase(progress.PhaseReady, 10*time.Second)

	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, faultsrv.GenerateBody(64*1024)) {
		t.Fatalf("stale .part not discarded: file len=%d", len(got))
	}
}

func TestURL_5xxStormThenRecover(t *testing.T) {
	// The first two GET attempts 503; the inner retry loop rides them
	// out and completes. transport_retries should advance.
	r, srv, dest, body, _ := urlRig(t, faultsrv.Config{
		SupportRange: true,
		FailFirstN:   2,
	}, true)
	st := r.waitPhase(progress.PhaseReady, 20*time.Second)

	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, body) {
		t.Fatalf("recovered file mismatch")
	}
	if srv.GetCount() < 3 {
		t.Errorf("expected >=3 GET attempts, got %d", srv.GetCount())
	}
	if st.TransportRetries == 0 {
		t.Errorf("transport_retries should be > 0 after a 5xx storm")
	}
}

func TestURL_Permanent404_FailsWithoutRetry(t *testing.T) {
	r, _, _, _, ad := urlRig(t, faultsrv.Config{ForceStatus: 404}, false)
	st := r.waitPhase(progress.PhaseFailed, 10*time.Second)

	if !strings.Contains(strings.ToLower(st.LastError), "not found") {
		t.Errorf("last_error = %q, want a clear 404 message", st.LastError)
	}
	if ad.regCalls.Load() != 0 {
		t.Errorf("Register must not run on a failed download (got %d calls)", ad.regCalls.Load())
	}
}

func TestURL_429_YieldsDegraded(t *testing.T) {
	// 429 is retriable; after the inner budget is exhausted the ensure
	// pass fails retriably and the manager sits in degraded.
	r, _, _, _, _ := urlRig(t, faultsrv.Config{ForceStatus: 429}, false)
	st := r.waitUntil(40*time.Second, func(s progress.State) bool {
		return s.Phase == progress.PhaseDegraded && s.RetryCount > 0
	})
	if st.LastError == "" {
		t.Error("last_error should be populated on a retriable 429 failure")
	}
}

func TestURL_SHAMismatch_RemovesFileAndDegrades(t *testing.T) {
	srv := faultsrv.New(faultsrv.Config{SupportRange: true})
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.bin")
	wrongSHA := strings.Repeat("0", 64)
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource(srv.URL+"/model.bin", wrongSHA, dest)}
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: &fakeAdapter{kind: config.EngineLlamaCpp}})

	st := r.waitUntil(10*time.Second, func(s progress.State) bool {
		return s.Phase == progress.PhaseDegraded && s.LastError != ""
	})
	if !strings.Contains(strings.ToLower(st.LastError), "verify") {
		t.Errorf("last_error = %q, want it to mention verify", st.LastError)
	}
	if _, err := os.Stat(dest); !os.IsNotExist(err) {
		t.Errorf("corrupt file should be removed on sha mismatch, stat err=%v", err)
	}
}

func TestURL_MissingLocalPath_FailsPermanently(t *testing.T) {
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource("https://example.com/x.bin", "", "")} // no LocalPath
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: &fakeAdapter{kind: config.EngineLlamaCpp}})
	st := r.waitPhase(progress.PhaseFailed, 5*time.Second)
	if !strings.Contains(st.LastError, "MODEL_SOURCE_LOCAL") {
		t.Errorf("last_error = %q, want it to mention MODEL_SOURCE_LOCAL", st.LastError)
	}
}

func TestURL_DestAlreadyComplete_Skips(t *testing.T) {
	body := faultsrv.GenerateBody(64 * 1024)
	srv := faultsrv.New(faultsrv.Config{Body: body, SupportRange: true})
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.bin")
	if err := os.WriteFile(dest, body, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource(srv.URL+"/model.bin", srv.BodySHA256(), dest)}
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: &fakeAdapter{kind: config.EngineLlamaCpp}})
	r.waitPhase(progress.PhaseReady, 10*time.Second)

	if srv.GetCount() != 0 {
		t.Errorf("complete dest should short-circuit GET, got %d GETs", srv.GetCount())
	}
}

func TestURL_HTTPS_WithTrustingClient(t *testing.T) {
	// https input mode: the lifecycle Downloader is wired with the
	// test server's trusting client so the full path runs over TLS.
	srv := faultsrv.NewTLS(faultsrv.Config{SupportRange: true})
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.bin")
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource(srv.URL+"/model.bin", srv.BodySHA256(), dest)}
	r := startManager(t, lifecycle.Options{
		Config:     cfg,
		Adapter:    &fakeAdapter{kind: config.EngineLlamaCpp},
		Downloader: fetch.New(srv.Client()),
	})
	r.waitPhase(progress.PhaseReady, 10*time.Second)
	got, _ := os.ReadFile(dest)
	if !bytes.Equal(got, faultsrv.GenerateBody(64*1024)) {
		t.Fatalf("https download mismatch")
	}
}

func TestURL_ForceRedownloads(t *testing.T) {
	// Dest already complete -> first ensure skips. A force retry must
	// wipe and re-fetch (GET count goes from 0 to >0).
	body := faultsrv.GenerateBody(64 * 1024)
	srv := faultsrv.New(faultsrv.Config{Body: body, SupportRange: true})
	t.Cleanup(srv.Close)
	dir := t.TempDir()
	dest := filepath.Join(dir, "model.bin")
	if err := os.WriteFile(dest, body, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := baseConfig(t, config.EngineLlamaCpp, "http://127.0.0.1:0")
	cfg.Sources = []config.ModelSource{urlSource(srv.URL+"/model.bin", srv.BodySHA256(), dest)}
	r := startManager(t, lifecycle.Options{Config: cfg, Adapter: &fakeAdapter{kind: config.EngineLlamaCpp}})
	r.waitPhase(progress.PhaseReady, 10*time.Second)
	if srv.GetCount() != 0 {
		t.Fatalf("precondition: expected 0 GETs, got %d", srv.GetCount())
	}

	r.mgr.RetryWith(lifecycle.RetryOptions{Force: true})
	r.waitUntil(10*time.Second, func(progress.State) bool { return srv.GetCount() > 0 })
}
