package lifecycle

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/internal/verify"
)

// Entering phase=download answers every /v1/* request with 503 for as
// long as the transfer takes, and asks the upstream how big it will be
// first. A pass that has nothing to fetch must not pay either: the
// health loop's self-heal runs against a model that is serving.
//
// It does not make the pass invisible. Every pass still moves through
// loading, which also gates /v1/*, so what this saves is a window
// proportional to the model and dependent on the network -- not the
// whole outage.
//
// The phase is read off the pass rather than the progress state because
// the pass moves on to loading and ready before it returns; the flag is
// the only record of whether it was ever entered.
func TestPass_ReusedBytesNeverEnterDownloadPhase(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)

	f.ensure(t)
	if f.mgr.pass.entered {
		t.Error("a pass with nothing to fetch entered phase=download")
	}
	if got := f.dl.hits.Load(); got != 1 {
		t.Errorf("downloader calls = %d, want the first pass's 1", got)
	}
}

// The budget estimate reaches the upstream: it HEADs every URL and asks
// the Hub about every repo. Deferring the phase defers that too, which
// is what keeps a warm-cache boot from depending on a network it does
// not need.
func TestPass_ReusedBytesNeverProbeTheUpstream(t *testing.T) {
	t.Parallel()
	var probes atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		probes.Add(1)
		w.Header().Set("Content-Length", "11")
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	f := newURLFixture(t, srv.URL, "", []byte(modelBody))
	f.ensure(t)
	before := probes.Load()
	if before == 0 {
		t.Fatal("setup: the first pass never estimated the budget")
	}

	f.ensure(t)
	if got := probes.Load(); got != before {
		t.Errorf("upstream probes = %d, want the first pass's %d", got, before)
	}
}

// The deferral must not cost a real download its phase, or the progress
// bar has nothing to draw on and /v1/* is served while bytes are still
// arriving.
func TestPass_ARealDownloadStillEntersDownloadPhase(t *testing.T) {
	t.Parallel()
	f := newURLFixture(t, "https://example.com/m.bin", "", []byte(modelBody))
	f.ensure(t)

	if !f.mgr.pass.entered {
		t.Error("a pass that fetched bytes never entered phase=download")
	}
}

// An ollama:// URL with a #sha256= fragment has already had those exact
// bytes hashed by the downloader, and the record keeps the result.
// Handing it to the adapter is what turns a second multi-gigabyte read
// on every boot into a map lookup.
func TestRegister_ReusesTheRecordedDigest(t *testing.T) {
	t.Parallel()
	sum := sha256.Sum256([]byte(modelBody))
	want := hex.EncodeToString(sum[:])

	runDir := t.TempDir()
	o := baseOptions(t, []config.ModelSource{ollamaURLSource("https://example.com/m.gguf", want)})
	o.Config.Runtime.RunDir = runDir
	ad := &fakeAdapter{kind: config.EngineOllama}
	o.Adapter = ad
	o.Config.Engine.Kind = config.EngineOllama
	o.Downloader = &fakeDownloader{files: map[string][]byte{"https://example.com/m.gguf": []byte(modelBody)}}
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	_, known := ad.registered()
	if calls := ad.regCalls.Load(); calls != 1 {
		t.Fatalf("Register calls = %d, want 1", calls)
	}
	dest := filepath.Join(runDir, "url-fetched", want+".gguf")
	if got := known[dest]; got != "sha256:"+want {
		t.Errorf("digest hint = %q, want %q", got, "sha256:"+want)
	}
}

// Without a declared digest the size level records none, and the
// adapter has to be told nothing rather than something wrong.
func TestRegister_NoRecordedDigestHintsNothing(t *testing.T) {
	t.Parallel()
	runDir := t.TempDir()
	o := baseOptions(t, []config.ModelSource{ollamaURLSource("https://example.com/m.gguf", "")})
	o.Config.Runtime.RunDir = runDir
	ad := &fakeAdapter{kind: config.EngineOllama}
	o.Adapter = ad
	o.Config.Engine.Kind = config.EngineOllama
	o.Downloader = &fakeDownloader{files: map[string][]byte{"https://example.com/m.gguf": []byte(modelBody)}}
	mgr, err := New(o)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := mgr.ensure(context.Background()); err != nil {
		t.Fatalf("ensure: %v", err)
	}

	if _, known := ad.registered(); len(known) != 0 {
		t.Errorf("digest hints = %v, want none", known)
	}
}

// A record that names a different file must not have its digest applied
// to this one. Ollama would reject the blob, but the wrong hint is the
// bug and the empty map is the contract.
func TestRegister_MismatchedRecordHintsNothing(t *testing.T) {
	t.Parallel()
	dest := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(dest, []byte(modelBody), 0o644); err != nil {
		t.Fatal(err)
	}
	writeSidecarNaming(t, dest+urlSidecarSuffix, "other.gguf")

	ad := &fakeAdapter{kind: config.EngineOllama}
	m := &Manager{
		opts:      Options{Adapter: ad, Manager: progress.New(time.Now())},
		installer: adapter.InstallerFor(ad),
	}
	if err := m.registerWithKnownDigests(context.Background(), dest); err != nil {
		t.Fatalf("registerWithKnownDigests: %v", err)
	}
	if _, known := ad.registered(); len(known) != 0 {
		t.Errorf("digest hints = %v, want none for a record about another file", known)
	}
}

// writeSidecarNaming writes a record whose single entry describes name,
// with a digest that must not be applied to anything else.
func writeSidecarNaming(t *testing.T, path, name string) {
	t.Helper()
	if err := verify.Write(path, verify.Sidecar{
		Source: verify.Source{Kind: verify.KindOllamaURL, URLSHA256: verify.URLIdentity("https://example.com/x")},
		Files:  []verify.File{{Path: name, Size: 1, SHA256: strings.Repeat("a", 64)}},
	}); err != nil {
		t.Fatalf("Write: %v", err)
	}
}
