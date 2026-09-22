//go:build download

package download

import (
	"strings"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter/ollama"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/fetch"
	"github.com/llm-init/llm-init/internal/lifecycle"
	"github.com/llm-init/llm-init/internal/progress"
	"github.com/llm-init/llm-init/tests/download/faultsrv"
	"github.com/llm-init/llm-init/tests/download/mockollama"
)

// ollamaNameRig wires a mock daemon + real ollama adapter for an
// ollama:// library-tag source.
func ollamaNameRig(t *testing.T, tag string, mcfg mockollama.Config) (*rig, *mockollama.Server) {
	t.Helper()
	daemon := mockollama.New(mcfg)
	t.Cleanup(daemon.Close)
	cfg := baseConfig(t, config.EngineOllama, daemon.URL)
	cfg.Sources = []config.ModelSource{ollamaNameSource(tag)}
	r := startManager(t, lifecycle.Options{
		Config:     cfg,
		Adapter:    ollama.NewAdapter(cfg),
		Downloader: fetch.New(nil),
	})
	return r, daemon
}

func TestOllamaName_PullHappyPath(t *testing.T) {
	r, daemon := ollamaNameRig(t, "qwen2.5:7b", mockollama.Config{})
	r.waitPhase(progress.PhaseReady, 10*time.Second)
	if daemon.PullCount() == 0 {
		t.Error("expected at least one /api/pull request")
	}
}

func TestOllamaName_UnknownTag_404(t *testing.T) {
	r, _ := ollamaNameRig(t, "does-not-exist:7b", mockollama.Config{PullStatus: 404})
	st := r.waitUntil(10*time.Second, func(s progress.State) bool {
		return s.Phase == progress.PhaseDegraded && s.LastError != ""
	})
	if !strings.Contains(strings.ToLower(st.LastError), "not found") {
		t.Errorf("last_error = %q, want an invalid-tag (not found) message", st.LastError)
	}
}

func TestOllamaName_Denied_403(t *testing.T) {
	r, _ := ollamaNameRig(t, "private/model:7b", mockollama.Config{PullStatus: 403})
	st := r.waitUntil(10*time.Second, func(s progress.State) bool {
		return s.Phase == progress.PhaseDegraded && s.LastError != ""
	})
	if !strings.Contains(strings.ToLower(st.LastError), "denied access") {
		t.Errorf("last_error = %q, want a permission (denied access) message", st.LastError)
	}
}

func TestOllamaName_NoSuccessFrame_RecoversViaTags(t *testing.T) {
	// Stream ends without status:success, but /api/tags shows the model
	// (the daemon finished writing): the adapter recovers.
	r, _ := ollamaNameRig(t, "qwen2.5:7b", mockollama.Config{
		PullOmitSuccess: true,
		ExistingModels:  []string{"qwen2.5:7b"},
	})
	r.waitPhase(progress.PhaseReady, 10*time.Second)
}

func TestOllamaName_NoSuccessFrame_AbsentModel_Degrades(t *testing.T) {
	r, _ := ollamaNameRig(t, "qwen2.5:7b", mockollama.Config{PullOmitSuccess: true})
	st := r.waitUntil(10*time.Second, func(s progress.State) bool {
		return s.Phase == progress.PhaseDegraded && s.LastError != ""
	})
	if st.LastError == "" {
		t.Error("last_error should be populated when pull ends without success and the model is absent")
	}
}

// ollamaURLRig wires a fault server (the blob source) plus a mock daemon
// (blob upload + create) for an ollama:// URL-overload source.
func ollamaURLRig(t *testing.T, fcfg faultsrv.Config, mcfg mockollama.Config) (*rig, *faultsrv.Server, *mockollama.Server) {
	t.Helper()
	src := faultsrv.New(fcfg)
	t.Cleanup(src.Close)
	daemon := mockollama.New(mcfg)
	t.Cleanup(daemon.Close)

	cfg := baseConfig(t, config.EngineOllama, daemon.URL)
	cfg.Sources = []config.ModelSource{ollamaURLSource(src.URL+"/model.gguf", src.BodySHA256())}
	r := startManager(t, lifecycle.Options{
		Config:     cfg,
		Adapter:    ollama.NewAdapter(cfg),
		Downloader: fetch.New(nil),
	})
	return r, src, daemon
}

func TestOllamaURL_DownloadThenRegister(t *testing.T) {
	r, src, daemon := ollamaURLRig(t, faultsrv.Config{SupportRange: true}, mockollama.Config{})
	r.waitPhase(progress.PhaseReady, 10*time.Second)
	if src.GetCount() == 0 {
		t.Error("expected the blob to be fetched from the URL")
	}
	if daemon.CreateCount() == 0 {
		t.Error("expected /api/create to register the blob")
	}
}

// A failed Adapter.Register during the loading phase is terminal (PhaseFailed,
// not retriable PhaseDegraded) since the phase-loading rework: the operator
// must POST /api/retry to re-run ensure.
func TestOllamaURL_CreateNoSuccess_Fails(t *testing.T) {
	r, _, _ := ollamaURLRig(t, faultsrv.Config{SupportRange: true}, mockollama.Config{CreateOmitSuccess: true})
	st := r.waitUntil(10*time.Second, func(s progress.State) bool {
		return s.Phase == progress.PhaseFailed && s.LastError != ""
	})
	if !strings.Contains(strings.ToLower(st.LastError), "create") {
		t.Errorf("last_error = %q, want it to mention the failed create", st.LastError)
	}
}

func TestOllamaURL_BlobPushDenied_Fails(t *testing.T) {
	r, _, _ := ollamaURLRig(t, faultsrv.Config{SupportRange: true}, mockollama.Config{BlobPushStatus: 403})
	st := r.waitUntil(10*time.Second, func(s progress.State) bool {
		return s.Phase == progress.PhaseFailed && s.LastError != ""
	})
	if st.LastError == "" {
		t.Error("last_error should be populated when the blob push is denied")
	}
}
