package dataplane

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

func newProgressManager() progress.Manager {
	return progress.New(time.Now())
}

// fakeAdapter is the only Adapter implementation that lives inside the
// dataplane package's test scope. Real adapters live in
// internal/adapter/{ollama,proxy} and import this package indirectly via
// dataplane.Mount; using them here would create a circular dependency
// chain in coverage measurement.
type fakeAdapter struct {
	calls atomic.Int32
	body  http.HandlerFunc
}

func (f *fakeAdapter) Kind() config.EngineKind         { return config.EngineOllama }
func (f *fakeAdapter) WaitAlive(context.Context) error { return nil }
func (f *fakeAdapter) AliveBeforeBoot() bool           { return true }
func (f *fakeAdapter) Ready(context.Context) (adapter.ReadyState, error) {
	return adapter.ReadyState{Alive: true, ModelExists: true}, nil
}
func (f *fakeAdapter) OpenAIHandler(config.Config) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.calls.Add(1)
		if f.body != nil {
			f.body(w, r)
			return
		}
		_, _ = w.Write([]byte("ok"))
	})
}
func (f *fakeAdapter) AnthropicHandler(config.Config) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("anthropic-ok"))
	})
}
func (f *fakeAdapter) EngineNativeStats(context.Context) (adapter.NativeStats, error) {
	return adapter.NativeStats{
		Source: "test",
		GPU:    adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown},
	}, nil
}

func newMux(t *testing.T, ready func() bool, mgr progress.Manager, fa *fakeAdapter) *http.ServeMux {
	t.Helper()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Ready:   ready,
		Manager: mgr,
		Adapter: fa,
		Config:  config.Config{Model: config.Model{Name: "m"}},
	})
	return mux
}

func TestMount_NotReady503(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mgr := newProgressManager()
	mgr.Update(func(s *progress.State) { s.Phase = progress.PhaseDownload })
	mux := newMux(t, func() bool { return false }, mgr, fa)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
		"/v1/chat/completions", strings.NewReader(`{}`)))

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d", rec.Code)
	}
	if rec.Header().Get("Retry-After") != "5" {
		t.Errorf("Retry-After=%q", rec.Header().Get("Retry-After"))
	}
	var body map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	errObj, _ := body["error"].(map[string]any)
	if errObj["code"] != "not_ready" {
		t.Errorf("code=%v", errObj["code"])
	}
	if errObj["phase"] != string(progress.PhaseDownload) {
		t.Errorf("phase=%v", errObj["phase"])
	}
	if fa.calls.Load() != 0 {
		t.Errorf("adapter called %d times despite 503", fa.calls.Load())
	}
}

func TestMount_ReadyDelegatesToAdapter(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newMux(t, func() bool { return true }, newProgressManager(), fa)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
		"/v1/chat/completions", strings.NewReader(`{}`)))
	if rec.Code != http.StatusOK || rec.Body.String() != "ok" {
		t.Errorf("status=%d body=%q", rec.Code, rec.Body.String())
	}
	if fa.calls.Load() != 1 {
		t.Errorf("calls=%d", fa.calls.Load())
	}
}

func TestMount_OpenWebUIAlias(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newMux(t, func() bool { return true }, nil, fa)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
		"/api/chat/completions", strings.NewReader(`{}`)))
	if rec.Code != http.StatusOK {
		t.Errorf("alias status=%d", rec.Code)
	}
	if fa.calls.Load() != 1 {
		t.Errorf("alias did not delegate, calls=%d", fa.calls.Load())
	}
}

// TestMount_NilReadyWithUnsafeOptInDefaultsTrue covers the legacy
// "always ready" behavior that pre-v1.0.4 was the silent default.
// Tests that don't care about readiness gating must now opt in
// explicitly with `AllowUnsafeNilReady: true`; production code must
// never reach this branch.
func TestMount_NilReadyWithUnsafeOptInDefaultsTrue(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := http.NewServeMux()
	Mount(mux, Options{
		Adapter:             fa,
		Config:              config.Config{},
		AllowUnsafeNilReady: true,
	})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("nil Ready with AllowUnsafeNilReady should default to always-ready, got %d", rec.Code)
	}
}

// TestMount_NilReadyWithoutOptInPanics is the regression test for the
// production-wiring footgun. Pre-v1.0.4 Mount silently treated nil
// Ready as always-ready, so a missing `Ready: lc.Ready` line in
// cmd/llm-init/main.go would route /v1/* to the engine before the
// lifecycle reached PhaseReady ? exactly what notReadyGuard exists to
// prevent. With the new contract, the same wiring bug fails fast at
// process boot.
func TestMount_NilReadyWithoutOptInPanics(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected Mount to panic on nil Ready without AllowUnsafeNilReady")
		}
		msg, _ := r.(string)
		if !strings.Contains(msg, "Ready is nil") {
			t.Errorf("panic message missing the wiring hint: %v", r)
		}
	}()
	Mount(http.NewServeMux(), Options{Adapter: fa, Config: config.Config{}})
}

func TestMount_ReadinessFlipBetweenRequests(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	var ready atomic.Bool
	mux := newMux(t, ready.Load, newProgressManager(), fa)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/x", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("first call: status=%d", rec.Code)
	}
	ready.Store(true)
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/x", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("post-flip: status=%d", rec.Code)
	}
}

// TestMount_ResponsesRouted proves /v1/responses reaches the adapter
// (with readiness gating) instead of a dataplane-level 501 stub.
func TestMount_ResponsesRouted(t *testing.T) {
	t.Run("ready", func(t *testing.T) {
		t.Parallel()
		fa := &fakeAdapter{}
		mux := newMux(t, func() bool { return true }, newProgressManager(), fa)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
			"/v1/responses", strings.NewReader(`{"input":"hi"}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("status=%d, want %d", rec.Code, http.StatusOK)
		}
		if fa.calls.Load() != 1 {
			t.Fatalf("adapter calls=%d want 1", fa.calls.Load())
		}
	})
	t.Run("not_ready", func(t *testing.T) {
		t.Parallel()
		fa := &fakeAdapter{}
		mgr := newProgressManager()
		mgr.Update(func(s *progress.State) { s.Phase = progress.PhaseDownload })
		mux := newMux(t, func() bool { return false }, mgr, fa)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
			"/v1/responses", strings.NewReader(`{"input":"hi"}`)))
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status=%d, want %d", rec.Code, http.StatusServiceUnavailable)
		}
		if fa.calls.Load() != 0 {
			t.Fatalf("adapter calls=%d want 0", fa.calls.Load())
		}
	})
}

// TestMount_MessagesRouted proves the dataplane-level /v1/messages
// registration dispatches into AnthropicHandler rather than the
// /v1/chat/completions catch-all. Without the dedicated route entry
// (mount.go), Anthropic clients would be routed through the OpenAI
// adapter chain which doesn't speak Anthropic shape.
func TestMount_MessagesRouted(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newMux(t, func() bool { return true }, newProgressManager(), fa)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"messages":[],"max_tokens":1}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if body := rec.Body.String(); body != "anthropic-ok" {
		t.Errorf("body=%q want anthropic-ok (means AnthropicHandler was hit)", body)
	}
}

// TestMount_MessagesNotReady503 proves /v1/messages is gated by
// NotReadyGuard just like /v1/chat/completions. Without the guard a
// client could hit Anthropic before the model is loaded.
func TestMount_MessagesNotReady503(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newMux(t, func() bool { return false }, newProgressManager(), fa)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status=%d want 503", rec.Code)
	}
}

// TestMount_CountTokensRouted proves /v1/messages/count_tokens is
// explicitly routed at the dataplane level and reaches the
// AnthropicHandler tree. Without the explicit entry the
// /v1/ catch-all would steal it.
func TestMount_CountTokensRouted(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newMux(t, func() bool { return true }, newProgressManager(), fa)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens",
		strings.NewReader(`{}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if body := rec.Body.String(); body != "anthropic-ok" {
		t.Errorf("body=%q want anthropic-ok (means AnthropicHandler was hit)", body)
	}
}

func TestMount_NotReadyNilManagerDoesNotPanic(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newMux(t, func() bool { return false }, nil, fa)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/x", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status=%d", rec.Code)
	}
	// phase serializes as "" when manager is nil - that is the contract.
	var body map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	errObj, _ := body["error"].(map[string]any)
	if errObj["phase"] != "" {
		t.Errorf("phase=%v want empty", errObj["phase"])
	}
}
