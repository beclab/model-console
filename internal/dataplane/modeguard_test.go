package dataplane

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

func newModeMux(t *testing.T, mode config.ModelType, ready func() bool, fa *fakeAdapter) *http.ServeMux {
	t.Helper()
	mux := http.NewServeMux()
	Mount(mux, Options{
		Ready:   ready,
		Manager: newProgressManager(),
		Adapter: fa,
		Config:  config.Config{Model: config.Model{Name: "m", Type: mode}},
	})
	return mux
}

func errorObj(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("body is not the error envelope: %v (%s)", err, rec.Body)
	}
	obj, _ := body["error"].(map[string]any)
	if obj == nil {
		t.Fatalf("no error object in %s", rec.Body)
	}
	return obj
}

// /v1/ is a catch-all, so an embedding application used to accept a chat
// request and hand it to an engine with no such route — while
// GET /api/endpoints was already reporting chat as unavailable. The
// client got that engine's 404, in that engine's format, saying nothing
// about this application.
func TestModeGuard_EmbeddingRefusesChatAndKeepsEmbeddings(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newModeMux(t, config.ModelEmbedding, func() bool { return true }, fa)

	for _, path := range []string{
		"/v1/chat/completions",
		"/v1/completions",
		"/v1/responses",
		"/v1/messages",
		"/api/chat/completions",
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusNotFound {
			t.Errorf("POST %s: status=%d, want 404", path, rec.Code)
			continue
		}
		obj := errorObj(t, rec)
		if obj["code"] != codeNotServed {
			t.Errorf("POST %s: code=%v, want %q", path, obj["code"], codeNotServed)
		}
		if obj["mode"] != string(config.ModelEmbedding) {
			t.Errorf("POST %s: mode=%v, want embedding", path, obj["mode"])
		}
		// Retrying is not going to help, and an OpenAI client backs off
		// on this header rather than reporting the error.
		if h := rec.Header().Get("Retry-After"); h != "" {
			t.Errorf("POST %s: Retry-After=%q on a permanent answer", path, h)
		}
	}
	if fa.calls.Load() != 0 {
		t.Errorf("engine was handed %d request(s) it has no route for", fa.calls.Load())
	}

	// What it does serve still reaches the engine.
	for _, r := range []struct{ method, path string }{
		{http.MethodPost, "/v1/embeddings"},
		{http.MethodGet, "/v1/models"},
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(r.method, r.path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusOK {
			t.Errorf("%s %s: status=%d, want 200; body=%s", r.method, r.path, rec.Code, rec.Body)
		}
	}
}

func TestModeGuard_RerankRefusesChatAndKeepsRerank(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newModeMux(t, config.ModelRerank, func() bool { return true }, fa)

	for _, path := range []string{
		"/v1/chat/completions",
		"/v1/completions",
		"/v1/embeddings",
		"/v1/responses",
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusNotFound {
			t.Errorf("POST %s: status=%d, want 404", path, rec.Code)
		}
	}
	if fa.calls.Load() != 0 {
		t.Errorf("engine was handed %d request(s) it has no route for", fa.calls.Load())
	}

	for _, r := range []struct{ method, path string }{
		{http.MethodPost, "/v1/rerank"},
		{http.MethodGet, "/v1/models"},
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(r.method, r.path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusOK {
			t.Errorf("%s %s: status=%d, want 200; body=%s", r.method, r.path, rec.Code, rec.Body)
		}
	}
}

func TestModeGuard_SystemOneOnlyServesTypedDecisionSurface(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newModeMux(t, config.ModelSystemOne, func() bool { return true }, fa)

	for _, path := range []string{
		"/v1/chat/completions",
		"/v1/embeddings",
		"/v1/rerank",
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusNotFound {
			t.Errorf("POST %s: status=%d, want 404", path, rec.Code)
		}
	}
	if fa.calls.Load() != 0 {
		t.Errorf("engine was handed %d request(s) it has no route for", fa.calls.Load())
	}

	for _, r := range []struct{ method, path string }{
		{http.MethodPost, "/v1/systemone"},
		{http.MethodGet, "/v1/models"},
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(r.method, r.path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusOK {
			t.Errorf("%s %s: status=%d, want 200; body=%s", r.method, r.path, rec.Code, rec.Body)
		}
	}
}

// A path this application will never serve says so while the model is
// still downloading too. 503 + Retry-After there would have a client
// wait for something that is not coming.
func TestModeGuard_RefusesBeforeReadiness(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newModeMux(t, config.ModelEmbedding, func() bool { return false }, fa)

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
		"/v1/chat/completions", strings.NewReader(`{}`)))

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status=%d, want 404 rather than a not-ready 503", rec.Code)
	}
	// The other order still holds: a path it does serve is 503 while booting.
	rec = httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
		"/v1/embeddings", strings.NewReader(`{}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Errorf("status=%d on a served path while not ready, want 503", rec.Code)
	}
}

// OCR forwards the async task contract to its engine, verbs included, so
// a task path is served whatever the method — and the id segment is not
// part of the comparison.
func TestModeGuard_OCRServesItsOwnSurface(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newModeMux(t, config.ModelOCR, func() bool { return true }, fa)

	for _, r := range []struct{ method, path string }{
		{http.MethodPost, "/v1/ocr"},
		{http.MethodGet, "/v1/ocr/queue"},
		{http.MethodGet, "/v1/ocr/jobs/abc123"},
		{http.MethodDelete, "/v1/ocr/jobs/abc123"},
		{http.MethodGet, "/v1/tasks"},
		{http.MethodDelete, "/v1/tasks/t-1"},
		{http.MethodGet, "/v1/tasks/t-1/result"},
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(r.method, r.path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusOK {
			t.Errorf("%s %s: status=%d, want 200; body=%s", r.method, r.path, rec.Code, rec.Body)
		}
	}

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost,
		"/v1/embeddings", strings.NewReader(`{}`)))
	if rec.Code != http.StatusNotFound {
		t.Errorf("embeddings on an OCR application: status=%d, want 404", rec.Code)
	}
}

// What a finished track can be asked about is the engine's decision, and
// a staged engine answers a lyrics alignment subresource this repository
// never enumerated. The gate refused it while the engine implemented it,
// so music forwards its whole surface and lets the engine refuse what it
// does not have.
func TestModeGuard_MusicForwardsEngineSubresources(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := newModeMux(t, config.ModelMusicGeneration, func() bool { return true }, fa)

	for _, r := range []struct{ method, path string }{
		{http.MethodGet, "/v1/models"},
		{http.MethodPost, "/v1/music/generations"},
		{http.MethodGet, "/v1/music/generations/g-1"},
		{http.MethodDelete, "/v1/music/generations/g-1"},
		{http.MethodGet, "/v1/music/generations/g-1/content"},
		{http.MethodGet, "/v1/music/generations/g-1/lyrics-alignment"},
		{http.MethodPost, "/v1/music/formats"},
		{http.MethodGet, "/v1/music/formats/fmt-1"},
	} {
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(r.method, r.path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusOK {
			t.Errorf("%s %s: status=%d, want 200; body=%s", r.method, r.path, rec.Code, rec.Body)
		}
	}
}

// audio and tts declare their surface at runtime — routes and socket
// streams that are not written down in this repository — music_generation
// declares its generation subresources the same way, and chat is open by
// design. A whitelist here would refuse the application's whole reason
// for existing, so these modes must stay pass-through.
func TestModeGuard_UnrestrictedModesForwardEverything(t *testing.T) {
	t.Parallel()
	for _, mode := range []config.ModelType{
		config.ModelChat, config.ModelAudio, config.ModelTTS, config.ModelTranslate,
		config.ModelMusicGeneration, "",
	} {
		fa := &fakeAdapter{}
		mux := newModeMux(t, mode, func() bool { return true }, fa)
		for _, path := range []string{
			"/v1/chat/completions",
			"/v1/audio/transcriptions",
			"/v1/embeddings",
			// The ElevenLabs-shaped voice surface a tts engine mounts.
			// It sits outside /v1/audio, so a guard keyed on that
			// prefix would let the OpenAI half through and 404 this.
			"/v1/voices/add",
			"/v1/text-to-voice/design",
			"/v1/text-to-speech/premade-abc",
		} {
			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{}`)))
			if rec.Code != http.StatusOK {
				t.Errorf("mode=%q POST %s: status=%d, want it forwarded", mode, path, rec.Code)
			}
		}
	}
}
