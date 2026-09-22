package moderoutes

import (
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

// The two readers ask different questions of the same table — the
// catalogue about a row, the /v1 gate about a request — and the whole
// point of one table is that they cannot disagree. A declared row whose
// path the gate refuses would put us back where we started, with a
// catalogue advertising an endpoint the port rejects.
func TestDeclaredRowsAreAlsoServedPaths(t *testing.T) {
	t.Parallel()
	for mode, routes := range declared {
		for _, r := range routes {
			concrete := strings.ReplaceAll(r.Path, "{id}", "x")
			if !ServesPath(mode, concrete) {
				t.Errorf("mode=%s declares %s %s but the gate refuses %s",
					mode, r.Method, r.Path, concrete)
			}
			if !Declares(mode, r.Method, r.Path) {
				t.Errorf("mode=%s: Declares disagrees with its own table for %s %s",
					mode, r.Method, r.Path)
			}
		}
	}
}

func TestRestricted(t *testing.T) {
	t.Parallel()
	for _, mode := range []config.ModelType{config.ModelEmbedding, config.ModelRerank, config.ModelOCR} {
		if !Restricted(mode) {
			t.Errorf("mode=%s should have a declared route set", mode)
		}
	}
	// Unrestricted is the safe default, and an unset mode has to land
	// there: it is what every test fixture and every pre-card config
	// leaves behind, and restricting it would refuse everything.
	for _, mode := range []config.ModelType{
		config.ModelChat, config.ModelAudio, config.ModelTTS, config.ModelTranslate,
		config.ModelMusicGeneration, "", "future-mode",
	} {
		if Restricted(mode) {
			t.Errorf("mode=%q must be pass-through", mode)
		}
		if ServesPath(mode, "/v1/embeddings") {
			t.Errorf("mode=%q: ServesPath must not claim a surface it has no table for", mode)
		}
	}
}

// music_generation was restricted once, and the whitelist 404'd the
// lyrics alignment subresource of a staged engine that implements it. The
// application answered by routing its shared entrance past llm-init,
// which cost Router the control plane it reads from that same host root —
// so re-restricting this mode does not merely refuse a subresource, it
// invites the workaround that takes the model off Router entirely.
func TestRestricted_MusicForwardsEngineSubresources(t *testing.T) {
	t.Parallel()
	if Restricted(config.ModelMusicGeneration) {
		t.Fatal("mode=music_generation became restricted; a generation subresource the engine implements would 404")
	}
	if ServesPath(config.ModelMusicGeneration, "/v1/music/generations/g-1") {
		t.Error("ServesPath must not claim a surface music_generation has no table for")
	}
}

// tts is the mode most at risk of being restricted by accident: its surface
// is the widest of the five, half of it lives outside /v1/audio, and none of
// it is written down in this repository. A whitelist would 404 the voice
// routes that are the application's whole point.
func TestRestricted_TTSForwardsTheVoiceSurface(t *testing.T) {
	t.Parallel()
	for _, p := range []string{
		"/v1/audio/speech",
		"/v1/audio/speech/clone",
		"/v1/voices",
		"/v1/voices/add",
		"/v1/voices/premade-abc/edit",
		"/v1/text-to-voice",
		"/v1/text-to-voice/design",
		"/v1/text-to-speech/premade-abc",
		"/v1/text-to-speech/premade-abc/stream",
	} {
		if Restricted(config.ModelTTS) {
			t.Fatalf("mode=tts became restricted; %s and the rest would 404", p)
		}
	}
}

func TestDeclares_MethodIsPartOfTheRow(t *testing.T) {
	t.Parallel()
	if Declares(config.ModelEmbedding, "GET", PathEmbeddings) {
		t.Error("embedding declares POST /v1/embeddings, not GET")
	}
	if !Declares(config.ModelEmbedding, "POST", PathEmbeddings) {
		t.Error("POST /v1/embeddings should be declared")
	}
	// AnyMethod exists because the async task contract is the engine's:
	// a verb it declares on a task path is that mode's route even though
	// the catalogue never listed it.
	if !Declares(config.ModelOCR, "PATCH", PathTask) {
		t.Error("a task route should accept any method the engine declares")
	}
	if Declares(config.ModelOCR, "PATCH", PathEmbeddings) {
		t.Error("AnyMethod must not leak past the task paths")
	}
	if !Declares(config.ModelRerank, "POST", PathRerank) {
		t.Error("POST /v1/rerank should be declared for rerank mode")
	}
	if Declares(config.ModelRerank, "POST", PathEmbeddings) {
		t.Error("rerank must not declare embeddings")
	}
}

func TestServesPath_RerankMode(t *testing.T) {
	t.Parallel()
	if !ServesPath(config.ModelRerank, "/v1/rerank") {
		t.Error("rerank should serve /v1/rerank")
	}
	if ServesPath(config.ModelRerank, "/v1/chat/completions") {
		t.Error("rerank must not serve chat")
	}
}

func TestServesPath_PatternMatching(t *testing.T) {
	t.Parallel()
	cases := []struct {
		path string
		want bool
	}{
		{"/v1/ocr/jobs/abc-123", true},
		{"/v1/tasks", true},
		{"/v1/tasks/t1", true},
		{"/v1/tasks/t1/result", true},
		// An id is exactly one segment: neither missing nor a subtree.
		{"/v1/ocr/jobs", false},
		{"/v1/ocr/jobs/", false},
		{"/v1/tasks/t1/result/extra", false},
		{"/v1/chat/completions", false},
		// A prefix of a served path is not a served path — the /v1/
		// mount is a catch-all, so this is the class of request the gate
		// exists to answer.
		{"/v1/ocrx", false},
		{"/v1", false},
	}
	for _, c := range cases {
		if got := ServesPath(config.ModelOCR, c.path); got != c.want {
			t.Errorf("ServesPath(ocr, %q) = %v, want %v", c.path, got, c.want)
		}
	}
}
