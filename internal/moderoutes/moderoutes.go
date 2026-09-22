// Package moderoutes says which of the /v1 data-plane paths an
// application in a given mode serves.
//
// It exists because two places needed that answer and each had its own
// copy of it. GET /api/endpoints marked chat unavailable on an embedding
// application, while the /v1/ mount was a catch-all that accepted
// POST /v1/chat/completions and forwarded it to an embedding engine —
// so the catalogue said the endpoint was not there and the port said it
// was, and the client got whatever 404 the engine spells in its own
// format rather than a statement about this application.
//
// Only the modes with a declared route set are restricted. chat, audio,
// tts, translate and music_generation pass everything through, because
// nothing here knows their full surface: audio's eleven routes and three
// socket streams are not written down in this repository at all — the
// engine declares them at runtime — and a whitelist that omitted them
// would refuse the application's entire reason for existing. tts is the
// same argument twice over: on top of the OpenAI /v1/audio/speech family
// it answers an ElevenLabs-shaped voice surface (/v1/voices,
// /v1/text-to-voice, /v1/text-to-speech/{voice_id}) that varies by
// engine base.
//
// music_generation was restricted until a staged engine answered a
// subresource under /v1/music/generations/{id} that this repository had
// never enumerated — lyrics alignment — and the gate 404'd it. The
// generation contract is the engine's: it decides what a finished track
// can be asked about, and a list written here is a list of the
// subresources that existed when it was written. The application worked
// around the gate by routing its shared entrance past llm-init
// altogether, which cost Router the control plane it reads from that
// same host root, so the whitelist was buying nothing and charging for
// it.
package moderoutes

import (
	"net/http"
	"strings"

	"github.com/llm-init/llm-init/internal/config"
)

// AnyMethod is the Method of a route whose verbs belong to the engine
// rather than to us. The async task contract is the engine's, and llm-init
// only forwards it, so a task route the engine declares with a method the
// catalogue never listed is still that mode's route.
const AnyMethod = ""

// Route is one row of a mode's declared surface. Path is a catalogue
// pattern, so it can carry a `{id}` placeholder.
type Route struct {
	Method string
	Path   string
}

// Catalogue patterns. Kept here rather than in the two callers so a path
// cannot be spelled one way for the catalogue and another for the gate.
// The music patterns have no gate left to agree with; they are still
// spelled here because GET /api/endpoints describes those rows.
const (
	PathModels           = "/v1/models"
	PathEmbeddings       = "/v1/embeddings"
	PathRerank           = "/v1/rerank"
	PathOCR              = "/v1/ocr"
	PathOCRQueue         = "/v1/ocr/queue"
	PathOCRJob           = "/v1/ocr/jobs/{id}"
	PathTasks            = "/v1/tasks"
	PathTask             = "/v1/tasks/{id}"
	PathTaskResult       = "/v1/tasks/{id}/result"
	PathMusicGenerations = "/v1/music/generations"
	PathMusicGeneration  = "/v1/music/generations/{id}"
	PathMusicContent     = "/v1/music/generations/{id}/content"
	PathMusicFormats     = "/v1/music/formats"
	PathMusicFormat      = "/v1/music/formats/{id}"
	PathMusicDrafts      = "/v1/music/drafts"
	PathMusicDraft       = "/v1/music/drafts/{id}"
)

var declared = map[config.ModelType][]Route{
	config.ModelEmbedding: {
		{http.MethodGet, PathModels},
		{http.MethodPost, PathEmbeddings},
	},
	config.ModelRerank: {
		{http.MethodGet, PathModels},
		{http.MethodPost, PathRerank},
	},
	config.ModelOCR: {
		{http.MethodGet, PathModels},
		{http.MethodPost, PathOCR},
		{http.MethodGet, PathOCRQueue},
		{http.MethodGet, PathOCRJob},
		{http.MethodDelete, PathOCRJob},
		{AnyMethod, PathTasks},
		{AnyMethod, PathTask},
		{AnyMethod, PathTaskResult},
	},
}

// Restricted reports whether mode has a declared route set. False means
// every path is forwarded and nothing here has an opinion about it.
func Restricted(mode config.ModelType) bool {
	_, ok := declared[mode]
	return ok
}

// Declares reports whether a catalogue row — an exact method and
// pattern, `{id}` included — is one this mode serves. Used by
// GET /api/endpoints, which is describing rows rather than answering
// requests.
func Declares(mode config.ModelType, method, pattern string) bool {
	for _, r := range declared[mode] {
		if r.Path != pattern {
			continue
		}
		if r.Method == AnyMethod || r.Method == method {
			return true
		}
	}
	return false
}

// ServesPath reports whether a concrete request path is one of this
// mode's routes.
//
// The method is deliberately not consulted. The claim being enforced is
// "this application does not serve chat", which is about paths; a wrong
// verb on a path the application does serve is the engine's 405 to give,
// and answering 404 there would be a worse lie than the one being fixed.
func ServesPath(mode config.ModelType, path string) bool {
	for _, r := range declared[mode] {
		if matches(r.Path, path) {
			return true
		}
	}
	return false
}

// matches compares a request path against a catalogue pattern, where
// `{id}` stands for exactly one non-empty segment.
func matches(pattern, path string) bool {
	if !strings.Contains(pattern, "{") {
		return pattern == path
	}
	p := strings.Split(pattern, "/")
	q := strings.Split(path, "/")
	if len(p) != len(q) {
		return false
	}
	for i := range p {
		if strings.HasPrefix(p[i], "{") && strings.HasSuffix(p[i], "}") {
			if q[i] == "" {
				return false
			}
			continue
		}
		if p[i] != q[i] {
			return false
		}
	}
	return true
}
