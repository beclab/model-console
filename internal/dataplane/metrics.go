package dataplane

import (
	"bufio"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/obs"
)

// observe wraps next so every served request increments
// `llm_init_dataplane_requests_total` with (route, status) labels and
// observes its duration in `llm_init_dataplane_request_duration_seconds`
// labelled by route. nil-safe: if metrics is nil the wrapper degrades
// to next directly so unit tests can omit it.
//
// Route is derived from `r.URL.Path` via `routeLabel` (see comment
// there) to keep label cardinality bounded — we never want a unique
// label per path so a malicious client can't blow up the registry by
// hitting `/v1/<random-uuid>` repeatedly.
func observe(metrics *obs.Metrics, next http.Handler) http.Handler {
	if metrics == nil {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		route := routeLabel(r.URL.Path)
		start := time.Now()
		sw := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(sw, r)
		elapsed := time.Since(start).Seconds()
		metrics.DataPlaneRequests.
			WithLabelValues(route, statusBucket(sw.status)).
			Inc()
		metrics.DataPlaneLatency.WithLabelValues(route).Observe(elapsed)
	})
}

// statusRecorder captures the HTTP status without buffering the body.
// Wrapping is required because http.ResponseWriter does not expose the
// status that WriteHeader was called with; without this we'd be unable
// to label the counter by 2xx/4xx/5xx.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (s *statusRecorder) WriteHeader(code int) {
	if !s.wroteHeader {
		s.status = code
		s.wroteHeader = true
	}
	s.ResponseWriter.WriteHeader(code)
}

// Write triggers an implicit WriteHeader(200) per net/http convention;
// mirror that so the counter reflects the right bucket when handlers
// stream body bytes without an explicit WriteHeader call.
func (s *statusRecorder) Write(b []byte) (int, error) {
	if !s.wroteHeader {
		s.status = http.StatusOK
		s.wroteHeader = true
	}
	return s.ResponseWriter.Write(b)
}

// Hijack forwards to the underlying writer so WebSocket upgrades work.
func (s *statusRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	hj, ok := s.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("dataplane: ResponseWriter %T is not an http.Hijacker", s.ResponseWriter)
	}
	return hj.Hijack()
}

// Flush forwards SSE/stream flushes to the underlying writer.
func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap exposes the underlying writer to http.ResponseController.
func (s *statusRecorder) Unwrap() http.ResponseWriter { return s.ResponseWriter }

// Prometheus label values for the dataplane metrics. These are
// dashboard-keyed strings; renaming any one of them rotates an
// operator-facing label and breaks existing Grafana queries, so they
// live as named constants and every reference in production +
// metrics_test.go threads through these names. goconst v2 picks up
// the literals as cross-cutting label values; named constants keep
// the file lint-clean and self-documenting.
const (
	// labelOther is the catch-all bucket shared by routeLabel (unknown
	// paths) and statusBucket (non-standard HTTP codes).
	labelOther = "other"

	// routeChatCompletions covers BOTH /v1/chat/completions and the
	// OpenWebUI /api/chat/completions alias — they collapse to the
	// same label because the wire shape, retry budget, and SSE
	// semantics are identical from the operator's POV.
	routeChatCompletions = "chat_completions"
	routeCompletions     = "completions"
	routeEmbeddings      = "embeddings"
	routeRerank          = "rerank"
	routeModels          = "models"
	routeResponses       = "responses"
	routeMessages        = "messages"

	// routeTasks covers the whole async task contract (/v1/tasks, /v1/tasks/{id},
	// /v1/tasks/{id}/result) and the engines' deprecated /v1/audio/tasks alias for
	// the same thing. Polling is the busiest path on an ocr or audio instance —
	// left in labelOther its rate and latency would be indistinguishable from everything
	// else llm-init happens to forward.
	routeTasks = "tasks"

	// The audio surface. An audio instance forwards every /v1/* path it is
	// given (audio and tts declare no route allowlist in moderoutes), so
	// before these existed a transcription, a synthesis and a speaker
	// diarization shared one label with everything else and an operator
	// could not tell which of them was failing.
	//
	// One label per capability rather than per path: the engines mount
	// several paths for a single capability (speech, speech/batch,
	// speech/stream) and splitting those adds cardinality without adding
	// an answer to any question an operator asks.
	routeAudioTranscriptions = "audio_transcriptions"
	routeAudioTranslations   = "audio_translations"
	routeAudioSTTStream      = "audio_stt_stream"
	routeAudioAlign          = "audio_align"
	routeAudioVAD            = "audio_vad"
	routeAudioDiarization    = "audio_diarization"
	routeAudioSpeakerEmbed   = "audio_speaker_embeddings"
	routeAudioEnhance        = "audio_enhance"
	routeAudioSpeech         = "audio_speech"
	routeAudioVoices         = "audio_voices"
	routeAudioTTS            = "audio_tts"
	routeAudioVoiceDesign    = "audio_voice_design"
	routeAudioHistory        = "audio_history"
	routeAudioS2S            = "audio_speech_to_speech"

	// routeAudioOther is the catch-all for the /v1/audio prefix. It is
	// distinct from labelOther on purpose: a path arriving here is one this
	// binary does not recognise but the engine may well serve, and that is
	// worth telling apart from an unknown /v1/ path.
	routeAudioOther = "audio_other"

	statusBucket1xx = "1xx"
	statusBucket2xx = "2xx"
	statusBucket3xx = "3xx"
	statusBucket4xx = "4xx"
	statusBucket5xx = "5xx"
)

// The request paths routeLabel recognises. Named for the same reason the
// labels above are: the mount and the label have to agree on the
// spelling, and a path that reaches routeLabel misspelled does not fail
// — it silently lands in labelOther, taking its rate and latency with
// it.
const (
	pathChatCompletions    = "/v1/chat/completions"
	pathAPIChatCompletions = "/api/chat/completions"
	pathCompletions        = "/v1/completions"
	pathEmbeddings         = "/v1/embeddings"
	pathRerank             = "/v1/rerank"
	pathModels             = "/v1/models"
	pathResponses          = "/v1/responses"
	pathMessages           = "/v1/messages"
	pathTasks              = "/v1/tasks"

	pathAudio          = "/v1/audio"
	pathAudioTasks     = "/v1/audio/tasks"
	pathVoices         = "/v1/voices"
	pathTextToSpeech   = "/v1/text-to-speech"
	pathTextToVoice    = "/v1/text-to-voice"
	pathSpeechToSpeech = "/v1/speech-to-speech"
	pathHistory        = "/v1/history"
)

// audioRoutes maps a path under /v1/audio to its capability label. Keyed by
// the suffix so the table reads like the engines' own route list.
var audioRoutes = map[string]string{
	"transcriptions": routeAudioTranscriptions,
	"translations":   routeAudioTranslations,
	"stream":         routeAudioSTTStream,
	"align":          routeAudioAlign,
	"vad":            routeAudioVAD,
	"diarization":    routeAudioDiarization,
	"diarize":        routeAudioDiarization,
	"embeddings":     routeAudioSpeakerEmbed,
	"enhance":        routeAudioEnhance,
	"speech":         routeAudioSpeech,
	"voices":         routeAudioVoices,
}

// audioRoute labels a path already known to sit under /v1/audio. Only the
// first suffix segment is read, so /v1/audio/speech/stream and
// /v1/audio/diarize/stream stay with the capability they belong to instead
// of becoming labels of their own.
func audioRoute(path string) string {
	suffix := strings.TrimPrefix(path, pathAudio+"/")
	if i := strings.IndexByte(suffix, '/'); i >= 0 {
		suffix = suffix[:i]
	}
	if label, ok := audioRoutes[suffix]; ok {
		return label
	}
	return routeAudioOther
}

// underPrefix reports whether path is prefix itself or something below it.
func underPrefix(path, prefix string) bool {
	return path == prefix || strings.HasPrefix(path, prefix+"/")
}

// routeLabel maps a URL path to a low-cardinality route label suitable
// for Prometheus. We collapse every adapter-handled path into one of
// the well-known OpenAI / OpenWebUI endpoints, with labelOther as the
// catch-all so unknown paths never explode label space.
func routeLabel(path string) string {
	switch {
	case path == pathChatCompletions:
		return routeChatCompletions
	case path == pathAPIChatCompletions:
		return routeChatCompletions
	case path == pathCompletions:
		return routeCompletions
	case path == pathEmbeddings:
		return routeEmbeddings
	case path == pathRerank:
		return routeRerank
	case underPrefix(path, pathModels):
		return routeModels
	case path == pathResponses:
		return routeResponses
	case underPrefix(path, pathMessages):
		return routeMessages
	// The audio alias is checked with the canonical path, before the audio
	// prefix below claims it: polling belongs with polling whichever
	// spelling the client used.
	case underPrefix(path, pathTasks) || underPrefix(path, pathAudioTasks):
		return routeTasks
	case underPrefix(path, pathAudio):
		return audioRoute(path)
	case underPrefix(path, pathVoices):
		return routeAudioVoices
	case underPrefix(path, pathTextToSpeech):
		return routeAudioTTS
	case underPrefix(path, pathTextToVoice):
		return routeAudioVoiceDesign
	case underPrefix(path, pathSpeechToSpeech):
		return routeAudioS2S
	case underPrefix(path, pathHistory):
		return routeAudioHistory
	default:
		return labelOther
	}
}

// statusBucket reduces an HTTP status code to its decade ("2xx", "4xx",
// "5xx"…) so the requests Counter does not get a label per integer
// status. Any code outside the standard 100-599 range maps to labelOther.
func statusBucket(code int) string {
	switch {
	case code >= 200 && code < 300:
		return statusBucket2xx
	case code >= 300 && code < 400:
		return statusBucket3xx
	case code >= 400 && code < 500:
		return statusBucket4xx
	case code >= 500 && code < 600:
		return statusBucket5xx
	case code >= 100 && code < 200:
		return statusBucket1xx
	default:
		return labelOther
	}
}
