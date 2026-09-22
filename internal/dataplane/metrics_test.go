package dataplane

import (
	"bufio"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/obs"
)

// hijackableRW records whether Hijack/Flush reached the inner writer.
type hijackableRW struct {
	http.ResponseWriter
	hijacked, flushed bool
}

func (h *hijackableRW) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	h.hijacked = true
	return nil, nil, nil
}
func (h *hijackableRW) Flush() { h.flushed = true }

// TestStatusRecorder_PassesThroughHijackFlush: WS/SSE pass through.
func TestStatusRecorder_PassesThroughHijackFlush(t *testing.T) {
	t.Parallel()
	base := &hijackableRW{ResponseWriter: httptest.NewRecorder()}
	sr := &statusRecorder{ResponseWriter: base, status: http.StatusOK}

	hj, ok := interface{}(sr).(http.Hijacker)
	if !ok {
		t.Fatal("statusRecorder must implement http.Hijacker for WS upgrades")
	}
	if _, _, err := hj.Hijack(); err != nil || !base.hijacked {
		t.Fatalf("Hijack not forwarded: err=%v hijacked=%v", err, base.hijacked)
	}
	fl, ok := interface{}(sr).(http.Flusher)
	if !ok {
		t.Fatal("statusRecorder must implement http.Flusher for SSE")
	}
	fl.Flush()
	if !base.flushed {
		t.Fatal("Flush not forwarded to underlying writer")
	}
	if u, ok := interface{}(sr).(interface{ Unwrap() http.ResponseWriter }); !ok || u.Unwrap() != base {
		t.Fatal("Unwrap must return the underlying ResponseWriter")
	}
}

// TestObserve_CountsAndLabelsByRouteAndStatus drives a few requests
// through Mount with metrics wired and asserts both the counter and
// histogram capture the (route, status) cardinality we expect. This
// is the only test that locks the Prometheus contract introduced in
// v1.0.3 — without it future refactors of routeLabel could silently
// re-introduce high-cardinality labels.
func TestObserve_CountsAndLabelsByRouteAndStatus(t *testing.T) {
	t.Parallel()
	m := obs.NewMetrics()
	fa := &fakeAdapter{}
	mux := http.NewServeMux()
	Mount(mux, Options{
		Ready:   func() bool { return true },
		Adapter: fa,
		Metrics: m,
	})

	cases := []struct {
		method, path     string
		wantRoute, wantS string
		wantCode         int
	}{
		{http.MethodPost, "/v1/chat/completions", routeChatCompletions, statusBucket2xx, http.StatusOK},
		{http.MethodPost, "/api/chat/completions", routeChatCompletions, statusBucket2xx, http.StatusOK}, // alias collapses
		{http.MethodGet, "/v1/models", routeModels, statusBucket2xx, http.StatusOK},
		{http.MethodPost, "/v1/responses", routeResponses, statusBucket2xx, http.StatusOK},
		{http.MethodPost, "/v1/completions", routeCompletions, statusBucket2xx, http.StatusOK},
		{http.MethodPost, "/v1/messages", routeMessages, statusBucket2xx, http.StatusOK},
		// The async task contract: three paths, one label, and the id must not reach it.
		{http.MethodGet, "/v1/tasks", routeTasks, statusBucket2xx, http.StatusOK},
		{http.MethodGet, "/v1/tasks/tsk_9f21c0b4e7a8", routeTasks, statusBucket2xx, http.StatusOK},
		{http.MethodGet, "/v1/tasks/tsk_9f21c0b4e7a8/result", routeTasks, statusBucket2xx, http.StatusOK},
	}
	for _, tc := range cases {
		req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(`{}`))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != tc.wantCode {
			t.Errorf("%s %s status=%d want %d", tc.method, tc.path, rec.Code, tc.wantCode)
		}
	}

	// chat_completions should have 2 hits (v1 + alias) under 2xx.
	if got := counter(t, m, "llm_init_dataplane_requests_total", map[string]string{
		"route": routeChatCompletions, "status": statusBucket2xx,
	}); got != 2 {
		t.Errorf("chat_completions/2xx = %v want 2", got)
	}
	if got := counter(t, m, "llm_init_dataplane_requests_total", map[string]string{
		"route": routeModels, "status": statusBucket2xx,
	}); got != 1 {
		t.Errorf("models/2xx = %v want 1", got)
	}
	if got := counter(t, m, "llm_init_dataplane_requests_total", map[string]string{
		"route": routeResponses, "status": statusBucket2xx,
	}); got != 1 {
		t.Errorf("responses/2xx = %v want 1", got)
	}
	if got := counter(t, m, "llm_init_dataplane_requests_total", map[string]string{
		"route": routeMessages, "status": statusBucket2xx,
	}); got != 1 {
		t.Errorf("messages/2xx = %v want 1", got)
	}
	// Three task paths, one label: a per-id label would make this 3 series, not 1.
	if got := counter(t, m, "llm_init_dataplane_requests_total", map[string]string{
		"route": routeTasks, "status": statusBucket2xx,
	}); got != 3 {
		t.Errorf("tasks/2xx = %v want 3", got)
	}
	// Latency histogram must have at least one observation per route.
	if got := histogramCount(t, m, "llm_init_dataplane_request_duration_seconds", "route", routeChatCompletions); got != 2 {
		t.Errorf("chat_completions latency count = %d want 2", got)
	}
}

// TestObserve_NotReadyStill5xxCounted asserts that a 503 from
// notReadyGuard counts as a dataplane request — the failure mode
// dashboards most want to detect ("how often did clients see 503
// during a download?").
func TestObserve_NotReadyStill5xxCounted(t *testing.T) {
	t.Parallel()
	m := obs.NewMetrics()
	fa := &fakeAdapter{}
	mux := http.NewServeMux()
	Mount(mux, Options{
		Ready:   func() bool { return false },
		Manager: newProgressManager(),
		Adapter: fa,
		Metrics: m,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d", rec.Code)
	}
	if got := counter(t, m, "llm_init_dataplane_requests_total", map[string]string{
		"route": routeChatCompletions, "status": statusBucket5xx,
	}); got != 1 {
		t.Errorf("503 not_ready should be counted under chat_completions/5xx, got %v", got)
	}
}

// TestRouteLabel_AudioSurfaceIsNotCollapsedIntoOther pins the labels an
// audio instance is measured by. Audio declares no route allowlist, so the
// binary forwards every /v1/* path it is handed; before these labels
// existed a failing synthesis and a failing transcription were the same
// series, and an operator had to read engine logs to tell them apart.
func TestRouteLabel_AudioSurfaceIsNotCollapsedIntoOther(t *testing.T) {
	t.Parallel()
	cases := []struct {
		path, want string
	}{
		// OpenAI shape.
		{"/v1/audio/transcriptions", routeAudioTranscriptions},
		{"/v1/audio/translations", routeAudioTranslations},
		{"/v1/audio/stream", routeAudioSTTStream},
		{"/v1/audio/align", routeAudioAlign},
		{"/v1/audio/vad", routeAudioVAD},
		{"/v1/audio/diarization", routeAudioDiarization},
		{"/v1/audio/enhance", routeAudioEnhance},
		{"/v1/audio/speech", routeAudioSpeech},
		// Speaker embeddings live under /v1/audio and must not be
		// confused with the text /v1/embeddings route.
		{"/v1/audio/embeddings", routeAudioSpeakerEmbed},
		{"/v1/embeddings", routeEmbeddings},
		// A capability with several paths keeps one label.
		{"/v1/audio/speech/batch", routeAudioSpeech},
		{"/v1/audio/speech/stream", routeAudioSpeech},
		{"/v1/audio/speech/clone", routeAudioSpeech},
		{"/v1/audio/diarize/stream", routeAudioDiarization},
		// ElevenLabs shape.
		{"/v1/voices", routeAudioVoices},
		{"/v1/voices/add", routeAudioVoices},
		{"/v1/voices/v_123/settings/edit", routeAudioVoices},
		{"/v1/text-to-speech/v_123", routeAudioTTS},
		{"/v1/text-to-speech/v_123/stream", routeAudioTTS},
		{"/v1/text-to-voice", routeAudioVoiceDesign},
		{"/v1/text-to-voice/design", routeAudioVoiceDesign},
		{"/v1/speech-to-speech/v_123", routeAudioS2S},
		{"/v1/history", routeAudioHistory},
		{"/v1/history/h_123/audio", routeAudioHistory},
		// The deprecated task alias belongs with polling, not with audio.
		{"/v1/audio/tasks", routeTasks},
		{"/v1/audio/tasks/tsk_1/result", routeTasks},
		{"/v1/tasks/tsk_1/result", routeTasks},
		// A path this binary does not know is still recognisably audio.
		{"/v1/audio/something-new", routeAudioOther},
		{"/v1/something-else", labelOther},
	}
	for _, tc := range cases {
		if got := routeLabel(tc.path); got != tc.want {
			t.Errorf("routeLabel(%q) = %q want %q", tc.path, got, tc.want)
		}
	}
}

// TestRouteLabel_IDsNeverBecomeLabels is the cardinality guard: every id
// the audio surface carries in its path must collapse into its route.
func TestRouteLabel_IDsNeverBecomeLabels(t *testing.T) {
	t.Parallel()
	seen := map[string]struct{}{}
	for _, id := range []string{"v_1", "v_2", "v_3"} {
		seen[routeLabel("/v1/text-to-speech/"+id)] = struct{}{}
		seen[routeLabel("/v1/voices/"+id)] = struct{}{}
		seen[routeLabel("/v1/history/"+id+"/audio")] = struct{}{}
	}
	if len(seen) != 3 {
		t.Fatalf("three routes over three ids produced %d labels: %v", len(seen), seen)
	}
}

// TestObserve_NilMetricsIsPassthrough proves the nil-safe contract: a
// Mount without Metrics still serves traffic, just without observability.
func TestObserve_NilMetricsIsPassthrough(t *testing.T) {
	t.Parallel()
	fa := &fakeAdapter{}
	mux := http.NewServeMux()
	Mount(mux, Options{
		Ready:   func() bool { return true },
		Adapter: fa,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Errorf("status=%d", rec.Code)
	}
}

// counter looks up a labelled counter sample. Returns -1 on miss.
func counter(t *testing.T, m *obs.Metrics, name string, labels map[string]string) float64 {
	t.Helper()
	mfs, err := m.Registry().Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, sample := range mf.GetMetric() {
			ok := true
			for k, v := range labels {
				match := false
				for _, l := range sample.GetLabel() {
					if l.GetName() == k && l.GetValue() == v {
						match = true
						break
					}
				}
				if !match {
					ok = false
					break
				}
			}
			if !ok {
				continue
			}
			if c := sample.GetCounter(); c != nil {
				return c.GetValue()
			}
		}
	}
	return -1
}

// histogramCount returns the sample count of a labelled histogram or
// -1 on miss. Used to assert "the route was observed N times".
func histogramCount(t *testing.T, m *obs.Metrics, name, labelName, labelValue string) uint64 {
	t.Helper()
	mfs, err := m.Registry().Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		for _, sample := range mf.GetMetric() {
			match := false
			for _, l := range sample.GetLabel() {
				if l.GetName() == labelName && l.GetValue() == labelValue {
					match = true
					break
				}
			}
			if !match {
				continue
			}
			if h := sample.GetHistogram(); h != nil {
				return h.GetSampleCount()
			}
		}
	}
	return 0
}
