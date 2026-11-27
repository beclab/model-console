package comfyconsole

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

func TestStudioMediaGeneration(t *testing.T) {
	token := strings.Repeat("s", 32)
	cfg := &AppConfig{
		Models: []Model{{Name: "model.safetensors", Size: 1}},
		APIWorkflows: []APIWorkflow{{
			ID: "t2i", AspectRatios: []string{"1:1", "16:9"}, DefaultAspect: "1:1", Resolution: "2K",
		}},
	}
	media, err := NewMediaConfig(cfg, "h3-image", "image", token)
	if err != nil {
		t.Fatal(err)
	}
	var creates atomic.Int32
	var studioReady atomic.Bool
	studio := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Model-Console-Token") != "" {
			t.Error("Model Console credential reached Studio")
		}
		switch r.URL.Path {
		case "/system_stats":
			w.WriteHeader(http.StatusOK)
			return
		case "/studio/api/health":
			if !studioReady.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
			}
			return
		}
		if r.Header.Get("X-Bfl-User") != "alice" {
			http.NotFound(w, r)
			return
		}
		switch r.URL.Path {
		case "/studio/api/jobs":
			creates.Add(1)
			var body struct {
				Mode   string
				Prompt string
				Params map[string]any
				Assets json.RawMessage
			}
			if r.Method != http.MethodPost || json.NewDecoder(r.Body).Decode(&body) != nil ||
				body.Mode != "t2i" || body.Prompt != "cat" ||
				body.Params["aspect"] != "16:9" || body.Params["resolution"] != "2K" ||
				body.Params["seed"] != float64(5) || body.Assets != nil {
				t.Errorf("unexpected Studio request: %+v", body)
			}
			_, _ = io.WriteString(w, `{"jobs":[{"id":"job-1","status":"queued"}]}`)
		case "/studio/api/jobs/job-1":
			_, _ = io.WriteString(w, `{"id":"job-1","status":"done","outputs":["image-1"]}`)
		case "/studio/api/jobs/unrecognized":
			_, _ = io.WriteString(w, `{"id":"unrecognized","status":"something-new"}`)
		case "/studio/api/media/image-1":
			w.Header().Set("Content-Type", "image/png")
			_, _ = io.WriteString(w, "image bytes")
		default:
			t.Errorf("unexpected Studio request: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer studio.Close()

	console := New(Options{Config: cfg, Media: media, ComfyURL: studio.URL})
	handler := console.Handler()
	request := func(method, path, user, credential, body string, status int, contains string) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("X-Model-Console-Token", credential)
		req.Header.Set("X-Model-Console-User", user)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, req)
		if response.Code != status || !strings.Contains(response.Body.String(), contains) {
			t.Errorf("%s %s as %s: got %d %s, want %d and %q", method, path, user, response.Code, response.Body, status, contains)
		}
		return response
	}

	const create = "/api/v1/generations"
	const content = create + "/job-1/content?outputId="
	const body = `{"workflowId":"t2i","prompt":"cat","params":{"aspectRatio":"16:9","seed":5}}`
	card := request("GET", "/api/model-spec", "", "", "", 200, `"mode":"image_generation"`)
	for _, field := range []string{`"name":"h3-image"`, `"protocol":"studio-v1"`, `"workflow_id":"t2i"`, `"operations":["generate"]`} {
		if !strings.Contains(card.Body.String(), field) {
			t.Errorf("missing %s in model card: %s", field, card.Body)
		}
	}
	request("GET", "/api/progress", "", "", "", 200, `"phase":"download"`)
	request("POST", create, "alice", token, body, 503, "model_not_ready")
	console.items[0].state = StateDone
	console.backend.Store(console.probe(context.Background()))
	request("GET", "/api/progress", "", "", "", 200, `"phase":"loading"`)
	request("POST", create, "alice", token, body, 503, "model_not_ready")
	studioReady.Store(true)
	console.backend.Store(console.probe(context.Background()))
	request("GET", "/api/progress", "", "", "", 200, `"phase":"ready"`)
	request("GET", "/api/v1/workflows", "", "", "", 404, "")
	request("POST", create, "alice", "invalid", body, 401, "unauthorized")
	request("POST", create, "", token, body, 401, "unauthorized")
	request("POST", create, "alice", token, strings.Replace(body, "16:9", "4:5", 1), 400, "invalid_params")
	request("POST", create, "alice", token, strings.Replace(body, `"seed":5`, `"resolution":"4K"`, 1), 400, "invalid_params")
	request("POST", create, "alice", token, strings.Replace(body, `"seed":5`, `"seed":-1`, 1), 400, "invalid_request")
	request("POST", create, "alice", token, strings.Replace(body, `"params"`, `"inputs":[],"params"`, 1), 400, "invalid_request")
	request("POST", create, "alice", token, body, 202, `"id":"job-1"`)
	request("GET", create+"/job-1", "alice", token, "", 200, `"outputs":[{"id":"image-1"}]`)
	request("GET", create+"/job-1", "bob", token, "", 404, "studio_rejected")
	request("GET", create+"/unrecognized", "alice", token, "", 502, "invalid_studio_response")
	request("GET", create+"/unrecognized/content?outputId=image-1", "alice", token, "", 502, "invalid_studio_response")
	request("GET", content+"different-image", "alice", token, "", 404, "not_found")
	request("GET", content+"image-1", "bob", token, "", 404, "studio_rejected")
	image := request("GET", content+"image-1", "alice", token, "", 200, "image bytes")
	if image.Header().Get("Content-Type") != "image/png" || creates.Load() != 1 {
		t.Fatalf("content type = %q, Studio creates = %d", image.Header().Get("Content-Type"), creates.Load())
	}
}

func TestStudioPixelSizesSelectAResolutionTier(t *testing.T) {
	flow := APIWorkflow{
		Operation: operationGenerate, DefaultAspect: "1:1", AspectRatios: []string{"1:1"},
		Resolution: "1K", Resolutions: []string{"1K", "2K"},
		ResolutionPixels: map[string]int64{"1K": 1024 * 1024, "2K": 2048 * 2048},
	}
	var request generationRequest
	for _, tc := range []struct {
		pixels int64
		tier   string
	}{{1024 * 1024, "1K"}, {2048 * 2048, "2K"}, {1536 * 1536, ""}} {
		request.Params.PixelCount = &tc.pixels
		params, err := flow.generationParams(request, config.ModelImage)
		if tc.tier == "" {
			if err == nil {
				t.Fatal("unsupported size was accepted")
			}
			continue
		}
		if err != nil || params["resolution"] != tc.tier {
			t.Fatalf("pixels %d: got %v, %v; want resolution %s", tc.pixels, params, err, tc.tier)
		}
	}
}

func TestStudioDownloadWaitsForProgressAndRejectsTruncation(t *testing.T) {
	studio := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/studio/api/jobs/job":
			_, _ = io.WriteString(w, `{"id":"job","status":"done","outputs":["slow","short"]}`)
		case "/studio/api/media/slow":
			w.Header().Set("Content-Length", "4")
			for i := 0; i < 4; i++ {
				time.Sleep(80 * time.Millisecond)
				_, _ = io.WriteString(w, "x")
				w.(http.Flusher).Flush()
			}
		case "/studio/api/media/short":
			w.Header().Set("Content-Length", "4")
			_, _ = io.WriteString(w, "x")
		default:
			http.NotFound(w, r)
		}
	}))
	defer studio.Close()
	console := New(Options{Config: &AppConfig{}, ComfyURL: studio.URL, StudioClient: &http.Client{Timeout: 250 * time.Millisecond}})
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.SetPathValue("id", "job")
		console.getGenerationContent(w, r, "alice")
	}))
	defer proxy.Close()
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(proxy.URL + "?outputId=slow")
	if err != nil {
		t.Fatal(err)
	}
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil || resp.StatusCode != http.StatusOK || string(body) != "xxxx" {
		t.Fatalf("slow download: status %d, body %q, error %v", resp.StatusCode, body, err)
	}
	resp, err = client.Get(proxy.URL + "?outputId=short")
	if err == nil {
		_, err = io.ReadAll(resp.Body)
		resp.Body.Close()
	}
	if err == nil {
		t.Fatal("download reported success for incomplete content")
	}
}

func TestStudioVideoGeneration(t *testing.T) {
	token := strings.Repeat("s", 32)
	cfg := &AppConfig{APIWorkflows: []APIWorkflow{{
		ID: "t2v", AspectRatios: []string{"16:9", "9:16"}, DefaultAspect: "16:9",
		Resolution: "480p", Resolutions: []string{"480p", "720p"},
		Duration: &videoDuration{Default: 5, Min: 2, Max: 15},
	}}}
	media, err := NewMediaConfig(cfg, "minimax-h3", "video", token)
	if err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int32
	studio := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/system_stats", "/studio/api/health":
			w.WriteHeader(http.StatusOK)
		case "/studio/api/jobs":
			calls.Add(1)
			var job struct {
				Mode   string
				Params map[string]any
			}
			err := json.NewDecoder(r.Body).Decode(&job)
			if err != nil || r.Method != http.MethodPost || r.Header.Get("X-Bfl-User") != "alice" ||
				job.Mode != "t2v" || job.Params["aspect"] != "16:9" ||
				job.Params["resolution"] != "720p" || job.Params["duration"] != float64(8) {
				t.Errorf("unexpected video job: %+v, %v", job, err)
			}
			_, _ = io.WriteString(w, `{"jobs":[{"id":"video-job","status":"queued"}]}`)
		case "/studio/api/jobs/video-job":
			_, _ = io.WriteString(w, `{"id":"video-job","status":"done","outputs":["mp4"]}`)
		case "/studio/api/media/mp4":
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = io.WriteString(w, "mp4 bytes")
		default:
			t.Errorf("unexpected Studio request %s", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer studio.Close()

	console := New(Options{Config: cfg, Media: media, ComfyURL: studio.URL})
	console.backend.Store(console.probe(context.Background()))
	handler := console.Handler()
	request := func(method, path, body string, code int, contains string) *httptest.ResponseRecorder {
		t.Helper()
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("X-Model-Console-Token", token)
		req.Header.Set("X-Model-Console-User", "alice")
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != code || !strings.Contains(rec.Body.String(), contains) {
			t.Fatalf("%s %s: status %d, body %s", method, path, rec.Code, rec.Body)
		}
		return rec
	}
	const create = "/api/v1/generations"
	const body = `{"workflowId":"t2v","prompt":"cat","params":{"resolution":"720p","durationSeconds":8}}`
	request("GET", "/api/model-spec", "", 200, `"mode":"video_generation"`)
	request("POST", create, strings.Replace(body, ":8", ":1", 1), 400, "invalid_params")
	request("POST", create, strings.Replace(body, ":8", ":5.5", 1), 400, "invalid_request")
	request("POST", create, strings.Replace(body, "720p", "1080p", 1), 400, "invalid_params")
	request("POST", create, strings.Replace(body, "t2v", "i2v", 1), 400, "invalid_workflow")
	request("POST", create, body, 202, `"id":"video-job"`)
	request("GET", create+"/video-job", "", 200, `"outputs":[{"id":"mp4"}]`)
	out := request("GET", create+"/video-job/content?outputId=mp4", "", 200, "mp4 bytes")
	if calls.Load() != 1 || out.Header().Get("Content-Type") != "video/mp4" {
		t.Fatalf("Studio creates = %d, download headers = %v", calls.Load(), out.Header())
	}
}

func TestStudioJobStates(t *testing.T) {
	for studio, want := range map[string]string{
		"queued": "queued", "running": "in_progress", "done": "completed",
		"cancelled": "canceled", "failed": "failed", "interrupted": "failed",
	} {
		job := studioJob{ID: "job-1", Status: studio, Outputs: []string{"output-1"}}
		got, ok := job.generation(httptest.NewRecorder())
		if !ok || got.Status != want || (got.ErrorMessage != "") != (want == "failed") ||
			(len(got.Outputs) == 1) != (want == "completed") {
			t.Errorf("Studio state %q: got %+v, ok=%t; want %q", studio, got, ok, want)
		}
	}
}
