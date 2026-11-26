package comfyconsole

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

func TestStudioUploadsGenerationInputs(t *testing.T) {
	const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zt9sAAAAASUVORK5CYII="
	image := "data:image/png;base64," + png
	for _, tc := range []struct {
		name, mode, operation, workflow, resolution string
		inputs                                      *generationInputs
		want                                        []studioAsset
	}{
		{operationEdit, string(config.ModelImage), operationEdit, operationEdit, "2K", &generationInputs{Images: []string{image}, Mask: image},
			[]studioAsset{{ID: assetReferenceImage, Role: assetReferenceImage}, {ID: "mask", Role: "mask"}}},
		{"first frame", "video", operationFirstFrame, "i2v", "480p", &generationInputs{Images: []string{png}},
			[]studioAsset{{ID: assetFirstFrame, Role: assetFirstFrame}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			flow := APIWorkflow{ID: tc.workflow, Operation: tc.operation, Resolution: tc.resolution}
			main := APIWorkflow{ID: "text", Resolution: tc.resolution, AspectRatios: []string{"1:1"}, DefaultAspect: "1:1"}
			if tc.mode == "video" {
				flow.Duration = &videoDuration{Default: 5, Min: 2, Max: 15}
				main.Duration = flow.Duration
			}
			cfg := &AppConfig{APIWorkflows: []APIWorkflow{main, flow}}
			token := strings.Repeat("s", 32)
			media, err := NewMediaConfig(cfg, "model", tc.mode, token)
			if err != nil {
				t.Fatal(err)
			}
			studio := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/system_stats" || r.URL.Path == "/studio/api/health" {
					w.WriteHeader(http.StatusOK)
					return
				}
				if r.Header.Get("X-Bfl-User") != "alice" || r.Header.Get("X-Model-Console-Token") != "" {
					t.Error("wrong Studio identity")
				}
				switch r.URL.Path {
				case "/studio/api/uploads":
					if err := r.ParseMultipartForm(1 << 20); err != nil {
						t.Error(err)
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					file, header, err := r.FormFile("file")
					if err != nil {
						t.Error(err)
						w.WriteHeader(http.StatusBadRequest)
						return
					}
					defer file.Close()
					data := make([]byte, 8)
					if _, err := file.Read(data); err != nil || base64.StdEncoding.EncodeToString(data) != "iVBORw0KGgo=" {
						t.Errorf("bad uploaded bytes: %x, %v", data, err)
					}
					writeJSON(w, http.StatusOK, map[string]string{"kind": string(config.ModelImage), "id": strings.TrimSuffix(header.Filename, ".png")})
				case "/studio/api/jobs":
					var job struct {
						Mode   string
						Assets []studioAsset
					}
					if err := json.NewDecoder(r.Body).Decode(&job); err != nil {
						t.Error(err)
					}
					if job.Mode != tc.workflow || len(job.Assets) != len(tc.want) {
						t.Errorf("unexpected Studio job: %+v", job)
					} else {
						for i, asset := range tc.want {
							if job.Assets[i] != asset {
								t.Errorf("unexpected asset %+v", job.Assets[i])
							}
						}
					}
					writeJSON(w, http.StatusOK, map[string]any{"jobs": []studioJob{{ID: "asset-job", Status: "queued"}}})
				default:
					http.NotFound(w, r)
				}
			}))
			defer studio.Close()
			console := New(Options{Config: cfg, Media: media, ComfyURL: studio.URL})
			console.backend.Store(console.probe(context.Background()))
			handler := console.Handler()
			post := func(inputs *generationInputs) *httptest.ResponseRecorder {
				t.Helper()
				body, err := json.Marshal(generationRequest{WorkflowID: tc.workflow, Prompt: "test", Inputs: inputs})
				if err != nil {
					t.Fatal(err)
				}
				request := httptest.NewRequest(http.MethodPost, "/api/v1/generations", strings.NewReader(string(body)))
				request.Header.Set("X-Model-Console-Token", token)
				request.Header.Set("X-Model-Console-User", "alice")
				rec := httptest.NewRecorder()
				handler.ServeHTTP(rec, request)
				return rec
			}
			if rec := post(nil); rec.Code != http.StatusBadRequest {
				t.Fatalf("missing image returned %d: %s", rec.Code, rec.Body)
			}
			if rec := post(tc.inputs); rec.Code != http.StatusAccepted || !strings.Contains(rec.Body.String(), "asset-job") {
				t.Fatalf("upload and submit returned %d: %s", rec.Code, rec.Body)
			}
		})
	}
}

func TestReferenceInputsRequireVisualAssetAndKeepRoles(t *testing.T) {
	const png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zt9sAAAAASUVORK5CYII="
	const video = "data:video/mp4;base64,Y2xpcA=="
	const audio = "data:audio/wav;base64,c291bmQ="
	reference := APIWorkflow{Operation: operationReference, MaxImages: 5}
	frame := APIWorkflow{Operation: operationFirstFrame, MaxImages: 2}
	for _, tc := range []struct {
		name   string
		flow   APIWorkflow
		inputs generationInputs
		roles  []string
	}{
		{"two ordered frames", frame, generationInputs{Images: []string{png, png}}, []string{"first_frame", "last_frame"}},
		{"image, clip, and audio", reference, generationInputs{Images: []string{png}, Video: video, Audio: audio}, []string{"ref_image", "ref_video", "ref_audio"}},
		{"audio alone", reference, generationInputs{Audio: audio}, nil},
		{"video in frame operation", frame, generationInputs{Images: []string{png}, Video: video}, nil},
		{"image labeled as video", reference, generationInputs{Video: png}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assets, err := prepareGenerationInputs(&tc.inputs, tc.flow)
			if tc.roles == nil {
				if err == nil {
					t.Fatal("invalid asset combination was accepted")
				}
				return
			}
			if err != nil || len(assets) != len(tc.roles) {
				t.Fatalf("assets = %v, err = %v", assets, err)
			}
			for idx, want := range tc.roles {
				if assets[idx].role != want {
					t.Errorf("asset %d role = %q, want %q", idx, assets[idx].role, want)
				}
			}
		})
	}
}
