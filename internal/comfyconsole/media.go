package comfyconsole

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"
	"strings"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/progress"
)

const (
	mediaContract       = "studio-v1"
	generationQueued    = "queued"
	generationCompleted = "completed"
	generationFailed    = "failed"
)

var (
	modelNamePattern  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
	studioUserPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9_.-]{0,63}$`)
	studioIDPattern   = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,128}$`)
)

// APIWorkflow describes the text-to-image workflow exposed by this app.
type APIWorkflow struct {
	ID            string   `json:"id"`
	AspectRatios  []string `json:"aspectRatios"`
	DefaultAspect string   `json:"defaultAspect"`
	Resolution    string   `json:"resolution"`
}

// MediaConfig binds an installed image model to its Studio workflow.
type MediaConfig struct {
	name     string
	token    string
	workflow APIWorkflow
}

// NewMediaConfig validates the settings used to publish a model to Router.
func NewMediaConfig(cfg *AppConfig, name, mode, token string) (*MediaConfig, error) {
	if !modelNamePattern.MatchString(name) {
		return nil, errors.New("MODEL_NAME must be a stable model identifier")
	}
	if mode != string(config.ModelImage) {
		return nil, errors.New("MODEL_MODE must be image for the text-to-image API")
	}
	if len(token) < 32 {
		return nil, errors.New("COMFY_API_TOKEN must be at least 32 characters")
	}
	if len(cfg.APIWorkflows) != 1 {
		return nil, errors.New("app config must declare one apiWorkflows entry")
	}
	flow := cfg.APIWorkflows[0]
	if !groupIDRe.MatchString(flow.ID) || flow.Resolution == "" || flow.DefaultAspect == "" || !slices.Contains(flow.AspectRatios, flow.DefaultAspect) {
		return nil, errors.New("apiWorkflows entry requires a valid ID, resolution, and supported default aspect ratio")
	}
	return &MediaConfig{name: name, token: token, workflow: flow}, nil
}

func (c *Console) registerMedia(mux *http.ServeMux) {
	media := c.opts.Media
	mux.HandleFunc("GET /api/model-spec", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, config.ModelSpec{
			Name: media.name,
			Mode: "image_generation",
			Extensions: map[string]any{
				"creative": map[string]any{
					"media": "image", "operations": []string{"generate"},
					"canonical_fields": []string{"prompt", "n", "seed", "output.size", "output.aspect_ratio", "output.resolution"},
				},
				"model_console_media": map[string]string{"protocol": mediaContract, "workflow_id": media.workflow.ID},
			},
		})
	})
	mux.HandleFunc("GET /api/progress", c.mediaProgress)
	mux.HandleFunc("POST /api/v1/generations", c.requireMediaUser(c.createGeneration))
	mux.HandleFunc("GET /api/v1/generations/{id}", c.requireMediaUser(c.getGeneration))
	mux.HandleFunc("GET /api/v1/generations/{id}/content", c.requireMediaUser(c.getGenerationContent))
}

func (c *Console) mediaProgress(w http.ResponseWriter, _ *http.Request) {
	st := c.Status()
	phase := progress.PhaseDownload
	switch {
	case st.Ready:
		phase = progress.PhaseReady
	case st.Models.State == StateError:
		phase = progress.PhaseFailed
	case st.Models.State == StateDone:
		phase = progress.PhaseLoading
	}
	writeJSON(w, http.StatusOK, struct {
		Phase progress.Phase `json:"phase"`
		Ready bool           `json:"ready"`
	}{Phase: phase, Ready: st.Ready})
}

func (c *Console) requireMediaUser(handler func(http.ResponseWriter, *http.Request, string)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		user := r.Header.Get("X-Model-Console-User")
		if subtle.ConstantTimeCompare([]byte(r.Header.Get("X-Model-Console-Token")), []byte(c.opts.Media.token)) != 1 || !studioUserPattern.MatchString(user) {
			mediaError(w, http.StatusUnauthorized, "unauthorized", "A Router credential and a valid user are required")
			return
		}
		handler(w, r, user)
	}
}

type generationRequest struct {
	WorkflowID string `json:"workflowId"`
	Prompt     string `json:"prompt"`
	Params     struct {
		AspectRatio string  `json:"aspectRatio"`
		Resolution  string  `json:"resolution"`
		Seed        *uint64 `json:"seed"`
	} `json:"params"`
}

type generation struct {
	ID           string             `json:"id"`
	Status       string             `json:"status"`
	Outputs      []generationOutput `json:"outputs,omitempty"`
	ErrorMessage string             `json:"errorMessage,omitempty"`
}

type generationOutput struct {
	ID string `json:"id"`
}

type studioJob struct {
	ID      string   `json:"id"`
	Status  string   `json:"status"`
	Outputs []string `json:"outputs"`
}

func (c *Console) createGeneration(w http.ResponseWriter, r *http.Request, user string) {
	var in generationRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 128<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&in); err != nil || decoder.Decode(new(any)) != io.EOF {
		mediaError(w, http.StatusBadRequest, "invalid_request", "Invalid generation request")
		return
	}
	flow := c.opts.Media.workflow
	if in.WorkflowID != flow.ID || strings.TrimSpace(in.Prompt) == "" {
		mediaError(w, http.StatusBadRequest, "invalid_workflow", "A published workflow and a prompt are required")
		return
	}
	aspect := in.Params.AspectRatio
	if aspect == "" {
		aspect = flow.DefaultAspect
	}
	if !slices.Contains(flow.AspectRatios, aspect) || (in.Params.Resolution != "" && in.Params.Resolution != flow.Resolution) {
		mediaError(w, http.StatusBadRequest, "invalid_params", "The requested aspect ratio or resolution is unsupported")
		return
	}
	if !c.Status().Ready {
		mediaError(w, http.StatusServiceUnavailable, "model_not_ready", "Model weights or Studio are still starting")
		return
	}
	params := map[string]any{"aspect": aspect, "resolution": flow.Resolution}
	if in.Params.Seed != nil {
		params["seed"] = *in.Params.Seed
	}
	body, err := json.Marshal(map[string]any{"mode": flow.ID, "prompt": in.Prompt, "params": params})
	if err != nil {
		mediaError(w, http.StatusInternalServerError, "generation_failed", "Could not encode Studio request")
		return
	}
	var result struct {
		Jobs []studioJob `json:"jobs"`
	}
	if !c.studioJSON(w, r, user, http.MethodPost, "/jobs", body, &result) {
		return
	}
	if len(result.Jobs) != 1 {
		mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Studio did not return one job")
		return
	}
	if gen, ok := result.Jobs[0].generation(w); ok {
		writeJSON(w, http.StatusAccepted, gen)
	}
}

func (c *Console) getGeneration(w http.ResponseWriter, r *http.Request, user string) {
	if gen, ok := c.readGeneration(w, r, user); ok {
		writeJSON(w, http.StatusOK, gen)
	}
}

func (c *Console) getGenerationContent(w http.ResponseWriter, r *http.Request, user string) {
	gen, ok := c.readGeneration(w, r, user)
	if !ok {
		return
	}
	id := r.URL.Query().Get("outputId")
	if !slices.Contains(gen.Outputs, generationOutput{ID: id}) {
		mediaError(w, http.StatusNotFound, "not_found", "No matching generation output")
		return
	}
	resp := c.studioCall(w, r, user, http.MethodGet, "/media/"+id, nil)
	if resp == nil {
		return
	}
	defer resp.Body.Close()
	for _, key := range []string{"Content-Type", "Content-Length", "Content-Disposition"} {
		if value := resp.Header.Get(key); value != "" {
			w.Header().Set(key, value)
		}
	}
	w.Header().Set("Cache-Control", "private, no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_, _ = io.Copy(w, resp.Body)
}

func (c *Console) readGeneration(w http.ResponseWriter, r *http.Request, user string) (generation, bool) {
	var job studioJob
	id := r.PathValue("id")
	if !studioIDPattern.MatchString(id) {
		mediaError(w, http.StatusNotFound, "not_found", "No matching generation")
		return generation{}, false
	}
	if !c.studioJSON(w, r, user, http.MethodGet, "/jobs/"+id, nil, &job) {
		return generation{}, false
	}
	if job.ID != id {
		mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Studio returned a different job ID")
		return generation{}, false
	}
	return job.generation(w)
}

func (job studioJob) generation(w http.ResponseWriter) (generation, bool) {
	if !studioIDPattern.MatchString(job.ID) {
		mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Studio returned an invalid job ID")
		return generation{}, false
	}
	gen := generation{ID: job.ID}
	switch job.Status {
	case generationQueued:
		gen.Status = generationQueued
	case "running":
		gen.Status = "in_progress"
	case "done":
		gen.Status = generationCompleted
		for _, id := range job.Outputs {
			if !studioIDPattern.MatchString(id) {
				mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Studio returned an invalid output ID")
				return generation{}, false
			}
			gen.Outputs = append(gen.Outputs, generationOutput{ID: id})
		}
	case "cancelled":
		gen.Status = "canceled"
	case generationFailed, "interrupted":
		gen.Status = generationFailed
		gen.ErrorMessage = "Studio generation failed"
	default:
		mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Studio returned an unknown job status")
		return generation{}, false
	}
	return gen, true
}

func (c *Console) studioJSON(w http.ResponseWriter, r *http.Request, user, method, path string, body []byte, result any) bool {
	resp := c.studioCall(w, r, user, method, path, body)
	if resp == nil {
		return false
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(result); err != nil {
		mediaError(w, http.StatusBadGateway, "invalid_studio_response", "Could not read Studio response")
		return false
	}
	return true
}

func (c *Console) studioCall(w http.ResponseWriter, r *http.Request, user, method, path string, body []byte) *http.Response {
	req, err := http.NewRequestWithContext(r.Context(), method, c.opts.ComfyURL+"/studio/api"+path, bytes.NewReader(body))
	if err != nil {
		mediaError(w, http.StatusBadGateway, "studio_unavailable", "Could not prepare Studio request")
		return nil
	}
	req.Header.Set("X-Bfl-User", user)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.opts.StudioClient.Do(req)
	if err != nil {
		mediaError(w, http.StatusBadGateway, "studio_unavailable", "Could not reach Studio")
		return nil
	}
	if resp.StatusCode == http.StatusOK {
		return resp
	}
	resp.Body.Close()
	status := resp.StatusCode
	if status < 400 || status >= 500 {
		status = http.StatusBadGateway
	}
	mediaError(w, status, "studio_rejected", fmt.Sprintf("Studio returned HTTP %d", resp.StatusCode))
	return nil
}

func mediaError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, map[string]any{"error": map[string]string{"code": code, "message": message}})
}
