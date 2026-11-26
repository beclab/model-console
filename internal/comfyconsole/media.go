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
	operationGenerate   = "generate"
	operationEdit       = "edit"
	operationFirstFrame = "first_last_frame"
	operationReference  = "reference"
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

// APIWorkflow describes the text-driven workflow exposed by this app.
type APIWorkflow struct {
	MaxImages     int            `json:"maxImages,omitempty"`
	ID            string         `json:"id"`
	Operation     string         `json:"operation,omitempty"`
	AspectRatios  []string       `json:"aspectRatios"`
	DefaultAspect string         `json:"defaultAspect"`
	Resolution    string         `json:"resolution"`
	Resolutions   []string       `json:"resolutions,omitempty"`
	Duration      *videoDuration `json:"durationSeconds,omitempty"`
}

type videoDuration struct {
	Default int `json:"default"`
	Min     int `json:"min"`
	Max     int `json:"max"`
}

// MediaConfig binds an installed model to its Studio workflow.
type MediaConfig struct {
	name       string
	mode       config.ModelType
	token      string
	workflows  map[string]APIWorkflow
	operations []string
}

// NewMediaConfig validates the settings used to publish a model to Router.
func NewMediaConfig(cfg *AppConfig, name, mode, token string) (*MediaConfig, error) {
	if !modelNamePattern.MatchString(name) {
		return nil, errors.New("MODEL_NAME must be a stable model identifier")
	}
	mediaMode := config.ModelType(mode)
	if mediaMode != config.ModelImage && mediaMode != config.ModelVideo {
		return nil, errors.New("MODEL_MODE must be image or video for the Studio API")
	}
	if len(token) < 32 {
		return nil, errors.New("COMFY_API_TOKEN must be at least 32 characters")
	}
	media := &MediaConfig{name: name, mode: mediaMode, token: token, workflows: make(map[string]APIWorkflow)}
	seenIDs := make(map[string]bool)
	for _, flow := range cfg.APIWorkflows {
		if flow.Operation == "" {
			flow.Operation = operationGenerate
		}
		if err := flow.validate(mediaMode); err != nil {
			return nil, err
		}
		if _, exists := media.workflows[flow.Operation]; exists || seenIDs[flow.ID] {
			return nil, errors.New("apiWorkflows must have distinct workflow IDs and operations")
		}
		media.workflows[flow.Operation] = flow
		media.operations = append(media.operations, flow.Operation)
		seenIDs[flow.ID] = true
	}
	if _, exists := media.workflows[operationGenerate]; !exists {
		return nil, errors.New("apiWorkflows must publish a generate workflow")
	}
	return media, nil
}

func (flow APIWorkflow) validate(mode config.ModelType) error {
	switch flow.Operation {
	case operationGenerate, operationReference:
		if flow.Operation == operationReference && mode != config.ModelVideo {
			return errors.New("the reference workflow requires a video model")
		}
		if !slices.Contains(flow.AspectRatios, flow.DefaultAspect) || flow.DefaultAspect == "" {
			return errors.New("the workflow requires a supported default aspect ratio")
		}
	case operationEdit:
		if mode != config.ModelImage {
			return errors.New("the edit workflow requires an image model")
		}
	case operationFirstFrame:
		if mode != config.ModelVideo {
			return errors.New("the frame workflow requires a video model")
		}
	default:
		return fmt.Errorf("unsupported Studio workflow operation %q", flow.Operation)
	}
	if flow.MaxImages < 0 || flow.MaxImages > 10 ||
		(flow.Operation == operationFirstFrame && flow.MaxImages > 2) ||
		(flow.Operation == operationGenerate && flow.MaxImages != 0) {
		return errors.New("invalid maxImages for the Studio workflow")
	}
	if !groupIDRe.MatchString(flow.ID) || flow.Resolution == "" {
		return errors.New("apiWorkflows entry requires a valid ID and resolution")
	}
	if len(flow.Resolutions) != 0 && !slices.Contains(flow.Resolutions, flow.Resolution) {
		return errors.New("the default resolution must be one of the declared resolutions")
	}
	if mode == config.ModelVideo {
		duration := flow.Duration
		if duration == nil || duration.Min <= 0 || duration.Default < duration.Min || duration.Default > duration.Max {
			return errors.New("the video workflow requires valid durationSeconds limits and default")
		}
	}
	return nil
}

func (c *Console) registerMedia(mux *http.ServeMux) {
	media := c.opts.Media
	fields := []string{"prompt", "seed", "output.aspect_ratio", "output.resolution"}
	if media.mode == config.ModelVideo {
		fields = append(fields, "output.duration_seconds")
	} else {
		fields = append(fields, "n", "output.size")
	}
	workflows := make(map[string]string, len(media.workflows))
	maxImages := 0
	for operation, flow := range media.workflows {
		workflows[operation] = flow.ID
		if operation != operationGenerate {
			maxImages = max(maxImages, flow.imageLimit())
		}
	}
	if len(workflows) > 1 {
		fields = append(fields, "inputs.images")
	}
	if _, exists := workflows[operationEdit]; exists {
		fields = append(fields, "inputs.mask")
	}
	if _, exists := workflows[operationReference]; exists {
		fields = append(fields, "inputs.video", "inputs.audio")
	}
	mux.HandleFunc("GET /api/model-spec", func(w http.ResponseWriter, _ *http.Request) {
		creative := map[string]any{"media": string(media.mode), "operations": media.operations, "canonical_fields": fields}
		if maxImages > 0 {
			creative["inputs"] = map[string]any{"reference_images": map[string]int{"max": maxImages}}
		}
		writeJSON(w, http.StatusOK, config.ModelSpec{
			Name: media.name,
			Mode: string(media.mode) + "_generation",
			Extensions: map[string]any{
				"creative": creative,
				"model_console_media": map[string]any{
					"protocol": mediaContract, "workflow_id": workflows[operationGenerate], "workflows": workflows,
				},
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
	WorkflowID string            `json:"workflowId"`
	Prompt     string            `json:"prompt"`
	Inputs     *generationInputs `json:"inputs,omitempty"`
	Params     struct {
		AspectRatio string  `json:"aspectRatio"`
		Resolution  string  `json:"resolution"`
		Seed        *uint64 `json:"seed"`
		Duration    *int    `json:"durationSeconds"`
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
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, studioMediaBodyLimit))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&in)
	if err == nil {
		if trailingErr := decoder.Decode(new(any)); trailingErr != io.EOF {
			err = trailingErr
			if err == nil {
				err = errors.New("multiple JSON values")
			}
		}
	}
	if err != nil {
		var oversized *http.MaxBytesError
		if errors.As(err, &oversized) {
			mediaError(w, http.StatusRequestEntityTooLarge, "payload_too_large", "Generation request exceeds 28 MiB")
			return
		}
		mediaError(w, http.StatusBadRequest, "invalid_request", "Invalid generation request")
		return
	}
	media := c.opts.Media
	var flow APIWorkflow
	for _, candidate := range media.workflows {
		if candidate.ID == in.WorkflowID {
			flow = candidate
			break
		}
	}
	if flow.ID == "" || strings.TrimSpace(in.Prompt) == "" {
		mediaError(w, http.StatusBadRequest, "invalid_workflow", "A published workflow and a prompt are required")
		return
	}
	params, err := flow.generationParams(in, media.mode)
	if err != nil {
		mediaError(w, http.StatusBadRequest, "invalid_params", err.Error())
		return
	}
	if !c.Status().Ready {
		mediaError(w, http.StatusServiceUnavailable, "model_not_ready", "Model weights or Studio are still starting")
		return
	}
	inputs, err := prepareGenerationInputs(in.Inputs, flow)
	if err != nil {
		mediaError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	job := map[string]any{"mode": flow.ID, "prompt": in.Prompt, "params": params}
	if len(inputs) != 0 {
		assets, ok := c.uploadGenerationInputs(w, r, user, inputs)
		if !ok {
			return
		}
		job["assets"] = assets
	}
	body, err := json.Marshal(job)
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

func (flow APIWorkflow) generationParams(in generationRequest, mode config.ModelType) (map[string]any, error) {
	aspect := in.Params.AspectRatio
	if aspect == "" {
		aspect = flow.DefaultAspect
	}
	resolution := in.Params.Resolution
	if resolution == "" {
		resolution = flow.Resolution
	}
	resolutionOK := resolution == flow.Resolution
	if len(flow.Resolutions) != 0 {
		resolutionOK = slices.Contains(flow.Resolutions, resolution)
	}
	if (aspect != "" && !slices.Contains(flow.AspectRatios, aspect)) || !resolutionOK {
		return nil, errors.New("the requested aspect ratio or resolution is unsupported")
	}
	params := map[string]any{"resolution": resolution}
	if aspect != "" {
		params["aspect"] = aspect
	}
	if mode == config.ModelVideo {
		duration := flow.Duration.Default
		if in.Params.Duration != nil {
			duration = *in.Params.Duration
		}
		if duration < flow.Duration.Min || duration > flow.Duration.Max {
			return nil, errors.New("the requested duration is unsupported")
		}
		params["duration"] = duration
	} else if in.Params.Duration != nil {
		return nil, errors.New("duration is only supported for video")
	}
	if in.Params.Seed != nil {
		params["seed"] = *in.Params.Seed
	}
	return params, nil
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
	return c.studioRequest(w, r, user, method, path, "application/json", bytes.NewReader(body))
}

func (c *Console) studioRequest(w http.ResponseWriter, r *http.Request, user, method, path, contentType string, body io.Reader) *http.Response {
	req, err := http.NewRequestWithContext(r.Context(), method, c.opts.ComfyURL+"/studio/api"+path, body)
	if err != nil {
		mediaError(w, http.StatusBadGateway, "studio_unavailable", "Could not prepare Studio request")
		return nil
	}
	req.Header.Set("X-Bfl-User", user)
	req.Header.Set("Content-Type", contentType)
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
