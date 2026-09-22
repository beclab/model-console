package ollama

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

// handleCompletion is the /v1/completions translator. It maps onto Ollama
// /api/generate, which is the legacy single-prompt completion endpoint
// (no chat templating). Field map:
//
//	prompt          → prompt
//	max_tokens      → options.num_predict
//	temperature     → options.temperature
//	top_p           → options.top_p
//	stop            → options.stop (string|[]string normalised)
//	stream          → stream (boolean)
//
// Tools and chat-shaped messages have no analogue here; if the caller
// wants chat semantics they should use /v1/chat/completions.
func (a *Adapter) handleCompletion(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, "bad_request", "read body: "+err.Error())
		return
	}
	defer r.Body.Close()

	var openaiReq map[string]any
	if err := json.Unmarshal(body, &openaiReq); err != nil {
		writeJSONError(w, http.StatusBadRequest, "bad_request", "invalid JSON: "+err.Error())
		return
	}

	stream, _ := openaiReq[keyStream].(bool)
	upstream := translateCompletionRequest(openaiReq, a.cfg)

	reqBody, err := json.Marshal(upstream)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "internal_error",
			"marshal upstream: "+err.Error())
		return
	}

	resp, err := a.client.PostJSON(r.Context(), "/api/generate", reqBody)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "adapter_unreachable", err.Error())
		return
	}
	defer resp.Body.Close()

	if stream {
		streamCompletionResponse(w, resp.Body, a.cfg.Model.Name)
		return
	}
	collectCompletionResponse(w, resp.Body, a.cfg.Model.Name)
}

func translateCompletionRequest(openai map[string]any, cfg config.Config) map[string]any {
	out := map[string]any{
		keyModel:  cfg.PrimaryOllamaTag(),
		keyStream: false,
	}
	if v, ok := openai[keyStream].(bool); ok {
		out[keyStream] = v
	}
	if v, ok := openai["prompt"].(string); ok {
		out["prompt"] = v
	}

	options := map[string]any{}
	mapNumberOption(options, openai, "temperature", "temperature")
	mapNumberOption(options, openai, "top_p", "top_p")
	mapNumberOption(options, openai, "max_tokens", "num_predict")
	mapNumberOption(options, openai, "frequency_penalty", "frequency_penalty")
	mapNumberOption(options, openai, "presence_penalty", "presence_penalty")
	mapNumberOption(options, openai, "seed", "seed")
	if stop, ok := openai["stop"]; ok {
		options["stop"] = normalizeStop(stop)
	}
	applyInferenceInjection(out, options, cfg)
	if len(options) > 0 {
		out["options"] = options
	}
	return out
}

// ollamaGenerateChunk is the subset of /api/generate frame we use.
type ollamaGenerateChunk struct {
	Response        string `json:"response"`
	Done            bool   `json:"done"`
	PromptEvalCount int64  `json:"prompt_eval_count"`
	EvalCount       int64  `json:"eval_count"`
}

func streamCompletionResponse(w http.ResponseWriter, body io.Reader, modelName string) {
	flusher, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	if flusher != nil {
		flusher.Flush()
	}
	id := fmt.Sprintf("cmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 4<<20)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var fr ollamaGenerateChunk
		if err := json.Unmarshal(line, &fr); err != nil {
			continue
		}
		// Streaming frames send the per-chunk text; the terminal frame
		// blanks Text and sets finish_reason. Pre-typed-struct refactor
		// this branched on fr.Done to mutate a map literal.
		choice := OpenAICompletionChoice{Index: 0, Text: fr.Response}
		if fr.Done {
			choice.Text = ""
			choice.FinishReason = finishReasonStop
		}
		chunk := OpenAICompletionResponse{
			ID:      id,
			Object:  objectTextCompletion,
			Created: created,
			Model:   modelName,
			Choices: []OpenAICompletionChoice{choice},
		}
		writeSSEChunk(w, chunk)
		if flusher != nil {
			flusher.Flush()
		}
		if fr.Done {
			break
		}
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

func collectCompletionResponse(w http.ResponseWriter, body io.Reader, modelName string) {
	data, err := io.ReadAll(io.LimitReader(body, 8<<20))
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "adapter_5xx",
			"read upstream: "+err.Error())
		return
	}
	var fr ollamaGenerateChunk
	if err := json.Unmarshal(data, &fr); err != nil {
		// fallback: pick last NDJSON line
		fr = lastGenerateFrame(data)
	}
	resp := OpenAICompletionResponse{
		ID:      fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
		Object:  objectTextCompletion,
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []OpenAICompletionChoice{
			{Index: 0, Text: fr.Response, FinishReason: finishReasonStop},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     fr.PromptEvalCount,
			CompletionTokens: fr.EvalCount,
			TotalTokens:      fr.PromptEvalCount + fr.EvalCount,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func lastGenerateFrame(data []byte) ollamaGenerateChunk {
	var out ollamaGenerateChunk
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var fr ollamaGenerateChunk
		if err := json.Unmarshal([]byte(line), &fr); err == nil {
			out = fr
		}
	}
	return out
}
