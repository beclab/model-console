package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// handleEmbed translates /v1/embeddings to /api/embed (with a fallback to
// the legacy /api/embeddings endpoint). The two
// upstream endpoints differ in field names and response shape:
//
//	/api/embed (Ollama >=0.2):  {"model":"...","input":[...]} → {"embeddings":[[...],[...]]}
//	/api/embeddings (legacy):   {"model":"...","prompt":"..."} → {"embedding":[...]}
//
// We try /api/embed first because it is the only one that supports the
// batched array-input form OpenAI clients commonly send.
func (a *Adapter) handleEmbed(w http.ResponseWriter, r *http.Request) {
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
	inputs := normalizeEmbedInput(openaiReq["input"])
	if len(inputs) == 0 {
		writeJSONError(w, http.StatusBadRequest, "bad_request",
			"input must be a string or non-empty array of strings")
		return
	}

	embeddings, err := a.embedTry(r.Context(), inputs)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "adapter_unreachable", err.Error())
		return
	}
	resp := OpenAIEmbeddingResponse{
		Object: objectList,
		Model:  a.cfg.Model.Name,
		Data:   buildEmbedData(embeddings),
		Usage:  OpenAIUsage{PromptTokens: 0, TotalTokens: 0},
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// embedTry calls /api/embed first; on 404 / unexpected status it falls
// back to /api/embeddings sending one prompt at a time and stitching the
// results back together.
//
// "model" sent upstream is the daemon tag (upstreamModel()); the alias
// the OpenAI client sees lives in the response envelope built in
// handleEmbed.
func (a *Adapter) embedTry(ctx context.Context, inputs []string) ([][]float64, error) {
	upstream := a.upstreamModel()
	body, err := json.Marshal(map[string]any{
		keyModel: upstream,
		"input":  inputs,
	})
	if err != nil {
		return nil, err
	}
	resp, err := a.client.PostJSON(ctx, "/api/embed", body)
	if err == nil {
		defer resp.Body.Close()
		var out struct {
			Embeddings [][]float64 `json:"embeddings"`
		}
		if decErr := json.NewDecoder(resp.Body).Decode(&out); decErr != nil {
			return nil, fmt.Errorf("decode /api/embed: %w", decErr)
		}
		if len(out.Embeddings) == len(inputs) {
			return out.Embeddings, nil
		}
		// Daemon returned a non-zero count that doesn't match — fall
		// through to the legacy path which is one-prompt-at-a-time and
		// guaranteed to align.
	}

	// Fallback: legacy /api/embeddings with a single prompt per call.
	out := make([][]float64, 0, len(inputs))
	for _, in := range inputs {
		legacyBody, _ := json.Marshal(map[string]any{
			keyModel: upstream,
			"prompt": in,
		})
		resp, lerr := a.client.PostJSON(ctx, "/api/embeddings", legacyBody)
		if lerr != nil {
			return nil, fmt.Errorf("/api/embeddings fallback: %w", lerr)
		}
		var legacy struct {
			Embedding []float64 `json:"embedding"`
		}
		decErr := json.NewDecoder(resp.Body).Decode(&legacy)
		resp.Body.Close()
		if decErr != nil {
			return nil, fmt.Errorf("decode /api/embeddings: %w", decErr)
		}
		out = append(out, legacy.Embedding)
	}
	return out, nil
}

// normalizeEmbedInput coerces OpenAI's "input" (string | []string |
// nil) into a flat []string suitable for /api/embed's "input" array.
// Non-string array elements are rendered with fmt.Sprint as a best-effort
// fallback (some clients send token-id arrays which we can't translate).
func normalizeEmbedInput(v any) []string {
	switch x := v.(type) {
	case string:
		if strings.TrimSpace(x) == "" {
			return nil
		}
		return []string{x}
	case []any:
		out := make([]string, 0, len(x))
		for _, e := range x {
			if s, ok := e.(string); ok {
				out = append(out, s)
			} else {
				out = append(out, fmt.Sprint(e))
			}
		}
		return out
	default:
		return nil
	}
}

func buildEmbedData(embeddings [][]float64) []OpenAIEmbeddingDatum {
	out := make([]OpenAIEmbeddingDatum, 0, len(embeddings))
	for i, e := range embeddings {
		out = append(out, OpenAIEmbeddingDatum{
			Object:    objectEmbedding,
			Index:     i,
			Embedding: e,
		})
	}
	return out
}
