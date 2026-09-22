//go:build integration

// Package clipembed_integration exercises deploy/compose/clipembed.yml against
// a real llm-init + IREmbeddingServer pair for CLIP dual-tower models.
// Env-gated: skip unless LLM_INIT_INTEGRATION_CLIPEMBED=1.
//
// Local invocation:
//
//	LLM_INIT_INTEGRATION_CLIPEMBED=1 \
//	    EMBED_IMAGE=embed-server:ov-intel-amd64 \
//	    go test -tags integration -timeout 45m ./tests/integration/clipembed/...
package clipembed_integration

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// logClipEmbedModels dumps GET /v1/models so failures like model_not_found
// show what the upstream actually advertises.
func logClipEmbedModels(t *testing.T, baseURL string, modelEnv map[string]string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/models", nil)
	if err != nil {
		t.Logf("GET /v1/models: build request: %v", err)
		return
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Logf("GET /v1/models: request failed: %v", err)
		return
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Logf("GET /v1/models status=%d: read body: %v", resp.StatusCode, err)
		return
	}

	var pretty json.RawMessage
	if err := json.Unmarshal(raw, &pretty); err != nil {
		t.Logf("GET /v1/models status=%d body=%s", resp.StatusCode, raw)
		return
	}
	formatted, err := json.MarshalIndent(pretty, "", "  ")
	if err != nil {
		t.Logf("GET /v1/models status=%d body=%s", resp.StatusCode, raw)
		return
	}
	t.Logf("GET /v1/models status=%d:\n%s", resp.StatusCode, formatted)

	modelName := envOr("MODEL_NAME", modelEnv["MODEL_NAME"])
	modelID := envOr("MODEL_ID", modelEnv["MODEL_ID"])
	req2, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("%s/v1/models/%s", baseURL, modelName), nil)
	if err == nil {
		if resp2, err := http.DefaultClient.Do(req2); err == nil {
			body2, _ := io.ReadAll(resp2.Body)
			resp2.Body.Close()
			t.Logf("GET /v1/models/%s status=%d body=%s", modelName, resp2.StatusCode, body2)
		}
	}
	req3, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("%s/v1/models/%s", baseURL, modelID), nil)
	if err == nil {
		if resp3, err := http.DefaultClient.Do(req3); err == nil {
			body3, _ := io.ReadAll(resp3.Body)
			resp3.Body.Close()
			t.Logf("GET /v1/models/%s status=%d body=%s", modelID, resp3.StatusCode, body3)
		}
	}
}
