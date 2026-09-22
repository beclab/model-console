//go:build integration

package clipembed_integration

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llm-init/llm-init/tests/integration/shared"
)

// clipModelEnv defaults for Jina CLIP v2 split (OpenVINO) from the unified HF
// repo beclab/jina-clip-v2-split. MODEL_NAME must match ModelSpec.model_name
// for MODEL_ID (see IREmbeddingServer embed_core registry).
var clipModelEnv = map[string]string{
	"MODEL_NAME":    "jina-clip-v2",
	"MODEL_ID":      "jina-clip-v2-split-ov",
	"MODEL_MODE":    "embedding",
	"MODEL_SOURCE":  `hf://beclab/jina-clip-v2-split --revision main --exclude "onnx/**" --subdir openvino`,
	"LLM_INIT_PORT": "8080",
	"EMBED_PORT":    "8080",
	"EMBED_DEVICE":  "auto",
	"LOG_LEVEL":     "info",
	"LOG_FORMAT":    "json",
}

// clipCase is one image id with two English captions (en.json value).
type clipCase struct {
	ID       string
	Image    string
	Captions []string
}

const (
	clipMinImageTextSim = 0.15 // floor for matched image↔caption pairs
	clipMinRetrievalGap = 0.05 // own caption must beat best wrong caption by this
)

// TestClipEmbed_FullStack exercises Jina CLIP v2 split dual-tower via
// POST /v1/embeddings: text input (string) and image input (object + data URL).
//
// Gate: LLM_INIT_INTEGRATION_CLIPEMBED=1.
//
// Local invocation:
//
//	LLM_INIT_INTEGRATION_CLIPEMBED=1 \
//	    EMBED_IMAGE=embed-server:ov-intel-amd64 \
//	    go test -tags integration -timeout 45m -run TestClipEmbed \
//	    ./tests/integration/clipembed/...
func TestClipEmbed_FullStack(t *testing.T) {
	if os.Getenv("LLM_INIT_INTEGRATION_CLIPEMBED") == "" {
		t.Skip("set LLM_INIT_INTEGRATION_CLIPEMBED=1 to enable CLIP integration test")
	}
	if os.Getenv("LLM_INIT_INTEGRATION_DISABLED") != "" {
		t.Skip("LLM_INIT_INTEGRATION_DISABLED set")
	}

	for k, v := range clipModelEnv {
		if os.Getenv(k) == "" {
			t.Setenv(k, v)
		}
	}
	port := envOr("LLM_INIT_PORT", clipModelEnv["LLM_INIT_PORT"])
	shared.AssertPortFree(t, port)

	fixtureDir := shared.RepoPath(t, "tests/integration/clipembed")
	cases := loadClipCases(t, filepath.Join(fixtureDir, "en.json"), fixtureDir)

	composePath := shared.RepoPath(t, "deploy/compose/clipembed.yml")
	stack := shared.Up(t, composePath)

	ready := stack.WaitReadyWithProgressLog(t, 30*time.Minute)
	t.Logf("ready in %s", ready)

	logClipEmbedModels(t, stack.BaseURL, clipModelEnv)

	model := envOr("MODEL_NAME", clipModelEnv["MODEL_NAME"])
	t.Logf("CLIP model=%q MODEL_ID=%q cases=%d", model, envOr("MODEL_ID", clipModelEnv["MODEL_ID"]), len(cases))

	type textEmb struct {
		caption string
		vec     []float64
	}
	imageVecs := make(map[string][]float64)
	textByCase := make(map[string][]textEmb)

	for _, c := range cases {
		t.Logf("embedding image %s", c.ID)
		imgVec := shared.FetchImageEmbeddingVector(t, stack.BaseURL, model, c.Image)
		if len(imgVec) != 1024 {
			t.Logf("note: image %s dim=%d (jina-clip-v2 expects 1024)", c.ID, len(imgVec))
		}
		imageVecs[c.ID] = imgVec

		for i, cap := range c.Captions {
			t.Logf("embedding text %s caption[%d]: %q", c.ID, i, cap)
			txtVec := shared.FetchEmbeddingVector(t, stack.BaseURL, model, cap)
			textByCase[c.ID] = append(textByCase[c.ID], textEmb{caption: cap, vec: txtVec})
		}
	}

	for _, c := range cases {
		imgVec := imageVecs[c.ID]
		for _, te := range textByCase[c.ID] {
			sim := shared.CosineSimilarity(imgVec, te.vec)
			t.Logf("matched  image=%s caption=%q cosine=%.4f", c.ID, te.caption, sim)
			if sim < clipMinImageTextSim {
				t.Errorf("image %s vs own caption %q: cosine %.4f < %.2f",
					c.ID, te.caption, sim, clipMinImageTextSim)
			}

			bestWrong := -1.0
			var bestWrongCap string
			for _, other := range cases {
				if other.ID == c.ID {
					continue
				}
				for _, wrong := range textByCase[other.ID] {
					ws := shared.CosineSimilarity(imgVec, wrong.vec)
					if ws > bestWrong {
						bestWrong = ws
						bestWrongCap = wrong.caption
					}
				}
			}
			t.Logf("retrieval image=%s caption=%q own=%.4f best_wrong=%.4f (from %q) gap=%.4f",
				c.ID, te.caption, sim, bestWrong, bestWrongCap, sim-bestWrong)
			if sim < bestWrong+clipMinRetrievalGap {
				t.Errorf("image %s caption %q: own sim %.4f should exceed best wrong %.4f by >= %.2f",
					c.ID, te.caption, sim, bestWrong, clipMinRetrievalGap)
			}
		}
	}

	shared.RecordBaseline(t, "clipembed", ready, 0)
}

func loadClipCases(t *testing.T, enJSON, fixtureDir string) []clipCase {
	t.Helper()
	raw, err := os.ReadFile(enJSON)
	if err != nil {
		t.Fatalf("read %s: %v", enJSON, err)
	}
	var m map[string][]string
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("parse %s: %v", enJSON, err)
	}
	out := make([]clipCase, 0, len(m))
	for id, captions := range m {
		if len(captions) == 0 {
			t.Fatalf("case %s: no captions", id)
		}
		img := filepath.Join(fixtureDir, id+".png")
		if _, err := os.Stat(img); err != nil {
			t.Fatalf("image for case %s: %v", id, err)
		}
		out = append(out, clipCase{ID: id, Image: img, Captions: captions})
	}
	if len(out) == 0 {
		t.Fatal("en.json produced no cases")
	}
	return out
}
