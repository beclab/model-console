package lifecycle

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

func TestParseHFResolveURL(t *testing.T) {
	t.Parallel()
	repo, rev, base, ok := parseHFResolveURL(
		"https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-IQ2_XXS.gguf",
		"https://huggingface.co",
	)
	if !ok || repo != "unsloth/Qwen3.5-9B-GGUF" || rev != "main" || base != "Qwen3.5-9B-UD-IQ2_XXS.gguf" {
		t.Fatalf("parse = (%q %q %q %v)", repo, rev, base, ok)
	}
	_, _, _, ok = parseHFResolveURL("https://example.com/a/b", "https://huggingface.co")
	if ok {
		t.Fatal("expected non-HF host to fail")
	}
}

func TestEstimateHFResolveURLList_E2Files(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
			{"type":"file","path":"Qwen3.5-9B-UD-IQ2_XXS.gguf","size":300},
			{"type":"file","path":"mmproj-BF16.gguf","size":50}
		]`))
	}))
	defer srv.Close()

	urls := []string{
		"https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-IQ2_XXS.gguf",
		"https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-BF16.gguf",
	}
	cfg := config.Config{HFEndpoint: srv.URL}
	total, complete := estimateHFResolveURLList(context.Background(), cfg, urls)
	if !complete || total != 350 {
		t.Fatalf("total=%d complete=%v, want 350 true", total, complete)
	}
}
