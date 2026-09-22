package diag

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
)

// stubAdapter lets the diag package's pure helpers be tested without
// pulling in either of the real adapter packages (which would
// duplicate fixtures already covered by their own test files).
type stubAdapter struct {
	stats adapter.NativeStats
	err   error
	kind  config.EngineKind
}

func (s *stubAdapter) Kind() config.EngineKind         { return s.kind }
func (s *stubAdapter) WaitAlive(context.Context) error { return nil }
func (s *stubAdapter) AliveBeforeBoot() bool           { return true }
func (s *stubAdapter) Ready(context.Context) (adapter.ReadyState, error) {
	return adapter.ReadyState{Alive: true, ModelExists: true}, nil
}
func (s *stubAdapter) OpenAIHandler(config.Config) http.Handler    { return nil }
func (s *stubAdapter) AnthropicHandler(config.Config) http.Handler { return nil }
func (s *stubAdapter) EngineNativeStats(context.Context) (adapter.NativeStats, error) {
	return s.stats, s.err
}

func fixedNow() time.Time {
	return time.Date(2026, 5, 17, 12, 0, 0, 0, time.UTC)
}

// TestBuildGPUReport_FullFlow asserts every wire field is populated
// from the corresponding NativeStats / config input.
func TestBuildGPUReport_FullFlow(t *testing.T) {
	t.Parallel()
	vram := int64(5_000_000_000)
	model := int64(5_000_000_000)
	stub := &stubAdapter{
		kind: config.EngineOllama,
		stats: adapter.NativeStats{
			Source:  "ollama_ps",
			Payload: []byte(`{"models":[{"name":"q","size":5000000000,"size_vram":5000000000}]}`),
			GPU: adapter.GPUResidencyHints{
				Mode:       adapter.GPUModeFull,
				VRAMBytes:  &vram,
				ModelBytes: &model,
			},
		},
	}
	cfg := config.Config{
		Engine: config.Engine{Kind: config.EngineOllama},
		Model:  config.Model{Name: "qwen2.5:7b"},
	}
	got, err := BuildGPUReport(context.Background(), stub, cfg, fixedNow)
	if err != nil {
		t.Fatalf("BuildGPUReport: %v", err)
	}
	if got.SchemaVersion != SchemaVersion {
		t.Errorf("SchemaVersion = %d", got.SchemaVersion)
	}
	if got.EngineKind != "ollama" {
		t.Errorf("EngineKind = %q", got.EngineKind)
	}
	if got.Model.Name != "qwen2.5:7b" {
		t.Errorf("Model.Name = %q", got.Model.Name)
	}
	if got.GPU.Source != "ollama_ps" {
		t.Errorf("GPU = %+v", got.GPU)
	}
	if got.GPU.VRAMBytes == nil || got.GPU.ModelBytes == nil {
		t.Errorf("VRAMBytes/ModelBytes lost in translation: %+v", got.GPU)
	}
	if string(got.EngineNative) == "" {
		t.Errorf("EngineNative payload lost")
	}
	if !got.GeneratedAt.Equal(fixedNow().UTC()) {
		t.Errorf("GeneratedAt = %v", got.GeneratedAt)
	}
}

// TestBuildGPUReport_WarningsFlow: NativeStats.Warnings flow into
// the wire shape verbatim.
func TestBuildGPUReport_WarningsFlow(t *testing.T) {
	t.Parallel()
	stub := &stubAdapter{
		kind: config.EngineLlamaCpp,
		stats: adapter.NativeStats{
			Source: "unavailable",
			GPU:    adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown},
			Warnings: []string{
				"llamacpp: compose contract violated; ENGINE_LLAMACPP_FIT=\"on\"",
			},
		},
	}
	got, _ := BuildGPUReport(context.Background(), stub,
		config.Config{Engine: config.Engine{Kind: config.EngineLlamaCpp}}, fixedNow)
	if got.GPU.Source != "unavailable" {
		t.Errorf("Source = %q want unavailable (mode field was removed in v1.1.0)", got.GPU.Source)
	}
	if len(got.Warnings) != 1 || got.Warnings[0] == "" {
		t.Errorf("Warnings = %v", got.Warnings)
	}
}

// TestBuildGPUReport_ErrorPropagates: engine unreachable surfaces
// as an error to the caller (handler returns 503).
func TestBuildGPUReport_ErrorPropagates(t *testing.T) {
	t.Parallel()
	stub := &stubAdapter{
		kind: config.EngineSGLang,
		err:  errors.New("connection refused"),
	}
	if _, err := BuildGPUReport(context.Background(), stub,
		config.Config{Engine: config.Engine{Kind: config.EngineSGLang}}, fixedNow); err == nil {
		t.Error("expected error to propagate")
	}
}

// TestTranslateResidency_DefaultsAndPropagation locks the small
// translation function, especially the "Source empty -> unavailable"
// fallback so a mis-built NativeStats still satisfies the wire
// contract.
func TestTranslateResidency_DefaultsAndPropagation(t *testing.T) {
	t.Parallel()
	t.Run("empty_source_becomes_unavailable", func(t *testing.T) {
		t.Parallel()
		got := translateResidency(adapter.NativeStats{
			GPU: adapter.GPUResidencyHints{Mode: adapter.GPUModeUnknown},
		})
		if got.Source != "unavailable" {
			t.Errorf("Source = %q, want unavailable", got.Source)
		}
	})
	t.Run("source_propagates_verbatim", func(t *testing.T) {
		t.Parallel()
		got := translateResidency(adapter.NativeStats{Source: "x"})
		if got.Source != "x" {
			t.Errorf("Source = %q, want %q", got.Source, "x")
		}
	})
}
