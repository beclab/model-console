package enginecapacity

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/llm-init/llm-init/internal/config"
)

// blockOf renders a Capacity the way a card reloaded from disk carries it:
// a generic map with float64 numbers, not the struct that was written.
func blockOf(t *testing.T, c Capacity) map[string]any {
	t.Helper()
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var block map[string]any
	if err := json.Unmarshal(data, &block); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return block
}

func TestFromExtensions_ReadsWhatWasWritten(t *testing.T) {
	want := Capacity{
		ContextSize:    12288,
		MaxConcurrency: 4,
		PoolTokens:     12288,
		Source:         SourceLlamacppProps,
		ReportedAt:     time.Now().UTC().Truncate(time.Second),
	}
	got, ok := FromExtensions(map[string]any{ExtensionKey: blockOf(t, want)})
	if !ok {
		t.Fatal("FromExtensions returned no reading for a block it wrote itself")
	}
	if got.PoolTokens != want.PoolTokens || got.ContextSize != want.ContextSize ||
		got.MaxConcurrency != want.MaxConcurrency || got.Source != want.Source {
		t.Errorf("FromExtensions = %+v, want %+v", got, want)
	}
	if !got.ReportedAt.Equal(want.ReportedAt) {
		t.Errorf("ReportedAt = %v, want %v", got.ReportedAt, want.ReportedAt)
	}
}

// TestFromExtensions_RejectsAnUnattributedBlock: these numbers get enforced
// against live traffic, and a block with no source and no timestamp cannot be
// told apart from somebody's guess.
func TestFromExtensions_RejectsAnUnattributedBlock(t *testing.T) {
	for name, block := range map[string]any{
		"no source":     map[string]any{"pool_tokens": 12288, "reported_at": time.Now()},
		"no timestamp":  map[string]any{"pool_tokens": 12288, "source": "llamacpp_props"},
		"not an object": "12288",
		"empty":         map[string]any{},
	} {
		t.Run(name, func(t *testing.T) {
			if _, ok := FromExtensions(map[string]any{ExtensionKey: block}); ok {
				t.Error("FromExtensions accepted a block that is not a reading")
			}
		})
	}
}

func TestFromExtensions_AbsentIsAbsent(t *testing.T) {
	for name, ext := range map[string]map[string]any{
		"nil map":    nil,
		"empty map":  {},
		"other keys": {"translate": map[string]any{}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, ok := FromExtensions(ext); ok {
				t.Error("FromExtensions found a reading where there is none")
			}
		})
	}
}

// TestPoolTokensOf covers the preference the probe exists for: what the engine
// allocated beats what the flags asked for, and the flags are still there when
// nothing answered.
func TestPoolTokensOf(t *testing.T) {
	args, err := config.ParseEngineArgs(config.EngineLlamaCpp, "-c 12288")
	if err != nil {
		t.Fatalf("ParseEngineArgs: %v", err)
	}

	base := config.Config{}
	base.Engine.Kind = config.EngineLlamaCpp
	base.Engine.Args = args

	t.Run("declaration when nothing was reported", func(t *testing.T) {
		got, ok := PoolTokensOf(base)
		if !ok || got != 12288 {
			t.Errorf("PoolTokensOf = (%d, %v), want (12288, true)", got, ok)
		}
	})

	t.Run("a reading wins", func(t *testing.T) {
		cfg := base
		cfg.Spec.Extensions = map[string]any{
			ExtensionKey: blockOf(t, Capacity{
				PoolTokens: 8192,
				Source:     SourceLlamacppProps,
				ReportedAt: time.Now(),
			}),
		}
		got, ok := PoolTokensOf(cfg)
		if !ok || got != 8192 {
			t.Errorf("PoolTokensOf = (%d, %v), want the reported 8192", got, ok)
		}
	})

	t.Run("a reading without a pool total falls back", func(t *testing.T) {
		// llama.cpp's /props is exactly this case: it publishes the window
		// and the slot count and no pool total, so -c is left to say what
		// the pool is.
		cfg := base
		cfg.Spec.Extensions = map[string]any{
			ExtensionKey: blockOf(t, Capacity{
				ContextSize:    12288,
				MaxConcurrency: 4,
				Source:         SourceLlamacppProps,
				ReportedAt:     time.Now(),
			}),
		}
		got, ok := PoolTokensOf(cfg)
		if !ok || got != 12288 {
			t.Errorf("PoolTokensOf = (%d, %v), want the declared 12288", got, ok)
		}
	})

	t.Run("nothing knows", func(t *testing.T) {
		// vLLM sizes its pool by a profiling run, so no flag states it and
		// a gate must not be given a number to enforce.
		cfg := config.Config{}
		cfg.Engine.Kind = config.EngineVLLM
		if got, ok := PoolTokensOf(cfg); ok {
			t.Errorf("PoolTokensOf = (%d, true), want no answer", got)
		}
	})
}
