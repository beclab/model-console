package factory

import (
	"context"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/adapter"
	"github.com/llm-init/llm-init/internal/config"
)

func TestNew_Ollama(t *testing.T) {
	t.Parallel()
	cfg := config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: "http://ollama:11434"},
		Model:  config.Model{Name: "qwen"},
	}
	a, err := New(cfg, nil)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if a == nil {
		t.Fatal("nil adapter")
	}
	if a.Kind() != config.EngineOllama {
		t.Errorf("Kind = %v", a.Kind())
	}
	// Spot-check that adapter.Adapter is satisfied — Kind+OpenAIHandler
	// reach is enough; deeper paths are exercised in each sub-package.
	if a.OpenAIHandler(cfg) == nil {
		t.Error("OpenAIHandler nil")
	}
	// Ollama keeps its own store, so it is the one engine a model has
	// to be pushed into. Resolving to the no-op installer here would
	// leave an ollama:// source downloading nothing and serving 404.
	if any(adapter.InstallerFor(a)) != any(a) {
		t.Error("ollama adapter is not its own installer")
	}
}

func TestNew_ProxyKinds(t *testing.T) {
	t.Parallel()
	for _, k := range []config.EngineKind{
		config.EngineVLLM, config.EngineLlamaCpp, config.EngineSGLang,
		config.EngineEmbed, config.EngineClipEmbed, config.EngineOCR, config.EngineRerank,
	} {
		k := k
		t.Run(string(k), func(t *testing.T) {
			t.Parallel()
			a, err := New(config.Config{
				Engine: config.Engine{Kind: k, URL: "http://up:8000"},
				Model:  config.Model{Name: "m"},
			}, nil)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if a.Kind() != k {
				t.Errorf("Kind = %v", a.Kind())
			}
			// These engines have nothing to install: the bytes on the
			// path they were launched with are the whole story. The
			// installer still answers, so lifecycle never asks which
			// kind of adapter it has.
			inst := adapter.InstallerFor(a)
			if any(inst) == any(a) {
				t.Error("proxy adapter answered as its own installer")
			}
			if err := inst.Pull(context.Background(), "ref", nil); err != nil {
				t.Errorf("Pull: %v", err)
			}
			if err := inst.Register(context.Background(), nil, nil, nil); err != nil {
				t.Errorf("Register: %v", err)
			}
		})
	}
}

func TestNew_ProxyBadURL(t *testing.T) {
	t.Parallel()
	_, err := New(config.Config{
		Engine: config.Engine{Kind: config.EngineVLLM, URL: ":not a url"},
		Model:  config.Model{Name: "m"},
	}, nil)
	if err == nil {
		t.Error("expected error for bad URL")
	}
}

func TestNew_EmptyKindDownloadOnly(t *testing.T) {
	t.Parallel()
	a, err := New(config.Config{
		Engine: config.Engine{Kind: ""},
		Model:  config.Model{Name: "m"},
	}, nil)
	if err != nil {
		t.Fatalf("New: %v (empty kind must select the download-only adapter)", err)
	}
	if a == nil {
		t.Fatal("nil adapter for empty kind")
	}
	if a.Kind() != "" {
		t.Errorf("Kind = %q, want empty", a.Kind())
	}
	if a.AliveBeforeBoot() {
		t.Error("download-only adapter must not be alive before boot")
	}
}

func TestNew_UnknownKind(t *testing.T) {
	t.Parallel()
	_, err := New(config.Config{
		Engine: config.Engine{Kind: config.EngineKind("wat"), URL: "http://x"},
		Model:  config.Model{Name: "m"},
	}, nil)
	if err == nil {
		t.Error("expected error")
	}
	if !strings.Contains(err.Error(), "wat") {
		t.Errorf("error should name kind: %v", err)
	}
}

func TestNew_ReturnsAdapterInterface(t *testing.T) {
	t.Parallel()
	// New's return type is already adapter.Adapter; the compile-time
	// `var _ adapter.Adapter = a` assertion was redundant (and tripped
	// staticcheck QF1011). Asserting non-nil is what actually matters
	// at runtime — a future refactor that returned (nil, nil) on a
	// valid config would otherwise pass.
	a, err := New(config.Config{
		Engine: config.Engine{Kind: config.EngineOllama, URL: "http://x"},
		Model:  config.Model{Name: "m"},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a == nil {
		t.Fatal("New returned nil adapter without error")
	}
}
