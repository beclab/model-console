package config

import (
	"strings"
	"testing"
)

// Not parallel: captureSlog swaps the default logger.
func TestNormalizeSpecForEngine_WarnsOnOversubscribedKVPool(t *testing.T) {
	cases := []struct {
		name       string
		engineArgs string
		wantWarn   bool
		wantPhrase string
	}{
		{
			name:       "explicit slots sharing one pool",
			engineArgs: "-c 12288 -np 2 -kvu",
			wantWarn:   true,
			wantPhrase: "every slot may claim the whole pool",
		},
		{
			// The base application's own documented example. Nothing in it
			// asks for four slots or a shared pool, and it gets both.
			name:       "no slot count at all",
			engineArgs: "-c 8192 -ngl all -fa on",
			wantWarn:   true,
			wantPhrase: "no -np was given",
		},
		{
			name:       "a per-slot ceiling is the fix, so no warning",
			engineArgs: "-c 12288 -np 2 -kvu --kv-unified-per-slot 4096",
			wantWarn:   false,
		},
		{
			// The hole: the flag is present but still lets every slot
			// claim the whole pool.
			name:       "a per-slot ceiling equal to the pool is not a ceiling",
			engineArgs: "-c 12288 -np 2 -kvu --kv-unified-per-slot 12288",
			wantWarn:   true,
			wantPhrase: "every slot may claim the whole pool",
		},
		{
			name:       "one slot cannot oversubscribe itself",
			engineArgs: "-c 12288 -np 1 -kvu",
			wantWarn:   false,
		},
		{
			name:       "split mode gives each slot a hard share",
			engineArgs: "-c 12288 -np 2",
			wantWarn:   false,
		},
		{
			name:       "opting out of the shared pool needs an explicit slot count too",
			engineArgs: "-c 8192 -ngl all -np 2 -no-kvu",
			wantWarn:   false,
		},
		{
			// -no-kvu on its own is overruled by the auto-slot branch, so
			// the operator who wrote it still has the problem.
			name:       "opting out without pinning the slot count opts out of nothing",
			engineArgs: "-c 8192 -ngl all -no-kvu",
			wantWarn:   true,
			wantPhrase: "no -np was given",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &Config{
				Engine: Engine{Kind: EngineLlamaCpp},
				Spec:   ModelSpec{Name: "m", Mode: "chat", EngineArgs: tc.engineArgs},
			}
			logs := captureSlog(t, func() {
				if _, err := NormalizeSpecForEngine(cfg); err != nil {
					t.Fatalf("NormalizeSpecForEngine: %v", err)
				}
			})
			gotWarn := strings.Contains(logs, "oversubscribed")
			if gotWarn != tc.wantWarn {
				t.Fatalf("warned = %v, want %v; logs:\n%s", gotWarn, tc.wantWarn, logs)
			}
			if !tc.wantWarn {
				return
			}
			if !strings.Contains(logs, tc.wantPhrase) {
				t.Errorf("want the message to say %q; logs:\n%s", tc.wantPhrase, logs)
			}
			// A warning nobody can act on is noise, so the three ways out
			// travel with it.
			for _, remedy := range []string{"--kv-unified-per-slot", "-np 1", "-np <n> -no-kvu"} {
				if !strings.Contains(logs, remedy) {
					t.Errorf("want %q offered as a remedy; logs:\n%s", remedy, logs)
				}
			}
		})
	}
}

// The other three engines' schedulers degrade instead of failing, so the
// warning must not follow them.
func TestNormalizeSpecForEngine_NoKVWarningForOtherEngines(t *testing.T) {
	cases := []struct {
		kind EngineKind
		args string
	}{
		{EngineVLLM, "--max-model-len 8192 --max-num-seqs 4"},
		{EngineSGLang, "--context-length 8192 --max-running-requests 4"},
		{EngineOllama, "OLLAMA_NUM_CTX=8192 OLLAMA_NUM_PARALLEL=4"},
	}
	for _, tc := range cases {
		t.Run(string(tc.kind), func(t *testing.T) {
			cfg := &Config{
				Engine: Engine{Kind: tc.kind},
				Spec:   ModelSpec{Name: "m", Mode: "chat", EngineArgs: tc.args},
			}
			logs := captureSlog(t, func() {
				if _, err := NormalizeSpecForEngine(cfg); err != nil {
					t.Fatalf("NormalizeSpecForEngine: %v", err)
				}
			})
			if strings.Contains(logs, "oversubscribed") {
				t.Errorf("unexpected KV warning for %s; logs:\n%s", tc.kind, logs)
			}
		})
	}
}
