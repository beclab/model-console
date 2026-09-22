package dataplane

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
)

func mustArgs(t *testing.T, kind config.EngineKind, raw string) config.EngineArgs {
	t.Helper()
	args, err := config.ParseEngineArgs(kind, raw)
	if err != nil {
		t.Fatalf("ParseEngineArgs(%q): %v", raw, err)
	}
	return args
}

func snapshotOf(t *testing.T, kind config.EngineKind, raw string) func() config.Config {
	t.Helper()
	cfg := config.Config{}
	cfg.Engine.Kind = kind
	cfg.Engine.Args = mustArgs(t, kind, raw)
	cfg.Engine.URL = "http://localhost:8080"
	return func() config.Config { return cfg }
}

// TestKVBudgetForMountsOnEngineKindAlone pins what the mount decision is
// allowed to depend on.
//
// The llama.cpp row with a single slot is the regression. The flags used to
// decide this too, so a deployment that booted with -np 1 got no gate -- and
// when an operator removed -np, which turns the unified pool on across four
// slots, the engine was relaunched and the gate was still absent for the rest
// of the process's life. Sixteen overlapping prompts then failed with the
// engine's 500 rather than this package's 503.
func TestKVBudgetForMountsOnEngineKindAlone(t *testing.T) {
	tests := []struct {
		name string
		kind config.EngineKind
		args string
		want bool
	}{
		{
			name: "llamacpp wanting accounting at boot",
			kind: config.EngineLlamaCpp,
			args: "-c 12288",
			want: true,
		},
		{
			name: "llamacpp not wanting accounting at boot",
			kind: config.EngineLlamaCpp,
			args: "-c 12288 -np 1 -kvu",
			want: true,
		},
		{
			// vLLM and SGLang share a cache too, and both refuse cleanly
			// when it is full rather than dropping their neighbours.
			name: "vllm",
			kind: config.EngineVLLM,
			args: "--max-model-len 12288 --max-num-seqs 4",
			want: false,
		},
		{
			name: "sglang",
			kind: config.EngineSGLang,
			args: "--context-length 12288",
			want: false,
		},
		{
			name: "ollama",
			kind: config.EngineOllama,
			args: "",
			want: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := KVBudgetFor(KVBudgetOptions{
				Snapshot:   snapshotOf(t, tc.kind, tc.args),
				PoolTokens: func() (int, bool) { return 12288, true },
			})
			if (got != nil) != tc.want {
				t.Errorf("KVBudgetFor(%s %q) mounted = %v, want %v",
					tc.kind, tc.args, got != nil, tc.want)
			}
		})
	}
}

func TestKVBudgetForNeedsBothItsReaders(t *testing.T) {
	if got := KVBudgetFor(KVBudgetOptions{
		PoolTokens: func() (int, bool) { return 12288, true },
	}); got != nil {
		t.Error("a gate was mounted with no way to read the configuration")
	}
	if got := KVBudgetFor(KVBudgetOptions{
		Snapshot: snapshotOf(t, config.EngineLlamaCpp, "-c 12288"),
	}); got != nil {
		t.Error("a gate was mounted with no way to read the pool size")
	}
}

// TestKVBudgetWanted covers which flags the gate applies to. The negative rows
// matter more than the positive ones: accounting for a pool the engine already
// refuses cleanly for buys two round trips and a queue in exchange for nothing.
func TestKVBudgetWanted(t *testing.T) {
	tests := []struct {
		name string
		args string
		want bool
	}{
		{
			// The configuration that reads like declining concurrency and
			// is the opposite: with no -np the server takes four slots and
			// turns the unified pool on, so this is the case most
			// deployments arrive at without meaning to.
			name: "no slot count",
			args: "-c 12288",
			want: true,
		},
		{
			name: "explicitly unified across slots",
			args: "-c 12288 -np 4 -kvu",
			want: true,
		},
		{
			// Each slot owns a hard share, so the engine refuses an
			// over-long prompt at admission with a 400 naming the limit --
			// already attributed to the request that caused it.
			name: "split across slots",
			args: "-c 12288 -np 4 -no-kvu",
			want: false,
		},
		{
			// The engine enforcing the same invariant one level down, and
			// doing it better: against a prompt it has already tokenized.
			// The four shares add up to the pool, so the gate stays off.
			name: "unified with a per-slot ceiling",
			args: "-c 12288 -np 4 -kvu --kv-unified-per-slot 3072",
			want: false,
		},
		{
			// Presence of the flag is not enough: each slot may still
			// claim the whole pool.
			name: "per-slot ceiling still oversubscribes the pool",
			args: "-c 12288 -np 2 -kvu --kv-unified-per-slot 12288",
			want: true,
		},
		{
			// One slot cannot oversubscribe a pool it is alone in.
			name: "single slot",
			args: "-c 12288 -np 1 -kvu",
			want: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := kvBudgetWanted(mustArgs(t, config.EngineLlamaCpp, tc.args))
			if got != tc.want {
				t.Errorf("kvBudgetWanted(%q) = %v, want %v", tc.args, got, tc.want)
			}
		})
	}
}

// TestTheMountedGateRereadsTheFlags is the other half of the regression. The
// gate is now mounted for every llama.cpp deployment, which only helps if what
// it was given reads the flags again per request instead of closing over the
// copy they had at boot.
func TestTheMountedGateRereadsTheFlags(t *testing.T) {
	cfg := config.Config{}
	cfg.Engine.Kind = config.EngineLlamaCpp
	cfg.Engine.URL = "http://localhost:8080"
	cfg.Engine.Args = mustArgs(t, config.EngineLlamaCpp, "-c 12288 -np 1 -kvu")

	reads := 0
	guard := KVBudgetFor(KVBudgetOptions{
		Snapshot: func() config.Config {
			reads++
			return cfg
		},
		PoolTokens: func() (int, bool) { return 12288, true },
	})
	if guard == nil {
		t.Fatal("no gate was mounted for a llama.cpp deployment")
	}
	atMount := reads

	served := false
	handler := guard.Wrap(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		served = true
	}))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hello"}]}`))
	handler.ServeHTTP(httptest.NewRecorder(), req)

	if !served {
		t.Error("the request did not reach the handler behind the gate")
	}
	if reads <= atMount {
		t.Errorf("the configuration was read %d times at mount and never again; "+
			"the gate is deciding on the flags it saw at boot", atMount)
	}
}
