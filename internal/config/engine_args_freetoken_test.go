package config

import (
	"reflect"
	"testing"
)

// TestParseEngineArgs_FreeToken covers the two halves of the contract the
// freetoken base chart depends on: the MoE-placement flags an operator
// writes are normalised onto the model card, and everything else is kept
// verbatim in Unknown so the wrapper still word-splits it into argv.
//
// The pass-through half is the one worth a test. freetokenKnownFlags is
// deliberately short -- FreeToken resolves most of its own configuration
// from the checkpoint -- so an operator reaching for a flag this table
// has never heard of is the normal case, not the exception.
func TestParseEngineArgs_FreeToken(t *testing.T) {
	t.Parallel()
	got, err := ParseEngineArgs(EngineFreeToken,
		"--moe-backend hybrid --moe-cpu-threads 16 --kv-reserve-tokens 8192 "+
			"--max-running-requests 2 --moe-cache-auto --some-future-flag x")
	if err != nil {
		t.Fatalf("ParseEngineArgs: %v", err)
	}
	want := map[string]string{
		"moe_backend":          "hybrid",
		"moe_cpu_threads":      "16",
		"kv_reserve_tokens":    "8192",
		"max_running_requests": "2",
		"moe_cache_auto":       knownFlagPresent,
	}
	if !reflect.DeepEqual(got.Known, want) {
		t.Errorf("Known = %v, want %v", got.Known, want)
	}
	if wantUnknown := []string{"--some-future-flag", "x"}; !reflect.DeepEqual(got.Unknown, wantUnknown) {
		t.Errorf("Unknown = %v, want %v", got.Unknown, wantUnknown)
	}
}

// TestFreeTokenDerivations pins which of the three capacity numbers
// FreeToken's flags can supply. Only the concurrency width can be read
// off a launch flag: the context window has no flag at all, and the pool
// is resized at runtime against the expert cache, so both must report
// "unknown" rather than a number the engine never agreed to.
func TestFreeTokenDerivations(t *testing.T) {
	t.Parallel()
	args, err := ParseEngineArgs(EngineFreeToken,
		"--max-running-requests 4 --kv-reserve-tokens 8192")
	if err != nil {
		t.Fatalf("ParseEngineArgs: %v", err)
	}
	if n, ok := DeriveMaxConcurrency(EngineFreeToken, args); !ok || n != 4 {
		t.Errorf("DeriveMaxConcurrency = (%d, %v), want (4, true)", n, ok)
	}
	if n, ok := DeriveContextSize(EngineFreeToken, args); ok {
		t.Errorf("DeriveContextSize = (%d, true), want unknown", n)
	}
	if n, ok := DerivePoolTokens(EngineFreeToken, args); ok {
		t.Errorf("DerivePoolTokens = (%d, true), want unknown", n)
	}
}

func TestLoadFreeToken(t *testing.T) {
	cfg, err := Load(mapEnv(minimalEnv(map[string]string{
		"ENGINE_KIND":  "freetoken",
		"MODEL_SOURCE": "hf://Qwen/Qwen3.6-35B-A3B",
		"ENGINE_ARGS":  "--moe-strategy hybrid --max-running-requests 4",
	})))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Engine.Kind != EngineFreeToken || cfg.Engine.URL != "http://freetoken:1919" {
		t.Fatalf("unexpected engine: %+v", cfg.Engine)
	}
	if got := cfg.Engine.Args.Known["moe_backend"]; got != "hybrid" {
		t.Errorf("moe_backend = %q", got)
	}
}
