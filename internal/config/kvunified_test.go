package config

import "testing"

func TestKVUnifiedEffective(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		raw  string
		want bool
	}{
		{"bare short flag", "-c 12288 -kvu -np 2", true},
		{"bare long flag", "-c 12288 --kv-unified -np 2", true},
		// The value peek must not read the following flag as -kvu's value.
		// If it did, kv_unified would hold "-np" -- neither truthy nor
		// falsey -- and the answer would fall through to the auto rule,
		// which says false here because -np is explicit.
		{"flag immediately before -np", "-kvu -np 2 -c 12288", true},
		{"negated short flag", "-c 12288 -no-kvu -np 2", false},
		{"negated long flag", "-c 12288 --no-kv-unified -np 2", false},
		{"negated wins over positive", "-c 12288 -kvu -no-kvu -np 2", false},
		{"explicit slots opt out", "-c 12288 -np 2", false},
		{"explicit single slot opts out", "-c 104448 -np 1", false},

		// The reason this file exists: no -np at all means four slots
		// sharing one pool, which reads from the flags like the most
		// conservative configuration available.
		{"no slot count is auto", "-c 131072 -ngl all -fa on", true},
		{"upstream's explicit auto", "-c 131072 -np -1", true},
		{"empty args", "", true},

		// The auto override runs after the flags, so opting out without
		// also pinning the slot count opts out of nothing.
		{"negation alone does not survive the auto override", "-c 8192 -no-kvu", true},
		{"negated env alone does not survive it either", "LLAMA_ARG_NO_KV_UNIFIED=1", true},

		{"env positive", "LLAMA_ARG_CTX_SIZE=12288 LLAMA_ARG_KV_UNIFIED=1", true},
		{"env positive spelled on", "LLAMA_ARG_KV_UNIFIED=on LLAMA_ARG_N_PARALLEL=2", true},
		{"env positive spelled enabled", "LLAMA_ARG_KV_UNIFIED=enabled LLAMA_ARG_N_PARALLEL=2", true},
		{"env falsey", "LLAMA_ARG_KV_UNIFIED=0 LLAMA_ARG_N_PARALLEL=2", false},
		{"env falsey spelled off", "LLAMA_ARG_KV_UNIFIED=off LLAMA_ARG_N_PARALLEL=2", false},
		{"env falsey spelled disabled", "LLAMA_ARG_KV_UNIFIED=disabled LLAMA_ARG_N_PARALLEL=2", false},
		// Upstream reads the negated env as off on presence alone, so even
		// a value that looks like "on" turns the pool off.
		{"negated env is off whatever its value", "LLAMA_ARG_NO_KV_UNIFIED=0 LLAMA_ARG_N_PARALLEL=2", false},
		{"negated env beats positive env", "LLAMA_ARG_KV_UNIFIED=1 LLAMA_ARG_NO_KV_UNIFIED=1 LLAMA_ARG_N_PARALLEL=2", false},
		// A value upstream would have thrown on leaves the flag without
		// effect, so the slot count is what decides.
		{"unclassifiable value with an auto slot count", "LLAMA_ARG_KV_UNIFIED=maybe", true},
		{"unclassifiable value with explicit slots", "LLAMA_ARG_KV_UNIFIED=maybe LLAMA_ARG_N_PARALLEL=2", false},

		{"per-slot ceiling does not itself enable the pool", "-c 12288 -np 2 --kv-unified-per-slot 4096", false},
		{"per-slot ceiling alongside the pool", "-c 12288 -np 2 -kvu --kv-unified-per-slot 4096", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			args, err := ParseEngineArgs(EngineLlamaCpp, tc.raw)
			if err != nil {
				t.Fatalf("ParseEngineArgs(%q): %v", tc.raw, err)
			}
			if got := KVUnifiedEffective(args); got != tc.want {
				t.Errorf("KVUnifiedEffective(%q) = %v, want %v", tc.raw, got, tc.want)
			}
		})
	}
}

func TestLlamacppSlots(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw       string
		wantAuto  bool
		wantSlots int
	}{
		{"-np 1", false, 1},
		{"-np 8", false, 8},
		{"--parallel 2", false, 2},
		{"LLAMA_ARG_N_PARALLEL=6", false, 6},
		{"-c 131072", true, 4},
		{"-np -1", true, 4},
		// A bare -np is not a width; upstream would have failed to parse
		// the missing value, so nothing running can disagree with 4.
		{"-c 65536 -np", true, 4},
		{"", true, 4},
	}
	for _, tc := range cases {
		args, err := ParseEngineArgs(EngineLlamaCpp, tc.raw)
		if err != nil {
			t.Fatalf("ParseEngineArgs(%q): %v", tc.raw, err)
		}
		if got := LlamacppSlotsAuto(args); got != tc.wantAuto {
			t.Errorf("LlamacppSlotsAuto(%q) = %v, want %v", tc.raw, got, tc.wantAuto)
		}
		if got := LlamacppSlots(args); got != tc.wantSlots {
			t.Errorf("LlamacppSlots(%q) = %d, want %d", tc.raw, got, tc.wantSlots)
		}
	}
}

func TestLlamacppKVUnifiedPerSlot(t *testing.T) {
	t.Parallel()
	cases := []struct {
		raw  string
		want int
		ok   bool
	}{
		{"-kvu --kv-unified-per-slot 4096", 4096, true},
		{"-kvu LLAMA_ARG_KV_UNIFIED_PER_SLOT=8192", 8192, true},
		{"-kvu", 0, false},
		// Bare, so it parses as knownFlagPresent rather than a number.
		{"-kvu --kv-unified-per-slot", 0, false},
		{"-kvu --kv-unified-per-slot 0", 0, false},
	}
	for _, tc := range cases {
		args, err := ParseEngineArgs(EngineLlamaCpp, tc.raw)
		if err != nil {
			t.Fatalf("ParseEngineArgs(%q): %v", tc.raw, err)
		}
		got, ok := LlamacppKVUnifiedPerSlot(args)
		if got != tc.want || ok != tc.ok {
			t.Errorf("LlamacppKVUnifiedPerSlot(%q) = (%d, %v), want (%d, %v)", tc.raw, got, ok, tc.want, tc.ok)
		}
	}
}

func TestLlamacppKVOversubscribed(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		raw  string
		want bool
	}{
		{"explicit slots sharing one pool", "-c 12288 -np 2 -kvu", true},
		{"per-slot ceiling that fits", "-c 12288 -np 2 -kvu --kv-unified-per-slot 4096", false},
		{"per-slot ceiling that fits four slots", "-c 12288 -np 4 -kvu --kv-unified-per-slot 3072", false},
		{"per-slot ceiling still oversubscribes the pool", "-c 12288 -np 2 -kvu --kv-unified-per-slot 12288", true},
		{"single slot", "-c 12288 -np 1 -kvu", false},
		{"split mode", "-c 12288 -np 2", false},
		{"auto four slots with no -np", "-c 12288", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			args, err := ParseEngineArgs(EngineLlamaCpp, tc.raw)
			if err != nil {
				t.Fatalf("ParseEngineArgs(%q): %v", tc.raw, err)
			}
			if got := LlamacppKVOversubscribed(args); got != tc.want {
				t.Errorf("LlamacppKVOversubscribed(%q) = %v, want %v", tc.raw, got, tc.want)
			}
		})
	}
}
