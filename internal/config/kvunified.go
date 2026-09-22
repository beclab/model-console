package config

import (
	"log/slog"
	"strconv"
)

// This file answers one question about a llama.cpp deployment: is the KV
// cache one pool shared by every slot, or is it cut into per-slot shares?
// Nothing else in ENGINE_ARGS depends on it, and everything about how much
// context a single request may use does.
//
// The two modes fail differently, which is why the answer has to be exact.
// Split mode gives each slot a hard quota, so an over-long prompt is refused
// at admission with a 400 naming the limit. Unified mode admits any prompt
// that fits the whole pool and only discovers the pool is oversubscribed
// mid-decode, where the recovery path releases every slot that is currently
// generating -- so one over-long request takes its neighbours down with it.

// llamacppTruthy mirrors common_arg_utils::is_truthy (common/arg.cpp).
// There is no falsey set to match it, because everything outside this one
// resolves to the same answer here: the falsey spellings (off, disabled,
// false, 0) mean the pool is not unified, and a value in neither set makes
// upstream throw rather than pick a side -- so it names no engine that
// could be running to contradict us.
//
// knownFlagPresent is one of upstream's spellings as well as our marker for
// a bare flag, so the one entry covers both `-kvu` and `-kvu true`.
var llamacppTruthy = map[string]bool{
	"on": true, "enabled": true, knownFlagPresent: true, "1": true,
}

// LlamacppSlotsAuto reports whether llama.cpp will pick the slot count
// itself. The server's own default for -np is -1, not the 1 that
// common_params carries (common/arg.cpp sets n_parallel = -1 for
// LLAMA_EXAMPLE_SERVER), so leaving the flag out selects auto rather than
// single-slot.
//
// A present-but-unusable value counts as auto for two reasons. `-np -1` is
// upstream's explicit spelling of auto, and our value peek rejects tokens
// beginning with '-' so it arrives here as knownFlagPresent. Anything else
// unparseable, including `-np 0`, makes the engine refuse to start, so no
// running engine ever contradicts what we say about it.
func LlamacppSlotsAuto(args EngineArgs) bool {
	raw, ok := args.GetString("parallel")
	if !ok {
		return true
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return true
	}
	return n < 0
}

// KVUnifiedEffective reports whether the llama.cpp server will run with one
// unified KV buffer shared across all sequences.
//
// The first branch is the one no flag table shows, and it is not a fallback:
// tools/server/server.cpp:152-157 runs after the flags are parsed and sets
// n_parallel = 4 *and* kv_unified = true whenever the slot count is auto,
// with no regard for what was written. Two consequences that surprise people
// in opposite directions -- the configuration that looks like it declined
// concurrency altogether is the one that gets four slots sharing a single
// pool with no per-slot ceiling, and -no-kvu on its own does not opt out of
// that, because the override lands after it.
func KVUnifiedEffective(args EngineArgs) bool {
	if LlamacppSlotsAuto(args) {
		return true
	}
	// With the slot count explicit, the flags decide, and the negated form
	// wins over the positive one. On the command line upstream resolves a
	// contradiction by order, which a flat key map cannot preserve; for the
	// env form upstream itself gives the negation unconditional precedence,
	// and following that is better than guessing at an ordering we did not
	// record.
	if _, ok := args.GetString("no_kv_unified"); ok {
		return false
	}
	if v, ok := args.GetString("kv_unified"); ok && llamacppTruthy[v] {
		return true
	}
	// Bare `-kvu` is recorded as knownFlagPresent, which is truthy, so
	// reaching here means either no flag or a value upstream would have
	// thrown on -- and the engine's own default is off.
	return false
}

// LlamacppKVUnifiedPerSlot returns the per-slot context ceiling set by
// --kv-unified-per-slot, and whether it was set to a usable value. It is the
// only way to keep a unified pool from being oversubscribed, because it caps
// what any single slot may take out of it.
func LlamacppKVUnifiedPerSlot(args EngineArgs) (int, bool) {
	n, ok := args.GetInt("kv_unified_per_slot")
	if !ok || n <= 0 {
		return 0, false
	}
	return n, true
}

// LlamacppSlots returns the number of server slots the engine will run with.
// The auto branch is 4 by upstream's own arithmetic, not a guess.
func LlamacppSlots(args EngineArgs) int {
	if LlamacppSlotsAuto(args) {
		return llamacppAutoSlots
	}
	n, _ := args.GetInt("parallel")
	return n
}

// llamacppAutoSlots is what tools/server/server.cpp:155 resolves an auto slot
// count to.
const llamacppAutoSlots = 4

// LlamacppKVOversubscribed reports whether a unified llama.cpp pool lets its
// slots promise more context in total than the pool holds.
//
// A per-slot ceiling only opts out when the slots' shares add up to no more
// than the pool. Presence of --kv-unified-per-slot is not enough: an explicit
// -c plus a ceiling equal to the whole pool still lets every slot claim it.
// When the pool cannot be read and there is no usable ceiling, the answer is
// the same as today — treat it as oversubscribed. When the pool cannot be
// read but a ceiling is set, deriveLlamacppPoolTokens already sizes the pool
// as perSlot*slots if -c is absent, and that product cannot exceed itself.
func LlamacppKVOversubscribed(args EngineArgs) bool {
	if !KVUnifiedEffective(args) {
		return false
	}
	slots := LlamacppSlots(args)
	if slots <= 1 {
		return false
	}
	perSlot, hasPerSlot := LlamacppKVUnifiedPerSlot(args)
	if !hasPerSlot {
		return true
	}
	pool, ok := deriveLlamacppPoolTokens(args)
	if !ok {
		return true
	}
	return perSlot*slots > pool
}

// warnKVUnifiedOversubscribed logs when a llama.cpp deployment lets its slots
// promise more context in total than the pool holds.
//
// The test is not a threshold anyone has to tune. With one shared pool and no
// usable per-slot ceiling — or a ceiling that still lets every slot claim the
// whole pool — n slots can each be granted the full -c, and
// n_slots * n_ctx > n_ctx holds by definition as soon as there is more than
// one slot.
//
// It is a warning rather than a refusal because the configuration works
// perfectly until two large requests overlap, and plenty of deployments never
// see that. What they do see when it happens is every in-flight request
// failing at once, including ones already streaming, which is why this is
// worth saying out loud at deploy time rather than leaving to be discovered.
func warnKVUnifiedOversubscribed(cfg *Config) {
	if cfg == nil || cfg.Engine.Kind != EngineLlamaCpp {
		return
	}
	args := cfg.Engine.Args
	if !LlamacppKVOversubscribed(args) {
		return
	}
	slots := LlamacppSlots(args)

	// -no-kvu is spelled with -np because on its own it does nothing: the
	// auto-slot override that turns the pool on runs after it.
	attrs := []any{
		"engine_args", cfg.Spec.EngineArgs,
		"slots", slots,
		"remedies", "--kv-unified-per-slot <n> to cap what one slot may take; -np 1 for a single slot; -np <n> -no-kvu to give each slot its own share",
	}
	if total, ok := args.GetInt("ctx_size"); ok && total > 0 {
		attrs = append(attrs, "kv_pool_tokens", total)
	}

	if LlamacppSlotsAuto(args) {
		// The likeliest way to arrive here is by writing no slot count at
		// all, which reads like opting out of concurrency and is the
		// opposite: the server takes an unset -np as auto and turns the
		// unified pool on with it.
		slog.Warn("llama.cpp KV pool is oversubscribed: no -np was given, so the server runs 4 slots sharing one unified pool with no per-slot ceiling. Concurrent large prompts will fail together, releasing requests that were already generating",
			attrs...)
		return
	}
	slog.Warn("llama.cpp KV pool is oversubscribed: every slot may claim the whole pool, so concurrent large prompts will fail together, releasing requests that were already generating",
		attrs...)
}
