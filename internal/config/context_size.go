package config

// Context-window derivation from the launch flags already on the model
// card. There is deliberately no CONTEXT_SIZE env: the number a caller
// needs is the one the engine was actually launched with, and a second
// env would be free to disagree with ENGINE_ARGS. It did — a stale
// `-c 65536` on a shared cache PVC served a model whose chart declared
// 102K, and every consumer downstream believed the chart.
//
// Each engine spells the flag differently, so the parsed Known table
// (see engine_args_known.go) is read per Kind rather than by scanning
// the raw string. Which key holds the window is declared there too, in
// contextSizeKeys, beside the tables that define those key names.

// DeriveContextSize reports the per-request context window the engine
// will serve, read from the card's engine_args. The bool is false when
// the flags do not pin one down — the operator left it to the engine,
// which then takes it from the model's own metadata, a value llm-init
// cannot see from here. Callers must leave an existing card value alone
// in that case rather than writing a zero.
func DeriveContextSize(kind EngineKind, args EngineArgs) (int, bool) {
	if kind == EngineLlamaCpp {
		return deriveLlamacppContextSize(args)
	}
	key, ok := contextSizeKeys[kind]
	if !ok {
		return 0, false
	}
	n, ok := args.GetInt(key)
	if !ok || n <= 0 {
		return 0, false
	}
	return n, true
}

// deriveLlamacppContextSize reproduces the `n_ctx_slot` the server prints
// at startup. Unlike the other three engines, whose flags are already
// per-request, llama.cpp's -c is the size of the KV pool, and how much of
// it one request may use depends on whether that pool is shared.
//
// Split mode cuts the pool into per-slot shares, so a conversation gets
// n_ctx / n_parallel, padded up to a multiple of 256
// (src/llama-context.cpp:293-294). Unified mode does no division at all
// (same file, :291): every slot may take the whole pool, and the only
// thing that narrows that is --kv-unified-per-slot, which
// server-context.cpp's n_ctx_slot() applies as a min.
//
// Two ways this can still overstate what the engine serves, both invisible
// from the flags. n_ctx_slot() also caps at the model's training context,
// which lives in the GGUF; and in unified mode the number below is what one
// request may ask for, not what it can count on getting while its
// neighbours hold KV cells. Only the running engine can settle either --
// hence the /props probe.
//
// `-c 0` means "take the training context from the model file", which is
// not a number that can be read off the flags. `-c` left out entirely is
// different: it also defaults the pool to the training context, but paired
// with --kv-unified-per-slot the pool is instead sized to
// n_parallel * per_slot (tools/server/server.cpp:162-170), which does make
// the per-request window knowable.
func deriveLlamacppContextSize(args EngineArgs) (int, bool) {
	perSlot, hasPerSlot := LlamacppKVUnifiedPerSlot(args)

	total, ok := args.GetInt("ctx_size")
	if !ok {
		// No -c. The per-slot ceiling is then both the pool divisor and
		// the answer; without it the pool comes from the model file.
		if hasPerSlot {
			return perSlot, true
		}
		return 0, false
	}
	if total <= 0 {
		return 0, false
	}
	total = padUpTo256(total)

	if !KVUnifiedEffective(args) {
		return padUpTo256(total / LlamacppSlots(args)), true
	}
	if hasPerSlot && perSlot < total {
		return perSlot, true
	}
	return total, true
}

// padUpTo256 mirrors GGML_PAD(n, 256), which llama.cpp applies to both the
// pool size and each slot's share.
func padUpTo256(n int) int {
	const align = 256
	return (n + align - 1) / align * align
}

// syncContextSizeFromEngineArgs writes the derived window onto the card
// and reports whether it changed. engine_args wins over a stored
// context_size: the flags are what the engine is launched with, and a
// card that claims a larger window than the engine serves is how a
// caller ends up sending a prompt that gets truncated mid-message.
func syncContextSizeFromEngineArgs(cfg *Config) bool {
	if cfg == nil {
		return false
	}
	n, ok := DeriveContextSize(cfg.Engine.Kind, cfg.Engine.Args)
	if !ok || n == cfg.Spec.ContextSize {
		return false
	}
	cfg.Spec.ContextSize = n
	return true
}

// syncMaxOutputTokensFromContext gives a chat card an output ceiling it can
// actually honor and reports whether it changed.
//
// An unset ceiling, or one as large as the window, is read by clients as
// "reserve the whole window for the reply": Router publishes it as
// max_output_tokens and a client sizes max_tokens from it, which leaves no
// room for the prompt and asks the KV pool for a request's worth of tokens
// nobody will generate. A quarter of the window is what such a card gets
// instead; a ceiling the author set below the window is theirs and stays.
func syncMaxOutputTokensFromContext(cfg *Config) bool {
	if cfg == nil || cfg.Spec.Mode != string(ModelChat) || cfg.Spec.ContextSize <= 0 {
		return false
	}
	if cfg.Spec.MaxOutputToks > 0 && cfg.Spec.MaxOutputToks < cfg.Spec.ContextSize {
		return false
	}
	n := cfg.Spec.ContextSize / 4
	if n < 1 || n == cfg.Spec.MaxOutputToks {
		return false
	}
	cfg.Spec.MaxOutputToks = n
	return true
}
