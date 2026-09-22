package config

// Pool-size derivation, the third number beside context_size.go and
// max_concurrency.go and the one the other two cannot substitute for.
//
// A per-request window and a concurrency width do not multiply into a
// capacity. Two of these engines size the pool independently of both:
// llama.cpp's -c *is* the pool and the window is carved out of it, and
// SGLang computes the pool from a memory fraction and then derives the
// width from the pool. Only a caller that knows the pool can tell the
// difference between four requests that fit and four that oversubscribe.

// DerivePoolTokens reports how many tokens of KV cache the engine holds
// in total, read from the card's engine_args. The bool is false when the
// flags do not pin it down.
//
// It is false more often than the other two derivations, and for a
// reason that is the point of Probe rather than a gap in this table: the
// pool is the number these engines most often decide for themselves at
// startup, after measuring the hardware. SGLang sizes it from
// --mem-fraction-static unless --max-total-tokens is given, and vLLM
// profiles the model to count blocks, which is why vllm is absent here
// entirely.
func DerivePoolTokens(kind EngineKind, args EngineArgs) (int, bool) {
	switch kind {
	case EngineLlamaCpp:
		return deriveLlamacppPoolTokens(args)
	case EngineSGLang:
		// Upstream documents --max-total-tokens as a development and
		// debugging flag; when it is absent the pool comes from
		// --mem-fraction-static and the hardware, neither of which
		// resolves to a token count from here.
		n, ok := args.GetInt("max_total_tokens")
		if !ok || n <= 0 {
			return 0, false
		}
		return n, true
	case EngineOllama:
		// Ollama launches its runner with num_ctx * num_parallel
		// (server/sched.go effectiveLlamaServerContext), so the pool is
		// the product -- but only when both are pinned. It also lowers
		// either one silently when memory is short, which is what makes
		// the /api/ps reading worth more than this.
		ctx, okCtx := args.GetInt("num_ctx")
		par, okPar := args.GetInt("num_parallel")
		if !okCtx || !okPar || ctx <= 0 || par <= 0 {
			return 0, false
		}
		return ctx * par, true
	default:
		return 0, false
	}
}

// deriveLlamacppPoolTokens reads the pool straight off -c, which is what
// that flag means: the whole KV cache, out of which n_ctx_slot is a share
// (see deriveLlamacppContextSize for the other half of the arithmetic).
//
// `-c 0` and an absent -c both defer to the model's training context,
// which lives in the GGUF and cannot be read from here. The one exception
// is --kv-unified-per-slot, which makes the server size the pool as
// n_parallel * per_slot instead (tools/server/server.cpp:162-170).
func deriveLlamacppPoolTokens(args EngineArgs) (int, bool) {
	total, ok := args.GetInt("ctx_size")
	if ok && total > 0 {
		return padUpTo256(total), true
	}
	if perSlot, hasPerSlot := LlamacppKVUnifiedPerSlot(args); hasPerSlot {
		return padUpTo256(perSlot * LlamacppSlots(args)), true
	}
	return 0, false
}
