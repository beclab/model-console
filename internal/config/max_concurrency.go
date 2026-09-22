package config

// Concurrency derivation from the launch flags already on the model
// card, in the shape of context_size.go and for the same reason: the
// only trustworthy statement about what the engine does is the flags it
// was launched with, so there is no MAX_CONCURRENCY env to disagree
// with them.
//
// Unlike context_size, the derived width is not written to the card. A
// card is read by whichever llm-init the application was installed
// with, ParseModelSpecBytes refuses a top-level field that build does
// not know, and a card on the shared cache PVC outlives the app that
// wrote it — so a width stored by a newer build is a boot failure for
// an older one, on the ordinary path where an app moves between the
// test and the release Market index. Callers derive it when they need
// it instead.
//
// The llamacpp branch reads its slot count through kvunified.go rather
// than straight off the flag, because an absent -np is not the absence of
// a width there; see LlamacppSlotsAuto.
//
// What the number is for: a single-slot engine does not refuse the
// second caller, it makes them wait. Measured on a llama.cpp running
// `-np 1`, twenty concurrent requests all returned 200, the deferred
// gauge peaked at nineteen, and the one that waited 118 seconds
// succeeded. Nothing upstream of the engine could see any of that, so
// the wait looked like a slow model to every layer above — and the
// answer to a slow model (pick a bigger box) is the opposite of the
// answer to a queue (raise the parallelism, or admit fewer callers).

// DeriveMaxConcurrency reports how many requests the engine will work
// on at once, read from the card's engine_args. The bool is false when
// the flags do not pin it down, and no default is substituted in that
// case.
//
// Substituting one would mean guessing for three of the four engines,
// because none of them defaults to something readable from here: vllm
// goes to 256, and 1024 on H100/H200; ollama to four or one depending on
// free memory; and sglang to a figure it computes from token capacity at
// startup, which it will also apply over an explicit flag. Two of those
// depend on the hardware and one overrides the operator, so a guess is
// wrong in a direction nobody can see.
//
// llamacpp is the exception, and the reason is that its default is not a
// guess: the server resolves an absent -np by assignment, to exactly four
// slots (tools/server/server.cpp:155), on any hardware. Reporting it is
// the difference between four callers being served at once and four
// callers looking like one slow model — which is the whole point of the
// number, and the more so here because those four slots also share one KV
// pool.
func DeriveMaxConcurrency(kind EngineKind, args EngineArgs) (int, bool) {
	if kind == EngineLlamaCpp && LlamacppSlotsAuto(args) {
		return LlamacppSlots(args), true
	}
	key, ok := maxConcurrencyKeys[kind]
	if !ok {
		return 0, false
	}
	n, ok := args.GetInt(key)
	if !ok || n < 1 {
		return 0, false
	}
	return n, true
}
