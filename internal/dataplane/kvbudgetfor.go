package dataplane

import (
	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/dataplane/kvbudget"
)

// KVBudgetOptions is what KVBudgetFor needs to build a gate, and what the
// gate needs to keep answering for itself afterwards.
type KVBudgetOptions struct {
	// Snapshot returns the configuration as it stands now. Read once here
	// for the engine kind, which comes from the environment and no edit
	// moves, and then per request for the launch flags and the card's
	// output ceiling, which are on the card and do move.
	Snapshot func() config.Config

	// PoolTokens reports the KV pool's size, preferring what the engine
	// said over what the flags declared. A function rather than a value
	// because the measured size does not exist until the engine has
	// started and been probed.
	PoolTokens func() (int, bool)
}

// KVBudgetFor returns the gate for this deployment, or nil when there is
// nothing a gate could do for it.
//
// Nil is the answer for every engine except llama.cpp, and the engine kind is
// the only thing settled here. Whether the gate applies is a question about
// the launch flags, which are editable, so it is asked per request -- see
// kvBudgetWanted.
//
// vLLM and SGLang both refuse cleanly when their pool is full, so neither
// needs this even though both share a cache across requests.
func KVBudgetFor(opts KVBudgetOptions) *kvbudget.Guard {
	if opts.Snapshot == nil || opts.PoolTokens == nil {
		return nil
	}
	boot := opts.Snapshot()
	if boot.Engine.Kind != config.EngineLlamaCpp {
		return nil
	}

	return kvbudget.New(kvbudget.Options{
		Active: func() bool {
			return kvBudgetWanted(opts.Snapshot().Engine.Args)
		},
		Capacity: opts.PoolTokens,
		// The engine's address is where it listens, which a card edit
		// does not move.
		Tokenizer: &kvbudget.Tokenizer{
			BaseURL: boot.Engine.URL,
		},
		MaxOutputTokens: func() int {
			return opts.Snapshot().Spec.MaxOutputToks
		},
	})
}

// kvBudgetWanted reports whether these flags describe the one configuration
// the gate is for: a unified KV pool shared by more than one slot, whose
// slots together can claim more than the pool holds.
//
// Split mode gives each slot a hard share that the engine itself enforces
// with a 400 naming the limit, so the failure is already attributed to the
// request that caused it and no accounting here could improve on that. A
// per-slot ceiling only opts out when the slots' shares add up to no more
// than the pool — it is then the engine enforcing the same invariant one
// level down, and enforcing it better, against a prompt it has already
// tokenized. An explicit -c plus --kv-unified-per-slot equal to the whole
// pool is still oversubscribed. A single slot cannot oversubscribe a pool
// it is alone in.
//
// Only the unified pool has the property that makes a gate worth its cost:
// admission checks a request against the whole pool, so the pool running out
// is discovered mid-decode and takes down every sequence generating at the
// time, including ones already streaming to somebody else.
func kvBudgetWanted(args config.EngineArgs) bool {
	return config.LlamacppKVOversubscribed(args)
}
