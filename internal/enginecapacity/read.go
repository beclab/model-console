package enginecapacity

import (
	"encoding/json"

	"github.com/llm-init/llm-init/internal/config"
)

// ExtensionKey is where a recorded Capacity lives inside the card's
// `extensions` object. A wire name shared with the Router, which reads the
// same block: renaming it is a break on both sides.
const ExtensionKey = "capacity"

// FromExtensions reads a recorded Capacity back off a card.
//
// This is the same block the Router parses, read here by the one consumer
// inside this process that needs it: the KV budget, which has to account
// against the pool the engine really allocated rather than the one the flags
// asked for. Round-tripping through JSON rather than walking the map is what
// keeps this reader correct when a field is added to Capacity.
func FromExtensions(extensions map[string]any) (Capacity, bool) {
	raw, ok := extensions[ExtensionKey]
	if !ok {
		return Capacity{}, false
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return Capacity{}, false
	}
	var c Capacity
	if err := json.Unmarshal(data, &c); err != nil {
		return Capacity{}, false
	}
	// A block with no source or no timestamp is not a reading. Both come
	// free from the writer, so their absence means the block was written by
	// something else -- and these numbers are indistinguishable from a
	// guess once they are being enforced against live traffic.
	if c.Source == "" || c.ReportedAt.IsZero() {
		return Capacity{}, false
	}
	return c, true
}

// PoolTokensOf reports the KV pool's size for a card, preferring what the
// engine reported over what the flags declared.
//
// The preference is the whole point of the probe. llama.cpp is the one engine
// whose -c does state the pool outright, and even there the model's training
// context can cap it below what was asked for -- and the other three size the
// pool from a memory fraction or a profiling run that no flag predicts.
func PoolTokensOf(cfg config.Config) (int, bool) {
	if c, ok := FromExtensions(cfg.Spec.Extensions); ok && c.PoolTokens > 0 {
		return c.PoolTokens, true
	}
	return config.DerivePoolTokens(cfg.Engine.Kind, cfg.Engine.Args)
}
