// Package runtimecfg owns the one copy of config.Config that changes
// while the process runs.
//
// Everything else takes a value copy at boot and keeps it: the adapter,
// the data plane, diag and translate all read a configuration that
// stopped moving when main assembled them. Exactly one part of it moves
// afterwards — the model card, replaced by PUT /api/model-spec — and the
// control plane is the only reader that has to see the new value.
//
// That write used to be two assignments in an HTTP handler
// (`s.opts.Config.Spec = spec`) against a struct that GET /api/config,
// GET /api/endpoints and /healthz read from other goroutines, unlocked,
// with maps inside it. Store makes the write serial and the read a
// snapshot.
//
// One invariant makes a snapshot safe to hand out: Store never mutates a
// map or a slice reachable from the config it holds. A change replaces
// whole values, so a reader holding an older snapshot keeps reading the
// older card instead of watching one change under it.
package runtimecfg

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/enginecapacity"
	"github.com/llm-init/llm-init/internal/handoff"
)

// ErrInvalidSpec marks a card the running engine cannot accept. The
// caller's input is wrong, so an HTTP caller answers 400.
var ErrInvalidSpec = errors.New("invalid model spec")

// ErrPersist marks a card that was valid but could not be made durable.
var ErrPersist = errors.New("persist model spec")

// Store is the single writer of the runtime configuration.
type Store struct {
	mu  sync.RWMutex
	cfg config.Config

	// handoff is held rather than derived per call so RUN_DIR has one
	// meaning: the store publishes engine_args through the same object
	// the restart endpoint signals through, and a test that points one
	// somewhere else cannot leave the two disagreeing about where the
	// wrapper is listening.
	handoff *handoff.Tracker
}

// New takes ownership of cfg. A nil tracker is built from
// cfg.Runtime.RunDir with no liveness probe, which is the honest
// arrangement for a store assembled without an engine to watch: restarts
// are then signaled and never confirmed.
func New(cfg config.Config, h *handoff.Tracker) *Store {
	if h == nil {
		h = handoff.NewTracker(handoff.Config{RunDir: cfg.Runtime.RunDir})
	}
	return &Store{cfg: cfg, handoff: h}
}

// Snapshot returns the current configuration by value. Callers may read
// it freely and must not write to anything it points at.
func (s *Store) Snapshot() config.Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg
}

// Handoff is the RUN_DIR channel to the engine wrapper.
func (s *Store) Handoff() *handoff.Tracker { return s.handoff }

// Applied describes what a card change achieved.
type Applied struct {
	// Spec is the card as stored, which is not necessarily the card
	// that was submitted: a mode may require a flag the caller omitted.
	Spec config.ModelSpec

	// EngineArgsChanged reports that the flags the engine launches with
	// are now different. It is the only reason to ask the wrapper for a
	// restart, and it is computed against the card this store was
	// holding rather than against anything on disk.
	EngineArgsChanged bool

	// PendingAppRestart names the card fields whose new values are
	// stored and served but not in force: something was derived from
	// them when main assembled the process, and only a restart of the
	// application re-derives it. Empty for a card that changed nothing
	// of that kind, which is the ordinary case.
	PendingAppRestart []string
}

// ApplySpec is the whole sequence that makes a new card real: normalise
// it against the engine, write it to disk, publish the launch flags for
// the wrapper, and only then swap it into memory. Requesting the restart
// is left to the caller, which is the one that has to describe the
// outcome to somebody.
//
// The lock is held across the disk writes, not just the swap. Two
// concurrent PUTs that interleaved would leave the card on disk and the
// flags in RUN_DIR describing different requests.
func (s *Store) ApplySpec(spec config.ModelSpec) (Applied, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := checkRename(s.cfg.Spec.Name, spec.Name); err != nil {
		return Applied{}, err
	}
	if s.cfg.Engine.Kind == config.EngineAudio &&
		!equalSupports(s.cfg.Spec.Supports, spec.Supports) {
		return Applied{}, fmt.Errorf(
			"%w: audio supports are fixed at startup by MODEL_SUPPORTS and the engine spec",
			ErrInvalidSpec)
	}

	next := s.cfg
	next.Spec = spec
	if next.Spec.Name == "" {
		// The card validator does not require a name, and the alias the
		// data plane rewrites to is read off the card at boot. Storing a
		// nameless card would leave the next boot serving MODEL_NAME
		// while Router registers nothing.
		next.Spec.Name = s.cfg.Spec.Name
	}
	if _, err := config.NormalizeSpecForEngine(&next); err != nil {
		return Applied{}, fmt.Errorf("%w: %w", ErrInvalidSpec, err)
	}
	if path := next.Runtime.ModelSpecPath; path != "" {
		if err := config.WriteModelSpecFile(path, next.Spec); err != nil {
			return Applied{}, fmt.Errorf("%w: %w", ErrPersist, err)
		}
	}
	if err := s.handoff.WriteEngineArgs(next.Spec.EngineArgs); err != nil {
		return Applied{}, fmt.Errorf("%w: %w", ErrPersist, err)
	}

	applied := Applied{
		Spec:              next.Spec,
		EngineArgsChanged: s.cfg.Spec.EngineArgs != next.Spec.EngineArgs,
		PendingAppRestart: pendingAppRestart(s.cfg.Spec, next.Spec),
	}
	s.cfg = next
	return applied, nil
}

// RecordCapacity publishes what the engine says it can hold onto the
// model card, under extensions.capacity. It reports whether the card
// changed.
//
// It goes under extensions rather than beside context_size for a reason
// that is about the card outliving the application, not about taste. A
// card is read by whichever llm-init the app was installed with,
// ParseModelSpecBytes refuses a top-level field that build does not know,
// and the card sits on the shared cache PVC — so a new top-level field is
// a boot failure for an older build on the ordinary path where an app
// moves between the test and the release Market index. extensions is a
// free-form map every build already tolerates and Router already carries
// through.
//
// A reading that only differs in ReportedAt is not a change. The reporter
// re-probes whenever the engine restarts, so the timestamp means "when
// these numbers were first observed", and rewriting the card to advance a
// clock would put a write on the shared PVC on every tick.
func (s *Store) RecordCapacity(c enginecapacity.Capacity) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	block := capacityBlock(c)
	if sameCapacity(s.cfg.Spec.Extensions[enginecapacity.ExtensionKey], block) {
		return false, nil
	}

	next := s.cfg
	// Replace the map rather than writing into it: a snapshot handed out
	// earlier is still being read, and this store's one invariant is that
	// nothing reachable from a published config mutates.
	ext := make(map[string]any, len(s.cfg.Spec.Extensions)+1)
	for k, v := range s.cfg.Spec.Extensions {
		ext[k] = v
	}
	ext[enginecapacity.ExtensionKey] = block
	next.Spec.Extensions = ext

	if path := next.Runtime.ModelSpecPath; path != "" {
		if err := config.WriteModelSpecFile(path, next.Spec); err != nil {
			return false, fmt.Errorf("%w: %w", ErrPersist, err)
		}
	}
	s.cfg = next
	return true, nil
}

// reportedAtKey is the one field of the block that is expected to move
// without meaning anything changed.
const reportedAtKey = "reported_at"

// sameCapacity compares two capacity blocks by everything except when
// they were read. A stored block that is not a JSON object at all — a
// hand-edited card, or one written by something else — counts as
// different, so this process replaces it rather than leaving a shape no
// consumer can read.
func sameCapacity(stored any, next map[string]any) bool {
	storedMap, ok := stored.(map[string]any)
	if !ok {
		return false
	}
	return reflect.DeepEqual(withoutKey(storedMap, reportedAtKey), withoutKey(next, reportedAtKey))
}

func withoutKey(m map[string]any, key string) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if k == key {
			continue
		}
		out[k] = v
	}
	return out
}

// capacityBlock renders a Capacity as the generic map the card round-trips
// through. Going through JSON is what makes the comparison above hold:
// a card reloaded from disk carries map[string]any with float64 numbers,
// and comparing that against a struct would report a change every boot.
func capacityBlock(c enginecapacity.Capacity) map[string]any {
	data, err := json.Marshal(c)
	if err != nil {
		return nil
	}
	var block map[string]any
	if err := json.Unmarshal(data, &block); err != nil {
		return nil
	}
	return block
}

func equalSupports(before, after map[string]bool) bool {
	for key, value := range before {
		if value != after[key] {
			return false
		}
	}
	for key, value := range after {
		if value != before[key] {
			return false
		}
	}
	return true
}

// pendingAppRestart lists the fields that changed and cannot take hold
// until the application restarts.
//
// Only the model card moves at runtime, and only the control plane reads
// it from the store. Everything main assembled holds a config by value:
// the mode decided which routes were mounted and which adapter wraps the
// engine, the reasoning gate was read once, and the translate language
// catalogue was passed into that package at construction. Writing those
// values into the store would not reach any of them — it would only make
// GET /api/endpoints describe routes that are not there.
//
// So the change is stored and served, which is right (Router's copy of
// the card should match the operator's edit), and this is the part that
// says the running process has not adopted it.
func pendingAppRestart(before, after config.ModelSpec) []string {
	var out []string
	if before.Mode != after.Mode {
		out = append(out, "mode")
	}
	if before.Supports[supportsReasoning] != after.Supports[supportsReasoning] {
		out = append(out, "supports."+supportsReasoning)
	}
	// Only the translate sub-object is wired at boot. The rest of
	// extensions is metadata Router reads back off the card, and
	// _unknown_flag is written by the validator itself — comparing the
	// whole map would report a restart for a key this process added.
	if !reflect.DeepEqual(before.Extensions["translate"], after.Extensions["translate"]) {
		out = append(out, "extensions.translate")
	}
	// The rules are description everywhere except one place: the proxy
	// adapter reads the enum ones at construction and refuses a request
	// naming a value outside them. Editing the list therefore changes
	// what the data plane accepts — after a restart, and not before, so
	// an operator who narrows a ladder and immediately tests it has to
	// be told why the old value still went through.
	if !reflect.DeepEqual(before.ParameterRules, after.ParameterRules) {
		out = append(out, "parameter_rules")
	}
	return out
}

// supportsReasoning is the one capability key with a runtime derivation
// behind it (Config.Model.ThinkSupported); the rest are description
// Router reads straight off the card.
const supportsReasoning = "supports_reasoning"

// checkRename refuses a card that renames the model.
//
// The name is the one field on the card that other things are keyed on
// rather than merely described by. The data plane rewrites every request
// to the alias it was built with, ollama-native answers 404 for anything
// else, and Router registers a model row per name — so accepting a rename
// would leave Router routing a name this process rejects, and the old row
// still there beside it. None of that resolves until a restart, which is
// the moment the card is read again anyway.
//
// An empty incoming name cannot happen through the API (the validator
// requires one) and is treated as "not a rename" rather than as one, so a
// caller that assembles a card some other way is not told the wrong
// thing about which field is at fault.
func checkRename(current, incoming string) error {
	if current == "" || incoming == "" || current == incoming {
		return nil
	}
	return fmt.Errorf(
		"%w: model name is fixed for the life of the process (serving %q, card says %q); "+
			"edit the card on disk and restart the app to rename the model",
		ErrInvalidSpec, current, incoming)
}
