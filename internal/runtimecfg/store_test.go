package runtimecfg

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/llm-init/llm-init/internal/config"
	"github.com/llm-init/llm-init/internal/handoff"
)

// newStore builds a store over a vLLM config whose card and run dir live
// in a temp dir, which is the shape the control plane hands it.
func newStore(t *testing.T) (*Store, string) {
	t.Helper()
	env := map[string]string{
		"ENGINE_KIND":  "vllm",
		"MODEL_NAME":   "qwen2.5-7b",
		"MODEL_MODE":   "chat",
		"MODEL_SOURCE": "hf://Qwen/Qwen2.5-7B-Instruct --revision 0123456789abcdef0123456789abcdef01234567",
	}
	cfg, err := config.Load(func(k string) string { return env[k] })
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	dir := t.TempDir()
	cfg.Runtime.RunDir = dir
	cfg.Runtime.ModelSpecPath = filepath.Join(dir, "model-spec.json")
	return New(cfg, nil), dir
}

func newAudioStore(t *testing.T) (*Store, string) {
	t.Helper()
	env := map[string]string{
		"ENGINE_KIND":    "audio",
		"MODEL_NAME":     "qwen3-asr",
		"MODEL_MODE":     "audio",
		"MODEL_SOURCE":   "hf://Qwen/Qwen3-ASR-1.7B",
		"MODEL_SUPPORTS": "supports_stt",
	}
	cfg, err := config.Load(func(k string) string { return env[k] })
	if err != nil {
		t.Fatalf("config: %v", err)
	}
	cfg.Spec = parseSpec(t,
		`{"name":"qwen3-asr","mode":"audio","supports":{"supports_stt":true}}`)
	dir := t.TempDir()
	cfg.Runtime.RunDir = dir
	cfg.Runtime.ModelSpecPath = filepath.Join(dir, "model-spec.json")
	return New(cfg, nil), dir
}

func parseSpec(t *testing.T, jsonSrc string) config.ModelSpec {
	t.Helper()
	spec, err := config.ParseModelSpecBytes([]byte(jsonSrc), "test")
	if err != nil {
		t.Fatalf("parse spec: %v", err)
	}
	return spec
}

func TestApplySpec_DerivesContextSizeLikeBootDoes(t *testing.T) {
	t.Parallel()
	s, dir := newStore(t)

	applied, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len 4096"}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if applied.Spec.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096 derived from the flags",
			applied.Spec.ContextSize)
	}
	if got := s.Snapshot().Engine.Args.Known["max_model_len"]; got != "4096" {
		t.Errorf("parsed flags = %q, want the card reparsed", got)
	}

	// The window has to reach disk too: Router reads the card, and the
	// only other place it is derived is the next boot.
	data, err := os.ReadFile(filepath.Join(dir, "model-spec.json"))
	if err != nil {
		t.Fatalf("read card: %v", err)
	}
	var onDisk config.ModelSpec
	if err := json.Unmarshal(data, &onDisk); err != nil {
		t.Fatalf("decode card: %v", err)
	}
	if onDisk.ContextSize != 4096 {
		t.Errorf("on-disk ContextSize = %d, want 4096", onDisk.ContextSize)
	}
}

func TestApplySpec_ReportsWhetherTheEngineNeedsRelaunching(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)
	card := `{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len 4096"}`

	first, err := s.ApplySpec(parseSpec(t, card))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if !first.EngineArgsChanged {
		t.Error("first apply set flags where there were none; that is a change")
	}

	// Same flags, different pricing: nothing the engine launches with
	// moved, so a restart would cost an outage for no reason.
	again, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len 4096",`+
			`"pricing":{"input_cost_per_token":"0.000001"}}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if again.EngineArgsChanged {
		t.Error("EngineArgsChanged on a card whose flags are identical")
	}
}

func TestApplySpec_RejectedCardChangesNothing(t *testing.T) {
	t.Parallel()
	s, dir := newStore(t)
	before := s.Snapshot()

	// An unbalanced quote: the flags cannot be split into tokens, so
	// there is nothing to hand the engine.
	_, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{},"engine_args":"--max-model-len '4096"}`))
	if !errors.Is(err, ErrInvalidSpec) {
		t.Fatalf("err = %v, want ErrInvalidSpec", err)
	}
	if s.Snapshot().Spec.EngineArgs != before.Spec.EngineArgs {
		t.Errorf("served card moved on a rejected apply: %q",
			s.Snapshot().Spec.EngineArgs)
	}
	if _, err := os.Stat(filepath.Join(dir, "model-spec.json")); !os.IsNotExist(err) {
		t.Errorf("rejected card reached disk (stat err=%v)", err)
	}
	if _, err := os.Stat(filepath.Join(dir, handoff.EngineArgsName)); !os.IsNotExist(err) {
		t.Errorf("rejected flags reached the wrapper (stat err=%v)", err)
	}
}

func TestApplySpec_RefusesToRenameTheServedModel(t *testing.T) {
	t.Parallel()
	s, dir := newStore(t)

	_, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b-instruct","mode":"chat","supports":{}}`))
	if !errors.Is(err, ErrInvalidSpec) {
		t.Fatalf("err = %v, want ErrInvalidSpec", err)
	}
	if got := s.Snapshot().Model.Name; got != "qwen2.5-7b" {
		t.Errorf("served alias = %q, want the name the process started with", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "model-spec.json")); !os.IsNotExist(err) {
		t.Errorf("renamed card reached disk (stat err=%v)", err)
	}
}

func TestApplySpec_AudioRefusesSupportsChanges(t *testing.T) {
	t.Parallel()
	s, dir := newAudioStore(t)

	_, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen3-asr","mode":"audio","supports":{"supports_stt":false}}`))
	if !errors.Is(err, ErrInvalidSpec) {
		t.Fatalf("err = %v, want ErrInvalidSpec", err)
	}
	if !strings.Contains(err.Error(), "audio supports are fixed at startup") {
		t.Fatalf("err = %q, want startup capability explanation", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "model-spec.json")); !os.IsNotExist(err) {
		t.Errorf("rejected supports change reached disk (stat err=%v)", err)
	}
}

func TestApplySpec_AudioAllowsUnchangedSupportsAndOtherFields(t *testing.T) {
	t.Parallel()
	s, _ := newAudioStore(t)

	applied, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen3-asr","mode":"audio","supports":{"supports_stt":true,"supports_vad":false},`+
			`"pricing":{"input_cost_per_second":"0.001"}}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if applied.Spec.Pricing["input_cost_per_second"] != "0.001" {
		t.Fatalf("pricing was not applied: %#v", applied.Spec.Pricing)
	}
}

// A nameless card is legal JSON — only mode is required — and would
// leave the next boot with no alias to serve.
func TestApplySpec_NamelessCardKeepsTheNameItWasServing(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)

	applied, err := s.ApplySpec(parseSpec(t, `{"mode":"chat","supports":{}}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if applied.Spec.Name != "qwen2.5-7b" {
		t.Errorf("stored name = %q, want the one already being served", applied.Spec.Name)
	}
}

func TestApplySpec_NamesTheFieldsOnlyARestartCanApply(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)

	// Metadata Router reads straight off the card: nothing derived it,
	// so nothing is waiting on a restart.
	quiet, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{},"max_output_tokens":512}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if len(quiet.PendingAppRestart) != 0 {
		t.Errorf("PendingAppRestart = %v on a pure-metadata edit", quiet.PendingAppRestart)
	}

	// Mode picked the mounted routes, supports_reasoning the thinking
	// gate, extensions.translate the language catalogue — all at boot.
	loud, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"translate","supports":{"supports_reasoning":true},`+
			`"extensions":{"translate":{"languages":["en","zh"]}}}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	want := []string{"mode", "supports.supports_reasoning", "extensions.translate"}
	if !reflect.DeepEqual(loud.PendingAppRestart, want) {
		t.Errorf("PendingAppRestart = %v, want %v", loud.PendingAppRestart, want)
	}

	// The card is served regardless: Router's copy should match what the
	// operator wrote, restart pending or not.
	if s.Snapshot().Spec.Mode != "translate" {
		t.Error("a pending restart must not stop the card from being stored")
	}
}

// Narrowing a ladder changes what the data plane accepts, because the proxy
// adapter reads the enum rules at construction. An operator who edits the
// list and immediately sends the value they just removed gets it through, and
// this is the only thing that tells them why.
func TestApplySpec_EditingTheRulesWaitsForARestart(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)

	applied, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{},"parameter_rules":[`+
			`{"name":"reasoning_effort","type":"string","options":["low","high"]}]}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if !slices.Contains(applied.PendingAppRestart, "parameter_rules") {
		t.Errorf("PendingAppRestart = %v, want parameter_rules named", applied.PendingAppRestart)
	}

	// Re-submitting the same list is not an edit.
	again, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{},"parameter_rules":[`+
			`{"name":"reasoning_effort","type":"string","options":["low","high"]}]}`))
	if err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}
	if slices.Contains(again.PendingAppRestart, "parameter_rules") {
		t.Errorf("PendingAppRestart = %v on an identical card", again.PendingAppRestart)
	}
}

// A snapshot is handed out by value and read without a lock, so a change
// landing afterwards must not be visible through it. This is what makes
// every handler's `cfg := s.config()` safe.
func TestSnapshot_IsNotAWindowOntoLaterChanges(t *testing.T) {
	t.Parallel()
	s, _ := newStore(t)
	if _, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{"supports_function_calling":true}}`)); err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}

	held := s.Snapshot()
	if _, err := s.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{"supports_vision":true},`+
			`"engine_args":"--max-model-len 8192"}`)); err != nil {
		t.Fatalf("ApplySpec: %v", err)
	}

	if held.Spec.EngineArgs != "" {
		t.Errorf("held snapshot picked up new flags: %q", held.Spec.EngineArgs)
	}
	if !held.Spec.Supports["supports_function_calling"] {
		t.Error("held snapshot lost the supports map it was taken with")
	}
	if held.Spec.Supports["supports_vision"] {
		t.Error("held snapshot's supports map was mutated in place")
	}
}

func TestApplySpec_UnwritableCardPathFailsLoudly(t *testing.T) {
	t.Parallel()
	s, dir := newStore(t)

	// A directory where the card file should be: the write cannot
	// succeed, and serving a card that was never stored would leave
	// Router and the next boot disagreeing.
	cfg := s.Snapshot()
	cfg.Runtime.ModelSpecPath = filepath.Join(dir, "blocked")
	if err := os.MkdirAll(cfg.Runtime.ModelSpecPath, 0o755); err != nil {
		t.Fatal(err)
	}
	blocked := New(cfg, nil)

	_, err := blocked.ApplySpec(parseSpec(t,
		`{"name":"qwen2.5-7b","mode":"chat","supports":{}}`))
	if !errors.Is(err, ErrPersist) {
		t.Fatalf("err = %v, want ErrPersist", err)
	}
}
