package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/llm-init/llm-init/internal/handoff"
)

const supportsTTS = "supports_tts"

// ModelSpec is the v2 ProviderModelSpec shape Router already consumes
// (stored as JSONB in `provider_models.model_spec`). Exposed by
// `GET /api/model-spec` so Router can discover the deployed model's
// capabilities directly from the sidecar.
//
// Field-for-field mirror of
// `router/backend/internal/providers/predefined/model.go::ProviderModelSpec`.
// JSON tags match the Router wire shape verbatim; do not rename without
// changing the Router consumer in lockstep.
//
// Population: the required identity (name/mode/supports) is synthesised
// from env (MODEL_NAME / MODEL_MODE / MODEL_SUPPORTS) by loadSpec, with
// defaults for the rest. EngineArgs is seeded from ENGINE_ARGS on first
// boot. On boot ReconcileModelSpecFile persists that seed to
// Runtime.ModelSpecPath when no file exists, or reloads an existing file
// (disk wins). PUT /api/model-spec rewrites the file at runtime.
//
// EngineArgs is the model-card SSOT for inference-engine launch flags
// (same raw string shape as the ENGINE_ARGS env). Parsed Known/Unknown
// views live on Config.Engine.Args, not on the card. Non-LLM kinds
// (embed/clipembed/audio/ocr/rerank/systemone) require an empty string.
type ModelSpec struct {
	Name           string            `json:"name"`
	Mode           string            `json:"mode"`
	Label          map[string]string `json:"label,omitempty"`
	Description    map[string]string `json:"description,omitempty"`
	Supports       map[string]bool   `json:"supports,omitempty"`
	Pricing        map[string]string `json:"pricing,omitempty"`
	ParameterRules []ParameterRule   `json:"parameter_rules,omitempty"`
	// ContextSize is derived from EngineArgs whenever those flags pin a
	// window down, overwriting whatever was stored; see context_size.go.
	// A written value survives only for engines and configurations whose
	// flags leave the window to the model file.
	ContextSize   int `json:"context_size,omitempty"`
	MaxOutputToks int `json:"max_output_tokens,omitempty"`
	// EngineArgs omits omitempty so an intentional empty string still
	// writes the JSON key. Reconcile only backfills from ENGINE_ARGS when
	// the key is absent (legacy files); omitempty would delete the key
	// and resurrect chart env on every boot after Dashboard clear.
	EngineArgs string         `json:"engine_args"`
	Extensions map[string]any `json:"extensions,omitempty"`
}

// ParameterRule mirrors
// `router/backend/internal/providers/predefined/model.go::ParameterRule`.
// Default / Min / Max are kept loosely typed because Dify / LiteLLM YAML
// emit numeric values as int OR float depending on the rule, and
// Router's parser accepts both.
type ParameterRule struct {
	Name        string            `json:"name"`
	UseTemplate string            `json:"use_template,omitempty"`
	Label       map[string]string `json:"label,omitempty"`
	Help        map[string]string `json:"help,omitempty"`
	Type        string            `json:"type"`
	Required    bool              `json:"required,omitempty"`
	Default     any               `json:"default,omitempty"`
	Min         *float64          `json:"min,omitempty"`
	Max         *float64          `json:"max,omitempty"`
	Precision   int               `json:"precision,omitempty"`
	Options     []string          `json:"options,omitempty"`
}

// MIRROR OF: the first 32 (non-audio) entries in
// router/backend/internal/core/supports/supports.go::AllSupports.
// The remaining 16 Router entries live in allAudioSupports below.
//
// When Router rotates `AllSupports`, mirror the change here and bump the
// list version comment. Router publishes the list as an ordered JSON array
// at router/backend/internal/core/supports/all_supports.json, kept honest
// by a test on that side, so the comparison is mechanical even though
// nothing here can fetch it: Router is a private repository, and a CI job
// that reads it would need a credential this one does not have.
//
// A key Router withdraws leaves here too, and a card that still declares
// it gets the same treatment as any other unrecognized key: warned about,
// kept in the spec, forwarded. Router folds `supports_thinking` onto
// `supports_reasoning` when it reads the card, so an old card keeps
// working — but this list is what a card is written against, and offering
// a spelling nothing on either side prefers only produces more of them.
var allSupports = []string{
	// Core (8)
	"supports_vision",
	"supports_function_calling",
	"supports_parallel_function_calling",
	"supports_native_streaming",
	"supports_response_schema",
	"supports_reasoning",
	"supports_prompt_caching",
	"supports_web_search",
	// Multimodal (6)
	"supports_audio_input",
	"supports_audio_output",
	"supports_video_input",
	"supports_pdf_input",
	"supports_computer_use",
	"supports_url_context",
	// Reasoning + control tokens (4)
	"supports_reasoning_effort",
	"supports_assistant_prefill",
	"supports_tool_choice",
	"supports_tokenizer",
	// Sampling controls (7)
	"supports_system_messages",
	"supports_temperature",
	"supports_top_p",
	"supports_top_k",
	"supports_stop_sequences",
	"supports_frequency_penalty",
	"supports_presence_penalty",
	// Response shape (6)
	"supports_n",
	"supports_logprobs",
	"supports_seed",
	"supports_response_format",
	"supports_logit_bias",
	"supports_user",
	// Embedding (1). An embedding model that places images in the same
	// vector space as text (Jina CLIP, Cohere embed-*-image). Not
	// supports_vision, which is a chat model looking at an image.
	"supports_embedding_image_input",
}

// MIRROR OF:
// router/backend/internal/core/supports/supports.go::AllAudioSupports.
// Keep all 16 keys in exactly the same order. Nothing reads the Router
// source at test time: TestAllAudioSupports_RouterOrder pins the order
// against a literal in this repo, and TestDocsSupportsTableMatchesMirror
// pins the docs table against this slice. Drift from Router itself is
// caught by review, not by CI. Audio capabilities use supports_* like
// every other capability.
//
// The five synthesis keys belong to MODEL_MODE=tts, the rest to
// MODEL_MODE=audio. They share one list because they share one
// vocabulary and one engine image, not because they share a mode.
var allAudioSupports = []string{
	// Speech to text (3)
	"supports_stt",
	"supports_stt_stream",
	"supports_align",
	// Speaker (3)
	"supports_diar",
	"supports_diar_stream",
	"supports_speaker_embed",
	// Signal (2)
	"supports_vad",
	"supports_enhance",
	// Text to speech (5). tts is synthesis itself; tts_custom adds a
	// listable set of voices shipped with the weights; tts_clone takes a
	// reference clip; tts_design builds a voice from a description.
	supportsTTS,
	"supports_tts_clone",
	"supports_tts_design",
	"supports_tts_custom",
	"supports_tts_dialogue",
	// Audio-native LLM (3)
	"supports_audio_llm",
	"supports_audio_s2s",
	"supports_sound_fx",
}

// knownSupports is the whitelist: the core supports_* flags plus the audio ones.
var knownSupports = func() map[string]struct{} {
	m := make(map[string]struct{}, len(allSupports)+len(allAudioSupports))
	for _, k := range allSupports {
		m[k] = struct{}{}
	}
	for _, k := range allAudioSupports {
		m[k] = struct{}{}
	}
	return m
}()

// supports_reasoning is the canonical capability key surfaced from the
// MODEL_THINK_SUPPORTED scalar env when no JSON override is provided. Pulled
// to a constant so the conflict-warning + fallback paths cannot drift.
const supportsReasoningKey = "supports_reasoning"

// ParamRuleTypeString is the rule type for a parameter whose value is an
// enumerated string. The reasoning ladder is the one this repo authors
// itself, in config_load_model.go.
//
// Exported because the data plane checks requests against exactly these
// rules (internal/adapter/proxy/paramcheck.go), and a second spelling of
// the type would leave the validator and the check disagreeing about
// which rules exist.
const ParamRuleTypeString = "string"

// allowedParameterRuleTypes matches predefined.ParameterRuleType* in
// Router. The wire-rejected set is closed (small enum); we mirror it
// rather than accepting opaque strings because a typo here propagates
// into Router's UI as a non-renderable rule.
var allowedParameterRuleTypes = map[string]struct{}{
	"int":               {},
	"float":             {},
	ParamRuleTypeString: {},
	"boolean":           {},
	"tag":               {},
}

// pricingDecimalRE accepts the LiteLLM-style decimal-string-as-value form
// (`"0.0000015"` / `"0"` / `"-1"`). float64 would lose precision at the
// fractional-microcent level that matters for spend reconciliation; the
// Router's ParsePricing in router/backend/internal/billing/calc.go enforces
// the same shape.
var pricingDecimalRE = regexp.MustCompile(`^-?\d+(\.\d+)?$`)

// LoadModelSpecFile reads and validates the v1.1 packed
// model-spec.json file at the given path. Required: missing,
// unreadable, or malformed file is returned as a fail-fast error
// (config_invalid). Caller is responsible for the env-driven
// MODEL_NAME override (Load does this in loadSpec).
func LoadModelSpecFile(path string) (ModelSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return ModelSpec{}, fmt.Errorf("model-spec.json: %s: %w", path, err)
	}
	return ParseModelSpecBytes(data, path)
}

// WriteModelSpecFile atomically writes spec as indented JSON to path,
// creating the parent directory. Used to seed the file on first boot
// (ReconcileModelSpecFile) and by PUT /api/model-spec at runtime. Atomic
// (tmp + rename) so a concurrent reader never observes a partial file.
func WriteModelSpecFile(path string, spec ModelSpec) error {
	data, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return fmt.Errorf("model-spec.json: marshal: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("model-spec.json: mkdir %s: %w", filepath.Dir(path), err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("model-spec.json: write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("model-spec.json: rename %s: %w", path, err)
	}
	return nil
}

// modelSpecJSONHasEngineArgsKey reports whether raw model-spec.json
// contains a top-level "engine_args" key (including null / ""). Used to
// distinguish legacy files (missing key → backfill from env) from an
// intentional empty card value (key present → leave alone).
func modelSpecJSONHasEngineArgsKey(data []byte) bool {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(data, &probe); err != nil {
		return false
	}
	_, ok := probe["engine_args"]
	return ok
}

// ApplyEngineArgsFromSpec parses cfg.Spec.EngineArgs into cfg.Engine.Args
// for the configured Kind. Download-only (empty Kind) clears Args and
// ignores a non-empty card field. Call after Load/Reconcile/PUT so the
// parsed view matches the model card.
func ApplyEngineArgsFromSpec(cfg *Config) error {
	if cfg.Engine.Kind == "" {
		cfg.Engine.Args = EngineArgs{Known: map[string]string{}}
		return nil
	}
	args, err := ParseEngineArgs(cfg.Engine.Kind, cfg.Spec.EngineArgs)
	if err != nil {
		return fmt.Errorf("model-spec engine_args: %w", err)
	}
	cfg.Engine.Args = args
	return nil
}

// ReconcileModelSpecFile makes disk the source of truth for the model spec.
// If a file exists at cfg.Runtime.ModelSpecPath it is loaded, validated, and
// supersedes the env-seed (cfg.Spec is replaced and the derived Model fields
// re-applied). If absent, the env-seed cfg.Spec is written there so it
// persists across restarts and the dashboard editor has a target. This is
// side-effecting (reads/writes disk); call once at boot after config.Load.
//
// After reconcile, Engine.Args is re-derived from Spec.EngineArgs and the
// wrapper handoff file is written under RunDir.
func ReconcileModelSpecFile(cfg *Config) error {
	path := cfg.Runtime.ModelSpecPath
	if path == "" {
		return applySpecSideEffects(cfg)
	}
	switch _, err := os.Stat(path); {
	case err == nil:
		spec, lerr := LoadModelSpecFile(path)
		if lerr != nil {
			return lerr
		}
		rewrote := false
		// MODEL_REASONING_EFFORT is the chart's
		// declaration, and every app installed before it has a card that
		// predates the env.
		if overlayReasoningEffortRuleFromEnv(&spec) {
			rewrote = true
		}
		// One-shot backfill only when the on-disk JSON lacks the
		// engine_args key entirely (legacy files). An explicit empty
		// string (or null) means the operator cleared the card — do not
		// re-apply ENGINE_ARGS env on every boot.
		if raw, rerr := os.ReadFile(path); rerr == nil &&
			!modelSpecJSONHasEngineArgsKey(raw) &&
			cfg.Spec.EngineArgs != "" {
			spec.EngineArgs = cfg.Spec.EngineArgs
			rewrote = true
		}
		if rewrote {
			if werr := WriteModelSpecFile(path, spec); werr != nil {
				return werr
			}
		}
		warnEngineArgsSeedIgnored(cfg.Spec.EngineArgs, spec.EngineArgs, path)
		warnModelNameSeedIgnored(cfg.ModelName, spec.Name, path)
		applyModelSpec(cfg, spec)
		slog.Info("model-spec loaded from disk", "path", path, "mode", spec.Mode)
		return applySpecSideEffects(cfg)
	case os.IsNotExist(err):
		if werr := WriteModelSpecFile(path, cfg.Spec); werr != nil {
			return werr
		}
		slog.Info("model-spec seeded to disk from env", "path", path, "mode", cfg.Spec.Mode)
		return applySpecSideEffects(cfg)
	default:
		return fmt.Errorf("model-spec.json: stat %s: %w", path, err)
	}
}

// warnEngineArgsSeedIgnored reports a card on disk whose launch flags
// differ from the ones the chart just handed us. Disk winning is the
// intended rule — it is how a Dashboard edit survives a restart — but
// the card outlives the application: it sits on the shared cache PVC
// and is still there after a reinstall or a chart upgrade. So an
// operator who raises ENGINE_ARGS in the chart and redeploys gets the
// old flags with nothing in the logs saying why, and every consumer of
// the model card downstream believes the chart. This line is the one
// place that difference is visible.
func warnEngineArgsSeedIgnored(seed, onDisk, path string) {
	if seed == "" || seed == onDisk {
		return
	}
	slog.Warn("ENGINE_ARGS from the chart is ignored; the model card on disk wins",
		"path", path,
		"card_engine_args", onDisk,
		"chart_engine_args", seed,
		"hint", "PUT /api/model-spec to change the card, or delete it to re-seed from the chart")
}

// warnModelNameSeedIgnored reports a card whose model name differs from
// the one the chart just declared. Disk wins here too, and the
// consequence is larger than for the launch flags: this name is what
// Router registers and what clients call, so a chart that renames the
// model has renamed nothing until the card is replaced.
func warnModelNameSeedIgnored(seed, onDisk, path string) {
	if seed == "" || onDisk == "" || seed == onDisk {
		return
	}
	slog.Warn("MODEL_NAME from the chart is ignored; the model card on disk wins",
		"path", path,
		"card_name", onDisk,
		"chart_name", seed,
		"hint", "the card outlives the app on the shared cache PVC; delete it to re-seed from the chart")
}

// EnsureModeRequiredEngineArgs applies card/mode constraints to
// Spec.EngineArgs (currently: llamacpp + embedding → ensure --embedding).
// Chart upgrades leave a hostPath model-spec whose engine_args key is
// present but missing the flag; without this merge the handoff would
// override the engine Deployment env and start in chat mode.
// Returns true when Spec.EngineArgs was mutated.
func EnsureModeRequiredEngineArgs(cfg *Config) bool {
	return ensureLlamacppEmbeddingArgs(cfg)
}

func ensureLlamacppEmbeddingArgs(cfg *Config) bool {
	if cfg == nil || cfg.Engine.Kind != EngineLlamaCpp {
		return false
	}
	if cfg.Spec.Mode != string(ModelEmbedding) {
		return false
	}
	if strings.Contains(cfg.Spec.EngineArgs, "--embedding") {
		return false
	}
	cfg.Spec.EngineArgs = strings.TrimSpace(cfg.Spec.EngineArgs + " --embedding")
	return true
}

// NormalizeSpecForEngine brings cfg.Spec and cfg.Engine.Args into
// agreement with the running ENGINE_KIND: it merges the flags a mode
// requires, reparses the launch flags, and derives context_size from
// them. It reports whether the card itself changed, which is how the
// caller learns the copy on disk has gone stale.
//
// Boot and PUT /api/model-spec both go through here. They used not to:
// the handler did the merge and the reparse inline and skipped the
// context_size derivation, so a card edited to `--ctx-size 8192` kept
// advertising the old window to Router until the next restart, while the
// identical card arriving from disk at boot reported the new one.
func NormalizeSpecForEngine(cfg *Config) (bool, error) {
	changed := EnsureModeRequiredEngineArgs(cfg)
	if changed {
		slog.Info("model-spec engine_args merged mode-required --embedding",
			"path", cfg.Runtime.ModelSpecPath)
	}
	if err := ApplyEngineArgsFromSpec(cfg); err != nil {
		return changed, err
	}
	if syncContextSizeFromEngineArgs(cfg) {
		changed = true
		slog.Info("model-spec context_size derived from engine_args",
			"engine", cfg.Engine.Kind, "context_size", cfg.Spec.ContextSize)
	}
	warnKVUnifiedOversubscribed(cfg)
	return changed, nil
}

// applySpecSideEffects normalises the card, writes it back when that
// changed it, and publishes the wrapper handoff. Boot only.
// When the handoff bytes change at boot (embedding merge, backfill, or
// card edit that raced ahead of the engine), bump engine_restart so a
// supervising wrapper that already launched with stale args relaunches.
func applySpecSideEffects(cfg *Config) error {
	specChanged, err := NormalizeSpecForEngine(cfg)
	if err != nil {
		return err
	}
	if specChanged && cfg.Runtime.ModelSpecPath != "" {
		if err := WriteModelSpecFile(cfg.Runtime.ModelSpecPath, cfg.Spec); err != nil {
			return err
		}
	}
	prev, prevMissing := handoff.ReadEngineArgs(cfg.Runtime.RunDir)
	if err := handoff.WriteEngineArgs(cfg.Runtime.RunDir, cfg.Spec.EngineArgs); err != nil {
		return err
	}
	changed := prevMissing || prev != cfg.Spec.EngineArgs
	if changed && cfg.Engine.Kind != "" && cfg.Runtime.RunDir != "" {
		// Boot-time bump: no observation. The engine is not up yet (its
		// wrapper is still blocked on the download sentinel), so there
		// is no dip to watch for — nothing to confirm, only stale args
		// to correct before the first launch.
		if _, err := handoff.Bump(cfg.Runtime.RunDir); err != nil {
			return err
		}
		slog.Info("engine_restart bumped after engine_args handoff change",
			"run_dir", cfg.Runtime.RunDir,
			"prev_missing", prevMissing,
			"args_len", len(cfg.Spec.EngineArgs))
	}
	return nil
}

// ParseModelSpecBytes decodes a model-spec.json payload from raw bytes
// and validates required fields + value ranges. Pulled out so tests
// can hit the validator without writing to disk.
func ParseModelSpecBytes(data []byte, source string) (ModelSpec, error) {
	var spec ModelSpec
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&spec); err != nil {
		return ModelSpec{}, fmt.Errorf("model-spec.json (%s): invalid JSON: %w", source, err)
	}
	return validateModelSpec(spec, source)
}

// validateModelSpec checks required fields + value ranges and normalises
// the spec (tags unknown supports keys, inits the supports map). Shared by
// the file decoder (ParseModelSpecBytes) and the env synthesis path so both
// enforce the same Router-facing contract.
func validateModelSpec(spec ModelSpec, source string) (ModelSpec, error) {
	var errs []error
	if spec.Mode == "" {
		errs = append(errs, fmt.Errorf("model-spec.json (%s): mode is required (chat | embedding | audio | tts | ocr | rerank | translate | music_generation | system_one)", source))
	} else if spec.Mode != string(ModelChat) && spec.Mode != string(ModelEmbedding) &&
		spec.Mode != string(ModelAudio) && spec.Mode != string(ModelTTS) &&
		spec.Mode != string(ModelOCR) && spec.Mode != string(ModelRerank) && spec.Mode != string(ModelTranslate) &&
		spec.Mode != string(ModelMusicGeneration) && spec.Mode != string(ModelSystemOne) {
		errs = append(errs, fmt.Errorf(
			"model-spec.json (%s): mode %q must be \"chat\", \"embedding\", \"audio\", \"tts\", \"ocr\", \"rerank\", \"translate\", \"music_generation\" or \"system_one\"",
			source, spec.Mode))
	}
	if spec.ContextSize < 0 {
		errs = append(errs, fmt.Errorf("model-spec.json (%s): context_size: %d must be >= 0", source, spec.ContextSize))
	}
	if spec.MaxOutputToks < 0 {
		errs = append(errs, fmt.Errorf("model-spec.json (%s): max_output_tokens: %d must be >= 0", source, spec.MaxOutputToks))
	}
	for k, v := range spec.Pricing {
		if !pricingDecimalRE.MatchString(v) {
			errs = append(errs, fmt.Errorf("model-spec.json (%s): pricing[%q]: %q is not a decimal string", source, k, v))
		}
	}
	for i, r := range spec.ParameterRules {
		if r.Name == "" {
			errs = append(errs, fmt.Errorf("model-spec.json (%s): parameter_rules[%d]: name is required", source, i))
		}
		if r.Type == "" {
			errs = append(errs, fmt.Errorf("model-spec.json (%s): parameter_rules[%d]: type is required", source, i))
		} else if _, ok := allowedParameterRuleTypes[r.Type]; !ok {
			errs = append(errs, fmt.Errorf(
				"model-spec.json (%s): parameter_rules[%d]: type %q must be one of int|float|string|boolean|tag",
				source, i, r.Type))
		}
	}
	if err := validateTTSSpeedExtension(spec.Extensions, source); err != nil {
		errs = append(errs, err)
	}
	if err := validateSystemOneExtension(spec, source); err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return ModelSpec{}, joinErrors(errs)
	}

	if unknown := unknownSupports(spec.Supports); len(unknown) > 0 {
		if spec.Extensions == nil {
			spec.Extensions = map[string]any{}
		}
		spec.Extensions["_unknown_flag"] = unknown
		slog.Warn("model_spec: unknown supports keys",
			"keys", unknown,
			"hint", "mirror Router AllSupports if these are intended")
	}

	if spec.Supports == nil {
		spec.Supports = map[string]bool{}
	}
	return spec, nil
}

func validateSystemOneExtension(spec ModelSpec, source string) error {
	value, exists := spec.Extensions["system_one"]
	if spec.Mode != string(ModelSystemOne) && !exists {
		return nil
	}
	if spec.Mode == string(ModelSystemOne) && spec.ContextSize <= 0 {
		return fmt.Errorf("model-spec.json (%s): system_one context_size must be > 0", source)
	}
	if !exists {
		return fmt.Errorf("model-spec.json (%s): extensions.system_one is required for mode system_one", source)
	}
	extension, ok := value.(map[string]any)
	if !ok {
		return fmt.Errorf("model-spec.json (%s): extensions.system_one must be an object", source)
	}
	if _, ok := extension["default_eligible"].(bool); !ok {
		return fmt.Errorf("model-spec.json (%s): extensions.system_one.default_eligible must be a boolean", source)
	}
	if err := validateSystemOneLimits(extension, source); err != nil {
		return err
	}
	return validateSystemOneLanguages(extension, source)
}

func validateSystemOneLimits(extension map[string]any, source string) error {
	for _, limit := range []struct {
		key      string
		min, max int
	}{
		{key: "max_choice_options", min: 2, max: 255},
		{key: "max_score_levels", min: 2, max: 10},
	} {
		key := limit.key
		value, ok := extension[key]
		if !ok {
			return fmt.Errorf("model-spec.json (%s): extensions.system_one.%s is required", source, key)
		}
		number, ok := modelSpecNumber(value)
		if !ok || math.IsNaN(number) || math.IsInf(number, 0) || math.Trunc(number) != number || number < float64(limit.min) || number > float64(limit.max) {
			return fmt.Errorf("model-spec.json (%s): extensions.system_one.%s must be an integer in [%d, %d]", source, key, limit.min, limit.max)
		}
	}
	return nil
}

func validateSystemOneLanguages(extension map[string]any, source string) error {
	languagesValue, ok := extension["languages"]
	if !ok {
		return fmt.Errorf("model-spec.json (%s): extensions.system_one.languages is required", source)
	}
	var languages []string
	switch values := languagesValue.(type) {
	case []string:
		languages = values
	case []any:
		languages = make([]string, 0, len(values))
		for _, value := range values {
			language, ok := value.(string)
			if !ok {
				return fmt.Errorf("model-spec.json (%s): extensions.system_one.languages must contain only strings", source)
			}
			languages = append(languages, language)
		}
	default:
		return fmt.Errorf("model-spec.json (%s): extensions.system_one.languages must be an array", source)
	}
	if len(languages) == 0 {
		return fmt.Errorf("model-spec.json (%s): extensions.system_one.languages must not be empty", source)
	}
	seen := map[string]struct{}{}
	for _, language := range languages {
		normalized := strings.TrimSpace(language)
		if normalized == "" || normalized == "*" {
			return fmt.Errorf("model-spec.json (%s): extensions.system_one.languages entries must be non-empty and cannot be wildcard %q", source, language)
		}
		if normalized != language {
			return fmt.Errorf("model-spec.json (%s): extensions.system_one.languages entry %q must not have surrounding whitespace", source, language)
		}
		if _, exists := seen[normalized]; exists {
			return fmt.Errorf("model-spec.json (%s): extensions.system_one.languages contains duplicate %q", source, language)
		}
		seen[normalized] = struct{}{}
	}
	return nil
}

func validateTTSSpeedExtension(extensions map[string]any, source string) error {
	tts, ok := extensions["tts"]
	if !ok {
		return nil
	}
	ttsObject, ok := tts.(map[string]any)
	if !ok {
		return fmt.Errorf("model-spec.json (%s): extensions.tts must be an object", source)
	}
	settings, ok := ttsObject["voice_settings"]
	if !ok {
		return nil
	}
	settingsObject, ok := settings.(map[string]any)
	if !ok {
		return fmt.Errorf("model-spec.json (%s): extensions.tts.voice_settings must be an object", source)
	}
	speed, ok := settingsObject["speed"]
	if !ok {
		return nil
	}
	speedObject, ok := speed.(map[string]any)
	if !ok {
		return fmt.Errorf("model-spec.json (%s): extensions.tts.voice_settings.speed must be an object", source)
	}
	values := make([]float64, 3)
	for i, key := range []string{"min", "default", "max"} {
		value, exists := speedObject[key]
		if !exists {
			return fmt.Errorf("model-spec.json (%s): extensions.tts.voice_settings.speed.%s is required", source, key)
		}
		number, ok := modelSpecNumber(value)
		if !ok || math.IsNaN(number) || math.IsInf(number, 0) || number <= 0 {
			return fmt.Errorf("model-spec.json (%s): extensions.tts.voice_settings.speed.%s must be a finite positive number", source, key)
		}
		values[i] = number
	}
	if values[0] > values[1] || values[1] > values[2] {
		return fmt.Errorf("model-spec.json (%s): extensions.tts.voice_settings.speed must satisfy min <= default <= max", source)
	}
	return nil
}

func modelSpecNumber(value any) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case float32:
		return float64(number), true
	case int:
		return float64(number), true
	case int64:
		return float64(number), true
	case json.Number:
		parsed, err := number.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

// unknownSupports returns the supports keys not in the local mirror of
// Router's known capabilities, sorted for deterministic test output.
func unknownSupports(m map[string]bool) []string {
	if len(m) == 0 {
		return nil
	}
	var out []string
	for k := range m {
		if _, ok := knownSupports[k]; !ok {
			out = append(out, k)
		}
	}
	if len(out) == 0 {
		return nil
	}
	// Insertion sort: tiny input (<32 keys), avoids pulling sort just for this.
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}
