package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/llm-init/llm-init/internal/reasoning"
	"github.com/llm-init/llm-init/internal/translate"
)

// loadModel reads only MODEL_NAME in v1.1. Model.Type and
// Model.ThinkSupported are derived from the env-seed spec in loadSpec.
// Model.Dir is synthesised from Sources[main] in loadSources.
func (l *loader) loadModel() {
	v, err := requireString(l.g, "MODEL_NAME")
	if err != nil {
		l.collect(err)
		return
	}
	v, err = parseModelName("MODEL_NAME", v)
	if err != nil {
		l.collect(err)
		return
	}
	l.c.ModelName = v
	l.c.Model.Name = v
}

// loadSources reads MODEL_SOURCE via LoadModelSources and populates
// cfg.Sources. Per-source dispatch is done by lifecycle.
func (l *loader) loadSources() {
	if strings.TrimSpace(l.g(removedModelSourceNumEnv)) != "" {
		return
	}
	sources, err := LoadModelSources(l.g)
	if err != nil {
		l.collect(err)
		return
	}
	l.c.Sources = sources

	if main := mainSource(sources); main != nil && main.Kind == KindURL && main.LocalPath != "" {
		l.c.Model.Dir = parentDir(main.LocalPath)
	}
}

// loadHFDeploy reads the deployment-level HF env (endpoint + token).
// Both are optional; HFEndpoint defaults to https://huggingface.co.
//
// HF_ENDPOINT / HF_TOKEN only matter when a source actually downloads
// from Hugging Face, so this loader is gated on the presence of an hf://
// source. ollama:// / url-only deployments leave HFEndpoint / HFToken
// zero and never validate HF_ENDPOINT.
func (l *loader) loadHFDeploy() {
	if !hasHFSource(l.c.Sources) {
		return
	}
	if v, err := parseHTTPURL("HF_ENDPOINT",
		stringDefault(l.g, "HF_ENDPOINT", defaultHFEndpoint)); err != nil {
		l.collect(err)
	} else {
		l.c.HFEndpoint = v
	}
	tok := l.g("HF_TOKEN")
	l.c.HFToken = tok
	l.c.HFTokenSet = strings.TrimSpace(tok) != ""
}

// hasHFSource reports whether any source downloads from Hugging Face.
func hasHFSource(sources []ModelSource) bool {
	for i := range sources {
		if sources[i].Kind == KindHF {
			return true
		}
	}
	return false
}

// loadSpec synthesises the env-seed ModelSpec from the required identity
// envs (MODEL_NAME / MODEL_MODE / MODEL_SUPPORTS) plus defaults for every
// other field. This seed is what gets persisted to Runtime.ModelSpecPath on
// first boot; an existing file on disk supersedes it (see
// ReconcileModelSpecFile). Empty MODEL_NAME from loadModel is OK here
// (loadModel collects its own error already).
func (l *loader) loadSpec() {
	var mode string
	if v, err := requireString(l.g, "MODEL_MODE"); err != nil {
		l.collect(err)
	} else if m, err := parseEnum("MODEL_MODE", v, allowedModelType); err != nil {
		l.collect(err)
	} else {
		mode = m
	}

	supports := parseSupportsCSV(l.g("MODEL_SUPPORTS"))

	// Seed extensions.translate from TRANSLATE_LANGUAGES / TRANSLATE_PAIRS
	// so first-boot model-spec.json (and Router GET /api/model-spec) carry
	// the chart-declared catalog. Disk still wins after Reconcile.
	var extensions map[string]any
	if seed := translate.SeedExtensionFromEnv(l.g); seed != nil {
		extensions = seed
	}
	if speed, err := seedTTSSpeedFromEnv(l.g); err != nil {
		l.collect(err)
	} else if speed != nil {
		if extensions == nil {
			extensions = map[string]any{}
		}
		extensions["tts"] = map[string]any{
			"voice_settings": map[string]any{"speed": speed},
		}
	}

	contextSize := 0
	if mode == string(ModelSystemOne) {
		limits, err := seedSystemOneLimitsFromEnv(l.g)
		if err != nil {
			l.collect(err)
		} else {
			contextSize = limits.contextSize
			if extensions == nil {
				extensions = map[string]any{}
			}
			extensions["system_one"] = map[string]any{
				"max_choice_options": limits.maxChoiceOptions,
				"max_score_levels":   limits.maxScoreLevels,
				"languages":          limits.languages,
				"default_eligible":   limits.defaultEligible,
			}
		}
	}

	// Seed parameter_rules with the reasoning ladder the chart declared.
	// Which levels exist is a joint fact of the model and the engine that
	// only this side knows, and Router projects it rather than inventing
	// it, so a deployment that has one has to say so.
	var rules []ParameterRule
	rule, rerr := reasoningEffortRuleFromEnv(l.g)
	if rerr != nil {
		l.collect(rerr)
	} else if rule != nil {
		rules = append(rules, *rule)
	}

	spec, err := validateModelSpec(ModelSpec{
		Name:           l.c.ModelName,
		Mode:           mode,
		Supports:       supports,
		ParameterRules: rules,
		EngineArgs:     strings.TrimSpace(l.g("ENGINE_ARGS")),
		ContextSize:    contextSize,
		Extensions:     extensions,
	}, "env (MODEL_MODE / MODEL_SUPPORTS)")
	if err != nil {
		l.collect(err)
		return
	}
	applyModelSpec(&l.c, spec)
	// Derive the parsed Engine.Args view from the card seed. Disk reconcile
	// in main may replace Spec.EngineArgs and re-apply.
	if err := ApplyEngineArgsFromSpec(&l.c); err != nil {
		l.collect(err)
	}
}

const (
	systemOneContextSizeEnv      = "SYSTEM_ONE_CONTEXT_SIZE"
	systemOneMaxChoiceOptionsEnv = "SYSTEM_ONE_MAX_CHOICE_OPTIONS"
	systemOneMaxScoreLevelsEnv   = "SYSTEM_ONE_MAX_SCORE_LEVELS"
	systemOneLanguagesEnv        = "SYSTEM_ONE_LANGUAGES"
	systemOneDefaultEligibleEnv  = "SYSTEM_ONE_DEFAULT_ELIGIBLE"
)

type systemOneSeedLimits struct {
	contextSize      int
	maxChoiceOptions int
	maxScoreLevels   int
	languages        []string
	defaultEligible  bool
}

func seedSystemOneLimitsFromEnv(g Getenv) (systemOneSeedLimits, error) {
	required := func(key string) (string, error) { return requireString(g, key) }
	contextRaw, err := required(systemOneContextSizeEnv)
	if err != nil {
		return systemOneSeedLimits{}, err
	}
	contextSize, err := parseInt(systemOneContextSizeEnv, contextRaw, 0)
	if err != nil {
		return systemOneSeedLimits{}, err
	}
	if contextSize <= 0 {
		return systemOneSeedLimits{}, fmt.Errorf("%s: %d must be > 0", systemOneContextSizeEnv, contextSize)
	}

	choiceRaw, err := required(systemOneMaxChoiceOptionsEnv)
	if err != nil {
		return systemOneSeedLimits{}, err
	}
	maxChoiceOptions, err := parseIntInRange(systemOneMaxChoiceOptionsEnv, choiceRaw, 0, 2, 255)
	if err != nil {
		return systemOneSeedLimits{}, err
	}

	scoreRaw, err := required(systemOneMaxScoreLevelsEnv)
	if err != nil {
		return systemOneSeedLimits{}, err
	}
	maxScoreLevels, err := parseIntInRange(systemOneMaxScoreLevelsEnv, scoreRaw, 0, 2, 10)
	if err != nil {
		return systemOneSeedLimits{}, err
	}

	languagesRaw, err := required(systemOneLanguagesEnv)
	if err != nil {
		return systemOneSeedLimits{}, err
	}
	defaultEligibleRaw, err := required(systemOneDefaultEligibleEnv)
	if err != nil {
		return systemOneSeedLimits{}, err
	}
	defaultEligible, err := parseBool(systemOneDefaultEligibleEnv, defaultEligibleRaw, false)
	if err != nil {
		return systemOneSeedLimits{}, err
	}
	seen := map[string]struct{}{}
	languages := make([]string, 0)
	for _, raw := range strings.Split(languagesRaw, ",") {
		language := strings.TrimSpace(raw)
		if language == "" {
			return systemOneSeedLimits{}, fmt.Errorf("%s: languages must be non-empty comma-separated values", systemOneLanguagesEnv)
		}
		if language == "*" {
			return systemOneSeedLimits{}, fmt.Errorf("%s: wildcard language %q is not allowed", systemOneLanguagesEnv, language)
		}
		if _, ok := seen[language]; ok {
			continue
		}
		seen[language] = struct{}{}
		languages = append(languages, language)
	}

	return systemOneSeedLimits{
		contextSize:      contextSize,
		maxChoiceOptions: maxChoiceOptions,
		maxScoreLevels:   maxScoreLevels,
		languages:        languages,
		defaultEligible:  defaultEligible,
	}, nil
}

const (
	ttsSpeedMinEnv     = "TTS_SPEED_MIN"
	ttsSpeedDefaultEnv = "TTS_SPEED_DEFAULT"
	ttsSpeedMaxEnv     = "TTS_SPEED_MAX"
)

func seedTTSSpeedFromEnv(g Getenv) (map[string]any, error) {
	raw := []string{
		strings.TrimSpace(g(ttsSpeedMinEnv)),
		strings.TrimSpace(g(ttsSpeedDefaultEnv)),
		strings.TrimSpace(g(ttsSpeedMaxEnv)),
	}
	if raw[0] == "" && raw[1] == "" && raw[2] == "" {
		return nil, nil
	}
	for i, value := range raw {
		if value == "" {
			return nil, fmt.Errorf("%s, %s and %s must be set together", ttsSpeedMinEnv, ttsSpeedDefaultEnv, ttsSpeedMaxEnv)
		}
		n, err := parseFloat([]string{ttsSpeedMinEnv, ttsSpeedDefaultEnv, ttsSpeedMaxEnv}[i], value, 0)
		if err != nil {
			return nil, err
		}
		raw[i] = fmt.Sprint(n)
	}
	min, _ := parseFloat(ttsSpeedMinEnv, raw[0], 0)
	def, _ := parseFloat(ttsSpeedDefaultEnv, raw[1], 0)
	max, _ := parseFloat(ttsSpeedMaxEnv, raw[2], 0)
	return map[string]any{"min": min, "default": def, "max": max}, nil
}

// Env names for the reasoning ladder. The options env is what makes the
// rule exist at all; the default env only describes which level the
// engine already uses when a client sends nothing, since nothing in this
// chain injects a rule's default into a request.
const (
	reasoningEffortEnv        = "MODEL_REASONING_EFFORT"
	reasoningEffortDefaultEnv = "MODEL_REASONING_EFFORT_DEFAULT"
)

// reasoningEffortRuleFromEnv builds the card's reasoning_effort rule from
// MODEL_REASONING_EFFORT (CSV of levels) plus the optional
// MODEL_REASONING_EFFORT_DEFAULT. Returns (nil, nil) when the options env
// is unset: a deployment that declares no ladder gets no rule, which is
// different from declaring an empty one.
//
// Unknown levels fail-fast, which an unknown MODEL_SUPPORTS key no longer
// does. A level the engine's chat template does not know is not a harmless
// extra option: Qwen3.8's template raises on anything outside its three, so
// the request errors out for whichever client trusted the card. An
// unrecognized capability key only overstates what the model can do, and
// every consumer of it already handles a key it has not heard of.
func reasoningEffortRuleFromEnv(g Getenv) (*ParameterRule, error) {
	raw := strings.TrimSpace(g(reasoningEffortEnv))
	def := reasoning.Normalize(g(reasoningEffortDefaultEnv))
	if raw == "" {
		if def != "" {
			return nil, fmt.Errorf(
				"%s: set without %s (a default level means nothing without the ladder it belongs to)",
				reasoningEffortDefaultEnv, reasoningEffortEnv)
		}
		return nil, nil
	}

	seen := map[string]struct{}{}
	var unknown []string
	for _, tok := range strings.Split(raw, ",") {
		lvl := reasoning.Normalize(tok)
		if lvl == "" {
			continue
		}
		if !reasoning.IsLevel(lvl) {
			unknown = append(unknown, lvl)
			continue
		}
		seen[lvl] = struct{}{}
	}
	if len(unknown) > 0 {
		return nil, fmt.Errorf(
			"%s: unknown level(s) %v (must be from %v; `thinking` is Router's literal for a model whose levels nobody published, not one an engine accepts)",
			reasoningEffortEnv, unknown, reasoning.Ladder)
	}
	if len(seen) == 0 {
		return nil, fmt.Errorf("%s: set but lists no level", reasoningEffortEnv)
	}

	// Ladder order, not the order they were typed: the option list is
	// rendered as a menu, and alphabetical would put high before low.
	options := make([]string, 0, len(seen))
	for _, lvl := range reasoning.Ladder {
		if _, ok := seen[lvl]; ok {
			options = append(options, lvl)
		}
	}

	rule := ParameterRule{
		Name:    reasoning.Param,
		Type:    ParamRuleTypeString,
		Options: options,
	}
	if def != "" {
		if _, ok := seen[def]; !ok {
			return nil, fmt.Errorf("%s: %q is not one of the declared levels %v",
				reasoningEffortDefaultEnv, def, options)
		}
		rule.Default = def
	}
	return &rule, nil
}

// overlayReasoningEffortRuleFromEnv makes the chart authoritative for the
// reasoning_effort rule on an existing card, leaving every other rule
// alone. Returns true when spec was mutated.
//
// Disk normally wins; the reasoning ladder is a deliberate exception because
// it is declared by the deployment. Every already-installed app has a card on
// a hostPath that predates the env, so a seed-only path would apply to new
// installs only. The cost is that an admin edit to this one rule through Router
// is reverted at the next restart; the env is where that decision belongs, and
// it is editable there.
func overlayReasoningEffortRuleFromEnv(spec *ModelSpec) bool {
	rule, err := reasoningEffortRuleFromEnv(os.Getenv)
	if err != nil || rule == nil {
		return false
	}
	for i := range spec.ParameterRules {
		if spec.ParameterRules[i].Name != rule.Name {
			continue
		}
		// Keep whatever label / help the card carries: those are prose
		// the env cannot express, and dropping them every boot would
		// make the rule less useful than it was.
		candidate := *rule
		candidate.Label = spec.ParameterRules[i].Label
		candidate.Help = spec.ParameterRules[i].Help
		prev, _ := json.Marshal(spec.ParameterRules[i])
		next, _ := json.Marshal(candidate)
		if bytes.Equal(prev, next) {
			return false
		}
		spec.ParameterRules[i] = candidate
		return true
	}
	spec.ParameterRules = append(spec.ParameterRules, *rule)
	return true
}

// applyModelSpec stores the resolved spec on the Config and derives the
// runtime fields that hang off it: Model.Name (the alias clients call and
// the data plane rewrites to), Model.Type (Router-facing classification)
// and Model.ThinkSupported (reasoning gate). Shared by loadSpec and
// ReconcileModelSpecFile so the disk-wins path re-derives consistently.
//
// The name has to be derived here, not left at MODEL_NAME. The card on
// disk is authoritative and outlives the application — it sits on the
// shared cache PVC — so a chart that changes MODEL_NAME meets a card
// still carrying the old one. Router registers what the card says; the
// data plane advertised and rewrote to what the env said, and
// ollama-native answered 404 for the name Router was routing.
func applyModelSpec(c *Config, spec ModelSpec) {
	c.Spec = spec
	if spec.Name != "" {
		c.Model.Name = spec.Name
	}
	if spec.Mode != "" {
		c.Model.Type = ModelType(spec.Mode)
	}
	if v, ok := spec.Supports[supportsReasoningKey]; ok {
		c.Model.ThinkSupported = v
	}
}

// parseSupportsCSV decodes the MODEL_SUPPORTS comma-separated enabled-key
// list into the supports map. Unlisted keys are false (absent).
//
// A key this mirror does not recognize is kept, and validateModelSpec tags
// it exactly as it tags an unrecognized key read off a card. The env used
// to fail fast instead, which put the chain's one policy difference on its
// least inspectable segment: an engine base's chart hands a manifest's
// MODEL_SUPPORTS through verbatim, so a typo in a published OlaresManifest
// -- or a key Router ratified before this mirror shipped -- stopped the
// container from booting, and said why only in its own log.
func parseSupportsCSV(raw string) map[string]bool {
	out := map[string]bool{}
	for _, tok := range strings.Split(raw, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		out[tok] = true
	}
	return out
}

// PrimarySource returns a pointer to the deployment's main ModelSource
// (the one with Role=="main", falling back to the first entry when no
// role was set). Returns nil for an empty Sources slice.
func (c Config) PrimarySource() *ModelSource {
	return mainSource(c.Sources)
}

// PrimarySourceKind returns the Kind of the primary source, or the
// empty SourceKind when Sources is empty (Load() guarantees at least
// one source on success, so callers using a successfully-loaded Config
// can treat the empty value as "boot config never validated").
func (c Config) PrimarySourceKind() SourceKind {
	if ms := c.PrimarySource(); ms != nil {
		return ms.Kind
	}
	return ""
}

// PrimaryOllamaTag returns the upstream Ollama daemon tag selected by the main
// source. Extras are preload-only and never become the inference target.
func (c Config) PrimaryOllamaTag() string {
	if main := c.PrimarySource(); main != nil &&
		main.Kind == KindOllama && main.OllamaTag != "" {
		return main.OllamaTag
	}
	if c.ModelName != "" {
		return c.ModelName
	}
	return c.Model.Name
}

// mainSource returns a pointer to the entry with Role==main, falling
// back to the first entry when no role was set (single-source case).
// The returned pointer indexes into the original slice.
func mainSource(sources []ModelSource) *ModelSource {
	for i := range sources {
		if sources[i].Role == RoleMain {
			return &sources[i]
		}
	}
	if len(sources) > 0 {
		return &sources[0]
	}
	return nil
}

// parentDir returns the directory portion of an absolute path or the
// empty string for an empty input. Avoids pulling path/filepath into
// the loader for one call site.
func parentDir(p string) string {
	if i := strings.LastIndexByte(p, '/'); i >= 0 {
		return p[:i]
	}
	return ""
}
