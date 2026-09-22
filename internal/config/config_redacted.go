package config

// Redacted returns a JSON-serialisable view of c with secrets masked,
// matching the /api/config schema (sources / engine.args /
// model_spec_file). Secrets that appear in source URLs (userinfo) are
// redacted by ParseOneSource at Load time; HF_TOKEN never round-trips
// (only the boolean HFTokenSet does).
func (c Config) Redacted() ConfigRedacted {
	out := ConfigRedacted{
		Engine: EngineRedacted{
			Kind:           c.Engine.Kind,
			URL:            c.Engine.URL,
			Args:           c.Engine.Args,
			MaxConcurrency: c.Engine.MaxConcurrency,
		},
		ModelName:     c.ModelName,
		ModelSpec:     c.Spec,
		ModelSpecFile: c.Runtime.ModelSpecPath,
		HFEndpoint:    c.HFEndpoint,
		HFTokenSet:    c.HFTokenSet,
		Runtime:       c.Runtime,
		Log:           c.Log,
	}
	if len(c.Sources) > 0 {
		out.Sources = make([]ModelSourceRedacted, 0, len(c.Sources))
		for _, ms := range c.Sources {
			out.Sources = append(out.Sources, ModelSourceRedacted{
				Index:          ms.Index,
				Role:           ms.Role,
				Kind:           ms.Kind,
				ViaOllama:      ms.ViaOllama(),
				RedactedSource: ms.RedactedSource,
				LocalPath:      ms.LocalPath,
			})
		}
	}
	return out
}

// ConfigRedacted is the JSON-marshal target for /api/config. Field
// names match the ConfigRedacted response.
type ConfigRedacted struct {
	Engine        EngineRedacted        `json:"engine"`
	Sources       []ModelSourceRedacted `json:"sources"`
	ModelName     string                `json:"model_name,omitempty"`
	ModelSpec     ModelSpec             `json:"model_spec"`
	ModelSpecFile string                `json:"model_spec_file"`
	HFEndpoint    string                `json:"hf_endpoint,omitempty"`
	HFTokenSet    bool                  `json:"hf_token_set"`
	Runtime       Runtime               `json:"runtime"`
	Log           Log                   `json:"log"`
}

// EngineRedacted mirrors ConfigRedacted.engine. Args is
// included verbatim - ENGINE_ARGS values may contain operator typos
// but never carry secrets (HF_TOKEN is a separate env).
type EngineRedacted struct {
	Kind           EngineKind `json:"kind"`
	URL            string     `json:"url"`
	Args           EngineArgs `json:"args"`
	MaxConcurrency int        `json:"max_concurrency,omitempty"`
}

// ModelSourceRedacted mirrors the public ModelSource shape. Internal fields
// (HFRepo / HFInclude / etc.) are intentionally omitted; the dashboard
// renders RedactedSource as the authoritative human form.
type ModelSourceRedacted struct {
	Index          int        `json:"index"`
	Role           string     `json:"role,omitempty"`
	Kind           SourceKind `json:"kind"`
	ViaOllama      bool       `json:"via_ollama"`
	RedactedSource string     `json:"redacted_source,omitempty"`
	LocalPath      string     `json:"local_path,omitempty"`
}
