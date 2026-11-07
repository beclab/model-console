// Package comfyconsole is llm-init's comfyui mode: the Model Console of a
// single-workflow ComfyUI app.
//
// It reads one app config shared with the Easy UI, downloads the models the
// workflow needs into ComfyUI's model tree (resumable, sha256-verified),
// probes ComfyUI's GET /system_stats, and reports both through
// GET /api/status. The app is ready only when every model is verified and
// ComfyUI answers. Nothing here touches the LLM lifecycle: the mode is
// entered through `llm-init comfyui` and shares only internal/fetch.
package comfyconsole

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path"
	"strings"
)

// AppConfig is the single source of truth for the app: the workflow the
// Easy UI submits and the models it needs. The console reads only Models;
// Workflow is kept raw so the Easy UI owns its schema.
type AppConfig struct {
	Workflow json.RawMessage `json:"workflow"`
	Models   []Model         `json:"models"`
}

// Model is one weight file. Dir is the ComfyUI model category directory
// (checkpoints, vae, clip, diffusion_models, ...) relative to the models
// root.
type Model struct {
	Name   string `json:"name"`
	Dir    string `json:"dir"`
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// RelPath is the model's path under the models root.
func (m Model) RelPath() string { return path.Join(m.Dir, m.Name) }

// LoadConfig reads and validates the app config. The file is JSON so the
// Easy UI can fetch the same bytes without a YAML parser.
func LoadConfig(file string) (*AppConfig, error) {
	raw, err := os.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("read app config: %w", err)
	}
	var cfg AppConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("parse app config %s: %w", file, err)
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("app config %s: %w", file, err)
	}
	return &cfg, nil
}

func (c *AppConfig) validate() error {
	if len(c.Models) == 0 {
		return errors.New("models is empty")
	}
	seen := make(map[string]bool, len(c.Models))
	for i, m := range c.Models {
		if m.Name == "" || strings.ContainsAny(m.Name, `/\`) || m.Name == "." || m.Name == ".." {
			return fmt.Errorf("models[%d]: invalid name %q", i, m.Name)
		}
		if m.Dir == "" || path.IsAbs(m.Dir) || path.Clean(m.Dir) != m.Dir || strings.HasPrefix(m.Dir, "..") {
			return fmt.Errorf("models[%d] %s: dir must be a clean relative path, got %q", i, m.Name, m.Dir)
		}
		u, err := url.Parse(m.URL)
		if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
			return fmt.Errorf("models[%d] %s: invalid url %q", i, m.Name, m.URL)
		}
		if len(m.SHA256) != 64 {
			return fmt.Errorf("models[%d] %s: sha256 must be 64 hex chars", i, m.Name)
		}
		if m.Size <= 0 {
			return fmt.Errorf("models[%d] %s: size must be positive", i, m.Name)
		}
		if seen[m.RelPath()] {
			return fmt.Errorf("models[%d]: duplicate path %s", i, m.RelPath())
		}
		seen[m.RelPath()] = true
	}
	return nil
}

// UseMirror points every huggingface.co model URL at a Hugging Face mirror
// (HF_ENDPOINT, e.g. Olares' OLARES_SYSTEM_HUGGINGFACE_SERVICE), keeping the
// /<repo>/resolve/<rev>/<path> part. It returns how many URLs changed. An
// empty endpoint, or huggingface.co itself, changes nothing.
func (c *AppConfig) UseMirror(endpoint string) (int, error) {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return 0, nil
	}
	base, err := url.Parse(endpoint)
	if err != nil || (base.Scheme != "http" && base.Scheme != "https") || base.Host == "" {
		return 0, fmt.Errorf("HF_ENDPOINT %q is not an http(s) URL", endpoint)
	}
	if base.Host == "huggingface.co" {
		return 0, nil
	}
	n := 0
	for i, m := range c.Models {
		u, err := url.Parse(m.URL)
		if err != nil || u.Host != "huggingface.co" {
			continue
		}
		u.Scheme, u.Host = base.Scheme, base.Host
		u.Path = strings.TrimRight(base.Path, "/") + u.Path
		c.Models[i].URL = u.String()
		n++
	}
	return n, nil
}
