package config

import (
	"errors"
	"strings"
	"testing"
)

// Each retired v1.0 llm-init env must fail-fast with a migration hint,
// even when the rest of the v1.1 surface is otherwise valid.
func TestLoad_RemovedEnvsFailFast(t *testing.T) {
	for _, re := range removedEnvs {
		re := re
		t.Run(re.key, func(t *testing.T) {
			_, err := Load(mapEnv(minimalEnv(map[string]string{re.key: "x"})))
			if err == nil {
				t.Fatalf("Load: want fail-fast for removed env %s, got nil", re.key)
			}
			var ce *ConfigError
			if !errors.As(err, &ce) {
				t.Fatalf("err = %T, want *ConfigError", err)
			}
			var saw bool
			for _, fe := range ce.Fields {
				if fe.Field == re.key {
					saw = true
				}
			}
			if !saw {
				t.Errorf("error %q lacks a FieldError for %s", err, re.key)
			}
		})
	}
}

// Standard third-party env names llm-init shares its environment with
// must NOT be rejected, since huggingface_hub / the ollama daemon read
// them legitimately.
func TestLoad_SharedThirdPartyEnvsIgnored(t *testing.T) {
	for _, key := range []string{"HF_HUB_CACHE", "HF_HOME_BASE", "OLLAMA_KEEP_ALIVE", "OLLAMA_DATA_ROOT"} {
		if _, err := Load(mapEnv(minimalEnv(map[string]string{key: "x"}))); err != nil {
			t.Errorf("Load with %s set: unexpected error %v", key, err)
		}
	}
}

// A removed env set alongside missing required envs surfaces both the
// migration hint and the normal required-field errors.
func TestLoad_RemovedEnvAlongsideMissingRequired(t *testing.T) {
	env := minimalEnv(map[string]string{"HF_REPO": "old/repo"})
	delete(env, "MODEL_SOURCE")
	_, err := Load(mapEnv(env))
	if err == nil {
		t.Fatal("Load: want error")
	}
	if !strings.Contains(err.Error(), "HF_REPO") {
		t.Errorf("error %q missing HF_REPO migration hint", err)
	}
}

// HF_MMPROJ_FILE's replacement is a MODEL_SOURCE segment carrying the
// inline --role mmproj flag, not a plain extra source: an extra lands in
// extra_model_path, which wrappers read as an engine-facing artefact.
func TestLoad_HFMmprojMigrationHintNamesRoleFlag(t *testing.T) {
	_, err := Load(mapEnv(minimalEnv(map[string]string{
		removedHFMMProjFileEnv: "mmproj.gguf",
	})))
	if err == nil {
		t.Fatal("expected HF_MMPROJ_FILE removal error")
	}
	for _, want := range []string{removedHFMMProjFileEnv, "MODEL_SOURCE", "--role mmproj"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error = %q, want migration guidance %q", err, want)
		}
	}
	if strings.Contains(err.Error(), "MODEL_SOURCE_NUM") {
		t.Errorf("error = %q must not point at the removed indexed contract", err)
	}
}
