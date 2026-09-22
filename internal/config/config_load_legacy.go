package config

import (
	"fmt"
	"strings"
)

// removedEnv pairs a retired env name with the operator-facing migration
// hint shown when it is still set. Most were retired by the v1.1.0
// "unified configuration cutover"; MODEL_SOURCE_NUM went with the
// comma-only MODEL_SOURCE cutover.
type removedEnv struct {
	key  string
	hint string
}

const (
	removedModelSourceNumEnv = "MODEL_SOURCE_NUM"
	removedHFMMProjFileEnv   = "HF_MMPROJ_FILE"
)

// hintUseEngineArgs is the shared migration hint for the v1.0 inference
// tuning envs (CONTEXT_LENGTH / REPEAT_PENALTY / REPEAT_LAST_N) that now
// flow through ENGINE_ARGS in engine-native syntax.
const hintUseEngineArgs = "set it via ENGINE_ARGS in engine-native syntax"

// removedEnvs lists llm-init-specific envs that are fail-fast. Silently
// ignoring them would leave an operator with a model that is mis-sourced
// or mis-described in ways they cannot detect from the logs, so we reject
// the boot and name the replacement.
//
// Deliberately EXCLUDED are standard third-party env names that
// llm-init shares its process / deployment environment with and must
// not claim:
//   - HF_HUB_CACHE, HF_HOME_BASE  -- read by huggingface_hub (the HF
//     download subprocess) to locate the shared cache.
//   - OLLAMA_KEEP_ALIVE, OLLAMA_DATA_ROOT -- owned by the ollama
//     daemon container.
//
// Those are simply ignored by llm-init (per CHANGELOG: only documented
// envs are read), since rejecting them would break valid deployments.
var removedEnvs = []removedEnv{
	{removedModelSourceNumEnv, "use comma-separated MODEL_SOURCE"},
	{"HF_REPO", "use MODEL_SOURCE=hf://<repo>"},
	{"HF_FILE", "use MODEL_SOURCE=hf://<repo> --include <file>"},
	{"HF_REVISION", "pin a revision inside MODEL_SOURCE=hf://<repo>@<rev>"},
	{removedHFMMProjFileEnv, "add a MODEL_SOURCE segment for the projector and mark it with --role mmproj"},
	{"MODEL_URL", "use MODEL_SOURCE=https://<url>"},
	{"EXPECTED_SHA256", "use MODEL_SOURCE=https://<url>#sha256=<sum>"},
	{"OLLAMA_MODEL", "use MODEL_SOURCE=ollama://<name>"},
	{"MODEL_DIR", "the model cache is now a deployment-managed mount"},
	{"ENGINE_URL", "the engine URL is derived from ENGINE_KIND"},
	{"CONTEXT_LENGTH", hintUseEngineArgs},
	{"REPEAT_PENALTY", hintUseEngineArgs},
	{"REPEAT_LAST_N", hintUseEngineArgs},
	{"GGUF_TEMPLATE_NAME", "templating now lives in the model / ENGINE_ARGS"},
	{"GGUF_TEMPLATE", "templating now lives in the model / ENGINE_ARGS"},
	{"GGUF_PARAMS", "sampling params now live in ENGINE_ARGS"},
	{"GGUF_SYSTEM", "the system prompt now lives in the model"},
	{"MODEL_TYPE", "use MODEL_MODE (chat|embedding)"},
	{"MODEL_THINK_SUPPORTED", "use MODEL_SUPPORTS (CSV of supports_* keys)"},
	{"MODEL_SPEC_JSON", "write the spec to MODEL_SPEC_PATH (model-spec.json), not an inline env"},
	{"ENGINE_GPU_MEMORY_UTILIZATION", "set --gpu-memory-utilization via ENGINE_ARGS"},
	{"ENGINE_CPU_OFFLOAD_GB", "set --cpu-offload-gb via ENGINE_ARGS"},
	{"ENGINE_LLAMACPP_FIT", "set llama.cpp flags via ENGINE_ARGS"},
	{"ENGINE_LLAMACPP_NGL", "set -ngl via ENGINE_ARGS"},
	{"VERIFY_INTERVAL", "ensure no longer runs on a timer; it runs on boot, on POST /api/retry and when the health loop finds the model de-registered"},
}

// loadRemovedEnvs fails fast when any retired llm-init env is still set,
// so a stale deployment surfaces a migration hint instead of booting with
// a silently-ignored knob. Runs first in Load() so the hint appears even
// when required envs are also absent.
func (l *loader) loadRemovedEnvs() {
	for _, re := range removedEnvs {
		if strings.TrimSpace(l.g(re.key)) != "" {
			l.collect(fmt.Errorf("%s: removed; %s", re.key, re.hint))
		}
	}
}
