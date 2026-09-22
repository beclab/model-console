#!/bin/sh
# vLLM engine wrapper.
#
# Waits for llm-init to publish the sentinel + model_path files, then
# launches vLLM's OpenAI-compatible API server with --model pointed at
# whatever path llm-init wrote. Runs under supervise_engine so Model
# Console can bump ${RUN_DIR}/engine_restart to relaunch with updated
# model-card engine_args (file handoff preferred over ENGINE_ARGS env).
#
# ENGINE_ARGS contract:
#   cmdline-style string in vLLM's native flag syntax, word-split into argv.

# shellcheck disable=SC2034
ENGINE=vllm
# Source the shared helpers: the sibling copy by default, or the one
# llm-init publishes into RUN_DIR when WRAPPER_COMMON says so. That copy
# can arrive late — the two containers are separate Deployments with no
# ordering between them — so waiting here is the normal path.
# Keep this block byte-identical across wrappers (tests/wrappers checks).
WRAPPER_COMMON=${WRAPPER_COMMON:-"$(dirname "$0")/common.sh"}
_wc=0
while [ ! -f "$WRAPPER_COMMON" ] && [ "$_wc" -lt "${WRAPPER_COMMON_WAIT:-3600}" ]; do
    printf '[wrapper] waiting for %s (elapsed=%ss)\n' "$WRAPPER_COMMON" "$_wc"
    sleep 5
    _wc=$((_wc + 5))
done
[ -f "$WRAPPER_COMMON" ] || { printf '[wrapper] %s never appeared\n' "$WRAPPER_COMMON" >&2; exit 1; }
unset _wc
# shellcheck source=common.sh
. "$WRAPPER_COMMON"

wait_for_sentinel
wait_for_engine_args_handoff
MODEL_PATH=$(read_model_path)

run_vllm() {
    load_engine_args_from_handoff
    log "starting vLLM with model=$MODEL_PATH name=${MODEL_NAME:-unset} args=${ENGINE_ARGS:-<empty>}"
    # vLLM's OpenAI server exposes Prometheus /metrics by default — no flag
    # needed. llm-init's /api/diag/gpu reads vllm:cache_config_info and
    # vllm:kv_cache_usage_perc from it, so do NOT pass --disable-log-stats
    # in ENGINE_ARGS or those metrics disappear.
    # shellcheck disable=SC2086
    exec python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "${MODEL_NAME:-llm-init-model}" \
        --host 0.0.0.0 \
        --port "${VLLM_PORT:-8000}" \
        ${ENGINE_ARGS:-} \
        "$@"
}

supervise_engine run_vllm "$@"
