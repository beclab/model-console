#!/bin/sh
# SGLang engine wrapper.
#
# Waits for sentinel + model_path, then launches SGLang under
# supervise_engine so Model Console can request a relaunch after
# model-card engine_args changes.

# shellcheck disable=SC2034
ENGINE=sglang
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

run_sglang() {
    load_engine_args_from_handoff
    log "starting SGLang with model-path=$MODEL_PATH name=${MODEL_NAME:-unset} args=${ENGINE_ARGS:-<empty>}"
    # shellcheck disable=SC2086
    exec python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --served-model-name "${MODEL_NAME:-llm-init-model}" \
        --host 0.0.0.0 \
        --port "${SGLANG_PORT:-30000}" \
        --enable-metrics \
        ${ENGINE_ARGS:-} \
        "$@"
}

supervise_engine run_sglang "$@"
