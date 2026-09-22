#!/bin/sh
# Ollama wrapper.
#
# Ollama's deployment model is the inverse of vLLM/SGLang/llama.cpp:
# the daemon owns the model bytes, llm-init drives /api/pull and
# /api/create. Therefore there is no sentinel to wait on -- Ollama
# starts immediately and llm-init connects to it.
#
# ENGINE_ARGS is a KEY=VALUE list of OLLAMA_* daemon env. Runs under
# supervise_engine so Model Console can relaunch after model-card edits.

# shellcheck disable=SC2034
ENGINE=ollama
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

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"

# No download sentinel for Ollama; still wait for model-card handoff so we
# do not briefly serve with chart ENGINE_ARGS while llm-init reconciles.
wait_for_engine_args_handoff

run_ollama() {
    engine_args_export_env
    log "starting ollama serve on $OLLAMA_HOST"
    exec ollama serve "$@"
}

supervise_engine run_ollama "$@"
