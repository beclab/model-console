#!/bin/sh
# Audio engine wrapper.
#
# Waits for llm-init's download sentinel, then runs the audio image's own
# Python wrapper under supervise_engine so POST /api/engine/restart can
# relaunch an engine that has wedged.
#
# Nothing here reads model_path or waits for the engine_args handoff: the
# audio bases resolve their own model directory, and Model Console's
# ENGINE_ARGS must be empty for this kind (config.ParseEngineArgs rejects
# a non-empty value), so the handoff is always empty and there is no card
# argument to wait for. The ENGINE_ARGS the chart sets on THIS container
# is the audio engine's own setting (wrapper/contract.py reads it) and the
# empty handoff is what leaves it intact.

# shellcheck disable=SC2034
ENGINE=audio
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

run_audio() {
    log "starting audio engine (audio-python -m wrapper.app)"
    exec audio-python -m wrapper.app "$@"
}

supervise_engine run_audio "$@"
