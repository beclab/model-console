#!/bin/sh
# OCR layout-service wrapper (paddle-hybrid).
#
# Waits for llm-init sentinel + /run/llm-init/extra_model_path, exports the
# first ROLE=extra path as MODEL_PATH (what ocr-layout reads) and
# LAYOUT_MODEL_DIR (L7 alias), then runs the layout binary under
# supervise_engine so POST /api/engine/restart can relaunch a wedged engine.
# ENGINE_ARGS must be empty for this kind, so there is no handoff to wait for.
#
# Required: MODEL_SOURCE must include a ROLE=extra HF/URL segment that lands
# an ONNX directory (e.g. PP-DocLayoutV3). A segment marked `--role mmproj`
# does not count — the projector never goes into extra_model_path.
#
# Optional env:
#   LAYOUT_BIN          -- default /usr/local/bin/ocr-layout (or $PATH)
#   LAYOUT_LISTEN       -- passed through to ocr-layout (default 0.0.0.0:8090)
#   EXTRA_MODEL_PATH_FILE / WAIT_* -- see common.sh

# shellcheck disable=SC2034
ENGINE=ocrlayout
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
wait_for_extra_model_path
LAYOUT_DIR=$(read_extra_model_path_first_line)

if [ -f "$LAYOUT_DIR" ]; then
    # URL ROLE=extra may land on a file inside the ONNX dir (e.g. config.json)
    # when integration tests skip re-download via MODEL_SOURCE_LOCAL.
    parent=$(dirname "$LAYOUT_DIR")
    log "extra path is a file; using parent directory $parent"
    LAYOUT_DIR=$parent
fi

if [ ! -d "$LAYOUT_DIR" ]; then
    log "LAYOUT_MODEL_DIR=$LAYOUT_DIR is not a directory"
    exit 1
fi

export LAYOUT_MODEL_DIR="$LAYOUT_DIR"
# OCRLayout (ocr-layout) requires MODEL_PATH = model *directory*.
export MODEL_PATH="$LAYOUT_DIR"
log "LAYOUT_MODEL_DIR=$LAYOUT_MODEL_DIR MODEL_PATH=$MODEL_PATH"

LAYOUT_BIN="${LAYOUT_BIN:-}"
if [ -z "$LAYOUT_BIN" ]; then
    if [ -x /usr/local/bin/ocr-layout ]; then
        LAYOUT_BIN=/usr/local/bin/ocr-layout
    else
        LAYOUT_BIN="$(command -v ocr-layout 2>/dev/null || true)"
    fi
fi
if [ -z "$LAYOUT_BIN" ] || [ ! -x "$LAYOUT_BIN" ]; then
    log "ocr-layout binary not found (checked /usr/local/bin/ocr-layout, \$PATH, \$LAYOUT_BIN)"
    exit 127
fi

# Default listen matches OCRLayout + OCRAdapter OCR_LAYOUT_URL (port 8090).
export LAYOUT_LISTEN="${LAYOUT_LISTEN:-0.0.0.0:8090}"

run_ocrlayout() {
    log "starting layout-service bin=$LAYOUT_BIN LAYOUT_LISTEN=$LAYOUT_LISTEN"
    exec "$LAYOUT_BIN" "$@"
}

supervise_engine run_ocrlayout "$@"
