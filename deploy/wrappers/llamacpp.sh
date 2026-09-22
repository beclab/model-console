#!/bin/sh
# llama.cpp server wrapper.
#
# Waits for llm-init to publish the sentinel + model_path, then launches
# llama-server.
#
# v1.1.0 ENGINE_ARGS contract:
#   llama.cpp accepts BOTH cmdline flags (`-c 8192 -ngl all`) AND a
#   parallel set of LLAMA_ARG_* env-vars (`LLAMA_ARG_CTX_SIZE=8192
#   LLAMA_ARG_N_GPU_LAYERS=all`). This wrapper auto-detects each
#   ENGINE_ARGS token and routes it to the matching mode:
#     -- Token starts with "-"                   -> appended to argv
#     -- Token matches LLAMA_ARG_*=val           -> exported env
#     -- Token matches some-other-NAME=val       -> warning, dropped
#     -- Else (positional value, e.g. "all")     -> appended to argv
#   Mixing both modes works but is discouraged for maintainability.
#
# Pre-v1.1.0 the wrapper hard-coded `--fit off -ngl all` to enforce a
# "100% on GPU" residency contract paired with ENGINE_LLAMACPP_FIT/NGL
# env mirrors on llm-init. v1.1.0 removed those mirrors: GPU residency is
# now read from n_gpu_layers inside ENGINE_ARGS, plus llama.cpp's own
# /metrics KV-cache occupancy (no nvidia-smi); operators put
# `-ngl all -fa on` in ENGINE_ARGS themselves when they want full-GPU.
#
# Required env:
#   MODEL_NAME -- served-model-name advertised on /v1/models.
# Optional env:
#   LLAMACPP_PORT -- default 8081 (8080 is taken by llm-init's control plane).
#   LLAMA_SERVER_BIN -- override binary path (default /app/llama-server
#                       or first match in $PATH).

# shellcheck disable=SC2034
ENGINE=llamacpp
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

# Resolve MODEL_PATH to a single GGUF file. llm-init's writeSentinel()
# records cfg.Sources[main].LocalPath which for HF can be the snapshot
# directory; llama.cpp's `-m` argument expects a single .gguf file.
# Split-GGUFs (Q5_K_M-00001-of-00007.gguf, ...) sort 00001 first
# lexically and llama-server auto-loads the rest.
if [ -d "$MODEL_PATH" ]; then
    GGUF_FILE=""
    for f in "$MODEL_PATH"/*.gguf; do
        if [ -f "$f" ]; then
            GGUF_FILE="$f"
            break
        fi
    done
    if [ -z "$GGUF_FILE" ]; then
        log "no .gguf file found under $MODEL_PATH (sentinel records the directory; llama.cpp needs a single file)"
        exit 1
    fi
    log "resolved MODEL_PATH directory $MODEL_PATH -> file $GGUF_FILE"
    MODEL_PATH="$GGUF_FILE"
fi

# Resolve llama-server binary. Order: env override -> /app/llama-server
# (current upstream image convention) -> $PATH lookup -> hard fail.
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-}"
if [ -z "$LLAMA_SERVER_BIN" ]; then
    if [ -x /app/llama-server ]; then
        LLAMA_SERVER_BIN=/app/llama-server
    else
        LLAMA_SERVER_BIN="$(command -v llama-server 2>/dev/null || true)"
    fi
fi
if [ -z "$LLAMA_SERVER_BIN" ] || [ ! -x "$LLAMA_SERVER_BIN" ]; then
    log "llama-server binary not found (checked /app/llama-server, \$PATH, \$LLAMA_SERVER_BIN); upstream image layout may have changed"
    exit 127
fi

# llama-server links against libllama-common.so.0 colocated under /app.
# Our wrapper replaces the image ENTRYPOINT with this shell script, so
# the loader no longer gets the upstream's resolution; prepend the
# binary's install dir (and optional lib/ subdir) to LD_LIBRARY_PATH.
LLAMA_SERVER_DIR=$(dirname "$LLAMA_SERVER_BIN")
export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [ -d "${LLAMA_SERVER_DIR}/lib" ]; then
    export LD_LIBRARY_PATH="${LLAMA_SERVER_DIR}/lib:${LD_LIBRARY_PATH}"
fi

run_llamacpp() {
    # Preserve container CMD extras before ENGINE_ARGS word-splitting.
    extra_argv="$*"
    load_engine_args_from_handoff
    # Auto-detect ENGINE_ARGS dual-track: each token is routed to either
    # argv (cmdline flags + positional values) or env (LLAMA_ARG_*=val).
    ARGV=""
    if [ -n "${ENGINE_ARGS:-}" ]; then
        # shellcheck disable=SC2086
        set -- $ENGINE_ARGS
        for tok in "$@"; do
            case "$tok" in
                -*)
                    ARGV="$ARGV $tok"
                    ;;
                LLAMA_ARG_*=*)
                    export "$tok"
                    log "engine_args export: $tok"
                    ;;
                *=*)
                    log "engine_args drop non-LLAMA_ARG env: $tok"
                    ;;
                *)
                    ARGV="$ARGV $tok"
                    ;;
            esac
        done
    fi

    log "starting llama.cpp server with -m=$MODEL_PATH alias=${MODEL_NAME:-unset} argv=${ARGV:-<empty>} (bin=$LLAMA_SERVER_BIN)"
    # --metrics turns on llama.cpp's Prometheus /metrics (off by default).
    # shellcheck disable=SC2086
    exec "$LLAMA_SERVER_BIN" \
        -m "$MODEL_PATH" \
        --alias "${MODEL_NAME:-llm-init-model}" \
        --host 0.0.0.0 \
        --port "${LLAMACPP_PORT:-8081}" \
        --metrics \
        $ARGV \
        $extra_argv
}

supervise_engine run_llamacpp "$@"
