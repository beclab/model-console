#!/bin/sh
# Shared helpers for engine wrappers. Source this from each engine
# wrapper to inherit the wait-on-sentinel + read-model-path +
# ENGINE_ARGS parsing logic so we do not maintain four near-identical
# copies. Pure POSIX sh -- runs in the engine container without bash.
#
# Implements engine waiting, restart supervision, and ENGINE_ARGS parsing.

set -eu

SENTINEL=${SENTINEL:-/run/llm-init/model_download_finish}
MODEL_PATH_FILE=${MODEL_PATH_FILE:-/run/llm-init/model_path}
# ROLE=extra paths (one absolute path per line). ocrlayout reads line 1.
EXTRA_MODEL_PATH_FILE=${EXTRA_MODEL_PATH_FILE:-/run/llm-init/extra_model_path}
# Model Console writes the model-card engine_args here (SSOT). When present,
# it overrides the chart ENGINE_ARGS env so wrappers follow the card.
ENGINE_ARGS_FILE=${ENGINE_ARGS_FILE:-/run/llm-init/engine_args}
# Model Console bumps this file's contents (generation string) to ask the
# wrapper to stop the engine child and relaunch with the latest handoff.
ENGINE_RESTART_FILE=${ENGINE_RESTART_FILE:-/run/llm-init/engine_restart}
WAIT_INTERVAL=${WAIT_INTERVAL:-5}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
# How long to wait for Model Console to write ENGINE_ARGS_FILE after the
# sentinel (or at ollama start). On timeout the wrapper falls back to the
# chart ENGINE_ARGS env.
ENGINE_ARGS_WAIT_TIMEOUT=${ENGINE_ARGS_WAIT_TIMEOUT:-60}
RESTART_POLL_INTERVAL=${RESTART_POLL_INTERVAL:-2}
# How long to wait after SIGTERM before SIGKILL on stop/restart.
ENGINE_STOP_GRACE_SECONDS=${ENGINE_STOP_GRACE_SECONDS:-15}

# log_<engine> "msg..." -- every wrapper sets ENGINE before sourcing so the
# log line is greppable across all four containers.
log() {
    printf '[wrapper-%s] %s\n' "${ENGINE:-engine}" "$*"
}

# wait_for_sentinel blocks until the sentinel file is present or
# WAIT_TIMEOUT seconds have elapsed. Polls every WAIT_INTERVAL seconds.
wait_for_sentinel() {
    elapsed=0
    while [ ! -f "$SENTINEL" ]; do
        if [ "$elapsed" -ge "$WAIT_TIMEOUT" ]; then
            log "timed out after ${WAIT_TIMEOUT}s waiting for $SENTINEL"
            exit 1
        fi
        log "waiting for $SENTINEL (elapsed=${elapsed}s)"
        sleep "$WAIT_INTERVAL"
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
    log "sentinel $SENTINEL found after ${elapsed}s"
}

# read_model_path echoes the contents of MODEL_PATH_FILE with the
# trailing newline stripped. The file is written by lifecycle.writeSentinel
# (sentinel.go) and is the single source of truth for "which file/dir the
# engine should load".
read_model_path() {
    if [ ! -f "$MODEL_PATH_FILE" ]; then
        log "model path file $MODEL_PATH_FILE not found"
        exit 1
    fi
    tr -d '\r\n' < "$MODEL_PATH_FILE"
}

# wait_for_extra_model_path blocks until EXTRA_MODEL_PATH_FILE exists.
# ensure() writes model_download_finish before extra_model_path, so a
# consumer that only waits on the sentinel can race; layout must wait here.
wait_for_extra_model_path() {
    elapsed=0
    while [ ! -f "$EXTRA_MODEL_PATH_FILE" ]; do
        if [ "$elapsed" -ge "$WAIT_TIMEOUT" ]; then
            log "timed out after ${WAIT_TIMEOUT}s waiting for $EXTRA_MODEL_PATH_FILE"
            exit 1
        fi
        log "waiting for $EXTRA_MODEL_PATH_FILE (elapsed=${elapsed}s)"
        sleep "$WAIT_INTERVAL"
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
    log "extra_model_path $EXTRA_MODEL_PATH_FILE found after ${elapsed}s"
}

# read_extra_model_path_first_line echoes the first non-empty line of
# EXTRA_MODEL_PATH_FILE (ROLE=extra only). Empty/missing first line exits 1.
read_extra_model_path_first_line() {
    if [ ! -f "$EXTRA_MODEL_PATH_FILE" ]; then
        log "extra model path file $EXTRA_MODEL_PATH_FILE not found"
        exit 1
    fi
    # Prefer head -n1 when available; fall back to sed for minimal images.
    line=$(head -n 1 "$EXTRA_MODEL_PATH_FILE" 2>/dev/null || sed -n '1p' "$EXTRA_MODEL_PATH_FILE")
    line=$(printf '%s' "$line" | tr -d '\r')
    if [ -z "$line" ]; then
        log "extra model path file $EXTRA_MODEL_PATH_FILE has empty first line"
        exit 1
    fi
    printf '%s' "$line"
}

# wait_for_engine_args_handoff polls until ENGINE_ARGS_FILE exists or
# ENGINE_ARGS_WAIT_TIMEOUT elapses. New llm-init writes the handoff during
# boot reconcile; waiting avoids launching on chart env while the card is
# still being seeded. Timeout is non-fatal (older llm-init / race).
wait_for_engine_args_handoff() {
    elapsed=0
    while [ ! -f "$ENGINE_ARGS_FILE" ]; do
        if [ "$elapsed" -ge "$ENGINE_ARGS_WAIT_TIMEOUT" ]; then
            log "timed out after ${ENGINE_ARGS_WAIT_TIMEOUT}s waiting for $ENGINE_ARGS_FILE; using chart ENGINE_ARGS"
            return 0
        fi
        log "waiting for $ENGINE_ARGS_FILE (elapsed=${elapsed}s)"
        sleep "$WAIT_INTERVAL"
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
    log "engine_args handoff $ENGINE_ARGS_FILE found after ${elapsed}s"
}

# load_engine_args_from_handoff prefers ${ENGINE_ARGS_FILE} written by
# Model Console from the model card. Non-empty contents replace ENGINE_ARGS.
# Empty/whitespace handoff (OCR/embed seed: llm-init must not own Args) keeps
# the chart ENGINE_ARGS env. Absent file also keeps chart env (older llm-init).
load_engine_args_from_handoff() {
    if [ -f "$ENGINE_ARGS_FILE" ]; then
        # tr strips a trailing newline.
        _handoff=$(tr -d '\r\n' < "$ENGINE_ARGS_FILE")
        if [ -n "$_handoff" ]; then
            ENGINE_ARGS=$_handoff
            export ENGINE_ARGS
            log "loaded ENGINE_ARGS from $ENGINE_ARGS_FILE (${#ENGINE_ARGS} bytes)"
        else
            # ENGINE_ARGS may be unset entirely: the embed / clipembed /
            # ocr charts do not declare it at all, and ${#ENGINE_ARGS} on
            # an unset name aborts under set -u. The audio charts DO set
            # it, for their own engine to read -- keeping it is the point
            # of this branch.
            _cur=${ENGINE_ARGS:-}
            log "engine_args handoff empty; keeping chart ENGINE_ARGS (${#_cur} bytes)"
            unset _cur
        fi
        unset _handoff
    fi
}

read_restart_gen() {
    if [ -f "$ENGINE_RESTART_FILE" ]; then
        tr -d '\r\n' < "$ENGINE_RESTART_FILE"
    else
        printf '0'
    fi
}

# stop_engine_child sends SIGTERM to the child's process group (fallback:
# the PID), waits ENGINE_STOP_GRACE_SECONDS, then SIGKILL.
#
# The group kill is the only thing that reaches a multi-process engine's
# workers (vLLM / SGLang), and in a Pod it is exactly the thing that does
# not work -- see the measured caveat in supervise_engine. So the PID this
# is given has to be the engine itself, which is why every run_<engine>
# function execs its binary: without that exec the PID is an intermediate
# subshell, killing it leaves the engine reparented to PID 1 still holding
# the listening port, and the relaunch cannot bind. Observed, not theorised:
# a restart left the first embed_server alive alongside the second.
stop_engine_child() {
    child=$1
    if [ -z "$child" ]; then
        return 0
    fi
    if ! kill -0 "$child" 2>/dev/null; then
        wait "$child" 2>/dev/null || true
        return 0
    fi
    log "stopping engine pid=$child (SIGTERM to process group)"
    # Negative PID = process group (job control via set -m below).
    kill -TERM -"$child" 2>/dev/null || kill -TERM "$child" 2>/dev/null || true
    elapsed=0
    while kill -0 "$child" 2>/dev/null; do
        if [ "$elapsed" -ge "$ENGINE_STOP_GRACE_SECONDS" ]; then
            log "engine pid=$child still alive after ${ENGINE_STOP_GRACE_SECONDS}s; SIGKILL"
            kill -KILL -"$child" 2>/dev/null || kill -KILL "$child" 2>/dev/null || true
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    wait "$child" 2>/dev/null || true
}

# supervise_engine CMD [args...] runs CMD as a child process-group leader
# and restarts it when Model Console bumps ENGINE_RESTART_FILE. Forwards
# SIGTERM/SIGINT to the child so kubelet/docker stop drains the engine
# instead of only killing the shell. Reloads ENGINE_ARGS from the handoff
# file before each launch.
supervise_engine() {
    # shellcheck disable=SC2034  # read by trap handlers
    _supervise_child=""
    _supervise_exiting=0

    _supervise_on_signal() {
        _supervise_exiting=1
        log "received stop signal; forwarding to engine"
        if [ -n "${_supervise_child}" ]; then
            stop_engine_child "$_supervise_child"
        fi
        exit 143
    }
    trap '_supervise_on_signal' TERM INT

    gen=$(read_restart_gen)
    while [ "$_supervise_exiting" -eq 0 ]; do
        load_engine_args_from_handoff
        log "launching engine (restart_gen=$gen)"
        # Job control makes the backgrounded job a process-group leader
        # (PID == PGID) so kill -TERM -$child works. Do NOT use
        # `setsid CMD`: CMD is typically a shell function (run_vllm /
        # run_llamacpp / …) and setsid can only execvp PATH binaries —
        # on Linux engine images that fails with "not found" and the
        # wrapper CrashLoops.
        #
        # Measured caveat: dash and busybox ash only put the job in its
        # own process group when the shell has a controlling terminal,
        # which a Pod does not have. `set -m` still succeeds, so on those
        # images the group kill in stop_engine_child fails with ESRCH and
        # the single-PID fallback is what actually runs. bash does it
        # without a tty.
        set -m 2>/dev/null || true
        "$@" &
        _supervise_child=$!
        child=$_supervise_child
        restarted=0
        while kill -0 "$child" 2>/dev/null; do
            sleep "$RESTART_POLL_INTERVAL"
            if [ "$_supervise_exiting" -eq 1 ]; then
                break
            fi
            new=$(read_restart_gen)
            if [ "$new" != "$gen" ]; then
                log "restart requested (gen $gen -> $new); stopping pid=$child"
                stop_engine_child "$child"
                _supervise_child=""
                gen=$new
                restarted=1
                break
            fi
        done
        if [ "$_supervise_exiting" -eq 1 ]; then
            break
        fi
        if [ "$restarted" -eq 1 ]; then
            # Settle so the previous listener releases the port (GPU
            # engines can take longer than one second; poll below).
            settle=0
            while [ "$settle" -lt 10 ]; do
                sleep 1
                settle=$((settle + 1))
                # Best-effort: if the old PID is gone, port is likely free.
                if ! kill -0 "$child" 2>/dev/null; then
                    break
                fi
            done
            continue
        fi
        # set -e would abort here on a non-zero status, losing the log line
        # that says why the container is going away.
        status=0
        wait "$child" || status=$?
        _supervise_child=""
        log "engine exited status=$status (no restart requested)"
        exit "$status"
    done
}

# engine_args_export_env transforms each KEY=VALUE token in $ENGINE_ARGS
# into an exported env var. Useful for engines whose only configuration
# surface is daemon-level env (Ollama). Non-KEY=VALUE tokens are logged
# and skipped, NOT failed -- callers may layer additional handling.
#
# Caveat: POSIX sh has no shlex; this uses default IFS word-splitting,
# so values containing whitespace must be wrapped in quotes that survive
# the env-var round-trip (rare; OLLAMA_* values are typically scalars).
engine_args_export_env() {
    load_engine_args_from_handoff
    args="${ENGINE_ARGS:-}"
    [ -z "$args" ] && return 0
    # shellcheck disable=SC2086
    set -- $args
    for tok in "$@"; do
        case "$tok" in
            *=*) export "$tok" && log "engine_args export: $tok" ;;
            *)   log "engine_args drop non-KV token: $tok" ;;
        esac
    done
}
