#!/bin/sh
# Behaviour of supervise_engine in deploy/wrappers/common.sh, driven with a
# fake engine: a restart request relaunches, a stop signal is forwarded and
# reported as 143, an engine that exits on its own passes its status through,
# and nothing relaunches when no restart was asked for.
set -eu

# shellcheck disable=SC1007  # CDPATH= is a deliberate empty assignment
repo_root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
common="$repo_root/deploy/wrappers/common.sh"

tmp=$(mktemp -d)
driver_pid=""

# A failing assertion leaves a supervisor and its fake engine running.
cleanup() {
    if [ -n "$driver_pid" ]; then
        kill -TERM "$driver_pid" 2>/dev/null || true
    fi
    rm -rf "$tmp"
}
trap cleanup EXIT HUP INT TERM

cat >"$tmp/driver.sh" <<'EOF'
#!/bin/sh
ENGINE=test
# shellcheck source=/dev/null
. "$COMMON_SH"
run_fake() {
    # exec, like every real run_<engine>: see the pid assertion below.
    exec "$FAKE_ENGINE"
}
supervise_engine run_fake
EOF

# A long-running engine that records each launch and each SIGTERM.
cat >"$tmp/engine-live" <<'EOF'
#!/bin/sh
printf 'launch\n' >>"$LAUNCH_LOG"
printf '%s\n' "$$" >>"$PID_LOG"
trap 'printf "term\n" >>"$TERM_LOG"; exit 143' TERM
while :; do
    sleep 1
done
EOF

# An engine that exits by itself with a distinctive status.
cat >"$tmp/engine-exit" <<'EOF'
#!/bin/sh
printf 'launch\n' >>"$LAUNCH_LOG"
printf '%s\n' "$$" >>"$PID_LOG"
exit 42
EOF

chmod +x "$tmp/driver.sh" "$tmp/engine-live" "$tmp/engine-exit"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

lines() {
    if [ -f "$1" ]; then
        wc -l <"$1" | tr -d ' '
    else
        printf '0'
    fi
}

# await_lines FILE N WHAT -- polls until FILE has N lines, or fails.
await_lines() {
    waited=0
    while [ "$(lines "$1")" -lt "$2" ]; do
        if [ "$waited" -ge 30 ]; then
            fail "timed out waiting for $3 (have $(lines "$1") of $2 lines)"
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

start_driver() {
    engine=$1
    rm -f "$tmp/launches" "$tmp/terms" "$tmp/restart" "$tmp/pids"
    env \
        COMMON_SH="$common" \
        FAKE_ENGINE="$tmp/$engine" \
        LAUNCH_LOG="$tmp/launches" \
        PID_LOG="$tmp/pids" \
        TERM_LOG="$tmp/terms" \
        ENGINE_ARGS_FILE="$tmp/engine_args" \
        ENGINE_RESTART_FILE="$tmp/restart" \
        RESTART_POLL_INTERVAL=1 \
        ENGINE_STOP_GRACE_SECONDS=5 \
        sh "$tmp/driver.sh" >"$tmp/driver.log" 2>&1 &
    driver_pid=$!
}

# 1. A restart request relaunches the engine.
start_driver engine-live
await_lines "$tmp/launches" 1 'the first launch'
first_engine_pid=$(sed -n '1p' "$tmp/pids")
printf '1\n' >"$tmp/restart"
await_lines "$tmp/launches" 2 'the relaunch after bumping engine_restart'
await_lines "$tmp/terms" 1 'the first engine to be stopped'

# 1a. The supervised pid must be the engine itself. If run_<engine> does not
#     exec, the pid is an intermediate subshell: the group kill would still
#     reach the engine on a shell that puts the job in its own process
#     group, and dash in a Pod does not, so the engine survives its own
#     restart holding the listening port and the relaunch cannot bind.
grep -q "stopping engine pid=$first_engine_pid " "$tmp/driver.log" \
    || fail "$(printf 'supervisor stopped a process that is not the engine (engine pid=%s):\n%s' \
        "$first_engine_pid" "$(grep 'stopping engine' "$tmp/driver.log")")"
if kill -0 "$first_engine_pid" 2>/dev/null; then
    fail "engine pid=$first_engine_pid survived its own restart"
fi

# 2. A stop signal is forwarded to the engine and reported as 143.
kill -TERM "$driver_pid"
status=0
wait "$driver_pid" || status=$?
[ "$status" -eq 143 ] || fail "supervisor exited $status after SIGTERM, want 143"
await_lines "$tmp/terms" 2 'the running engine to receive SIGTERM'
grep -q 'received stop signal' "$tmp/driver.log" \
    || fail "supervisor did not log the stop signal: $(cat "$tmp/driver.log")"

# 3. An engine that exits on its own passes its status through, exactly once.
start_driver engine-exit
status=0
wait "$driver_pid" || status=$?
[ "$status" -eq 42 ] || fail "supervisor exited $status, want the engine's 42"
got=$(lines "$tmp/launches")
[ "$got" -eq 1 ] || fail "engine was launched $got times without a restart request, want 1"
grep -q 'engine exited status=42 (no restart requested)' "$tmp/driver.log" \
    || fail "supervisor did not report the exit: $(cat "$tmp/driver.log")"

# 4. ENGINE_ARGS may be unset entirely: the kinds that must not carry args
#    have no such variable in their chart, and an empty handoff still logs
#    its length.
: >"$tmp/engine_args"
start_driver engine-exit
status=0
wait "$driver_pid" || status=$?
[ "$status" -eq 42 ] || fail "supervisor exited $status with an empty handoff, want 42"
grep -q 'handoff empty' "$tmp/driver.log" \
    || fail "empty handoff was not reported: $(cat "$tmp/driver.log")"

# 5. Every wrapper's run_<engine> execs its engine, for the reason in 1a.
#    Checked as a shape because the failure it prevents needs a real engine
#    holding a real port to reproduce.
for name in audio clipembed embed llamacpp ocrlayout ollama sglang vllm; do
    file="$repo_root/deploy/wrappers/$name.sh"
    body=$(sed -n "/^run_$name() {/,/^}/p" "$file")
    [ -n "$body" ] || fail "$name wrapper has no run_$name function"
    case $(printf '%s\n' "$body" | sed 's/#.*//') in
        *"    exec "*) ;;
        *) fail "$(printf 'run_%s does not exec its engine; the supervisor would track a subshell:\n%s' \
            "$name" "$body")" ;;
    esac
done

printf 'supervise_engine OK\n'
