#!/bin/sh
# Shape + argv checks for the four wrappers whose modes never carry
# engine_args: audio, embed, clipembed, ocrlayout.
#
# Each is launched against fake engine binaries so the assertions cover
# what the wrapper actually execs, not just what it looks like.
set -eu

# shellcheck disable=SC1007  # CDPATH= is a deliberate empty assignment
repo_root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
wrappers="$repo_root/deploy/wrappers"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

# calls FILE FUNC -- true when FILE mentions FUNC outside a comment.
calls() {
    sed 's/#.*//' "$1" | grep -qw -- "$2"
}

# config.ParseEngineArgs rejects a non-empty ENGINE_ARGS for the embed,
# clipembed, audio, ocr and rerank kinds, so these wrappers must not take engine
# arguments from the model card: waiting for the handoff would only stall
# every launch, and loading it would apply a value llm-init refuses to set.
for wrapper in audio embed clipembed ocrlayout rerank; do
    file="$wrappers/$wrapper.sh"
    sh -n "$file"
    for forbidden in wait_for_engine_args_handoff load_engine_args_from_handoff; do
        if calls "$file" "$forbidden"; then
            fail "$wrapper wrapper must not call $forbidden"
        fi
    done
    # The engine must be supervised, not exec'd: POST /api/engine/restart
    # bumps engine_restart and only supervise_engine reads it.
    last=$(grep -v '^[[:space:]]*$' "$file" | tail -n 1)
    want="supervise_engine run_$wrapper \"\$@\""
    if [ "$last" != "$want" ]; then
        fail "$wrapper wrapper must end with '$want', got '$last'"
    fi
done

# The audio bases resolve their own model directory; model_path is written
# for engines that take a --model flag, which this one does not.
if calls "$wrappers/audio.sh" read_model_path; then
    fail "audio wrapper must not call read_model_path"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
touch "$tmp/sentinel"
mkdir -p "$tmp/model"
printf '%s\n' "$tmp/model" >"$tmp/model_path"
printf '%s\n' "$tmp/model" >"$tmp/extra_model_path"

# fake_engine PATH -- records argv (one per line) and selected env.
fake_engine() {
    cat >"$1" <<'EOF'
#!/bin/sh
printf '%s\n' "$@" >"$FAKE_ENGINE_ARGV"
printf 'LAYOUT_MODEL_DIR=%s\nMODEL_PATH=%s\nLAYOUT_LISTEN=%s\n' \
    "${LAYOUT_MODEL_DIR:-}" "${MODEL_PATH:-}" "${LAYOUT_LISTEN:-}" \
    >"$FAKE_ENGINE_ARGV.env"
EOF
    chmod +x "$1"
}

fake_engine "$tmp/audio-python"
fake_engine "$tmp/embed_server"
fake_engine "$tmp/rerank-server"
fake_engine "$tmp/ocr-layout"

# run_wrapper NAME EXTRA_ENV... -- launches the wrapper with the fakes in
# place and leaves its output in $tmp/NAME.log, argv in $tmp/NAME.argv.
run_wrapper() {
    name=$1
    shift
    env \
        FAKE_ENGINE_ARGV="$tmp/$name.argv" \
        PATH="$tmp:$PATH" \
        SENTINEL="$tmp/sentinel" \
        MODEL_PATH_FILE="$tmp/model_path" \
        EXTRA_MODEL_PATH_FILE="$tmp/extra_model_path" \
        ENGINE_ARGS_FILE="$tmp/engine_args" \
        ENGINE_RESTART_FILE="$tmp/engine_restart" \
        WAIT_INTERVAL=0 \
        WAIT_TIMEOUT=0 \
        RESTART_POLL_INTERVAL=0 \
        "$@" \
        sh "$wrappers/$name.sh" >"$tmp/$name.log" 2>&1 \
        || fail "$name wrapper exited $? (log: $(cat "$tmp/$name.log"))"
    grep -q 'launching engine (restart_gen=' "$tmp/$name.log" \
        || fail "$name wrapper did not launch under supervise_engine: $(cat "$tmp/$name.log")"
}

expect_argv() {
    name=$1
    shift
    want=$(printf '%s\n' "$@")
    got=$(cat "$tmp/$name.argv")
    if [ "$got" != "$want" ]; then
        fail "$(printf '%s wrapper argv mismatch:\nwant:\n%s\ngot:\n%s' "$name" "$want" "$got")"
    fi
}

run_wrapper audio
expect_argv audio -m wrapper.app

run_wrapper embed EMBED_SERVER_BIN="$tmp/embed_server"
expect_argv embed ''

run_wrapper clipembed EMBED_SERVER_BIN="$tmp/embed_server"
expect_argv clipembed ''

run_wrapper rerank RERANK_SERVER_BIN="$tmp/rerank-server"
expect_argv rerank ''

run_wrapper ocrlayout LAYOUT_BIN="$tmp/ocr-layout"
expect_argv ocrlayout ''
want_env=$(printf 'LAYOUT_MODEL_DIR=%s\nMODEL_PATH=%s\nLAYOUT_LISTEN=0.0.0.0:8090\n' "$tmp/model" "$tmp/model")
got_env=$(cat "$tmp/ocrlayout.argv.env")
if [ "$got_env" != "$want_env" ]; then
    fail "$(printf 'ocrlayout wrapper env mismatch:\nwant:\n%s\ngot:\n%s' "$want_env" "$got_env")"
fi

printf 'wrappers: audio, embed, clipembed, ocrlayout, rerank OK\n'
