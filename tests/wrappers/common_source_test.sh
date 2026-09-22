#!/bin/sh
# Where a wrapper gets its helpers from.
#
# A Market chart points WRAPPER_COMMON at the copy llm-init publishes into
# RUN_DIR, so supervise_engine has one implementation across every
# application instead of forty hand-copies. The two containers are separate
# Deployments with no ordering between them, so that copy can arrive after
# the engine container has already started: waiting for it is the normal
# path, and a wrapper that exited instead would CrashLoop on a healthy
# deployment.
set -eu

# shellcheck disable=SC1007  # CDPATH= is a deliberate empty assignment
repo_root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
wrappers="$repo_root/deploy/wrappers"

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

# The preamble cannot live in common.sh -- it is what finds common.sh --
# so it is copied into every wrapper, and only a check keeps the copies
# from drifting.
first=""
for name in audio clipembed embed llamacpp ocrlayout ollama rerank sglang vllm; do
    file="$wrappers/$name.sh"
    block=$(sed -n '/^WRAPPER_COMMON=/,/^\. "\$WRAPPER_COMMON"$/p' "$file")
    [ -n "$block" ] || fail "$name wrapper has no WRAPPER_COMMON preamble"
    if [ -z "$first" ]; then
        first=$block
        first_name=$name
    elif [ "$block" != "$first" ]; then
        fail "$(printf '%s preamble differs from %s:\n%s\n---\n%s' \
            "$name" "$first_name" "$first" "$block")"
    fi
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
touch "$tmp/sentinel"
mkdir -p "$tmp/model"
printf '%s\n' "$tmp/model" >"$tmp/model_path"

cat >"$tmp/embed_server" <<'EOF'
#!/bin/sh
exit 0
EOF
chmod +x "$tmp/embed_server"

# A published copy carries a marker, so the assertions below distinguish
# "sourced the published file" from "fell back to the sibling copy and
# happened to work".
publish() {
    cp "$wrappers/common.sh" "$1"
    printf '\nlog "helpers came from the published copy"\n' >>"$1"
}

# run_embed LOG -- the embed wrapper with the fakes in place. Every path
# the helpers read is redirected into $tmp so nothing touches /run.
run_embed() {
    log=$1
    shift
    env \
        PATH="$tmp:$PATH" \
        EMBED_SERVER_BIN="$tmp/embed_server" \
        SENTINEL="$tmp/sentinel" \
        MODEL_PATH_FILE="$tmp/model_path" \
        ENGINE_ARGS_FILE="$tmp/engine_args" \
        ENGINE_RESTART_FILE="$tmp/engine_restart" \
        WAIT_INTERVAL=0 \
        WAIT_TIMEOUT=0 \
        RESTART_POLL_INTERVAL=0 \
        "$@" \
        sh "$wrappers/embed.sh" >"$log" 2>&1
}

# 1. The helpers are already there: no waiting, and the marker proves the
#    published copy won over the one sitting next to the script.
published="$tmp/published-common.sh"
publish "$published"
run_embed "$tmp/ready.log" WRAPPER_COMMON="$published" \
    || fail "wrapper exited $? with the helpers in place: $(cat "$tmp/ready.log")"
grep -q 'helpers came from the published copy' "$tmp/ready.log" \
    || fail "wrapper did not source WRAPPER_COMMON: $(cat "$tmp/ready.log")"
if grep -q 'waiting for' "$tmp/ready.log"; then
    fail "wrapper waited for helpers that were already present"
fi

# 2. The helpers arrive late. The wrapper must wait rather than exit: this
#    is llm-init still pulling its image while the engine Pod is up.
late="$tmp/late-common.sh"
( sleep 2; publish "$late" ) &
waiter=$!
run_embed "$tmp/late.log" WRAPPER_COMMON="$late" \
    || fail "wrapper exited $? waiting for late helpers: $(cat "$tmp/late.log")"
wait "$waiter" 2>/dev/null || true
grep -q "waiting for $late" "$tmp/late.log" \
    || fail "wrapper did not report waiting: $(cat "$tmp/late.log")"
grep -q 'helpers came from the published copy' "$tmp/late.log" \
    || fail "wrapper did not source the late helpers: $(cat "$tmp/late.log")"

# 3. They never arrive. Bounded by WRAPPER_COMMON_WAIT, and the failure
#    has to name the path -- a bare "not found" from sh sends the reader
#    looking for a bug in the wrapper instead of at the sibling container.
status=0
run_embed "$tmp/never.log" WRAPPER_COMMON="$tmp/absent-common.sh" \
    WRAPPER_COMMON_WAIT=0 || status=$?
[ "$status" -ne 0 ] || fail "wrapper succeeded with no helpers at all"
grep -q "$tmp/absent-common.sh never appeared" "$tmp/never.log" \
    || fail "wrapper did not say which file was missing: $(cat "$tmp/never.log")"

printf 'wrapper helper source: preamble identical, present/late/absent OK\n'
