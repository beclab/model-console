#!/usr/bin/env bash
# Tiny dispatcher for examples/local scenarios.
#
# Usage:
#   bash examples/local/run.sh <ID> [up|down|logs]
#
#   ID    : A1 A2 A3 A4 B1 B2 B3 C1 C2 D1 D2 D3 E1 E2 F1
#           (D2 has no env file; see D2-source-mismatch.md)
#   verb  : up      (default) -- docker compose up with the right file set
#           down            -- docker compose down -v (wipes the per-scenario volume)
#           logs            -- docker compose logs -f
#
# Environment overrides:
#   LLM_INIT_IMAGE   default: llm-init:local, auto-built on demand
#                    from the current source tree. Set to a different
#                    tag to point at an image you manage yourself --
#                    run.sh will then refuse to start if it is
#                    missing instead of building one for you.
#   SKIP_REBUILD     default: unset. Set to 1 to skip the auto-build
#                    even when the image is missing or stale (will
#                    warn and continue with whatever is on disk).
#   LLM_INIT_PORT    pin the host-side port for llm-init. Default comes
#                    from the scenario's .env (8080). When that port is
#                    already bound on the host, run.sh auto-picks a free
#                    one in 28080..28099 and falls back to a random
#                    30000..60000 ephemeral port; the banner prints the
#                    effective URL.
#   USE_GPU          NVIDIA passthrough for llamacpp / ollama scenarios.
#                    Tri-state:
#                        unset (default) -> auto: layer the GPU
#                              override when `nvidia-smi -L` succeeds
#                              on the host.
#                        1               -> force on (skip host probe;
#                              useful when nvidia-smi is not on PATH
#                              but Docker still has device access).
#                        0               -> force off (CPU mode,
#                              matches the upstream defaults in
#                              deploy/compose/*).
#                    vllm/sglang already require GPUs in their base
#                    compose files and ignore this flag.
#
# Auto-build on `up`:
#   Before bringing the stack up, run.sh decides whether the default
#   image llm-init:local needs rebuilding by:
#     1. checking whether it exists at all;
#     2. comparing its org.opencontainers.image.revision label
#        (injected via --build-arg COMMIT) to `git rev-parse HEAD`;
#     3. looking for uncommitted edits under cmd/, internal/, go.mod,
#        go.sum, Dockerfile (the paths Dockerfile actually copies in);
#     4. flagging images whose version label ends in -dirty.
#   On any of those, it runs `docker build` against the repo root
#   with VERSION/COMMIT/BUILD_TIME args matching Makefile. Docker's
#   own layer cache decides how much work that actually does (usually
#   seconds when nothing under cmd/ or internal/ changed).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/examples/local"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <ID> [up|down|logs]" >&2
    echo "       ID = A1 A2 A3 A4 B1 B2 B3 C1 C2 D1 D2 D3 E1 E2 F1" >&2
    echo "       D2 is a manual procedure; see D2-source-mismatch.md" >&2
    exit 2
fi

ID="$(printf '%s' "$1" | tr '[:lower:]' '[:upper:]')"
VERB="${2:-up}"
BASE_COMPOSE=""

# Every scenario layers a spec override that mounts the right full spec
# (chat / embedding / thinking / translate) and points MODEL_SPEC_PATH at it, so the
# mounted file wins over the MODEL_MODE env seed set in each .env.
SPEC_CHAT="$HERE/_spec-chat.compose.yml"
SPEC_EMBED="$HERE/_spec-embedding.compose.yml"
SPEC_THINK="$HERE/_spec-thinking.compose.yml"
SPEC_TRANSLATE="$HERE/_spec-translate.compose.yml"

case "$ID" in
    A1) ENGINE=ollama   ; ENV_FILE="$HERE/A1-ollama-name.env"      ; OVERRIDE="$SPEC_CHAT" ;;
    A2) ENGINE=llamacpp ; ENV_FILE="$HERE/A2-llamacpp-hf-gguf.env" ; OVERRIDE="$SPEC_CHAT" ;;
    A3) ENGINE=vllm     ; ENV_FILE="$HERE/A3-vllm-hf-repo.env"     ; OVERRIDE="$SPEC_CHAT" ;;
    A4) ENGINE=sglang   ; ENV_FILE="$HERE/A4-sglang-hf-repo.env"   ; OVERRIDE="$SPEC_CHAT" ;;
    B1) ENGINE=ollama   ; ENV_FILE="$HERE/B1-ollama-hf-gguf.env"   ; OVERRIDE="$SPEC_CHAT" ;;
    B2) ENGINE=ollama   ; ENV_FILE="$HERE/B2-ollama-url.env"       ; OVERRIDE="$SPEC_CHAT" ;;
    B3) ENGINE=llamacpp ; ENV_FILE="$HERE/B3-llamacpp-url.env"     ; OVERRIDE="$SPEC_CHAT" ;;
    C1) ENGINE=ollama   ; ENV_FILE="$HERE/C1-ollama-embedding.env" ; OVERRIDE="$SPEC_EMBED" ;;
    C2) ENGINE=llamacpp ; ENV_FILE="$HERE/C2-llamacpp-thinking.env"; OVERRIDE="$SPEC_THINK" ;;
    D1) ENGINE=llamacpp ; ENV_FILE="$HERE/D1-bad-repo.env"         ; OVERRIDE="$SPEC_CHAT" ;;
    D2) echo "D2 is a manual two-step procedure. Read: $HERE/D2-source-mismatch.md" >&2 ; exit 2 ;;
    D3) ENGINE=llamacpp ; ENV_FILE="$HERE/D3-hf-mirror.env"        ; OVERRIDE="$SPEC_CHAT" ;;
    E1) ENGINE=llamacpp ; ENV_FILE="$HERE/E1-multi-source.env"     ; OVERRIDE="$SPEC_CHAT" ;;
    E2) ENGINE=ollama   ; ENV_FILE="$HERE/E2-qwen35-vlm.env"       ; OVERRIDE="$SPEC_CHAT" ;;
    F1) ENGINE=llamacpp ; BASE_COMPOSE=translate ; ENV_FILE="$HERE/F1-llamacpp-translate.env" ; OVERRIDE="$SPEC_TRANSLATE" ;;
    *)  echo "unknown scenario: $ID (expected A1 A2 A3 A4 B1 B2 B3 C1 C2 D1 D2 D3 E1 E2 F1)" >&2 ; exit 2 ;;
esac

[[ -z "$BASE_COMPOSE" ]] && BASE_COMPOSE="$ENGINE"
BASE="$ROOT/deploy/compose/${BASE_COMPOSE}.yml"
PROJECT="llmlocal-$(echo "$ID" | tr '[:upper:]' '[:lower:]')"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "missing env file: $ENV_FILE" >&2
    exit 1
fi
if [[ ! -f "$BASE" ]]; then
    echo "missing base compose file: $BASE" >&2
    exit 1
fi

COMPOSE_ARGS=(-f "$BASE")
if [[ -n "$OVERRIDE" ]]; then
    if [[ ! -f "$OVERRIDE" ]]; then
        echo "missing override file: $OVERRIDE" >&2
        exit 1
    fi
    COMPOSE_ARGS+=(-f "$OVERRIDE")
fi

GPU_OVERRIDE_NAME=""
case "$ENGINE" in
    llamacpp) GPU_OVERRIDE_NAME="_llamacpp-gpu.compose.yml" ;;
    ollama)   GPU_OVERRIDE_NAME="_ollama-gpu.compose.yml"   ;;
esac

host_nvidia_available() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

GPU_ON=0
GPU_REASON=""
if [[ -n "$GPU_OVERRIDE_NAME" ]]; then
    case "${USE_GPU-}" in
        0)
            GPU_REASON='off (USE_GPU=0)'
            ;;
        1)
            GPU_ON=1
            GPU_REASON='forced via USE_GPU=1'
            ;;
        "")
            if host_nvidia_available; then
                GPU_ON=1
                GPU_REASON='NVIDIA auto-detected on host'
            else
                GPU_REASON='off (no nvidia-smi on host; set USE_GPU=1 to force)'
            fi
            ;;
        *)
            echo "warning: ignoring unrecognized USE_GPU=${USE_GPU} (expected 0 | 1 | unset)" >&2
            GPU_REASON="off (unrecognized USE_GPU=${USE_GPU})"
            ;;
    esac

    if [[ "$GPU_ON" == "1" ]]; then
        GPU_OVERRIDE="$HERE/$GPU_OVERRIDE_NAME"
        if [[ ! -f "$GPU_OVERRIDE" ]]; then
            echo "missing GPU override file: $GPU_OVERRIDE" >&2
            exit 1
        fi
        COMPOSE_ARGS+=(-f "$GPU_OVERRIDE")
    fi
fi

COMPOSE_ARGS+=(--env-file "$ENV_FILE" -p "$PROJECT")

# Resolve the effective image: parent shell env wins, otherwise read
# the scenario's image variable from the .env file. F1 uses
# TRANSLATE_LLM_INIT_IMAGE so the generic LLM_INIT_IMAGE cannot override it.
# Fail fast if the image
# is not present locally -- pulling a registry tag here would mean
# the operator silently observes upstream behaviour instead of the
# tree they just edited.
IMAGE_VARIABLE=LLM_INIT_IMAGE
if [[ "$ID" == "F1" ]]; then
    IMAGE_VARIABLE=TRANSLATE_LLM_INIT_IMAGE
fi
RESOLVED_IMAGE="${!IMAGE_VARIABLE:-}"
if [[ -z "$RESOLVED_IMAGE" ]]; then
    RESOLVED_IMAGE="$(grep -E "^${IMAGE_VARIABLE}=" "$ENV_FILE" | head -1 | cut -d= -f2-)"
fi

# `port_free` answers "can docker bind this on the host right now?".
# `connect` probes (nc -z, /dev/tcp) only see active listeners, but
# docker's failure mode at A2 was a bind on something already holding
# the port -- a bind probe is the right primitive. python3 ships with
# Docker Desktop's WSL distro and most dev hosts; otherwise we fall
# back to ss / netstat (listener-set checks) and finally to /dev/tcp.
port_free() {
    local p="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$p" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
try:
    s.bind(('127.0.0.1', int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
    elif command -v ss >/dev/null 2>&1; then
        ! ss -ltn "sport = :$p" 2>/dev/null | tail -n +2 | grep -q .
    elif command -v netstat >/dev/null 2>&1; then
        ! netstat -an 2>/dev/null | grep -qE "[.:]$p[[:space:]]+.*LISTEN"
    else
        ! (echo >"/dev/tcp/127.0.0.1/$p") >/dev/null 2>&1
    fi
}

resolve_host_port() {
    local desired="${LLM_INIT_PORT:-}"
    if [[ -z "$desired" && -f "$ENV_FILE" ]]; then
        desired="$(grep -E '^LLM_INIT_PORT=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '[:space:]')"
    fi
    [[ -z "$desired" ]] && desired=8080

    if port_free "$desired"; then
        printf '%s' "$desired"
        return 0
    fi

    echo ">> host port $desired is in use; picking a free one..." >&2
    local c picked=""
    for c in $(seq 28080 28099); do
        if port_free "$c"; then picked="$c"; break; fi
    done
    if [[ -z "$picked" ]]; then
        for c in $(awk 'BEGIN{srand(); for(i=0;i<40;i++) print int(30000+rand()*30000)}'); do
            if port_free "$c"; then picked="$c"; break; fi
        done
    fi
    if [[ -z "$picked" ]]; then
        echo "error: could not find a free TCP port" >&2
        exit 1
    fi
    echo ">> using LLM_INIT_PORT=$picked (override with LLM_INIT_PORT=...)" >&2
    printf '%s' "$picked"
}

EFFECTIVE_PORT=""
if [[ "$VERB" == "up" ]]; then
    EFFECTIVE_PORT="$(resolve_host_port)"
    export LLM_INIT_PORT="$EFFECTIVE_PORT"
fi

build_image() {
    local version commit build_time
    version=$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo dev)
    commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo none)
    build_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo ">> docker build -t $RESOLVED_IMAGE  (VERSION=$version COMMIT=$commit)" >&2
    docker build \
        --build-arg "VERSION=$version" \
        --build-arg "COMMIT=$commit" \
        --build-arg "BUILD_TIME=$build_time" \
        -t "$RESOLVED_IMAGE" \
        "$ROOT"
}

build_if_needed() {
    [[ "$VERB" == "up" ]] || return 0

    local image_exists=0
    docker image inspect "$RESOLVED_IMAGE" >/dev/null 2>&1 && image_exists=1

    # A custom LLM_INIT_IMAGE tag belongs to the operator. We do not
    # rebuild it; we only refuse to start when it is missing.
    if [[ "$RESOLVED_IMAGE" != "llm-init:local" ]]; then
        if [[ "$image_exists" -eq 0 ]]; then
            cat >&2 <<EOF
error: image '$RESOLVED_IMAGE' not present locally.
       $IMAGE_VARIABLE is set to a non-default tag, so run.sh will not
       build it for you. Build it yourself, or unset $IMAGE_VARIABLE
       to fall back to the auto-built llm-init:local.
EOF
            exit 1
        fi
        return 0
    fi

    local reason="" img_commit img_version cur_commit dirty_files
    if [[ "$image_exists" -eq 0 ]]; then
        reason="image '$RESOLVED_IMAGE' is not present locally"
    elif command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        img_commit=$(docker image inspect "$RESOLVED_IMAGE" \
            -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)
        img_version=$(docker image inspect "$RESOLVED_IMAGE" \
            -f '{{index .Config.Labels "org.opencontainers.image.version"}}' 2>/dev/null || true)
        cur_commit=$(git -C "$ROOT" rev-parse --short HEAD)
        dirty_files=$(git -C "$ROOT" status --porcelain -- \
            cmd internal go.mod go.sum Dockerfile 2>/dev/null || true)

        if [[ -z "$img_commit" || "$img_commit" == "<no value>" ]]; then
            reason="image has no revision label (built outside docker build?)"
        elif [[ "$img_commit" != "$cur_commit" ]]; then
            reason="image commit $img_commit != current HEAD $cur_commit"
        elif [[ -n "$dirty_files" ]]; then
            reason="working tree dirty under cmd/ internal/ go.mod go.sum Dockerfile"
        elif [[ "$img_version" == *-dirty ]]; then
            reason="last build was from a dirty tree (version=$img_version)"
        fi
    fi

    if [[ -z "$reason" ]]; then
        return 0
    fi

    if [[ "${SKIP_REBUILD:-0}" == "1" ]]; then
        echo ">> $reason" >&2
        echo ">> SKIP_REBUILD=1 set; continuing with the existing image" >&2
        return 0
    fi

    echo ">> $reason" >&2
    echo ">> auto-building (set SKIP_REBUILD=1 to bypass)" >&2
    if ! build_image; then
        echo ">> docker build failed; aborting" >&2
        exit 1
    fi
}

build_if_needed

echo "scenario : $ID"
echo "engine   : $ENGINE"
echo "project  : $PROJECT"
echo "env-file : $ENV_FILE"
[[ -n "$OVERRIDE" ]] && echo "override : $OVERRIDE"
echo "image    : $RESOLVED_IMAGE"
if [[ "$GPU_ON" == "1" ]]; then
    if [[ "$ENGINE" == "llamacpp" ]]; then
        echo "gpu      : on  (:server-cuda; $GPU_REASON)"
    else
        echo "gpu      : on  ($GPU_REASON)"
    fi
elif [[ -n "$GPU_OVERRIDE_NAME" ]]; then
    echo "gpu      : $GPU_REASON"
fi
[[ -n "$EFFECTIVE_PORT" ]] && echo "url      : http://localhost:$EFFECTIVE_PORT"
echo

case "$VERB" in
    up)   exec docker compose "${COMPOSE_ARGS[@]}" up ;;
    down) exec docker compose "${COMPOSE_ARGS[@]}" down -v ;;
    logs) exec docker compose "${COMPOSE_ARGS[@]}" logs -f ;;
    *)    echo "unknown verb: $VERB (expected up|down|logs)" >&2 ; exit 2 ;;
esac
