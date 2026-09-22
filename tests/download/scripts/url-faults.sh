#!/usr/bin/env bash
# Reproduce a direct-URL download fault against the real binary and show
# the resulting /api/progress message.
#
# Usage: bash url-faults.sh [404|429|503storm|reset|ok]
#
# Mirrors the TestURL_* cases in e2e (see ../README.md).
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

MODE="${1:-404}"
case "${MODE}" in
	404)     SRV_ARGS=(-force-status 404) ;;
	429)     SRV_ARGS=(-force-status 429) ;;
	503storm) SRV_ARGS=(-fail-first 2) ;;
	reset)   SRV_ARGS=(-drop-after 65536 -drop-first 1 -support-range) ;;
	ok)      SRV_ARGS=(-support-range) ;;
	*) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac

ensure_model_mode
BIN="$(build_binary)"
WORK="$(mktemp -d)"

echo "==> starting faultsrv (mode=${MODE})"
go run "${REPO_ROOT}/tests/download/cmd/faultsrv" -addr 127.0.0.1:9000 -size 1048576 "${SRV_ARGS[@]}" &
PIDS+=("$!")
wait_http "http://127.0.0.1:9000/model.bin" || { echo "faultsrv did not start" >&2; exit 1; }

echo "==> booting llm-init (url channel, llamacpp)"
PORT="${PORT}" \
MODEL_NAME=runbook-url \
ENGINE_KIND=llamacpp \
MODEL_SOURCE="http://127.0.0.1:9000/model.bin" \
MODEL_SOURCE_LOCAL="${WORK}/model.bin" \
RUN_DIR="${WORK}/run" \
	"${BIN}" &
PIDS+=("$!")

wait_http "${BASE}/livez" || { echo "control plane did not start" >&2; exit 1; }
poll_progress
