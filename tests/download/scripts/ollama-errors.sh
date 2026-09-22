#!/usr/bin/env bash
# Reproduce an Ollama pull failure against the real binary using the mock
# daemon, and show the operator-facing /api/progress message (invalid tag
# or denied pull).
#
# Usage: bash ollama-errors.sh [404|403|nosuccess|ok]
#
# Mirrors TestOllamaName_* (see ../README.md).
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

MODE="${1:-404}"
case "${MODE}" in
	404)       MOCK_ARGS=(-pull-status 404) ;;
	403)       MOCK_ARGS=(-pull-status 403) ;;
	nosuccess) MOCK_ARGS=(-pull-omit-success) ;;
	ok)        MOCK_ARGS=() ;;
	*) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac

ensure_model_mode
BIN="$(build_binary)"
WORK="$(mktemp -d)"

echo "==> starting mockollama (mode=${MODE})"
go run "${REPO_ROOT}/tests/download/cmd/mockollama" -addr 127.0.0.1:11434 "${MOCK_ARGS[@]}" &
PIDS+=("$!")
wait_http "http://127.0.0.1:11434/api/tags" || { echo "mockollama did not start" >&2; exit 1; }

echo "==> booting llm-init (ollama channel)"
PORT="${PORT}" \
MODEL_NAME=runbook-ollama \
ENGINE_KIND=ollama \
LLM_INIT_TEST_ENGINE_URL="http://127.0.0.1:11434" \
MODEL_SOURCE="ollama://qwen2.5:7b" \
RUN_DIR="${WORK}/run" \
	"${BIN}" &
PIDS+=("$!")

wait_http "${BASE}/livez" || { echo "control plane did not start" >&2; exit 1; }
poll_progress
