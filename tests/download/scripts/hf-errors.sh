#!/usr/bin/env bash
# Reproduce a HuggingFace download failure against the real binary using
# the fake `hf` CLI on PATH, and show the operator-facing /api/progress
# message (e.g. a gated repo or an unauthorized token).
#
# Usage: bash hf-errors.sh [repo|gated|token|revision|disk|network|ok]
#
# Mirrors TestHF_ErrorMessages_ForFrontend / TestHF_Subprocess_* (see ../README.md).
set -euo pipefail
cd "$(dirname "$0")"
source ./lib.sh

MODE="${1:-gated}"
FAKEHF_EXIT=1
case "${MODE}" in
	repo)     FAKEHF_STDERR='huggingface_hub.errors.RepositoryNotFoundError: 404' ;;
	gated)    FAKEHF_STDERR='GatedRepoError: Cannot access gated repo for url ...' ;;
	token)    FAKEHF_STDERR='401 Client Error: Unauthorized for url ...' ;;
	revision) FAKEHF_STDERR='RevisionNotFoundError: 404 for revision deadbeef' ;;
	disk)     FAKEHF_STDERR='OSError: [Errno 28] No space left on device' ;;
	network)  FAKEHF_STDERR='requests.exceptions.ConnectionError: HTTPSConnectionPool' ;;
	ok)       FAKEHF_STDERR=''; FAKEHF_EXIT=0 ;;
	*) echo "unknown mode: ${MODE}" >&2; exit 2 ;;
esac

ensure_model_mode
BIN="$(build_binary)"
WORK="$(mktemp -d)"

echo "==> building fake hf onto PATH"
HFBIN_DIR="$(mktemp -d)"
(cd "${REPO_ROOT}" && go build -o "${HFBIN_DIR}/hf" ./tests/download/cmd/fakehf)

echo "==> booting llm-init (hf channel, llamacpp) mode=${MODE}"
PATH="${HFBIN_DIR}:${PATH}" \
PORT="${PORT}" \
MODEL_NAME=runbook-hf \
ENGINE_KIND=llamacpp \
MODEL_SOURCE="hf://owner/repo" \
RUN_DIR="${WORK}/run" \
FAKEHF_STDERR="${FAKEHF_STDERR}" \
FAKEHF_EXIT="${FAKEHF_EXIT}" \
FAKEHF_COMMIT="1111111111111111111111111111111111111111" \
	"${BIN}" &
PIDS+=("$!")

wait_http "${BASE}/livez" || { echo "control plane did not start" >&2; exit 1; }
poll_progress
