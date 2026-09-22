#!/usr/bin/env bash
# Shared helpers for the download runbook scripts. Sourced, not executed.
# Linux/bash; requires `go` and `curl` on PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PORT="${PORT:-18090}"
BASE="http://127.0.0.1:${PORT}"

# Background PIDs to reap on exit.
PIDS=()
cleanup() {
	for pid in "${PIDS[@]:-}"; do
		[ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
	done
}
trap cleanup EXIT INT TERM

# ensure_model_mode exports MODEL_MODE so the real binary can seed its
# model-spec (it fail-fasts without MODEL_MODE). No /etc file / sudo is
# needed: each script sets RUN_DIR to a temp dir, so the spec is seeded to
# ${RUN_DIR}/model-spec.json (MODEL_SPEC_PATH default).
ensure_model_mode() {
	export MODEL_MODE="${MODEL_MODE:-chat}"
}

# build_binary compiles llm-init into a temp dir and echoes its path.
build_binary() {
	local out
	out="$(mktemp -d)/llm-init"
	(cd "${REPO_ROOT}" && go build -o "${out}" ./cmd/llm-init) >&2
	echo "${out}"
}

# wait_http waits until $1 returns any HTTP response (max ~10s).
wait_http() {
	local url="$1" i
	for i in $(seq 1 50); do
		if curl -fsS -o /dev/null "${url}" 2>/dev/null; then return 0; fi
		sleep 0.2
	done
	return 1
}

# poll_progress prints /api/progress a few times so the operator can see
# the phase + last_error settle.
poll_progress() {
	local i
	echo "==> polling ${BASE}/api/progress"
	for i in $(seq 1 8); do
		curl -fsS "${BASE}/api/progress" 2>/dev/null || true
		echo
		sleep 0.5
	done
}
