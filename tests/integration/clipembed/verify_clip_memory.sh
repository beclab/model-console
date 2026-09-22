#!/usr/bin/env bash
# Measure Jina CLIP v2 split (OpenVINO CPU) memory during compile vs first inference.
#
# Usage:
#   bash tests/integration/clipembed/verify_clip_memory.sh [compile-only|embed-only|both]
#
# Requires: docker, embed-server:ov-intel-amd64, HF cache with jina-clip-v2-split openvino snapshot.

set -euo pipefail

MODE="${1:-both}"
HF_CACHE="${HF_CACHE:-/olares/share/hdd0/llminit-model-cache}"
IMAGE="${EMBED_IMAGE:-embed-server:ov-intel-amd64}"
SNAP="${SNAP:-65d37401d92f5fdf216bb09fcded7eb562eb392b}"
MODEL_DIR="${MODEL_DIR:-$HF_CACHE/models--beclab--jina-clip-v2-split/snapshots/$SNAP/openvino}"
SAMPLE="${SAMPLE:-The cat sits on the mat.}"

run_case() {
  local name=$1
  local cmd=$2
  local cname="clip-mem-$name-$$"
  local stats="/tmp/${cname}.csv"
  local log="/tmp/${cname}.log"
  local done="/tmp/${cname}.done"

  cleanup() { docker rm -f "$cname" >/dev/null 2>&1 || true; rm -f "$done"; }
  trap cleanup RETURN

  docker run -d --name "$cname" --entrypoint sleep \
    -v "$HF_CACHE:/cache/hf/hub:ro" "$IMAGE" infinity >/dev/null

  echo "elapsed_s,rss" > "$stats"
  local oom_before
  oom_before=$(dmesg -T 2>/dev/null | grep -c 'Killed process.*embed_server' || true)

  (
    local t=0
    while docker inspect -f '{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; do
      docker stats "$cname" --no-stream --format '{{.MemUsage}}' 2>/dev/null \
        | awk -v t="$t" '{print t","$1}' >> "$stats"
      sleep 1
      t=$((t + 1))
      [[ -f "$done" ]] && { sleep 2; break; }
      [[ $t -gt 300 ]] && break
    done
  ) &
  local mon=$!

  local start end exit=0
  start=$(date +%s)
  if ! docker exec \
    -e EMBED_MODEL_DIR="/cache/hf/hub/models--beclab--jina-clip-v2-split/snapshots/$SNAP/openvino" \
    -e MODEL_ID=jina-clip-v2-split-ov \
    -e EMBED_DEVICE=cpu \
    "$cname" sh -c "$cmd" > "$log" 2>&1; then
    exit=$?
  fi
  touch "$done"
  end=$(date +%s)
  wait "$mon" 2>/dev/null || true

  local oom_after oom_new
  oom_after=$(dmesg -T 2>/dev/null | grep -c 'Killed process.*embed_server' || true)
  oom_new=$((oom_after - oom_before))

  echo ""
  echo "=== $name ==="
  echo "exit=$exit wall_s=$((end - start)) oom_new=$oom_new"
  echo "rss timeline (MiB/GiB):"
  tail -n +2 "$stats" | head -25
  if [[ $(wc -l < "$stats") -gt 26 ]]; then
    echo "..."
    tail -5 "$stats"
  fi
  echo "markers:"
  grep -E 'compile_model done|EmbeddingService ready|embed-only: success|compile-only: CLIP|failed|ERROR' "$log" || tail -3 "$log"
  if [[ "$exit" -eq 137 ]] || [[ "$oom_new" -gt 0 ]]; then
    echo ">>> SIGKILL/OOM likely (exit 137 or new dmesg OOM entry)"
    dmesg -T 2>/dev/null | grep 'Killed process.*embed_server' | tail -1 || true
  fi
  echo "log: $log"
}

[[ -d "$MODEL_DIR" ]] || { echo "missing MODEL_DIR: $MODEL_DIR" >&2; exit 1; }

echo "=== CLIP memory verification ==="
echo "image=$IMAGE"
echo "model_dir=$MODEL_DIR"
free -h | awk '/^Mem:/{printf "host_mem: used=%s total=%s avail=%s\n", $3, $2, $7}'
text_blob="$HF_CACHE/models--beclab--jina-clip-v2-split/blobs/7d2575d122583e2e132813ee3700c81d8d0d4b5eec7bdb23ac28422e31506d5c"
vision_blob="$HF_CACHE/models--beclab--jina-clip-v2-split/blobs/06752079750d79cc080ea64fd41f7fda5bff5135144fd68aa0f6d5cdea7f3d80"
echo "weights: text=$(numfmt --to=iec "$(stat -c%s "$text_blob")") vision=$(numfmt --to=iec "$(stat -c%s "$vision_blob")")"

case "$MODE" in
  compile-only) run_case compile_only '/usr/local/bin/embed_server --compile-only' ;;
  embed-only)   run_case embed_only "/usr/local/bin/embed_server --embed-only \"$SAMPLE\"" ;;
  both)
    run_case compile_only '/usr/local/bin/embed_server --compile-only'
    run_case embed_only "/usr/local/bin/embed_server --embed-only \"$SAMPLE\""
    ;;
  *) echo "mode must be compile-only|embed-only|both" >&2; exit 2 ;;
esac
