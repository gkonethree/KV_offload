#!/usr/bin/env bash
set -euo pipefail

echo "=== $(date) starting sparse 200 @ GPUs 5,6,7 ==="

TS=$(date +%s)
RUN=/data/apdesai/pd/skylight/sparse-verified-20pct-200-$TS
mkdir -p "$RUN"
split -n l/3 -d -a 1 /data/apdesai/pd/skylight/bench-resources/swebench-verified-200.txt "$RUN/chunk-"
wc -l "$RUN"/chunk-*
echo "$RUN" > /tmp/skylight-verified-sparse-200-latest.run

HF_TOKEN="$(hf auth token)"

# Mount repo src so harness fixes apply without rebuilding the image.
# Drop SKYLIGHT_SRC_MOUNT=0 to use whatever is baked into skylight:dev.
SRC_MOUNT=(-v /data/apdesai/pd/skylight/src:/skylight-src:ro -e PYTHONPATH=/skylight-src)
if [[ "${SKYLIGHT_SRC_MOUNT:-1}" == "0" ]]; then
  SRC_MOUNT=()
fi

launch_one() {
  local gpu="$1" chunk="$2" name="$3" out="$4"
  docker rm -f "$name" 2>/dev/null || true
  docker run --rm --gpus "device=${gpu}" --shm-size=16g \
    "${SRC_MOUNT[@]}" \
    -v /data/apdesai/cache:/root/.cache/huggingface \
    -v /run/user/200706/docker.sock:/var/run/docker.sock \
    -v "$chunk":/in/subset.txt:ro -v "$out":/out \
    -e SKYLIGHT_GPU_MEM_UTIL=0.95 -e HF_TOKEN="$HF_TOKEN" -e HF_HUB_DISABLE_XET=1 \
    -e SKYLIGHT_SPARSE_TOPK=0.20 -e SKYLIGHT_SPARSE_SINK=128 \
    -e SKYLIGHT_SPARSE_LOCAL=128 -e SKYLIGHT_SPARSE_CHANNEL_NUM=-1 \
    -e SKYLIGHT_METRICS_SAMPLING=0.01 \
    --name "$name" -d skylight:dev \
    python -m skylight.bench.agentic \
    --backend sparse --config-name sparse-20pct --profile verified-official \
    --model Qwen/Qwen3.5-27B --instances /in/subset.txt --output-dir /out \
    --topk 0.20 --sink 128 --local 128 --channel-num -1 \
    --max-model-len 131072 --server-timeout 900 --request-timeout-s 600
}

CID0=$(launch_one 5 "$RUN/chunk-0" skylight-gpu-0 "$RUN/gpu-0")
CID1=$(launch_one 6 "$RUN/chunk-1" skylight-gpu-1 "$RUN/gpu-1")
CID2=$(launch_one 7 "$RUN/chunk-2" skylight-gpu-2 "$RUN/gpu-2")

echo "=== launched RUN=$RUN ==="
echo "container ids: $CID0 $CID1 $CID2"
docker ps --filter name=skylight-gpu

# Wait by ID (names + --rm can race); do not abort the log if one already exited.
set +e
docker wait "$CID0"; RC0=$?
docker wait "$CID1"; RC1=$?
docker wait "$CID2"; RC2=$?
set -e
echo "exit codes: gpu-0=$RC0 gpu-1=$RC1 gpu-2=$RC2"

echo "=== all done $(date) ==="
echo "post-process: bash /data/apdesai/pd/skylight/scripts/post_process_run.sh $RUN"
