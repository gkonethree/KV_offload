#!/usr/bin/env bash
# Token-eviction comparison: pq vs pq + token eviction (FastKVzip, per-layer,
# order-preserving compaction) vs pq + token eviction (random scores).
# Qwen3-4B-Instruct-2507, LongBench-v2 8k prompt + 4096 decode, 70% offload,
# W=256, token staging store, 230 staging pages, batched FastKVzip gate, CUDA ADC kernel, warm-up mode (decode-only
# TPOT), default (non-deterministic) gather. Two interleaved rounds (order
# reversed in round 2), one process per run, same GPU.
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/tok_cmp_8k4k_v2
GPU=${GPU:-4}
export CUDA_VISIBLE_DEVICES=$GPU BENCH_MODEL=Qwen/Qwen3-4B-Instruct-2507 BENCH_DTYPE=bfloat16
export BENCH_PROMPT_FILE=$OUT/prompt_8k.txt BENCH_NEW_TOKENS=4096 BENCH_MAX_MODEL_LEN=12352
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1 BENCH_STAGING_SLOTS=230
export OFFLOAD_PACKAGE_STAGING_STORE=token
mkdir -p /tmp/opencode
CODE="bench_1req_off99.py offload_package/staging/token_store.py offload_package/staging/pool.py \
offload_package/staging/residency.py offload_package/integration/*.py offload_package/offload/manager.py \
offload_package/scoring/*.py offload_package/shared/types.py offload_package/config.py \
skylight/skylight_kernels/pq_optimized/*.py skylight/skylight_kernels/pq/codebook.py skylight/src/skylight/backend.py"

wait_for_space() {  # shared disk: don't start a run with < 3 GB free
  while [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -lt 3 ]; do
    echo "    $(date '+%T') waiting: / has <3 GB free"; sleep 60
  done
}

one() {  # one <tag> <config>
  local tag=$1 cfg=$2
  wait_for_space
  echo "=== $(date '+%F %T') start $tag ($cfg)"
  echo "code_md5 $(cat $CODE | md5sum | cut -c1-12)" > "$OUT/$tag.meta"
  echo "gpu_at_start $(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$OUT/$tag.meta"
  .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/$tag.log" 2>&1
  cp "/tmp/opencode/tokens_${cfg//+/_}.json" "$OUT/$tag.tokens.json" 2>/dev/null
  grep -E "decode-only TPOT|TTFT|KV blocks|FALLBACK|Traceback" "$OUT/$tag.log"
}

for cfg in pq pq+tok70 pq+tokrand70; do one "${cfg//+/_}_r1" "$cfg"; done
for cfg in pq+tokrand70 pq+tok70 pq; do one "${cfg//+/_}_r2" "$cfg"; done
.venv/bin/python $OUT/summarize.py | tee $OUT/summary.txt
echo "=== $(date '+%F %T') done"
