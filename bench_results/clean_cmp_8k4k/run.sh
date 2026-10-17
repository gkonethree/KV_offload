#!/usr/bin/env bash
# Clean comparison: pq vs pq + {paged_eviction, lru, random} offload.
# Qwen3-4B, LongBench-v2 8k prompt + 4096 decode, 70% offload, W=256,
# per-layer token recall + TOKEN-level staging store, 400 staging pages,
# CUDA ADC + fused norm scorer, warm-up mode (decode-only TPOT).
# Two interleaved rounds (order reversed in round 2), one process per run,
# all on the same GPU. Code checksums + GPU load logged before every run.
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/clean_cmp_8k4k
GPU=${GPU:-7}
export CUDA_VISIBLE_DEVICES=$GPU BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16
export BENCH_PROMPT_FILE=$OUT/prompt_8k.txt BENCH_NEW_TOKENS=4096 BENCH_MAX_MODEL_LEN=18000
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1 BENCH_STAGING_SLOTS=400
export OFFLOAD_PACKAGE_STAGING_STORE=token
CODE="bench_1req_off99.py offload_package/staging/token_store.py offload_package/staging/pool.py \
offload_package/staging/residency.py offload_package/integration/*.py offload_package/offload/manager.py \
offload_package/scoring/*.py offload_package/shared/types.py offload_package/config.py \
skylight/skylight_kernels/pq_optimized/*.py skylight/src/skylight/backend.py"

wait_for_space() {
  while [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -lt 2 ]; do
    echo "    $(date '+%T') waiting: / has <2 GB free"; sleep 60
  done
}

one() {  # one <tag> <config>
  local tag=$1 cfg=$2 try
  for try in 1 2 3; do
    wait_for_space
    echo "=== $(date '+%F %T') start $tag ($cfg, attempt $try)"
    echo "code_md5 $(cat $CODE | md5sum | cut -c1-12)" > "$OUT/$tag.meta"
    echo "gpu_at_start $(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$OUT/$tag.meta"
    .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/$tag.log" 2>&1
    if grep -q "decode-only TPOT" "$OUT/$tag.log"; then
      grep -E "decode-only TPOT|TTFT" "$OUT/$tag.log"; cat "$OUT/$tag.meta"; return
    fi
    echo "!!! $tag attempt $try failed; see $OUT/$tag.log"
    grep -q "Traceback" "$OUT/$tag.log" && ! grep -q "No space left" "$OUT/$tag.log" && return
  done
}

for cfg in pq pq+off70 pq+off70-lru pq+off70-rand; do one "${cfg//+/_}_r1" "$cfg"; done
for cfg in pq+off70-rand pq+off70-lru pq+off70 pq; do one "${cfg//+/_}_r2" "$cfg"; done
.venv/bin/python $OUT/summarize.py | tee $OUT/summary.txt
echo "=== $(date '+%F %T') done"
