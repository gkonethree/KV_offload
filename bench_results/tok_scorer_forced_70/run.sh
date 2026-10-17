#!/usr/bin/env bash
# FastKVzip vs random token eviction at 70% offload, with FORCED decoding:
# every config decodes the same reference tokens (pq, no offload, unforced, at
# the same top-k), so selections and fetches follow identical text. Each forced
# run also reports agreement of its own argmax with the reference and the
# reference-token NLL (quality vs pq).
# Qwen3-4B-Instruct-2507, LongBench 8k prompt + 4096 decode, W=256, warm-up
# mode, token kernels + ADC kernel, fixed staging.
#   GPU=0 TOPK=0.1 STAGING=336 bash run.sh
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/tok_scorer_forced_70
GPU=${GPU:?} TOPK=${TOPK:?} STAGING=${STAGING:?}
export CUDA_VISIBLE_DEVICES=$GPU BENCH_MODEL=Qwen/Qwen3-4B-Instruct-2507 BENCH_DTYPE=bfloat16
export BENCH_PROMPT_FILE=$OUT/prompt_8k.txt BENCH_NEW_TOKENS=${NEW:-4096} BENCH_MAX_MODEL_LEN=12352
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1 BENCH_TOPK=$TOPK
export BENCH_STAGING_SLOTS=$STAGING OFFLOAD_PACKAGE_STAGING_STORE=token
P="k${TOPK}s${STAGING}"
REF=$OUT/ref_$P.json
CODE="bench_1req_off99.py bench_force.py offload_package/staging/*.py offload_package/staging/csrc/*.cu \
offload_package/integration/*.py offload_package/scoring/*.py offload_package/config.py offload_package/timing.py \
skylight/skylight_kernels/pq_optimized/*.py skylight/skylight_kernels/pq/codebook.py skylight/src/skylight/backend.py"
wait_for_space() {
  while [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -lt 3 ]; do
    echo "    $(date '+%T') waiting: / has <3 GB free"; sleep 60
  done
}
one() {  # one <config> <tag>   (BENCH_FORCE_TOKENS set by caller or not)
  local cfg=$1 tag="${P}_${1//+/_}_$2"
  wait_for_space
  echo "=== $(date '+%F %T') start $tag"
  echo "code_md5 $(cat $CODE | md5sum | cut -c1-12)" > "$OUT/$tag.meta"
  echo "gpu_at_start $(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$OUT/$tag.meta"
  BENCH_TOKENS_OUT=$OUT/tokens_$tag.json .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/$tag.log" 2>&1
  grep -E "decode-only TPOT|KV blocks|forced:|FALLBACK|Traceback|^\[counts\]|bytes moved" "$OUT/$tag.log"
}
if [ ! -s "$REF" ]; then
  one pq ref && cp "$OUT/tokens_${P}_pq_ref.json" "$REF"
fi
export BENCH_FORCE_TOKENS=$REF
CFGS=${CONFIGS:-pq pq+tok70 pq+tokrand70}
for c in $CFGS; do one "$c" r1; done
for c in $(echo $CFGS | tr ' ' '\n' | tac); do one "$c" r2; done
echo "=== $(date '+%F %T') done"
