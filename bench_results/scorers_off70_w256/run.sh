#!/usr/bin/env bash
# Paged eviction vs LRU at 70% offload, per-layer token recall, W=256,
# Qwen3-4B on a LongBench-v2 sample (66f36490..., single-doc QA):
#   A) 8k prompt + 4k decode   (max_model_len 18000)
#   B) 16k prompt + 8k decode  (max_model_len 25600)
# FIXED staging = half the blocks offloaded at prefill (176 / 355 slots), so
# the offloaded set never fits: evictions + re-fetches from the first decode
# step. Warm-up mode: decode-only TPOT.
# One process per run, GPU 0, sequential. Waits for the scorer-overlap
# recording to finish first, and for >=10 GB free on / before each run.
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/scorers_off70_w256
while pgrep -u gaurav_kumar -f "bench_results/scorer_overlap/run.sh" >/dev/null; do sleep 15; done
export CUDA_VISIBLE_DEVICES=0 BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1
unset BENCH_STAGING_SLOTS

wait_for_space() {
  while [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -lt 10 ]; do
    echo "    $(date '+%T') waiting: / has <10 GB free"; sleep 60
  done
}

one() {  # one <tag> <config> <prompt file> <new tokens> <max_model_len> <staging slots>
  local tag=$1 cfg=$2 try
  if grep -q "decode-only TPOT" "$OUT/$tag.log" 2>/dev/null; then
    echo "=== skip $tag (already done)"; return
  fi
  for try in 1 2 3; do
    wait_for_space
    echo "=== $(date '+%F %T') start $tag ($cfg, attempt $try)"
    BENCH_PROMPT_FILE=$3 BENCH_NEW_TOKENS=$4 BENCH_MAX_MODEL_LEN=$5 BENCH_STAGING_SLOTS=$6 \
      .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/$tag.log" 2>&1
    if grep -q "decode-only TPOT" "$OUT/$tag.log"; then
      grep -E "prompt tokens|new tokens|decode-only TPOT|TTFT" "$OUT/$tag.log"; return
    fi
    echo "!!! $tag attempt $try failed (free: $(df -h --output=avail / | tail -1)); see $OUT/$tag.log"
    grep -q "Traceback" "$OUT/$tag.log" && ! grep -q "No space left" "$OUT/$tag.log" && return
  done
}

one 8k4k_paged  pq+off70     $OUT/prompt_8k.txt  4096 18000 176
one 8k4k_lru    pq+off70-lru $OUT/prompt_8k.txt  4096 18000 176
one 16k8k_paged pq+off70     $OUT/prompt_16k.txt 8192 25600 355
one 16k8k_lru   pq+off70-lru $OUT/prompt_16k.txt 8192 25600 355
.venv/bin/python $OUT/summarize.py | tee $OUT/summary.txt
echo "=== $(date '+%F %T') done"
