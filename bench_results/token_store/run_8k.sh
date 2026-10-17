#!/usr/bin/env bash
# Token store: paged eviction vs LRU, 8k prompt + 4k decode, 70% offload,
# W=256, same 400 staging slots as bench_results/scorers_off70_w256 (page store).
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/token_store
export CUDA_VISIBLE_DEVICES=${GPU:-5} BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1
export BENCH_PROMPT_FILE=bench_results/scorers_off70_w256/prompt_8k.txt BENCH_MAX_MODEL_LEN=18000 BENCH_STAGING_SLOTS=400
echo "=== $(date "+%F %T") smoke"
if ! BENCH_NEW_TOKENS=16 .venv/bin/python bench_1req_off99.py pq+off70 > $OUT/smoke_8k.log 2>&1 \
   || ! grep -q "decode-only TPOT" $OUT/smoke_8k.log; then
  echo "SMOKE FAILED"; tail -15 $OUT/smoke_8k.log; exit 1
fi
grep -E "decode-only TPOT" $OUT/smoke_8k.log
for pair in "8k4k_paged pq+off70" "8k4k_lru pq+off70-lru"; do
  set -- $pair
  echo "=== $(date "+%F %T") start $1"
  BENCH_NEW_TOKENS=4096 .venv/bin/python bench_1req_off99.py $2 > $OUT/$1.log 2>&1
  grep -E "decode-only TPOT|Traceback" $OUT/$1.log | head -3
done
.venv/bin/python $OUT/summarize.py | tee $OUT/summary.txt
echo "=== $(date "+%F %T") done"
