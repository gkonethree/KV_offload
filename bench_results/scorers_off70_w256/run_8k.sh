#!/usr/bin/env bash
# Paged eviction vs LRU: 8k LongBench prompt + 4k decode, 70% offload, W=256,
# per-layer token recall, FIXED 400 staging slots: above the per-layer need
# at prefill (~353 offloaded blocks, nearly all touched by each layer) but
# below the offloaded set later (grows to ~532) -> evictions + re-fetches.
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/scorers_off70_w256
while pgrep -u gaurav_kumar -f "fallback_det.py" >/dev/null; do sleep 10; done
export CUDA_VISIBLE_DEVICES=0 BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1
export BENCH_PROMPT_FILE=$OUT/prompt_8k.txt BENCH_MAX_MODEL_LEN=18000 BENCH_STAGING_SLOTS=400

wait_for_space() {
  while [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -lt 10 ]; do
    echo "    $(date '+%T') waiting: / has <10 GB free"; sleep 60
  done
}

wait_for_space
echo "=== $(date '+%F %T') smoke (pq+off70, 400 slots, 16 tokens)"
if ! BENCH_NEW_TOKENS=16 .venv/bin/python bench_1req_off99.py pq+off70 > "$OUT/smoke_8k.log" 2>&1 \
   || ! grep -q "decode-only TPOT" "$OUT/smoke_8k.log"; then
  echo "SMOKE FAILED -- see $OUT/smoke_8k.log"; tail -15 "$OUT/smoke_8k.log"; exit 1
fi
grep -E "decode-only TPOT" "$OUT/smoke_8k.log"

for pair in "8k4k_paged pq+off70" "8k4k_lru pq+off70-lru"; do
  set -- $pair; tag=$1; cfg=$2
  for try in 1 2 3; do
    wait_for_space
    echo "=== $(date '+%F %T') start $tag ($cfg, attempt $try)"
    BENCH_NEW_TOKENS=4096 .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/$tag.log" 2>&1
    if grep -q "decode-only TPOT" "$OUT/$tag.log"; then
      grep -E "decode-only TPOT|TTFT" "$OUT/$tag.log"; break
    fi
    echo "!!! $tag attempt $try failed; see $OUT/$tag.log"
    grep -q "Traceback" "$OUT/$tag.log" && ! grep -q "No space left" "$OUT/$tag.log" && break
  done
done
.venv/bin/python $OUT/summarize.py | tee $OUT/summary.txt
echo "=== $(date '+%F %T') done"
