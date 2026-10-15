#!/usr/bin/env bash
# Scorer comparison on Qwen3-4B: pq vs pq+50% offload with paged_eviction /
# lru / random. 8k prefill (390 reps x ~21 Qwen tokens), 2k decode, W=512.
# One process per config, sequential on one GPU so TPOT is comparable.
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/scorers_qwen3-4b_w512_8k2k
export CUDA_VISIBLE_DEVICES=0 BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16

echo "=== $(date '+%F %T') smoke test (pq+off50, short)"
if ! BENCH_PROMPT_REPS=100 BENCH_NEW_TOKENS=16 BENCH_INTERVAL=512 \
     .venv/bin/python bench_1req_off99.py pq+off50 > "$OUT/smoke.log" 2>&1; then
  echo "SMOKE TEST FAILED -- see $OUT/smoke.log"; tail -20 "$OUT/smoke.log"; exit 1
fi
grep -E "prompt tokens|new tokens|TPOT" "$OUT/smoke.log"

export BENCH_PROMPT_REPS=390 BENCH_NEW_TOKENS=2048 BENCH_INTERVAL=512
for cfg in pq pq+off50 pq+off50-lru pq+off50-rand; do
  echo "=== $(date '+%F %T') start $cfg"
  .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/${cfg//+/_}.log" 2>&1 \
    || echo "!!! $cfg exited with an error -- see $OUT/${cfg//+/_}.log"
  cp "/tmp/opencode/tokens_${cfg//+/_}.json" "$OUT/" 2>/dev/null
  grep -E "prompt tokens|new tokens|total time|TPOT|throughput|peak GPU" "$OUT/${cfg//+/_}.log"
done
.venv/bin/python "$OUT/summarize.py" | tee "$OUT/summary.txt"
echo "=== $(date '+%F %T') done"
