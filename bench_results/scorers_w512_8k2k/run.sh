#!/usr/bin/env bash
# Scorer comparison: pq vs pq+50% offload with paged_eviction / lru / random.
# 8k prefill (315 x ~26-token reps), 2k decode, W=512. One process per config.
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/scorers_w512_8k2k
export CUDA_VISIBLE_DEVICES=0 BENCH_PROMPT_REPS=315 BENCH_NEW_TOKENS=2048 BENCH_INTERVAL=512
for cfg in pq pq+off50 pq+off50-lru pq+off50-rand; do
  echo "=== $(date '+%F %T') start $cfg"
  .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/${cfg//+/_}.log" 2>&1
  cp "/tmp/opencode/tokens_${cfg//+/_}.json" "$OUT/" 2>/dev/null
  grep -E "prompt tokens|new tokens|total time|TPOT|throughput|peak GPU" "$OUT/${cfg//+/_}.log"
done
.venv/bin/python "$OUT/summarize.py" | tee "$OUT/summary.txt"
echo "=== $(date '+%F %T') done"
