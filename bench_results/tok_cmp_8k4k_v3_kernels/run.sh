#!/usr/bin/env bash
# Phase B (token-staging CUDA kernels) speed check, same setup as v2:
# Qwen3-4B-Instruct-2507, LongBench 8k + 4096 decode, 70% token offload,
# W=256, warm-up mode. Kernels on (default). 230 staging pages = v2
# (comparable), plus FastKVzip at 336 pages (auto size for 12.3k context).
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/tok_cmp_8k4k_v3_kernels
GPU=${GPU:-5}
export CUDA_VISIBLE_DEVICES=$GPU BENCH_MODEL=Qwen/Qwen3-4B-Instruct-2507 BENCH_DTYPE=bfloat16
export BENCH_PROMPT_FILE=$OUT/prompt_8k.txt BENCH_NEW_TOKENS=4096 BENCH_MAX_MODEL_LEN=12352
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1
export OFFLOAD_PACKAGE_STAGING_STORE=token
wait_for_space() {
  while [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -lt 3 ]; do
    echo "    $(date '+%T') waiting: / has <3 GB free"; sleep 60
  done
}
one() {  # one <tag> <config> <staging pages>
  wait_for_space
  echo "=== $(date '+%F %T') start $1 ($2, staging $3)"
  BENCH_STAGING_SLOTS=$3 .venv/bin/python bench_1req_off99.py "$2" > "$OUT/$1.log" 2>&1
  grep -E "decode-only TPOT|TTFT|KV blocks|FALLBACK|Traceback" "$OUT/$1.log"
  awk '/\[OFFLOAD TIMING\] FINAL/{f=1} f' "$OUT/$1.log" | grep -E "wrap\.|pool\.|bytes|tok\." 
}
one pq_tok70_s230 pq+tok70 230
one pq_tokrand70_s230 pq+tokrand70 230
one pq_tok70_s336 pq+tok70 336
echo "=== $(date '+%F %T') done"
