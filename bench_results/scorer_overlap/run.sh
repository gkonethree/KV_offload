#!/usr/bin/env bash
# Waits for the scorer sweep to exit, then records offload sets / hit rates.
cd /home/gaurav_kumar/KV_offload
while pgrep -u gaurav_kumar -f "bench_results/scorers_qwen3-4b_longbench/run.sh" >/dev/null; do sleep 15; done
echo "=== $(date '+%T') start recording"
export CUDA_VISIBLE_DEVICES=0 BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16 \
  BENCH_PROMPT_FILE=bench_results/scorers_qwen3-4b_longbench/prompt.txt \
  BENCH_NEW_TOKENS=2048 BENCH_INTERVAL=512
.venv/bin/python bench_results/scorer_overlap/record.py > bench_results/scorer_overlap/record.log 2>&1
grep -E "RECORDED|Traceback|Error" bench_results/scorer_overlap/record.log
echo "=== $(date '+%T') done"
