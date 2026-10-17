#!/usr/bin/env bash
# Token store exactness: 8k prompt, 70% offload, 400 staging slots, every
# layer verified for the first steps (ref_cos vs fp64, byte audit).
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/token_store
export CUDA_VISIBLE_DEVICES=${GPU:-5} BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16
export BENCH_INTERVAL=256 BENCH_WARMUP=1 SKYLIGHT_OFFLOAD_TIMING=1
export BENCH_PROMPT_FILE=bench_results/scorers_off70_w256/prompt_8k.txt BENCH_MAX_MODEL_LEN=18000 BENCH_STAGING_SLOTS=400
export BENCH_NEW_TOKENS=${TOKENS:-16} SKYLIGHT_PQ_VERIFY=1 SKYLIGHT_PQ_VERIFY_LAYERS=all SKYLIGHT_PQ_VERIFY_STEPS=${STEPS:-4}
.venv/bin/python bench_1req_off99.py ${CFG:-pq+off70} > $OUT/verify_8k.log 2>&1
echo "exit=$?" >> $OUT/verify_8k.log
