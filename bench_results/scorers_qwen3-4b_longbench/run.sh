#!/usr/bin/env bash
# Qwen3-4B scorer comparison on a REAL long-context input (LongBench-v2 sample
# 66f36490..., single-doc QA, truncated to 8192 tokens), 2k decode, W=512,
# 50% offload. Staging: auto (offload-aware estimate) vs 709 fixed (the old
# auto size). Offload timing on
# (host perf_counter only, no CUDA syncs). One process per run, one GPU.
set -u
cd /home/gaurav_kumar/KV_offload
OUT=bench_results/scorers_qwen3-4b_longbench
export CUDA_VISIBLE_DEVICES=0 BENCH_MODEL=Qwen/Qwen3-4B BENCH_DTYPE=bfloat16
export BENCH_PROMPT_FILE=$OUT/prompt.txt BENCH_NEW_TOKENS=2048 BENCH_INTERVAL=512
export SKYLIGHT_OFFLOAD_TIMING=1
# Warm-up mode: one-time PQ codebook fit + JIT paid before timing; prefix
# caching off; decode-only TPOT from per-request token timestamps.
export BENCH_WARMUP=1

wait_for_space() {  # block until / has >= 10 GB free (another user's job
  # periodically fills the disk; a run that starts on a full disk just dies)
  while [ "$(df --output=avail -B1G / | tail -1 | tr -d ' ')" -lt 10 ]; do
    echo "    $(date '+%T') waiting: / has <10 GB free"; sleep 60
  done
}

one() {  # one <config> <staging: auto|N>; skips finished runs, retries 3x
  local cfg=$1 st=$2 tag="${1//+/_}_st$2" try
  if grep -q "decode-only TPOT" "$OUT/$tag.log" 2>/dev/null; then
    echo "=== skip $cfg staging=$st (already done)"; return
  fi
  if [ "$st" = auto ]; then unset BENCH_STAGING_SLOTS; else export BENCH_STAGING_SLOTS=$st; fi
  for try in 1 2 3; do
    wait_for_space
    echo "=== $(date '+%F %T') start $cfg staging=$st (attempt $try)"
    .venv/bin/python bench_1req_off99.py "$cfg" > "$OUT/$tag.log" 2>&1
    if grep -q "decode-only TPOT" "$OUT/$tag.log"; then
      cp "/tmp/opencode/tokens_${cfg//+/_}.json" "$OUT/tokens_$tag.json" 2>/dev/null
      grep -E "decode-only TPOT|TTFT|total time" "$OUT/$tag.log"; return
    fi
    echo "!!! $cfg staging=$st attempt $try failed (free: $(df -h --output=avail / | tail -1)); see $OUT/$tag.log"
    grep -qE "Traceback" "$OUT/$tag.log" && ! grep -q "No space left" "$OUT/$tag.log" && return  # real crash: don't retry
  done
}

# Reuse a finished pq baseline (rerun only if its log has no TPOT).
wait_for_space
echo "=== $(date '+%F %T') smoke test (warm-up mode, pq+off50, 16 tokens)"
if ! BENCH_NEW_TOKENS=16 .venv/bin/python bench_1req_off99.py pq+off50 > "$OUT/smoke.log" 2>&1 \
   || ! grep -q "decode-only TPOT" "$OUT/smoke.log"; then
  echo "SMOKE TEST FAILED -- see $OUT/smoke.log"; tail -15 "$OUT/smoke.log"; exit 1
fi
grep -E "decode-only TPOT|TTFT" "$OUT/smoke.log"

one pq auto
for st in auto 709; do  # auto = offload-aware estimate (350 here); 709 = old auto size
  for cfg in pq+off50 pq+off50-lru pq+off50-rand; do one "$cfg" "$st"; done
done
unset BENCH_STAGING_SLOTS
.venv/bin/python "$OUT/summarize.py" | tee "$OUT/summary.txt"
echo "=== $(date '+%F %T') done"
