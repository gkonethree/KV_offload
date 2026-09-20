# H200 million-context crossover experiment

## Objective

Measure the maximum usable context on an H200 and identify the context and
generation lengths at which Skylight BMM beats the dense FlashInfer baseline.
This is a performance stress test; points beyond the model's native context
limit are valid throughput measurements but must be labeled RoPE-scaled.

## Fixed release inputs

- Model: `Qwen/Qwen3.5-9B`
- Skylight source snapshot: `f53bcfdcf105e7a9c4a4389f8a0f251df339e9ce`
- Separate kernels snapshot: `b88bd0f4758ebb90cd388786af3e4f4fbd9c3cde`
- H200 NVL, SM90, one visible GPU
- Dtype: `bfloat16`; block size: `16`; GDN prefill: `triton`
- BMM profile: `block_minmax`, target top-k `0.10`, sink `64`, local `64`,
  full score head (`channel_num=-1`)

## Phase 1 matrix

The context curve tests `64K, 128K, 256K, 512K, 768K, 1M` input tokens with
`128` generated tokens. Dense and BMM use identical requests. CUDA Graph is the
primary mode; eager is a diagnostic subset at `64K`, `256K`, and the largest
context that fits.

At every context, the runner probes batch sizes in descending order
`64, 32, 16, 8, 4, 2, 1` and keeps the largest batch that completes without an
OOM or failed request. The selected batch is then used for the dense/BMM pair.
The server uses `--gpu-memory-utilization 0.99`, `--max-num-seqs` equal to the
selected batch, and `max_model_len = context + generated + 512`.

If the model rejects a context above its native limit, the runner retries with
an explicit RoPE scaling override. The result records the native or scaled
mode and never presents a scaled point as a correctness result.

## Measurements and crossover rule

Each condition records mean and P99 TPOT, median and P99 TTFT, output
throughput, completed/failed requests, HBM usage, GPU utilization, memory
utilization, power, resolved commands, package/source revisions, and server
logs. A BMM point is a confirmed win only when it has zero failures, at least
5% lower mean TPOT, and at least 5% higher output throughput than its paired
dense point. Otherwise it is neutral or a regression.

The primary plots are:

1. Context length versus mean TPOT for dense and BMM, with the first confirmed
   crossover marked.
2. Context length versus BMM/dense output-throughput ratio.
3. Fixed-context generation-length curve using `64, 256, 512, 1024, 2048,
   4096` generated tokens at the largest stable batch for `32K` (or the
   largest lower context that fits).

## Artifacts and operation

All output goes under a UTC-named directory matching
`/workspace/berkeley/skylight/worklogs/skylight_vllm_h200_*`, with one directory
per hardware/backend/context/batch/mode condition. The runner
is detached with a durable log and PID file. It tears down each server before
the next condition, records OOM boundaries instead of retrying indefinitely,
and leaves the H200 idle after completion.

Phase 2 (large-batch saturation at several fixed contexts) is intentionally
separate so Phase 1 answers the maximum-context question first.
