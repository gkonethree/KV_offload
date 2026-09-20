# Skylight on Docker

The Dockerfile at the repo root is the canonical install path. This
doc covers building the image, running the SWE-bench agentic bench
end-to-end (single GPU and 8-GPU fan-out), reading outputs, and the
configuration knobs the harness exposes.

For a 5-minute smoke test, see the [README](../README.md). For
building from source on the host (no Docker), use the Dockerfile as
the source of truth and reproduce the same steps with `uv pip install`
into a local venv.

## Contents
- [Building the image](#building-the-image)
- [Running the bench](#running-the-bench)
  - [Single chunk on one GPU](#single-chunk-on-one-gpu)
  - [Full 8-GPU fan-out](#full-8-gpu-fan-out)
  - [Sparse backend](#sparse-backend)
  - [Watching a run](#watching-a-run)
  - [Aggregating after the run](#aggregating-after-the-run)
- [Reading outputs](#reading-outputs)
- [Configuration reference](#configuration-reference)

---

## Building the image

The Dockerfile pins both repo SHAs into the image (as Docker LABELs
and as `.skylight-sha` / `.skylight-kernels-sha` files on disk). Build
context is the **parent** directory containing both repos as siblings:

```bash
cd /path/to/parent  # contains skylight/ and skylight_kernels/ side by side
docker build -f skylight/Dockerfile \
  --build-arg SKYLIGHT_SHA=$(git -C skylight rev-parse HEAD) \
  --build-arg SKYLIGHT_KERNELS_SHA=$(git -C skylight_kernels rev-parse HEAD) \
  -t skylight:dev .
```

Wall time: 8-15 min cold (the native kernel build dominates). Image
size ~17 GB. Subsequent builds re-use cached layers; only the source
copy + kernel rebuild re-run when SHAs change.

Verify what was built:
```bash
docker inspect skylight:dev --format '{{.Config.Labels}}'
docker run --rm skylight:dev python -c "
import skylight, skylight_kernels
from sparse_oracle_topk_sink_local_optimized import _cuda_ops
print('OK')
"
```

The Dockerfile targets `sm_100` (B200) via
`TORCH_CUDA_ARCH_LIST="10.0"`. To target additional architectures, set
it to a space-separated list (e.g. `"10.0 9.0"` for B200 + H100).

### Why the explicit cu13 pin

The Dockerfile pins all `nvidia-cuda-*` sub-packages to 13.0.x
exactly. The naive resolver mixes 13.0.x runtime with 13.2.x
cccl/nvcc; cccl rejects the mismatch with "CUDA compiler and CUDA
toolkit headers are incompatible" during the native kernel build. The
explicit pin avoids this.

### Why the wheel-layout fixes

The pip-shipped cu13 wheels have three issues that block the native
build, all patched in-image:
1. Missing `lib64/` directory (linker expects `-L .../lib64`).
2. Missing `libcuda.so` stub in `lib/stubs/` (needed for build-time linking).
3. Versioned `.so.13` names without the unversioned `.so` symlinks the linker wants.

### Why `uv pip install --python`

`uv venv` does not ship a `pip` shim; calling `.venv/bin/pip` errors
with "No such file or directory". `uv pip install --python
/opt/skylight/.venv/bin/python ...` operates on the venv without
needing pip installed there.

---

## Running the bench

End-to-end flow per instance: vLLM server up -> mini-swe-agent runs N
LLM turns + bash tool calls inside a per-instance docker container ->
agent submits a patch -> swebench harness runs tests inside its own
docker container -> pass/fail.

The harness lives at `src/skylight/bench/agentic.py`.

### Single chunk on one GPU

```bash
mkdir -p /tmp/run-out

docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $PWD/bench-resources/swebench-subset-32.txt:/in/subset.txt:ro \
  -v /tmp/run-out:/out \
  -e SKYLIGHT_GPU_MEM_UTIL=0.95 \
  skylight:dev \
  python -m skylight.bench.agentic \
    --backend dense --model Qwen/Qwen3.5-27B \
    --instances /in/subset.txt --output-dir /out \
    --max-model-len 131072 --server-timeout 900 \
    --step-limit 100 --request-timeout-s 600
```

Wall time: ~60-90 min for 32 instances on one GPU.

### Full 8-GPU fan-out

Partitions the 32-instance subset into 8 chunks, one per GPU, all
running in parallel. Wall time: ~30-60 min for the whole 32.

```bash
TS=$(date +%s); RUN=/tmp/dense-$TS; mkdir -p $RUN
split -n l/8 -d -a 1 bench-resources/swebench-subset-32.txt $RUN/chunk-

for i in 0 1 2 3 4 5 6 7; do
  docker run --rm --gpus "\"device=$i\"" --shm-size=16g \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $RUN/chunk-$i:/in/subset.txt:ro \
    -v $RUN/gpu-$i:/out \
    -e SKYLIGHT_GPU_MEM_UTIL=0.95 \
    --name skylight-gpu-$i -d \
    skylight:dev \
    python -m skylight.bench.agentic \
      --backend dense --config-name dense --model Qwen/Qwen3.5-27B \
      --instances /in/subset.txt --output-dir /out \
      --max-model-len 131072 --server-timeout 900 \
      --step-limit 100 --request-timeout-s 600
done
echo "watch: $RUN ; logs:  docker logs -f skylight-gpu-<i>"
```

### Sparse backend

Same incantation with `--backend sparse` plus the sparse env vars and
CLI flags. Env vars:
```
-e SKYLIGHT_SPARSE_TOPK=0.30
-e SKYLIGHT_SPARSE_SINK=128
-e SKYLIGHT_SPARSE_LOCAL=128
-e SKYLIGHT_SPARSE_CHANNEL_NUM=-1
```
CLI: `--topk 0.30 --sink 128 --local 128 --channel-num -1`.

### Watching a run

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
docker ps                                 # skylight + per-instance eval containers
docker logs -f skylight-gpu-0             # one chunk's orchestrator
```

Kill all skylight containers:
```bash
docker ps -q --filter name=skylight-gpu | xargs -r docker kill
```

### Aggregating after the run

```bash
docker run --rm \
  -v /tmp/dense-<ts>:/run \
  skylight:dev \
  bash /opt/skylight/scripts/post_process_run.sh /run
```

Writes `/tmp/dense-<ts>/aggregate_outcomes.json` with the cluster-wide
outcome counter (see [Reading outputs](#reading-outputs) for the
bucket definitions).

---

## Reading outputs

Each chunk writes to `<output-dir>/`:

| File | What it is |
|---|---|
| `summary.json` | Headline. `n_solved`, `pass_at_1_over_attempted`, `pass_at_1_over_chunk`, full `resolved_ids`, `outcomes` counter, stamped config + knobs + git SHAs + pkg versions + hostname + ISO timestamp. |
| `per_instance.csv` | One row per chunk instance. Columns: `instance_id, in_preds, patch_len, patch_nonempty, eval_completed, resolved, empty_patch_swebench, error_swebench, dropped_by_agent, exit_status, outcome`. |
| `litellm_trace.jsonl` | Per-LLM-call categorized trace (always emitted via `SkylightLitellmModel`). Categories: `ok / timeout / context_overflow / other`. Each row has the full request + response + tokens. |
| `preds.json` | mini-swe-agent's per-instance patch (`{instance_id: {model_patch, ...}}`). |
| `openai__<model>__<run_id>.json` | swebench eval report (`resolved_ids / submitted_ids / empty_patch_ids / error_ids / completed_ids / incomplete_ids`). |
| `metrics.csv` / `progress.jsonl` | Per-tick vLLM `/metrics` scrape + live progress reporter ticks. |
| `server.log` / `agent.log` / `eval.log` | Raw stdout/stderr per subsystem. |

The `outcome` column in `per_instance.csv` is the canonical bucket:

| Outcome | Meaning |
|---|---|
| `resolved` | Agent submitted a patch; swebench eval passed all tests. |
| `submitted_not_resolved` | Patch existed; eval ran but instance not in `resolved_ids`. |
| `submitted_but_failed` | Patch was non-empty but swebench scored it as `empty_patch` (failed to apply, or didn't fix the bug). |
| `agent_gave_up_empty` | Agent exit was `Submitted` with an empty patch (agent gave up cleanly). |
| `lost_to_timeout` | A LiteLLM request hit `--request-timeout-s`; `SkylightLitellmModel` aborted the instance. |
| `lost_to_walltime` | Agent hit `wall_time_limit_seconds` (rare with current config). |
| `lost_to_context_overflow` | LLM call exceeded `--max-model-len`. |
| `lost_to_step_limit` | Agent hit `--step-limit` without producing a patch (`LimitsExceeded`). |
| `lost_to_docker_start` | SWE-bench agent docker container failed to start (`CalledProcessError`, exit 125). |
| `error_swebench` | swebench harness errored on this instance (docker pull fail, harness bug). |
| `dropped_by_agent` | Instance is in the chunk's input but never appeared in `preds.json` (mini-extra filtered it out, often because the ID isn't in the loaded SWE-bench Lite dataset). |
| `swebench_eval_inconsistent` | pytest reported pass but swebench report disagrees (rare; investigate `logs/run_evaluation/*/test_output.txt`). |
| `unknown` | exit_status outside our enumerated set; inspect `agent.log`. |

---

## Configuration reference

### CLI flags (`python -m skylight.bench.agentic --help`)

| Flag | Default | Notes |
|---|---|---|
| `--backend` | required | `dense` or `sparse` |
| `--model` | required | HF model id, e.g. `Qwen/Qwen3.5-27B` |
| `--instances` | required | Path to text file, one instance_id per line |
| `--output-dir` | `bench-results/agentic/<backend>_<ts>` | Per-chunk dir; all artifacts land here |
| `--max-model-len` | 131072 | vLLM `--max-model-len`; cap on prompt + completion |
| `--server-timeout` | 180 | seconds to wait for `/health` after spawning vLLM (set to 900 for cold first-run JIT) |
| `--step-limit` | 100 (or profile default) | Max agent turns per instance. Same value across configs = fair pass@1 comparison. |
| `--wall-time-limit-seconds` | 1800 (or profile default) | Per-instance wall clock cap (`TimeExceeded`). |
| `--profile` | None | `verified-official` (100 steps, 1800s) or `verified-rescue` (150 steps, 3600s). |
| `--retry-from-run` | None | Rerun failed IDs from a prior multi-GPU run dir (`gpu-*/per_instance.csv`). |
| `--request-timeout-s` | 900 | LiteLLM per-call HTTP timeout. `SkylightLitellmModel` aborts the instance on Timeout (no retry against hung server). |
| `--no-retry-docker` | off | Disable post-agent docker failure retry pass. |
| `--no-retry-eval` | off | Disable eval retry for `error_ids`. |
| `--trust-pytest-on-inconsistent` | off | Reclassify pytest-pass / swebench-fail as `resolved`. |
| `--config-name` | `=--backend` | Logical name stamped into `summary.json` |
| `--topk` / `--sink` / `--local` / `--channel-num` | None | Sparse knobs (ignored for `--backend dense`) |

### Env vars (set with `-e` in `docker run`)

| Var | Notes |
|---|---|
| `SKYLIGHT_GPU_MEM_UTIL` | Default 0.5. Bump to 0.9-0.95 to fit a 128K-context KV cache on B200. |
| `SKYLIGHT_SPARSE_TOPK` | Sparse only. Fraction of keys to keep per query (e.g. 0.30). |
| `SKYLIGHT_SPARSE_SINK` | Sparse only. Sink-token attention window (e.g. 128). |
| `SKYLIGHT_SPARSE_LOCAL` | Sparse only. Local-window attention (e.g. 128). |
| `SKYLIGHT_SPARSE_CHANNEL_NUM` | Sparse only. -1 disables per-channel sparsity. |
| `SKYLIGHT_METRICS_LOG_DIR` | Sparse only. Directory for `micro_metrics/micro_metrics.jsonl` (per-layer sparsity). |
| `SKYLIGHT_METRICS_SAMPLING` | Sparse only. Sampling rate for micro metrics (default `0.01`). |
| `SKYLIGHT_DOCKER_START_LOG` | Set automatically by orchestrator to `docker_start.log` in the output dir. |

### Pre-pull SWE-bench images (recommended)

Before a large run, pull all agent/eval images for your instance list:

```bash
python scripts/prep_swebench_images.py bench-resources/swebench-verified-500.txt
python scripts/prep_swebench_images.py bench-resources/swebench-verified-200.txt --workers 4
```

Matplotlib smoke test (Phase 0):

```bash
docker pull docker.io/swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-13989:latest
docker run --rm docker.io/swebench/sweb.eval.x86_64.matplotlib_1776_matplotlib-13989:latest echo ok
```

### Rescue rerun (optional second pass)

After an official run completes, rerun infra failures with the rescue profile:

```bash
python -m skylight.bench.agentic --profile verified-rescue \
  --retry-from-run /path/to/sparse-verified-RUN \
  --backend sparse --model Qwen/Qwen3.5-27B \
  --instances bench-resources/swebench-verified-500.txt \
  --output-dir /path/to/rescue/gpu-0 ...
```

The orchestrator also retries docker start failures and eval `error_ids`
automatically unless `--no-retry-docker` / `--no-retry-eval` are set.

At least one of TOPK/SINK/LOCAL/CHANNEL_NUM must be > 0 for `--backend
sparse`; the config loader enforces this at startup.

### mini-swe-agent overrides

We set the following overrides in
`src/skylight/bench/benchmarks/mini_swe_agent.py:build_agent_cmd`,
beyond mini-swe-agent's upstream `swebench.yaml`:
- `model.model_name=openai/<model>`
- `model.model_kwargs.api_base=http://localhost:<port>/v1`
- `model.model_kwargs.api_key=EMPTY`
- `model.model_kwargs.request_timeout=<--request-timeout-s>`
- `model.model_class=skylight.bench.litellm_model.SkylightLitellmModel` (always; required for `litellm_trace.jsonl`)
- `agent.step_limit=<--step-limit>`

Upstream `swebench.yaml` sets `agent.cost_limit=3.0` and
`environment.timeout=60` (bash command timeout); we don't override
these.
