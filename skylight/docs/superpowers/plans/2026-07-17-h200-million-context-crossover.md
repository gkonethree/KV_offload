# H200 Million-Context Crossover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with checkpoints.

**Goal:** Run a resumable H200 Phase 1 experiment that finds the maximum usable context through 1M tokens and identifies real dense/BMM crossover points.

**Architecture:** Add a small experiment-only Python orchestrator that invokes the existing `skylight.bench.run_serving` entry point for each condition, records GPU telemetry, and writes one immutable artifact directory per condition. It probes batch capacity before each dense/BMM pair, retries beyond the native context with an explicit RoPE override, and can resume by skipping complete result files. No serving or kernel implementation code changes are required.

**Tech Stack:** Python 3.12, existing `run_serving`/vLLM CLI, Bash setup, `nvidia-smi` CSV telemetry, JSON/CSV summaries.

## Global Constraints

- Model: `Qwen/Qwen3.5-9B`.
- Context points: `65536, 131072, 262144, 524288, 786432, 1048576` input tokens.
- Generated tokens: `128` for the context curve.
- BMM: `block_minmax`, top-k `0.10`, sink `64`, local `64`, channel number `-1`, block size `16`.
- Main mode: CUDA Graph; eager only at `64K`, `256K`, and the largest fitting point.
- GPU memory utilization: `0.99`; batches probed descending `64,32,16,8,4,2,1`.
- A BMM win requires zero failures, at least 5% lower mean TPOT, and at least 5% higher output throughput.
- H200 is classified as Hopper/SM90 and uses kernels revision `b88bd0f4758ebb90cd388786af3e4f4fbd9c3cde`.

---

### Task 1: Add the experiment orchestrator

**Files:**
- Create: `scripts/long_context_crossover.py`
- Test: `tests/test_long_context_crossover.py`

**Interfaces:**
- `build_condition_command(...) -> list[str]` returns the exact `run_serving` command for dense or BMM.
- `condition_key(...) -> str` returns a stable artifact key from mode, backend, context, batch, and generation length.
- `summarize_pair(dense: dict, bmm: dict) -> dict` returns TPOT/throughput ratios and `win`/`neutral`/`regression` classification.
- CLI accepts `--model`, `--output-root`, `--gpu`, `--contexts`, `--generated`, `--gpu-memory-utilization`, `--batch-candidates`, `--eager-contexts`, `--rope-overrides`, `--dry-run`, and `--resume`.

- [ ] **Step 1: Write failing tests** for command construction, stable paths, resume skipping, and crossover classification.
- [ ] **Step 2: Run `python3 -m pytest -q tests/test_long_context_crossover.py` and confirm the new tests fail because the module is absent.
- [ ] **Step 3: Implement the minimal orchestrator.** Use `subprocess.run` for each condition, write `status.json` even for OOM/nonzero exits, run an `nvidia-smi` sampler during each condition, and never reuse a partial result directory.
- [ ] **Step 4:** Run the focused tests and the existing qualification tests; require all to pass.
- [ ] **Step 5:** Run `python3 scripts/long_context_crossover.py --dry-run --contexts 65536,131072 --generated 128 --batch-candidates 4,2,1` and verify the emitted dense/BMM commands and artifact paths.
- [ ] **Step 6:** Commit as `feat: add long-context crossover experiment`.

### Task 2: Prepare the H200 worklog and environment

**Files/locations:**
- Remote worklog: a UTC-named `/workspace/berkeley/skylight/worklogs/skylight_vllm_h200_*` directory
- Remote node-local environment: `/opt/skylight-h200/venv`
- Remote compile cache: `/var/tmp/skylight-h200`

**Interfaces:**
- Source checkout is pinned to Skylight `f53bcfdcf105e7a9c4a4389f8a0f251df339e9ce`.
- Kernels checkout is pinned to `b88bd0f4758ebb90cd388786af3e4f4fbd9c3cde`.

- [ ] **Step 1:** Clone the two private repositories using `/root/.env` only for the Git authorization header; never print or persist credentials.
- [ ] **Step 2:** Run `SKYLIGHT_ENV_DIR=/opt/skylight-h200/venv SKYLIGHT_COMPILE_CACHE_DIR=/var/tmp/skylight-h200 ./setup.sh --hardware hopper`.
- [ ] **Step 3:** Verify H200/SM90 runtime JSON, kernel tests, imports, and the exact source/kernel revisions.
- [ ] **Step 4:** Copy the experiment script into the worklog and run its local dry-run on the H200.

### Task 3: Run Phase 1 detached and resumably

**Files/locations:**
- Results: the `crossover/` directory inside the UTC-named H200 worklog
- Runner log: `/opt/skylight-h200/phase1.log`
- Runner PID: `/opt/skylight-h200/phase1.pid`

**Interfaces:**
- Every condition writes `result.json`, `resolved.json`, `versions.json`, server/client logs, telemetry CSV, and `status.json`.
- The runner exits nonzero only after all planned conditions have been attempted; OOM points remain in the summary as capacity boundaries.

- [ ] **Step 1:** Launch the runner with `nohup` and redirected stdout/stderr so SSH disconnects cannot terminate it.
- [ ] **Step 2:** Verify the first context point starts and the runner has `PPID=1`.
- [ ] **Step 3:** Monitor result count, current context/batch, HBM usage, and failed/OOM conditions with short SSH probes.
- [ ] **Step 4:** After completion, build `summary.json`, `summary.csv`, and the three requested plots from the immutable condition artifacts.
- [ ] **Step 5:** Copy the finished H200 result tree locally and report the maximum native/scaled context and every confirmed BMM win.

### Task 4: Verification and handoff

- [ ] **Step 1:** Confirm every successful point has `failed == 0`, a complete telemetry file, and matching dense/BMM workload settings.
- [ ] **Step 2:** Confirm any scaled point is marked `rope_scaled=true` and is not presented as a native-context result.
- [ ] **Step 3:** Run `git diff --check`, the focused tests, and the full release qualification tests.
- [ ] **Step 4:** Leave the H200 running only while the detached Phase 1 process is active; shut it down after artifacts are copied.
