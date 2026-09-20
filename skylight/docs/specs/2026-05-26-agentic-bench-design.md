# Agentic Benchmark Harness — Design

- **Date:** 2026-05-26
- **Status:** Draft, pending user review
- **Owner:** krishna
- **Related tasks:** #8 (Phase 6 — SWE-bench Lite full sweep)

## Goal

Land a single command that runs the three sparse/dense configs against a
subset of SWE-bench Lite on one B200 node and emits a comparison table of
pass@1:

```
python -m skylight.bench.agentic_sweep \
    --benchmark mini-swe-agent \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --instances bench-resources/swebench-subset-32.txt \
    --configs dense,sparse-token,sparse-double
```

Output (stdout + `bench-results/agentic/sweep-summary.json`):

```
config              n_solved   n_total   pass@1   elapsed_s
------------------  ---------  --------  -------  ---------
dense                  N        32       0.XX       NNNN
sparse-token           N        32       0.XX       NNNN
sparse-double          N        32       0.XX       NNNN
```

Everything else in this spec exists only to make that command work and to
make each step on the path verifiable.

## Constraints

1. **Minimal new code.** Reuse `src/skylight/bench/run_serving.py` for the
   server lifecycle. Reuse `src/skylight/bench/sweep.py`'s table-formatting
   and sweep-loop shape. No new orchestration patterns invented.
2. **External deps, not vendored.** Agent and eval logic stay upstream:
   `mini-swe-agent` (`SWE-bench/mini-swe-agent`) and `swebench` (the
   official harness) installed via a pyproject extra. We write no agent
   loop and no eval container code.
3. **Single-node.** One B200 box (vb200-14). No node-agent, no
   control-server, no SQLite, no multi-node sharding. JSON files on disk.
4. **Incremental delivery.** Each repo state passes a verify gate before
   the next step starts. No single "everything lands at once" PR.
5. **Subset ladder.** Geometric: 32 → 64 → 128 → full. Each subset is a
   checked-in plain text file of instance IDs.

## Non-goals

Explicitly excluded from this design and the first implementation:

- Multi-node orchestration (HTTP node-agents, control-server, sharding).
- SQLite persistence — JSON files on disk are sufficient at our scale.
- Other agentic benchmarks (TAU-bench, AgentBench, HumanEval). The
  protocol in `benchmarks/base.py` is shaped so they could plug in, but
  no other adapter is written in the first PR.
- Parallel instance evaluation. Instances run sequentially within a
  config; the bench server batches requests internally.
- Custom subset-curation tooling. Subsets are plain text files;
  selecting different ones = edit the file or pass a different
  `--instances` argument.
- Per-instance docker image management. Defer to whatever
  `swebench.harness.run_evaluation` does upstream.

## Architecture

```
src/skylight/bench/
├── run_serving.py             # existing — server spawn/health/teardown helpers
├── sweep.py                   # existing — perf sweep orchestrator
├── agentic.py                 # NEW — one-config runner (server + agent + eval + summary)
├── agentic_sweep.py           # NEW — outer loop over configs, emits comparison table
├── progress.py                # NEW — metrics scraper + progress formatter (shared with sweep)
└── benchmarks/
    ├── __init__.py            # NEW — registry: name -> adapter
    ├── base.py                # NEW — AgenticBenchmark protocol
    └── mini_swe_agent.py      # NEW — adapter for mini-swe-agent + swebench eval

bench-resources/
└── swebench-subset-32.txt     # NEW — 32 curated SWE-bench Lite instance IDs

tests/
├── test_bench_agentic.py                       # NEW
├── test_bench_benchmarks_mini_swe_agent.py     # NEW
├── test_bench_agentic_sweep.py                 # NEW
└── test_bench_progress.py                      # NEW

pyproject.toml                 # MODIFIED — adds [project.optional-dependencies].agentic
```

### Per-run output layout

Each `run_one_config` call writes a self-contained directory:

```
bench-results/agentic/<run_id>/
├── config.json            # resolved config (model, backend, knobs, instances, timestamps)
├── orchestrator.log       # orchestrator's own log (server lifecycle, progress ticks, errors)
├── server.log             # captured stdout+stderr of skylight serve / vllm api_server
├── agent.log              # captured stdout+stderr of mini-swe-agent subprocess
├── eval.log               # captured stdout+stderr of swebench.harness.run_evaluation
├── patches.jsonl          # written by the agent, one record per instance attempted
├── metrics.csv            # timeseries of /metrics samples scraped every 5s
├── progress.jsonl         # one JSON record per progress tick (every 30s)
└── summary.json           # final {n_solved, n_total, pass_at_1, elapsed_s, errors}
```

Sweep runs add `bench-results/agentic/<sweep_id>/sweep-summary.json` aggregating
the per-config summaries plus a `sweep.log` for the outer-loop chatter.

### Reuse from existing modules

From `skylight.bench.run_serving`:
- `free_port()` — OS-allocated port for the server
- `build_serve_cmd(backend, model, port, max_model_len, enforce_eager)` —
  argv for `skylight serve` or `vllm api_server`
- `build_env(backend, topk, sink, local, channel_num)` — env vars for
  sparse knobs
- `wait_for_health(port, timeout, proc)` — poll /health
- `terminate_group(proc)` — SIGTERM → SIGKILL the server process group

From `skylight.bench.sweep`:
- The `format_table(results)` pattern (re-implemented for accuracy
  columns rather than perf columns; not literally reused, but the same
  shape).

## Components

### 1. `benchmarks/base.py` — protocol

```python
from typing import Protocol
from pathlib import Path


class AgenticBenchmark(Protocol):
    """Adapter shape every agentic benchmark must satisfy (duck-typed)."""
    name: str  # e.g. "mini-swe-agent"

    def default_instances_path(self) -> Path: ...
    def build_agent_cmd(
        self, model: str, base_url: str, instances: Path, out_dir: Path,
    ) -> list[str]: ...
    def build_eval_cmd(self, out_dir: Path) -> list[str]: ...
    def parse_results(self, out_dir: Path) -> dict: ...  # {n_solved, n_total, pass_at_1}
```

A registry in `benchmarks/__init__.py` maps `"mini-swe-agent"` →
`MiniSweAgent()` instance. Adding a new benchmark = new module that
implements this `Protocol` + one line in the registry. Adapters do not
need to inherit anything; structural typing is sufficient.

### 2. `benchmarks/mini_swe_agent.py` — first adapter

- `build_agent_cmd`: builds the `mini-swe-agent` CLI argv with
  `--model openai/<model> --base-url <server>/v1 --instances <file>
  --output <out>/patches.jsonl`.
- `build_eval_cmd`: builds `python -m swebench.harness.run_evaluation
  --predictions_path <out>/patches.jsonl --max_workers 1 --run_id <out_name>
  --instance_ids_path <file>`.
- `parse_results`: reads swebench's results JSON, returns
  `{n_solved, n_total, pass_at_1}`.
- `default_instances_path`: returns
  `bench-resources/swebench-subset-32.txt`.

Exact upstream CLI flags will be confirmed against
`mini-swe-agent`/`swebench` README during step 2 (manual dry run).
Adjustments here are local and contained.

### 3. `agentic.py` — one-config runner

```python
def run_one_config(
    benchmark: AgenticBenchmark,
    backend: str,            # "dense" | "sparse"
    model: str,
    instances: Path,
    output_dir: Path,
    max_model_len: int,
    server_timeout_s: float,
    topk: Optional[float] = None,
    sink: Optional[int] = None,
    local: Optional[int] = None,
    channel_num: Optional[int] = None,
) -> dict:
    """
    Lifecycle: spawn server → wait health → run agent → run eval → parse.
    Returns: {config, backend, n_solved, n_total, pass_at_1, elapsed_s, errors}.
    Caller is responsible for log file + summary.json writing.
    """
```

Internals:
1. `port = free_port()`
2. `proc = subprocess.Popen(build_serve_cmd(...), env=build_env(...),
   start_new_session=True)`
3. `wait_for_health(port, ...)`; if it fails, return `errors=["server_unhealthy"]`
4. `subprocess.run(benchmark.build_agent_cmd(...), capture stdout/stderr to file, timeout)`
5. `subprocess.run(benchmark.build_eval_cmd(...), capture, timeout)`
6. `summary = benchmark.parse_results(out_dir)`
7. `finally: terminate_group(proc)`

### 4. `agentic_sweep.py` — outer loop

Mirrors `sweep.py` for perf. Named CONFIGS table (`dense`, `sparse-token`,
`sparse-double`) plus per-axis overrides (`--topk`, `--sink`, `--local`,
`--channel-num`). For each config, calls `run_one_config`, accumulates
results, prints the comparison table, writes `sweep-summary.json`.

### 5. `progress.py` — observability

Shared by `agentic.py` (per-config) and `agentic_sweep.py` (cross-config
chatter). Three pieces:

- `class MetricsScraper(threading.Thread)`: GETs
  `http://localhost:<port>/metrics` every `interval_s` (default 5s),
  parses the few Prometheus metrics we care about
  (`vllm:generation_tokens_total`, `vllm:request_success_total`,
  `vllm:e2e_request_latency_seconds_*`, `skylight_observed_sparsity_fraction`),
  appends rows to `metrics.csv`, keeps the latest sample in memory for
  the progress formatter. Daemon thread; bails silently on scrape
  failures so the orchestrator never blocks on `/metrics` hiccups.

- `def format_progress_line(snapshot) -> str`: pure function. Takes a
  snapshot dict + counts (`patches_written`, `instances_total`,
  `elapsed_s`) and returns a single status line:

  ```
  [agentic] sparse-token  8/32 [████░░░░░░░] 25%  812 tok/s  spars=0.092  4m21s elapsed  ETA 13m18s
  ```

- `class ProgressReporter`: drives a periodic tick (default every
  30s) from the main thread, writes one `progress.jsonl` record + one
  stdout line per tick. Configurable cadence via
  `SKYLIGHT_BENCH_PROGRESS_INTERVAL_S` env var for tests / debugging.

**Runtime cost.** Scrape = one localhost HTTP GET every 5s, ~1-2ms each,
parsed with `_parse_prometheus_text` (~50 LOC, no regex). Progress tick
= file-stat on `patches.jsonl` + line count of a small in-memory deque +
one `print` + one `json.dumps`. Total well under 0.1% of wall clock on
the runs we care about (instance-level wall clocks are 5-15 GPU-min). No
NVTX, no Python `cProfile`, no in-process instrumentation that hurts the
hot path.

**Log levels.** Orchestrator uses Python `logging` configured to:
- `INFO` → orchestrator.log + stdout: server lifecycle, instance
  start/finish (parsed from agent.log lines), progress ticks, config
  transitions
- `WARNING` → server unhealthy, agent timeout, eval failure
- `DEBUG` → only when `SKYLIGHT_BENCH_DEBUG=1` is set: per-scrape
  payloads, raw subprocess return codes, etc. Off by default.

### 6. `bench-resources/swebench-subset-32.txt`

32 instance IDs from `princeton-nlp/SWE-bench_Lite`, spanning multiple
repos for failure-mode diversity. One ID per line:

```
django__django-12915
sympy__sympy-20590
scikit-learn__scikit-learn-13439
flask__flask-4992
sphinx-doc__sphinx-8595
... (27 more)
```

The exact 32 are picked once during step 1 and locked. Subsequent
ladders (`-64.txt`, `-128.txt`) are supersets.

## Data flow

### One-config run

```
caller (agentic_sweep or CLI)
    │
    ▼
agentic.run_one_config(benchmark, backend, ...)
    │
    ├─ free_port()                                  →  port=N
    ├─ build_serve_cmd, build_env                   →  argv, env
    ├─ subprocess.Popen(server, stdout=server.log)  →  proc
    ├─ wait_for_health(port, 180s)                  →  /health 200
    │
    ├─ scraper = MetricsScraper(port, interval_s=5).start()   ── daemon thread
    ├─ reporter = ProgressReporter(scraper, interval_s=30).start() ── daemon thread
    │      (every 30s: reads scraper snapshot + counts patches.jsonl lines,
    │       writes progress.jsonl row + stdout line)
    │
    ├─ subprocess.Popen(benchmark.build_agent_cmd(...), stdout=agent.log)  ── stream
    │      orchestrator concurrently:
    │        - tails agent.log to detect "instance X done" lines (best-effort)
    │        - lets the scraper+reporter run
    │      waits for agent proc exit
    │
    ├─ subprocess.run(benchmark.build_eval_cmd(…), stdout=eval.log)
    │      writes → bench-results/agentic/<run>/<swebench_results>.json
    │
    ├─ reporter.stop(); scraper.stop()              →  flush metrics.csv, progress.jsonl
    ├─ benchmark.parse_results(out_dir)             →  {n_solved, n_total, pass_at_1}
    ├─ terminate_group(proc)                        →  server gone
    └─ return summary dict
```

### Sweep

```
agentic_sweep
    │
    for config in [dense, sparse-token, sparse-double]:
    │       summary = run_one_config(config-specific args)
    │       results.append({"config": config, **summary})
    │
    print(format_accuracy_table(results))
    write bench-results/agentic/sweep-summary.json
```

## Error handling

| Failure mode                  | Behavior                                                                                  |
|-------------------------------|-------------------------------------------------------------------------------------------|
| Server doesn't come up        | `wait_for_health` raises; `run_one_config` returns `errors=["server_unhealthy"]`, skips agent/eval. Sweep marks config as failed, continues to next config. |
| Agent subprocess timeout      | Kill subprocess. Eval phase still runs on whatever `patches.jsonl` was written (mini-swe-agent flushes per-instance). Summary reflects partial coverage. |
| Eval subprocess failure       | Capture exit code + stderr in log. `parse_results` returns whatever the partial results JSON has; missing instances counted as un-solved (conservative). |
| Subset file missing/empty     | Hard error before any server spawn. `agentic.py` validates the instances path on entry.   |
| Server doesn't terminate cleanly | `terminate_group` escalates SIGTERM → SIGKILL after 15s, same as `run_serving.py`.     |
| Disk pressure (patches/eval logs) | `bench-results/agentic/<run>/` directory layout; user prunes manually. No auto-rotation. |
| `/metrics` scrape fails           | `MetricsScraper` swallows the exception, logs at DEBUG, continues. Progress ticks fall back to "tok/s=N/A spars=N/A" and still show instances-completed + elapsed. Never blocks the orchestrator. |
| Progress reporter thread crash    | Daemon thread; main thread is unaffected. Loss of progress lines is logged once at WARNING. Run completes with full `summary.json` regardless.  |

No partial-run resumption in v1. If a config fails halfway, you re-run
that config. Adding resumability is a separate spec.

## Testing strategy

### Unit tests (gated by `verify_commit.sh`)

Follow the existing `test_bench_run_serving.py` / `test_bench_sweep.py`
pattern: pure-Python helpers under test, no subprocess execution.

- `test_bench_benchmarks_mini_swe_agent.py`
  - `build_agent_cmd` includes expected flags (`--model`, `--base-url`,
    `--instances`, `--output`)
  - `build_eval_cmd` includes expected flags (`--predictions_path`,
    `--instance_ids_path`, `--run_id`)
  - `parse_results` handles complete / partial / missing results JSON
  - `default_instances_path` returns the checked-in 32-instance file

- `test_bench_agentic.py`
  - `run_one_config` with mocked subprocess + mocked benchmark returns
    expected summary
  - Server unhealthy → returns `errors=["server_unhealthy"]`, skips
    agent/eval
  - Agent timeout → still runs eval, summary marks partial

- `test_bench_agentic_sweep.py`
  - CONFIGS table contains dense + sparse-token + sparse-double
  - Per-axis overrides flow through (`--topk` etc.) — same shape as
    perf-sweep tests
  - `format_accuracy_table` includes pass@1 column, handles None entries
    (failed runs)

- `test_bench_progress.py`
  - `_parse_prometheus_text` extracts the four metrics we care about
    from a fixture `/metrics` body, ignores other lines, tolerates
    label clutter
  - `format_progress_line` renders the expected fields with sensible
    defaults when a metric is missing (`tok/s=N/A`, `spars=N/A`)
  - `ProgressReporter` writes one `progress.jsonl` record per tick
    (using a controllable clock fixture, not real time)

### Integration tests (manual, on vb200-14)

- Step 2: manual dry run of mini-swe-agent + swebench against a
  hand-started server, one instance.
- Step 4: smoke run of `agentic.py` on one instance, one config.
- Step 5: real `agentic.py` run on the 32-instance subset, dense
  baseline.
- Step 7: `agentic_sweep.py` milestone run, 32 × 3 configs.

## Sequenced plan

| # | Step                          | Code change                                                                                                          | Verify gate                                                                                                                                                                            | Commit? |
|---|-------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| 0 | Land perf bench               | Commit `run_serving.py`, `sweep.py`, 40 tests (this session's work)                                                  | `scripts/dev/verify_commit.sh vb200-14 tests/test_bench_run_serving.py tests/test_bench_sweep.py` green; two atomic commits land                                                       | yes     |
| 1 | Deps + subset                 | `pyproject.toml` adds `agentic` extra; `bench-resources/swebench-subset-32.txt` checked in                          | On vb200-14: `uv sync --extra agentic` clean; `python -c "import minisweagent, swebench"` works                                                                                       | yes     |
| 2 | Manual dry run                | Zero code. Start `skylight serve` manually, drive `mini-swe-agent` CLI on one instance, run swebench eval container | One instance produces pass/fail. Confirms: protocol compatibility, docker eval works on the host, file shapes are what `parse_results` expects                                          | no      |
| 3 | Adapter module                | `benchmarks/base.py` + `benchmarks/mini_swe_agent.py` + `test_bench_benchmarks_mini_swe_agent.py`                  | `verify_commit.sh vb200-14 tests/test_bench_benchmarks_mini_swe_agent.py` green                                                                                                          | yes     |
| 4 | Orchestrator (one config)     | `agentic.py` + `progress.py` + tests for both + smoke run on one instance                                          | Unit tests green via verify (`test_bench_agentic.py` + `test_bench_progress.py`); then on vb200-14: `python -m skylight.bench.agentic --backend dense --instances <1-instance>.txt` produces a real `summary.json` + populated `metrics.csv` + `progress.jsonl` + progress lines visible on stdout during the run | yes     |
| 5 | Real run, dense, 32 instances | No code; operational                                                                                                 | `bench-results/agentic/.../summary.json` with `{n_solved, n_total=32, pass@1, elapsed_s}` for dense                                                                                      | no      |
| 6 | Sweep module                  | `agentic_sweep.py` + `test_bench_agentic_sweep.py`                                                                  | `verify_commit.sh vb200-14 tests/test_bench_agentic_sweep.py` green                                                                                                                       | yes     |
| 7 | Milestone sweep, 32 × 3       | No code; operational                                                                                                 | Comparison table on stdout + `sweep-summary.json` with pass@1 for all three configs                                                                                                     | no      |
| 8 | Scale to 64 → 128 → full      | New subset files only                                                                                                | Same shape; bigger subset                                                                                                                                                              | yes (file)|

## Risk gates (stop-and-reconsider triggers)

- **Step 2** — if `mini-swe-agent` won't talk to `skylight serve` (tool
  calling, response format, streaming quirks), there's a protocol gap.
  Stop. Either: (a) configure mini-swe-agent for OpenAI-compatible
  no-native-tool-calling mode, (b) add a shim, or (c) reconsider agent
  choice. Don't push through.
- **Step 5** — if dense pass@1 on 32 instances is wildly off from
  public Qwen-Coder-30B-A3B SWE-bench Lite numbers (rough public range:
  25-35%), the harness is wrong, not the model. Investigate before
  running sparse.
- **Step 7** — if sparse pass@1 drops >5pp vs dense, that's *the*
  finding. Capture it cleanly. Run a 2% sparsity (`--topk 0.02`)
  comparison and longer subset before concluding.

## Open questions

(Filled during step 2 manual dry run; left empty here.)

- Exact `mini-swe-agent` CLI flags (confirm against upstream README).
- Exact `swebench.harness.run_evaluation` invocation (confirm).
- Whether mini-swe-agent supports `--instances <file>` directly or
  requires a config YAML; if the latter, adapter generates the YAML.

## References

- Existing pattern: `src/skylight/bench/run_serving.py`,
  `src/skylight/bench/sweep.py`
- Existing harness for reference (not ported):
  `skylight-org/skylight-backend@kk/swebench-kernels`,
  `skylight/agentic_benchmark/` — multi-node, SQLite, OpenHands. We
  deliberately do not port; we re-implement minimally.
- `SWE-bench/mini-swe-agent` (upstream agent)
- `princeton-nlp/SWE-bench` and the `swebench` PyPI package (eval
  harness)
