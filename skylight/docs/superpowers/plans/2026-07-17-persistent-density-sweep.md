# Persistent Density Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one dense server and one BMM server per density while benchmarking every requested concurrency against each persistent CUDA-graph server.

**Architecture:** Add a process-only `ServingSession` to the existing one-case serving runner, then build a focused YAML orchestrator around the existing command builders. Server-group artifacts own logs, versions, and telemetry; case artifacts own client results and resumable status. Density stays immutable for each server process.

**Tech Stack:** Python 3.12, pytest, PyYAML 6, vLLM 0.21 CLI, existing Skylight runtime and benchmark helpers.

## Global Constraints

- Develop and test in an isolated RunPod worktree; never modify or stop the active H200 sweep.
- Preserve all existing `run_serving` helper signatures and `ArtifactPaths`.
- Use CUDA graphs only; no eager cases.
- Do not mutate topk in a running server.
- Do not change `skylight_kernels`.
- Use one trial and no automatic measurement retries.
- Commit with `kumarkagrawal@gmail.com`.

---

### Task 1: Reusable serving process

**Files:**
- Modify: `src/skylight/bench/run_serving.py`
- Modify: `tests/test_bench_run_serving.py`

**Interfaces:**
- Produces: `ServingSession(serve_cmd, env, port, timeout, server_log=None)`
- Produces: `ServingSession.run_benchmark(command, client_log=None) -> subprocess.CompletedProcess`
- Preserves: `build_serve_cmd`, `build_env`, `build_benchmark_cmd`, `ArtifactPaths`, and `main`

- [ ] **Step 1: Write failing lifecycle tests**

Add tests proving one session starts one fake process, waits for health once,
runs two client commands, and terminates once:

```python
def test_serving_session_reuses_one_server_for_multiple_clients(monkeypatch, tmp_path):
    events = []
    process = FakeProcess()
    monkeypatch.setattr(run_serving.subprocess, "Popen",
                        lambda *a, **k: events.append("start") or process)
    monkeypatch.setattr(run_serving, "wait_for_health",
                        lambda *a: events.append("health"))
    monkeypatch.setattr(run_serving.subprocess, "run",
                        lambda cmd, **k: events.append(cmd[-1]) or
                        SimpleNamespace(returncode=0))
    monkeypatch.setattr(run_serving, "terminate_group",
                        lambda p: events.append("stop"))

    with run_serving.ServingSession(
        ["server"], {}, 8123, 30, tmp_path / "server.log"
    ) as session:
        session.run_benchmark(["client", "one"])
        session.run_benchmark(["client", "two"])

    assert events == ["start", "health", "one", "two", "stop"]
```

Add separate tests for teardown after a health exception, refusal to run a
client after the server exits, and preservation of the existing `main` tests.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_bench_run_serving.py -q
```

Expected: new tests fail because `ServingSession` does not exist.

- [ ] **Step 3: Implement the minimal lifecycle class**

Implement a context manager that opens the optional server log in append mode,
starts the process with `start_new_session=True`, calls `wait_for_health`,
checks `proc.poll()` before clients, and always calls `terminate_group`.
`run_benchmark` redirects to the optional client log and returns the completed
process without applying experiment policy.

Refactor `main` to call the class for its one benchmark command while retaining
all existing argument parsing and artifact generation.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 command. Expected: all serving tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/skylight/bench/run_serving.py tests/test_bench_run_serving.py
git -c user.email=kumarkagrawal@gmail.com commit \
  -m "refactor: add reusable serving session"
```

### Task 2: Strict density configuration and run identity

**Files:**
- Create: `src/skylight/bench/density_sweep.py`
- Create: `tests/test_bench_density_sweep.py`

**Interfaces:**
- Produces: `SweepConfig.from_yaml(path: Path) -> SweepConfig`
- Produces: `SweepConfig.groups() -> tuple[ServerGroup, ...]`
- Produces: `canonical_density(value: float) -> str`
- Produces: `semantic_fingerprint(payload: Mapping[str, object]) -> str`

- [ ] **Step 1: Write failing parser and grouping tests**

Create a minimal valid YAML fixture and assert:

```python
config = SweepConfig.from_yaml(path)
assert config.workload.concurrencies == (64, 1)
assert config.max_num_seqs == 64
assert [group.name for group in config.groups()] == [
    "dense", "topk-0.001", "topk-0.1"
]
```

Add focused tests rejecting duplicate mapping keys, unknown keys, duplicate
concurrencies/densities, invalid density, an eager mode, insufficient
`max_model_len`, and a mismatched canonical density path. Add a deterministic
SHA-256 fingerprint test independent of mapping order.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_bench_density_sweep.py -q
```

Expected: import fails because `density_sweep` does not exist.

- [ ] **Step 3: Implement immutable configuration objects**

Use frozen dataclasses for hardware, server, workload, BMM, group, and root
configuration. Implement a PyYAML `SafeLoader` subclass that rejects duplicate
mapping keys, explicit allowed-key checks at every schema level except
`hf_overrides`, and the numeric validation from the design.

Normalize densities with:

```python
def canonical_density(value: float) -> str:
    return format(value, ".12g")
```

Hash canonical JSON:

```python
def semantic_fingerprint(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 4: Verify GREEN**

Run the Task 2 command. Expected: all density configuration tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/skylight/bench/density_sweep.py tests/test_bench_density_sweep.py
git -c user.email=kumarkagrawal@gmail.com commit \
  -m "feat: add strict density sweep configuration"
```

### Task 3: Persistent orchestration, artifacts, resume, and summaries

**Files:**
- Modify: `src/skylight/bench/density_sweep.py`
- Modify: `tests/test_bench_density_sweep.py`

**Interfaces:**
- Produces: `run_sweep(config, output_root, session_factory=ServingSession) -> int`
- Produces: `case_complete(case_dir, fingerprint, concurrency) -> bool`
- Produces: `write_summary(output_root, config) -> None`
- Consumes: `ServingSession` and existing `run_serving` command/version helpers

- [ ] **Step 1: Write failing orchestration tests**

Use a fake session factory and fake benchmark results to prove:

- one dense plus two density groups create exactly three sessions;
- each session receives both client commands;
- every server command uses `--max-num-seqs 64`;
- every executed case uses the configured two client warmups;
- a fully completed matching group starts no session;
- a fingerprint mismatch raises before session startup;
- failed stale artifacts move under `attempts/`;
- a client failure continues only when the session remains healthy;
- summaries join BMM metrics to the same-concurrency dense result.

The central assertion is:

```python
assert [(s.group, s.client_warmups) for s in sessions] == [
    ("dense", [2, 2]),
    ("topk-0.001", [2, 2]),
    ("topk-0.1", [2, 2]),
]
```

- [ ] **Step 2: Verify RED**

Run the Task 2 test command. Expected: orchestration tests fail because the
runner functions do not exist.

- [ ] **Step 3: Implement the runner**

Set `CUDA_VISIBLE_DEVICES` before `configure_runtime(expected_hardware)`.
Validate `expected_name` using the detected CUDA device name. Collect a stable
run identity with timestamps removed, then create group and case fingerprints.

For each incomplete group:

1. Build one server command/environment.
2. Open one `ServingSession`.
3. Run incomplete concurrencies in configured order.
4. Use the configured warmups for every benchmark client.
5. Atomically write running/final statuses and summaries.
6. Stop only that session when the group finishes.

Preserve old artifacts in numbered `attempts/` directories before rerun.
Write `summary.json` and `summary.csv` after every completed case.

- [ ] **Step 4: Verify GREEN**

Run the Task 2 test command and Task 1 tests. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/skylight/bench/density_sweep.py tests/test_bench_density_sweep.py
git -c user.email=kumarkagrawal@gmail.com commit \
  -m "feat: reuse servers across density sweep concurrencies"
```

### Task 4: Checked configuration, dependency, and qualification gate

**Files:**
- Create: `experiments/h200-32k-density.yaml`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`
- Modify: `tests/test_bench_density_sweep.py`

**Interfaces:**
- Provides the checked H200 configuration and documented invocation.

- [ ] **Step 1: Write failing packaging/config tests**

Assert the checked YAML parses to 13 groups and 91 cases, uses Hopper/H200,
CUDA graphs, 99% memory, and the exact 32K matrix. Assert `pyproject.toml`
declares `pyyaml>=6,<7`.

- [ ] **Step 2: Verify RED**

Run the density tests. Expected: failure because the checked YAML and direct
dependency are absent.

- [ ] **Step 3: Add configuration and dependency**

Add the exact YAML from the approved design, the direct PyYAML dependency, and
a concise README command. Refresh the lockfile with:

```bash
uv lock
```

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_bench_run_serving.py \
  tests/test_bench_density_sweep.py \
  tests/test_bench_sweep.py \
  tests/test_long_context_crossover.py \
  tests/test_profiles.py
```

Expected: all tests pass.

Also run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intended files changed.

- [ ] **Step 5: Commit**

```bash
git add experiments/h200-32k-density.yaml pyproject.toml uv.lock README.md \
  tests/test_bench_density_sweep.py
git -c user.email=kumarkagrawal@gmail.com commit \
  -m "docs: add H200 persistent density experiment"
```

### Task 5: H200 equivalence smoke after the active queue

**Files:**
- No source changes unless the smoke exposes a defect.

- [ ] **Step 1: Wait for the active density queue to release the GPU**

Do not kill, pause, or modify `skylight-h200-32k-density`.

- [ ] **Step 2: Run the legacy and persistent compatibility matrix**

Run dense and BMM topk `0.10` at concurrencies `64` and `1` through both
lifecycles with identical 32K settings.

- [ ] **Step 3: Verify the acceptance gate**

Require zero failed requests, completed prompts equal requested concurrency,
and persistent throughput/mean TPOT within 5% of legacy for all comparisons.

- [ ] **Step 4: Record the result**

Write the comparison to the feature branch worklog. Do not launch the full
fast matrix until the gate passes.
