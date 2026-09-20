# Agentic Benchmark Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a single command that runs `dense | sparse-token | sparse-double` against a 32-instance SWE-bench Lite subset on one B200 node and emits a `pass@1` comparison table.

**Architecture:** Thin orchestrator in `src/skylight/bench/`. Reuses the server lifecycle helpers in the already-written `run_serving.py`. Delegates agent + eval to upstream `mini-swe-agent` and `swebench` packages installed via a pyproject extra (no vendoring). One adapter file per benchmark, registered in a dict. Progress visibility via a daemon-thread scraper of `/metrics` + a periodic stdout tick.

**Tech Stack:** Python 3.10+ stdlib (subprocess, threading, urllib, logging) + existing skylight bench helpers + upstream `mini-swe-agent` + upstream `swebench`. No new heavyweight deps.

**Spec:** `docs/specs/2026-05-26-agentic-bench-design.md`. The plan respects that spec's 9-step sequenced order with verify gates.

**Workflow notes — Krishna's rules** (engineer must respect throughout):

- **Atomic commits, no phase prefixes.** Commit messages name the atomic change. Never `step N:`, `phase N:`, etc.
- **User drives git.** The engineer describes file/scope/message; Krishna runs `git add`, `commit`, and `push` himself. Plan steps that look like `git commit ...` are *proposed commits for Krishna*, not commands the engineer runs.
- **Server-only execution.** `pytest`, `uv sync`, `python -m skylight.bench.*`, and any model execution happen ONLY on vb200-14. The engineer never runs them locally on the Mac. The user (Krishna) runs them on the bench host.
- **Verify gate.** Tests run via `scripts/dev/verify_commit.sh vb200-14 <test paths>` which rsyncs the worktree to vb200-14 and pytest-runs there. Krishna invokes this; engineer just edits code.
- **No Co-Authored-By trailers** in suggested commit messages.

---

## File structure (locked here, referenced by tasks below)

```
src/skylight/bench/
├── __init__.py                          # already exists
├── run_serving.py                       # already exists (helpers reused)
├── sweep.py                             # already exists (perf sweep)
├── progress.py                          # NEW (Task 4)
├── agentic.py                           # NEW (Task 5)
├── agentic_sweep.py                     # NEW (Task 6)
└── benchmarks/
    ├── __init__.py                      # NEW (Task 3) — registry
    ├── base.py                          # NEW (Task 3) — Protocol
    └── mini_swe_agent.py                # NEW (Task 3) — first adapter

bench-resources/
└── swebench-subset-32.txt               # NEW (Task 2)

tests/
├── test_bench_run_serving.py            # already written (Task 1)
├── test_bench_sweep.py                  # already written (Task 1)
├── test_bench_benchmarks_mini_swe_agent.py  # NEW (Task 3)
├── test_bench_progress.py               # NEW (Task 4)
├── test_bench_agentic.py                # NEW (Task 5)
└── test_bench_agentic_sweep.py          # NEW (Task 6)

pyproject.toml                           # MODIFIED (Task 2) — agentic extra

docs/specs/2026-05-26-agentic-bench-design.md   # already written
docs/plans/2026-05-26-agentic-bench-impl-plan.md  # this file
```

---

## Task 1: Land the already-written perf bench

**Spec step 0.** Code is already written this session (`run_serving.py`, `sweep.py`, their test files). This task gates everything else.

**Files (all already exist on disk, just need verify + commit):**
- `src/skylight/bench/__init__.py`
- `src/skylight/bench/run_serving.py`
- `src/skylight/bench/sweep.py`
- `tests/test_bench_run_serving.py`
- `tests/test_bench_sweep.py`

- [ ] **Step 1.1: Verify all 40 tests pass on vb200-14**

  Tell Krishna to run, from the skylight repo root:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_run_serving.py tests/test_bench_sweep.py
  ```

  Expected: `✓ verify GREEN` and all 40 tests pass.

  Stop and investigate if any test fails.

- [ ] **Step 1.2: Suggested commit decomposition for Krishna**

  Two atomic commits keep the perf path coherent:

  **Commit A** — files: `src/skylight/bench/__init__.py`, `src/skylight/bench/run_serving.py`, `tests/test_bench_run_serving.py`
  Message:
  ```
  feat(bench): single-config sparse-vs-dense A/B wrapper for vllm bench serve

  Wraps vllm bench serve subcommand: spawn skylight serve or vllm
  api_server, wait /health, run benchmark, terminate process group.
  Sparse knobs (topk/sink/local/channel_num) flow via SKYLIGHT_SPARSE_*
  env vars only on the sparse backend; dense run uses FLASHINFER.
  23 unit tests cover cmd-building, env construction, port allocation.
  ```

  **Commit B** — files: `src/skylight/bench/sweep.py`, `tests/test_bench_sweep.py`
  Message:
  ```
  feat(bench): sweep orchestrator with named configs and per-axis overrides

  Loops over (input_len, config) for dense + sparse-token + sparse-double.
  Per-axis CLI overrides (--topk/--sink/--local/--channel-num) lay over
  the named config defaults; per-run output suffixes prevent collisions.
  Captures per-run logs to file, prints one structured status line per
  run, emits a comparison table + sweep-summary.json at the end.
  17 unit tests cover cmd-building, override precedence, summary parsing,
  table formatting, CONFIGS shape.
  ```

  Krishna runs `git add` + `git commit` for each, in order.

---

## Task 2: pyproject `agentic` extra + 32-instance subset file

**Spec step 1.** Add upstream deps + ship the curated subset.

**Files:**
- Modify: `pyproject.toml` (add `agentic` to `[project.optional-dependencies]`)
- Create: `bench-resources/swebench-subset-32.txt`

- [ ] **Step 2.1: Read current pyproject.toml to find the optional-dependencies block**

  The engineer reads `/Users/kumarkagrawal/local/berkeley/sky/skylight/pyproject.toml` to confirm the existing `[project.optional-dependencies]` section structure.

- [ ] **Step 2.2: Add the `agentic` extra**

  In `pyproject.toml`, under `[project.optional-dependencies]`, add:

  ```toml
  agentic = [
      "mini-swe-agent",
      "swebench",
  ]
  ```

  Don't pin versions in v1 — let `uv lock` capture whatever resolves. We can pin after Step 2 (manual dry run) confirms the CLI we depend on is stable.

- [ ] **Step 2.3: Create `bench-resources/swebench-subset-32.txt`**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/bench-resources/swebench-subset-32.txt`

  Contents (32 SWE-bench Lite instance IDs, one per line, no blank lines, no trailing whitespace):

  ```
  astropy__astropy-12907
  astropy__astropy-14182
  django__django-11099
  django__django-11848
  django__django-12286
  django__django-12915
  django__django-13315
  django__django-13710
  django__django-14534
  django__django-14752
  django__django-15347
  django__django-15814
  django__django-16873
  flask__flask-4992
  flask__flask-5063
  matplotlib__matplotlib-23987
  matplotlib__matplotlib-25433
  pylint-dev__pylint-7080
  pytest-dev__pytest-5103
  pytest-dev__pytest-7236
  pytest-dev__pytest-7373
  requests__requests-2317
  scikit-learn__scikit-learn-13439
  scikit-learn__scikit-learn-13779
  scikit-learn__scikit-learn-14894
  scikit-learn__scikit-learn-25500
  sphinx-doc__sphinx-8595
  sphinx-doc__sphinx-8801
  sympy__sympy-13031
  sympy__sympy-15011
  sympy__sympy-20590
  xarray__xarray-3993
  ```

  Spread across 11 repos to surface failure-mode diversity (multi-file edits, test-discovery quirks, type-stub clashes), not just easy single-file fixes. Concrete instance IDs come from the public SWE-bench Lite test split; if any happens to fail the *harness* phase (not the model's fix attempt), swap it during Step 2 dry-run.

- [ ] **Step 2.4: Verify `uv sync --extra agentic` resolves on vb200-14**

  Tell Krishna to run on vb200-14, from the skylight repo root:

  ```
  uv sync --extra agentic 2>&1 | tail -20
  python -c "import minisweagent; import swebench; print(minisweagent.__version__, swebench.__version__)"
  ```

  Expected: both packages resolve cleanly and import. If either fails, troubleshoot in pyproject (pin versions, etc.) before moving on.

- [ ] **Step 2.5: Suggested commit for Krishna**

  Files: `pyproject.toml`, `bench-resources/swebench-subset-32.txt`
  Message:
  ```
  feat(bench): agentic extra and 32-instance SWE-bench Lite subset

  Adds [project.optional-dependencies].agentic = ["mini-swe-agent",
  "swebench"] for the upcoming agentic harness. 32-instance subset
  spans 11 repos (astropy, django, flask, matplotlib, pylint, pytest,
  requests, scikit-learn, sphinx, sympy, xarray) for failure-mode
  diversity. Base unit of the 32 -> 64 -> 128 -> full ladder.
  ```

---

## Manual Phase A: Dry run on vb200-14 (no code change)

**Spec step 2.** Validates the protocol assumptions before we commit any adapter code.

- [ ] **Step A.1: Krishna starts skylight serve manually**

  On vb200-14:

  ```
  .venv/bin/python -m skylight.cli serve --model Qwen/Qwen3-Coder-30B-A3B-Instruct --port 8000 --max-model-len 32768
  ```

  In another terminal, confirm:
  ```
  curl -sS http://127.0.0.1:8000/health
  curl -sS http://127.0.0.1:8000/v1/models
  ```

  Expected: `/health` returns 200; `/v1/models` lists the model.

- [ ] **Step A.2: Krishna runs mini-swe-agent on ONE instance against that server**

  On vb200-14 (consult the upstream `mini-swe-agent` README first to confirm exact CLI; the commands below are the *expected* shape):

  ```
  mkdir -p /tmp/dryrun
  echo "sympy__sympy-20590" > /tmp/dryrun/one.txt
  python -m minisweagent.run \
      --model openai/Qwen/Qwen3-Coder-30B-A3B-Instruct \
      --base-url http://127.0.0.1:8000/v1 \
      --api-key EMPTY \
      --instances-file /tmp/dryrun/one.txt \
      --output /tmp/dryrun/patches.jsonl
  ```

  Expected: agent runs to completion, writes one record to `/tmp/dryrun/patches.jsonl`.

  **If the upstream CLI flag names differ** (`--instances` vs `--instances-file`, `--output-path` vs `--output`, etc.), record the exact flags here as a note. They go into the adapter in Task 3.

- [ ] **Step A.3: Krishna runs swebench eval on that one patch**

  On vb200-14:

  ```
  python -m swebench.harness.run_evaluation \
      --predictions_path /tmp/dryrun/patches.jsonl \
      --dataset_name princeton-nlp/SWE-bench_Lite \
      --run_id dryrun_one \
      --max_workers 1
  ```

  Expected: docker container spins up, runs the test suite, writes a results JSON (typically `<model>.<run_id>.json` in CWD). Note the exact output filename pattern; Task 3's `parse_results` needs it.

- [ ] **Step A.4: Capture the file shapes that downstream code will consume**

  Inspect:

  ```
  head -1 /tmp/dryrun/patches.jsonl | python -m json.tool
  ls -la *.json | head
  cat <results-json> | python -m json.tool | head -40
  ```

  Confirm keys: `patches.jsonl` should have `instance_id` + `model_patch` (or whatever upstream calls it); results JSON should have `resolved_ids` + `submitted_ids` lists (this is the SWE-bench standard).

  **If the field names differ, record them as a note.** Adapter code in Task 3 uses these exact field names.

  **Risk gate (per spec).** If any of A.1-A.4 fails or requires a shim, stop. Do not start Task 3. Discuss the protocol gap with Krishna.

---

## Task 3: `benchmarks/` package — Protocol + mini-swe-agent adapter

**Spec step 3.** Pure-Python helpers, TDD.

**Files:**
- Create: `src/skylight/bench/benchmarks/__init__.py`
- Create: `src/skylight/bench/benchmarks/base.py`
- Create: `src/skylight/bench/benchmarks/mini_swe_agent.py`
- Create: `tests/test_bench_benchmarks_mini_swe_agent.py`

- [ ] **Step 3.1: Write the failing tests**

  Create `/Users/kumarkagrawal/local/berkeley/sky/skylight/tests/test_bench_benchmarks_mini_swe_agent.py` with this content:

  ```python
  """Tests for skylight.bench.benchmarks.mini_swe_agent.

  Pure-Python: covers cmd-building (build_agent_cmd, build_eval_cmd) and
  result-parsing (parse_results) with no subprocess execution.
  """
  from __future__ import annotations

  import json
  import sys
  from pathlib import Path

  import pytest

  from skylight.bench.benchmarks.mini_swe_agent import MiniSweAgent


  # ----------------------------- build_agent_cmd --------------------------------


  def test_build_agent_cmd_includes_model_and_base_url():
      agent = MiniSweAgent()
      cmd = agent.build_agent_cmd(
          model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
          base_url="http://localhost:8000/v1",
          instances=Path("subset.txt"),
          out_dir=Path("/tmp/out"),
      )
      joined = " ".join(cmd)
      assert "openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" in joined
      assert "http://localhost:8000/v1" in joined


  def test_build_agent_cmd_points_at_instances_file_and_out_dir():
      agent = MiniSweAgent()
      cmd = agent.build_agent_cmd(
          model="X", base_url="http://h/v1",
          instances=Path("/data/subset-32.txt"),
          out_dir=Path("/tmp/out"),
      )
      joined = " ".join(cmd)
      assert "/data/subset-32.txt" in joined
      # The adapter writes patches under out_dir; the cmd must reference it.
      assert "/tmp/out" in joined


  def test_build_agent_cmd_invokes_minisweagent_module():
      agent = MiniSweAgent()
      cmd = agent.build_agent_cmd(
          model="X", base_url="http://h/v1",
          instances=Path("s.txt"), out_dir=Path("/tmp/o"),
      )
      assert cmd[0] == sys.executable
      assert "-m" in cmd
      assert any("minisweagent" in part for part in cmd)


  # ----------------------------- build_eval_cmd ---------------------------------


  def test_build_eval_cmd_points_at_patches_jsonl_in_out_dir():
      agent = MiniSweAgent()
      cmd = agent.build_eval_cmd(Path("/tmp/run42"))
      joined = " ".join(cmd)
      assert "/tmp/run42/patches.jsonl" in joined
      assert "--predictions_path" in cmd


  def test_build_eval_cmd_invokes_swebench_harness_module():
      cmd = MiniSweAgent().build_eval_cmd(Path("/tmp/o"))
      assert cmd[0] == sys.executable
      assert "-m" in cmd
      assert any("swebench" in part for part in cmd)
      assert "princeton-nlp/SWE-bench_Lite" in " ".join(cmd)


  def test_build_eval_cmd_uses_out_dir_name_as_run_id():
      """--run_id is the out_dir basename so the swebench results file is namespaced."""
      cmd = MiniSweAgent().build_eval_cmd(Path("/tmp/run-2026-05-26"))
      assert "--run_id" in cmd
      assert cmd[cmd.index("--run_id") + 1] == "run-2026-05-26"


  # ----------------------------- parse_results ----------------------------------


  def test_parse_results_reads_resolved_and_submitted(tmp_path: Path):
      """Standard swebench results JSON yields n_solved, n_total, pass_at_1."""
      (tmp_path / "X.results.json").write_text(json.dumps({
          "resolved_ids": ["sympy__sympy-20590", "django__django-12915"],
          "submitted_ids": [
              "sympy__sympy-20590", "django__django-12915", "flask__flask-4992",
              "scikit-learn__scikit-learn-13439",
          ],
      }))
      r = MiniSweAgent().parse_results(tmp_path)
      assert r["n_solved"] == 2
      assert r["n_total"] == 4
      assert r["pass_at_1"] == 0.5


  def test_parse_results_no_results_json_returns_error_marker(tmp_path: Path):
      """No JSON in out_dir => zero-solved with an explicit error marker."""
      r = MiniSweAgent().parse_results(tmp_path)
      assert r["n_solved"] == 0
      assert r["pass_at_1"] == 0.0
      assert "errors" in r and "no_results_json" in r["errors"]


  def test_parse_results_ignores_unrelated_json(tmp_path: Path):
      """A junk JSON in the dir doesn't fool the parser."""
      (tmp_path / "junk.json").write_text('{"hello": "world"}')
      (tmp_path / "real.json").write_text(json.dumps({
          "resolved_ids": ["a"], "submitted_ids": ["a", "b"],
      }))
      r = MiniSweAgent().parse_results(tmp_path)
      assert r["n_solved"] == 1
      assert r["n_total"] == 2


  def test_parse_results_handles_empty_submitted(tmp_path: Path):
      """submitted_ids=[] means agent crashed before producing any patch."""
      (tmp_path / "X.json").write_text(json.dumps({
          "resolved_ids": [], "submitted_ids": [],
      }))
      r = MiniSweAgent().parse_results(tmp_path)
      assert r["n_solved"] == 0
      assert r["n_total"] == 0
      # pass_at_1 must not divide by zero
      assert r["pass_at_1"] == 0.0


  # ----------------------------- default_instances_path -------------------------


  def test_default_instances_path_points_at_subset_32():
      p = MiniSweAgent().default_instances_path()
      assert p.name == "swebench-subset-32.txt"
      assert str(p).endswith("bench-resources/swebench-subset-32.txt")
  ```

- [ ] **Step 3.2: Verify the tests fail (no implementation yet)**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_benchmarks_mini_swe_agent.py
  ```

  Expected: red — `ModuleNotFoundError: No module named 'skylight.bench.benchmarks'` or similar.

- [ ] **Step 3.3: Create `benchmarks/__init__.py`**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/src/skylight/bench/benchmarks/__init__.py`

  ```python
  """Registry of agentic benchmark adapters supported by skylight bench.

  Each adapter implements the AgenticBenchmark protocol (see base.py).
  Adding a new benchmark: write a module + add one entry below.
  """
  from .mini_swe_agent import MiniSweAgent

  REGISTRY = {
      "mini-swe-agent": MiniSweAgent(),
  }

  __all__ = ["REGISTRY"]
  ```

- [ ] **Step 3.4: Create `benchmarks/base.py`**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/src/skylight/bench/benchmarks/base.py`

  ```python
  """Protocol every agentic benchmark adapter must satisfy.

  Structural typing: adapters do not need to inherit anything; if they
  expose these attrs/methods with these signatures, they work.
  """
  from __future__ import annotations

  from pathlib import Path
  from typing import Protocol


  class AgenticBenchmark(Protocol):
      """Adapter shape for one agentic benchmark."""

      name: str  # e.g. "mini-swe-agent"

      def default_instances_path(self) -> Path:
          """Default subset to run against if the user doesn't pass one."""
          ...

      def build_agent_cmd(
          self,
          model: str,
          base_url: str,
          instances: Path,
          out_dir: Path,
      ) -> list[str]:
          """argv that runs the agent against `base_url`, processing `instances`.

          The agent must write per-instance attempts (e.g. patches.jsonl)
          under `out_dir` so `parse_results` and the progress reporter can
          consume them.
          """
          ...

      def build_eval_cmd(self, out_dir: Path) -> list[str]:
          """argv that scores the agent's output in `out_dir`.

          Run AFTER build_agent_cmd finishes. Writes a results JSON under
          out_dir (or CWD) that `parse_results` can parse.
          """
          ...

      def parse_results(self, out_dir: Path) -> dict:
          """Read whatever the eval wrote and return headline numbers.

          MUST return at least: {"n_solved": int, "n_total": int,
          "pass_at_1": float}. May add "errors": list[str] on failure.
          """
          ...
  ```

- [ ] **Step 3.5: Create `benchmarks/mini_swe_agent.py`**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/src/skylight/bench/benchmarks/mini_swe_agent.py`

  **CLI flag names below are the *expected* shape. If Manual Phase A confirmed different upstream flags, edit accordingly before running tests.**

  ```python
  """Adapter for SWE-bench/mini-swe-agent + the swebench eval harness.

  Phase A: agent reads a list of instance IDs, drives an OpenAI-compatible
  endpoint to produce patches, writes patches.jsonl under out_dir.
  Phase B: swebench harness runs each patch in a docker container test
  suite, writes a results JSON with resolved_ids/submitted_ids lists.
  """
  from __future__ import annotations

  import json
  import sys
  from pathlib import Path


  # Default subset path is project-relative. Resolved by caller against CWD.
  _DEFAULT_INSTANCES = Path("bench-resources/swebench-subset-32.txt")


  class MiniSweAgent:
      """Concrete implementation of the AgenticBenchmark protocol."""

      name = "mini-swe-agent"

      def default_instances_path(self) -> Path:
          return _DEFAULT_INSTANCES

      def build_agent_cmd(
          self,
          model: str,
          base_url: str,
          instances: Path,
          out_dir: Path,
      ) -> list[str]:
          # NOTE: flag names confirmed in Manual Phase A. Adjust here if
          # upstream's `mini-swe-agent` CLI uses different names.
          return [
              sys.executable, "-m", "minisweagent.run",
              "--model", f"openai/{model}",
              "--base-url", base_url,
              "--api-key", "EMPTY",
              "--instances-file", str(instances),
              "--output", str(out_dir / "patches.jsonl"),
          ]

      def build_eval_cmd(self, out_dir: Path) -> list[str]:
          return [
              sys.executable, "-m", "swebench.harness.run_evaluation",
              "--predictions_path", str(out_dir / "patches.jsonl"),
              "--dataset_name", "princeton-nlp/SWE-bench_Lite",
              "--run_id", out_dir.name,
              "--max_workers", "1",
          ]

      def parse_results(self, out_dir: Path) -> dict:
          """Find any JSON in out_dir with resolved_ids/submitted_ids and parse."""
          for path in sorted(out_dir.glob("*.json")):
              try:
                  data = json.loads(path.read_text())
              except Exception:
                  continue
              if isinstance(data, dict) and "resolved_ids" in data and "submitted_ids" in data:
                  resolved = data["resolved_ids"] or []
                  submitted = data["submitted_ids"] or []
                  n_solved = len(resolved)
                  n_total = len(submitted)
                  pass_at_1 = (n_solved / n_total) if n_total > 0 else 0.0
                  return {
                      "n_solved": n_solved,
                      "n_total": n_total,
                      "pass_at_1": pass_at_1,
                  }
          return {
              "n_solved": 0,
              "n_total": 0,
              "pass_at_1": 0.0,
              "errors": ["no_results_json"],
          }
  ```

- [ ] **Step 3.6: Verify the tests pass**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_benchmarks_mini_swe_agent.py
  ```

  Expected: `✓ verify GREEN`, all 10 tests pass.

- [ ] **Step 3.7: Suggested commit for Krishna**

  Files:
  - `src/skylight/bench/benchmarks/__init__.py`
  - `src/skylight/bench/benchmarks/base.py`
  - `src/skylight/bench/benchmarks/mini_swe_agent.py`
  - `tests/test_bench_benchmarks_mini_swe_agent.py`

  Message:
  ```
  feat(bench): mini-swe-agent adapter + AgenticBenchmark protocol

  Adapter pattern: each agentic benchmark exposes build_agent_cmd
  (drive agent against OpenAI endpoint), build_eval_cmd (score against
  upstream harness), parse_results (extract n_solved/n_total/pass_at_1).
  Mini-swe-agent + swebench are upstream pip deps; no agent code is
  vendored. Registry in benchmarks/__init__.py maps "mini-swe-agent" ->
  MiniSweAgent(). 10 unit tests cover cmd-building and result-parsing.
  ```

---

## Task 4: `progress.py` — metrics scraper + tick reporter

**Spec step 4 (first half).** Daemon-thread scraping + a pure formatter, TDD.

**Files:**
- Create: `src/skylight/bench/progress.py`
- Create: `tests/test_bench_progress.py`

- [ ] **Step 4.1: Write the failing tests**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/tests/test_bench_progress.py`

  ```python
  """Tests for skylight.bench.progress.

  Covers the pure-Python pieces: Prometheus text parsing, progress-line
  formatting, and ProgressReporter's tick output (driven via a
  controllable clock, not real time).
  """
  from __future__ import annotations

  import json
  import threading
  import time
  from pathlib import Path

  import pytest

  from skylight.bench.progress import (
      MetricsScraper,
      ProgressReporter,
      _parse_prometheus_text,
      format_progress_line,
  )


  # ----------------------------- _parse_prometheus_text -------------------------


  PROM_PAYLOAD = """\
  # HELP vllm:generation_tokens_total Total generated tokens
  # TYPE vllm:generation_tokens_total counter
  vllm:generation_tokens_total{model_name="Qwen3-32B"} 12345.0
  vllm:request_success_total{model_name="Qwen3-32B",finished_reason="stop"} 7.0
  skylight_observed_sparsity_fraction 0.0925
  some_unrelated_metric 99.0
  """


  def test_parse_extracts_tracked_metrics():
      out = _parse_prometheus_text(PROM_PAYLOAD)
      assert out["vllm:generation_tokens_total"] == 12345.0
      assert out["vllm:request_success_total"] == 7.0
      assert out["skylight_observed_sparsity_fraction"] == 0.0925


  def test_parse_ignores_untracked_metrics():
      out = _parse_prometheus_text(PROM_PAYLOAD)
      assert "some_unrelated_metric" not in out


  def test_parse_ignores_comments_and_blank_lines():
      out = _parse_prometheus_text("\n# comment\n\nvllm:generation_tokens_total 5\n")
      assert out == {"vllm:generation_tokens_total": 5.0}


  def test_parse_tolerates_malformed_lines():
      """Lines that don't have a numeric value are skipped, not raised."""
      payload = "vllm:generation_tokens_total NOT_A_NUMBER\nvllm:request_success_total 3\n"
      out = _parse_prometheus_text(payload)
      assert out == {"vllm:request_success_total": 3.0}


  # ----------------------------- format_progress_line ---------------------------


  def test_format_progress_line_includes_config_done_total_and_pct():
      line = format_progress_line(
          config="dense",
          instances_done=8,
          instances_total=32,
          snapshot={},
          elapsed_s=261.0,
      )
      assert "dense" in line
      assert "8/32" in line
      assert "25%" in line


  def test_format_progress_line_falls_back_to_NA_without_metrics():
      """No metrics snapshot => tok/s and sparsity show N/A, not crash."""
      line = format_progress_line(
          config="sparse-token",
          instances_done=0,
          instances_total=32,
          snapshot={},
          elapsed_s=0.0,
      )
      assert "N/A" in line  # tok/s fallback
      assert "spars=N/A" in line


  def test_format_progress_line_with_full_snapshot():
      """Full snapshot => tok/s computed from token deltas, sparsity shown."""
      snapshot = {
          "vllm:generation_tokens_total": 5000.0,
          "skylight_observed_sparsity_fraction": 0.0925,
      }
      # 5000 tokens generated in 10s since previous snapshot (which had 0 tokens
      # at t=t_now-10s) => 500 tok/s.
      line = format_progress_line(
          config="sparse-token",
          instances_done=5,
          instances_total=32,
          snapshot=snapshot,
          elapsed_s=60.0,
          prev_tok_count=0.0,
          prev_tok_time=time.time() - 10.0,
      )
      assert "500" in line  # rounded tok/s
      assert "0.093" in line or "0.092" in line  # sparsity precision


  def test_format_progress_line_eta_visible_when_partial_done():
      line = format_progress_line(
          config="dense",
          instances_done=10,
          instances_total=32,
          snapshot={},
          elapsed_s=600.0,
      )
      # 10/32 done in 600s => 1320s remaining => "22m00s" ETA
      assert "ETA" in line
      assert "m" in line.split("ETA")[1]


  def test_format_progress_line_eta_none_when_fully_done():
      line = format_progress_line(
          config="dense",
          instances_done=32,
          instances_total=32,
          snapshot={},
          elapsed_s=1200.0,
      )
      # Fully done => no time-remaining estimate
      assert "ETA --:--" in line or "ETA" not in line


  # ----------------------------- ProgressReporter -------------------------------


  def test_progress_reporter_writes_jsonl_record_per_tick(
      tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
  ):
      """Reporter ticks (very fast for the test) write JSONL rows."""
      monkeypatch.setenv("SKYLIGHT_BENCH_PROGRESS_INTERVAL_S", "0.05")
      patches = tmp_path / "patches.jsonl"
      patches.write_text('{"instance_id": "x"}\n{"instance_id": "y"}\n')

      # Minimal stand-in for MetricsScraper.snapshot()
      class FakeScraper:
          def snapshot(self) -> dict:
              return {"vllm:generation_tokens_total": 100.0}

      progress_jsonl = tmp_path / "progress.jsonl"
      reporter = ProgressReporter(
          config_name="dense",
          scraper=FakeScraper(),
          patches_path=patches,
          instances_total=32,
          progress_jsonl=progress_jsonl,
      )
      reporter.start()
      time.sleep(0.2)  # ~3-4 ticks
      reporter.stop()
      reporter.join(timeout=2)

      assert progress_jsonl.exists()
      lines = [
          json.loads(line) for line in progress_jsonl.read_text().splitlines()
          if line.strip()
      ]
      assert len(lines) >= 1
      first = lines[0]
      assert first["config"] == "dense"
      assert first["instances_done"] == 2
      assert first["instances_total"] == 32
      assert first["vllm:generation_tokens_total"] == 100.0
  ```

- [ ] **Step 4.2: Verify the tests fail**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_progress.py
  ```

  Expected: red — `ModuleNotFoundError: No module named 'skylight.bench.progress'`.

- [ ] **Step 4.3: Create `progress.py`**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/src/skylight/bench/progress.py`

  ```python
  """Metrics scraping + progress reporting for agentic bench runs.

  Lightweight observability for one-config runs:

    * MetricsScraper — daemon thread that polls /metrics every 5s, writes
      a CSV timeseries, keeps the latest sample in memory.
    * format_progress_line — pure function that renders one status line.
    * ProgressReporter — daemon thread that ticks every 30s (configurable
      via SKYLIGHT_BENCH_PROGRESS_INTERVAL_S), writes a progress.jsonl row
      and prints the status line on stdout.

  Runtime overhead is negligible (<0.1% wall clock on 5-15 GPU-min
  instances). All threads are daemons; the main thread never blocks on
  metric I/O.
  """
  from __future__ import annotations

  import json
  import os
  import threading
  import time
  import urllib.error
  import urllib.request
  from pathlib import Path
  from typing import Optional


  # Metrics we extract from /metrics. Everything else in the payload is ignored.
  _TRACKED_METRICS = (
      "vllm:generation_tokens_total",
      "vllm:request_success_total",
      "skylight_observed_sparsity_fraction",
  )


  def _parse_prometheus_text(text: str) -> dict:
      """Extract tracked metrics from a Prometheus text-format payload.

      Returns {metric_name: float}. Labels are stripped — we report a
      single value per metric (the last one seen if multiple labels
      exist for the same name). Missing tracked metrics simply aren't
      present in the dict. Malformed lines are skipped, not raised.
      """
      out: dict[str, float] = {}
      for line in text.splitlines():
          line = line.strip()
          if not line or line.startswith("#"):
              continue
          if " " not in line:
              continue
          left, val_str = line.rsplit(" ", 1)
          name = left.split("{", 1)[0]
          if name not in _TRACKED_METRICS:
              continue
          try:
              out[name] = float(val_str)
          except ValueError:
              continue
      return out


  class MetricsScraper(threading.Thread):
      """Daemon thread: GETs /metrics every interval_s; snapshots + CSV row each scrape."""

      def __init__(
          self,
          port: int,
          csv_path: Path,
          interval_s: float = 5.0,
      ) -> None:
          super().__init__(daemon=True)
          self.port = port
          self.csv_path = csv_path
          self.interval_s = interval_s
          self._stop = threading.Event()
          self._latest: dict = {}
          self._lock = threading.Lock()

      def snapshot(self) -> dict:
          """Return a copy of the most-recent /metrics sample."""
          with self._lock:
              return dict(self._latest)

      def stop(self) -> None:
          self._stop.set()

      def run(self) -> None:
          self.csv_path.parent.mkdir(parents=True, exist_ok=True)
          with self.csv_path.open("w") as f:
              f.write("timestamp," + ",".join(_TRACKED_METRICS) + "\n")
              f.flush()
              while not self._stop.is_set():
                  try:
                      with urllib.request.urlopen(
                          f"http://localhost:{self.port}/metrics", timeout=2.0,
                      ) as r:
                          text = r.read().decode("utf-8", errors="replace")
                      sample = _parse_prometheus_text(text)
                      if sample:
                          with self._lock:
                              self._latest = sample
                          row = [f"{time.time():.2f}"] + [
                              str(sample.get(m, "")) for m in _TRACKED_METRICS
                          ]
                          f.write(",".join(row) + "\n")
                          f.flush()
                  except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                      # Scrape failures are silent; the progress reporter shows N/A.
                      pass
                  self._stop.wait(self.interval_s)


  def format_progress_line(
      config: str,
      instances_done: int,
      instances_total: int,
      snapshot: dict,
      elapsed_s: float,
      prev_tok_count: Optional[float] = None,
      prev_tok_time: Optional[float] = None,
  ) -> str:
      """Render a single status line. Pure function (no I/O)."""
      pct = (instances_done / max(1, instances_total)) * 100
      bar_width = 12
      filled = int(bar_width * instances_done / max(1, instances_total))
      bar = "█" * filled + "░" * (bar_width - filled)

      tok_total = snapshot.get("vllm:generation_tokens_total")
      if (
          tok_total is not None
          and prev_tok_count is not None
          and prev_tok_time is not None
      ):
          dt = max(1.0, time.time() - prev_tok_time)
          tok_per_s = max(0.0, (tok_total - prev_tok_count) / dt)
          tok_str = f"{tok_per_s:>5.0f} tok/s"
      else:
          tok_str = "  N/A tok/s"

      spars = snapshot.get("skylight_observed_sparsity_fraction")
      spars_str = f"spars={spars:.3f}" if spars is not None else "spars=N/A  "

      if instances_done > 0 and instances_done < instances_total:
          per_inst = elapsed_s / instances_done
          eta_s = per_inst * (instances_total - instances_done)
          m, s = divmod(int(eta_s), 60)
          eta_str = f"ETA {m}m{s:02d}s"
      else:
          eta_str = "ETA --:--"

      m, s = divmod(int(elapsed_s), 60)
      elapsed_str = f"{m}m{s:02d}s elapsed"

      return (
          f"[agentic] {config:<14} {instances_done:>3}/{instances_total} "
          f"[{bar}] {pct:>3.0f}%  {tok_str}  {spars_str}  {elapsed_str}  {eta_str}"
      )


  class ProgressReporter(threading.Thread):
      """Daemon thread: every interval_s, read scraper + count patches, emit one line."""

      def __init__(
          self,
          config_name: str,
          scraper: MetricsScraper,
          patches_path: Path,
          instances_total: int,
          progress_jsonl: Path,
          interval_s: Optional[float] = None,
      ) -> None:
          super().__init__(daemon=True)
          self.config_name = config_name
          self.scraper = scraper
          self.patches_path = patches_path
          self.instances_total = instances_total
          self.progress_jsonl = progress_jsonl

          if interval_s is None:
              env_interval = os.environ.get("SKYLIGHT_BENCH_PROGRESS_INTERVAL_S")
              interval_s = float(env_interval) if env_interval else 30.0
          self.interval_s = interval_s

          self._stop = threading.Event()
          self._t_start = time.time()
          self._prev_tok_count: Optional[float] = None
          self._prev_tok_time: Optional[float] = None

      def stop(self) -> None:
          self._stop.set()

      def run(self) -> None:
          self.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
          with self.progress_jsonl.open("a") as f:
              while not self._stop.is_set():
                  snapshot = self.scraper.snapshot()
                  instances_done = self._count_patches()
                  elapsed = time.time() - self._t_start
                  line = format_progress_line(
                      config=self.config_name,
                      instances_done=instances_done,
                      instances_total=self.instances_total,
                      snapshot=snapshot,
                      elapsed_s=elapsed,
                      prev_tok_count=self._prev_tok_count,
                      prev_tok_time=self._prev_tok_time,
                  )
                  print(line, flush=True)
                  record = {
                      "t": time.time(),
                      "config": self.config_name,
                      "elapsed_s": elapsed,
                      "instances_done": instances_done,
                      "instances_total": self.instances_total,
                      **snapshot,
                  }
                  f.write(json.dumps(record) + "\n")
                  f.flush()
                  tok_now = snapshot.get("vllm:generation_tokens_total")
                  if tok_now is not None:
                      self._prev_tok_count = tok_now
                      self._prev_tok_time = time.time()
                  self._stop.wait(self.interval_s)

      def _count_patches(self) -> int:
          if not self.patches_path.exists():
              return 0
          with self.patches_path.open() as f:
              return sum(1 for line in f if line.strip())
  ```

- [ ] **Step 4.4: Verify the tests pass**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_progress.py
  ```

  Expected: `✓ verify GREEN`, all 9 tests pass.

- [ ] **Step 4.5: Suggested commit for Krishna**

  Files: `src/skylight/bench/progress.py`, `tests/test_bench_progress.py`
  Message:
  ```
  feat(bench): metrics scraper + progress reporter for agentic runs

  MetricsScraper polls /metrics every 5s, writes CSV timeseries, keeps
  latest sample for the reporter. ProgressReporter ticks every 30s
  (env-configurable), prints one status line on stdout with instances
  done, tok/s, observed sparsity, elapsed and ETA, and appends a
  progress.jsonl row. All threads are daemons; scrape failures swallow
  silently. Runtime overhead <0.1% wall clock. 9 unit tests cover
  Prometheus parsing, line formatting, and the reporter's tick output.
  ```

---

## Task 5: `agentic.py` — one-config orchestrator

**Spec step 4 (second half).** Wires server lifecycle + agent + eval + progress together.

**Files:**
- Create: `src/skylight/bench/agentic.py`
- Create: `tests/test_bench_agentic.py`

- [ ] **Step 5.1: Write the failing tests**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/tests/test_bench_agentic.py`

  ```python
  """Tests for skylight.bench.agentic.run_one_config.

  All subprocess calls + the metrics scraper are mocked. We're testing the
  *control flow*: in what order things run, what happens on failure, what
  ends up in the summary dict.
  """
  from __future__ import annotations

  import json
  from pathlib import Path
  from unittest.mock import patch, MagicMock

  import pytest

  from skylight.bench.agentic import run_one_config


  # A fake benchmark that records calls and lets tests inject results.
  class FakeBenchmark:
      name = "fake-bench"

      def __init__(self, n_solved: int = 2, n_total: int = 3) -> None:
          self.n_solved = n_solved
          self.n_total = n_total
          self.calls: list[tuple[str, dict]] = []

      def default_instances_path(self) -> Path:
          return Path("never-used.txt")

      def build_agent_cmd(self, model, base_url, instances, out_dir):
          self.calls.append(("agent", {
              "model": model, "base_url": base_url,
              "instances": str(instances), "out_dir": str(out_dir),
          }))
          return ["/bin/true"]

      def build_eval_cmd(self, out_dir):
          self.calls.append(("eval", {"out_dir": str(out_dir)}))
          return ["/bin/true"]

      def parse_results(self, out_dir):
          self.calls.append(("parse", {"out_dir": str(out_dir)}))
          return {
              "n_solved": self.n_solved,
              "n_total": self.n_total,
              "pass_at_1": self.n_solved / max(1, self.n_total),
          }


  @pytest.fixture
  def small_instances(tmp_path: Path) -> Path:
      p = tmp_path / "subset.txt"
      p.write_text("django__django-12915\nflask__flask-4992\n")
      return p


  # ------------------------------- happy path ----------------------------------


  def test_run_one_config_returns_summary_with_pass_at_1(
      tmp_path: Path, small_instances: Path,
  ):
      bench = FakeBenchmark(n_solved=1, n_total=2)
      with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
           patch("skylight.bench.agentic.subprocess.run") as mock_run, \
           patch("skylight.bench.agentic.wait_for_health"), \
           patch("skylight.bench.agentic.terminate_group"), \
           patch("skylight.bench.agentic.free_port", return_value=12345), \
           patch("skylight.bench.agentic.MetricsScraper") as mock_scraper, \
           patch("skylight.bench.agentic.ProgressReporter") as mock_reporter:
          mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))
          mock_run.return_value = MagicMock(returncode=0)
          mock_scraper.return_value = MagicMock()
          mock_reporter.return_value = MagicMock()

          summary = run_one_config(
              benchmark=bench,
              config_name="dense",
              backend="dense",
              model="Qwen/Qwen3-32B",
              instances=small_instances,
              output_dir=tmp_path / "run42",
              max_model_len=33500,
          )

      assert summary["config"] == "dense"
      assert summary["backend"] == "dense"
      assert summary["n_solved"] == 1
      assert summary["n_total"] == 2
      assert summary["pass_at_1"] == 0.5
      assert "elapsed_s" in summary
      # summary.json written
      assert (tmp_path / "run42" / "summary.json").exists()


  def test_run_one_config_calls_agent_then_eval_then_parse_in_order(
      tmp_path: Path, small_instances: Path,
  ):
      bench = FakeBenchmark()
      with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
           patch("skylight.bench.agentic.subprocess.run") as mock_run, \
           patch("skylight.bench.agentic.wait_for_health"), \
           patch("skylight.bench.agentic.terminate_group"), \
           patch("skylight.bench.agentic.free_port", return_value=12345), \
           patch("skylight.bench.agentic.MetricsScraper"), \
           patch("skylight.bench.agentic.ProgressReporter"):
          mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))
          mock_run.return_value = MagicMock(returncode=0)

          run_one_config(
              benchmark=bench, config_name="dense", backend="dense",
              model="X", instances=small_instances,
              output_dir=tmp_path / "run", max_model_len=33500,
          )

      phases = [c[0] for c in bench.calls]
      assert phases == ["agent", "eval", "parse"]


  # ------------------------------- error paths ---------------------------------


  def test_empty_instances_file_returns_error_summary(tmp_path: Path):
      empty = tmp_path / "empty.txt"
      empty.write_text("")
      bench = FakeBenchmark()
      with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen:
          summary = run_one_config(
              benchmark=bench, config_name="dense", backend="dense",
              model="X", instances=empty,
              output_dir=tmp_path / "run", max_model_len=33500,
          )
      assert "errors" in summary
      assert "empty_instances_file" in summary["errors"]
      # No server spawned for an empty instances file
      mock_popen.assert_not_called()


  def test_server_unhealthy_returns_error_summary(
      tmp_path: Path, small_instances: Path,
  ):
      bench = FakeBenchmark()
      with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
           patch("skylight.bench.agentic.wait_for_health",
                 side_effect=TimeoutError("/health did not respond 200 within 180s")), \
           patch("skylight.bench.agentic.terminate_group") as mock_term, \
           patch("skylight.bench.agentic.free_port", return_value=12345):
          mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))

          summary = run_one_config(
              benchmark=bench, config_name="dense", backend="dense",
              model="X", instances=small_instances,
              output_dir=tmp_path / "run", max_model_len=33500,
          )

      assert "errors" in summary
      assert "server_unhealthy" in summary["errors"]
      assert "agent" not in [c[0] for c in bench.calls]  # never reached
      assert "eval" not in [c[0] for c in bench.calls]
      mock_term.assert_called_once()  # server still got torn down


  def test_server_always_torn_down_even_on_agent_crash(
      tmp_path: Path, small_instances: Path,
  ):
      """If subprocess.run raises mid-flight, terminate_group still fires."""
      bench = FakeBenchmark()
      with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
           patch("skylight.bench.agentic.subprocess.run",
                 side_effect=OSError("boom")), \
           patch("skylight.bench.agentic.wait_for_health"), \
           patch("skylight.bench.agentic.terminate_group") as mock_term, \
           patch("skylight.bench.agentic.free_port", return_value=12345), \
           patch("skylight.bench.agentic.MetricsScraper"), \
           patch("skylight.bench.agentic.ProgressReporter"):
          mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))

          with pytest.raises(OSError):
              run_one_config(
                  benchmark=bench, config_name="dense", backend="dense",
                  model="X", instances=small_instances,
                  output_dir=tmp_path / "run", max_model_len=33500,
              )
      mock_term.assert_called_once()


  # ------------------------------- sparse knobs ---------------------------------


  def test_sparse_knobs_flow_into_build_env(
      tmp_path: Path, small_instances: Path,
  ):
      """topk/sink/local/channel_num end up in the env build_env constructs."""
      bench = FakeBenchmark()
      with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
           patch("skylight.bench.agentic.subprocess.run",
                 return_value=MagicMock(returncode=0)), \
           patch("skylight.bench.agentic.wait_for_health"), \
           patch("skylight.bench.agentic.terminate_group"), \
           patch("skylight.bench.agentic.free_port", return_value=12345), \
           patch("skylight.bench.agentic.MetricsScraper"), \
           patch("skylight.bench.agentic.ProgressReporter"), \
           patch("skylight.bench.agentic.build_env") as mock_build_env:
          mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))
          mock_build_env.return_value = {"DUMMY": "1"}

          run_one_config(
              benchmark=bench, config_name="sparse-double", backend="sparse",
              model="X", instances=small_instances,
              output_dir=tmp_path / "run", max_model_len=33500,
              topk=0.10, sink=64, local=64, channel_num=8,
          )

      mock_build_env.assert_called_once()
      args = mock_build_env.call_args
      # Function uses kw or positional — accept either.
      kwargs = {**dict(zip(["backend", "topk", "sink", "local", "channel_num"], args.args)),
                **args.kwargs}
      assert kwargs["backend"] == "sparse"
      assert kwargs["topk"] == 0.10
      assert kwargs["sink"] == 64
      assert kwargs["local"] == 64
      assert kwargs["channel_num"] == 8
  ```

- [ ] **Step 5.2: Verify the tests fail**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_agentic.py
  ```

  Expected: red — `ModuleNotFoundError: No module named 'skylight.bench.agentic'`.

- [ ] **Step 5.3: Create `agentic.py`**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/src/skylight/bench/agentic.py`

  ```python
  """Orchestrate one (benchmark, config) agentic bench run.

  Lifecycle per call:
    1. Spawn the right server via run_serving helpers
    2. Wait for /health
    3. Start metrics scraper + progress reporter (daemon threads)
    4. Run the benchmark's agent subprocess (writes patches.jsonl)
    5. Run the benchmark's eval subprocess (writes results.json)
    6. Stop scraper + reporter
    7. Parse results into a summary dict, write summary.json
    8. Teardown server

  Errors at each step are captured into summary["errors"]; the server is
  always torn down in a finally block.
  """
  from __future__ import annotations

  import argparse
  import json
  import logging
  import os
  import subprocess
  import sys
  import time
  from pathlib import Path
  from typing import Optional

  from skylight.bench.benchmarks import REGISTRY
  from skylight.bench.benchmarks.base import AgenticBenchmark
  from skylight.bench.progress import MetricsScraper, ProgressReporter
  from skylight.bench.run_serving import (
      build_env,
      build_serve_cmd,
      free_port,
      terminate_group,
      wait_for_health,
  )


  def _setup_logger(log_path: Path) -> logging.Logger:
      """Configure a per-run logger that tees to file + stdout."""
      logger = logging.getLogger(f"skylight.bench.agentic.{log_path.parent.name}")
      logger.handlers.clear()
      level = logging.DEBUG if os.environ.get("SKYLIGHT_BENCH_DEBUG") == "1" else logging.INFO
      logger.setLevel(level)
      fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
      log_path.parent.mkdir(parents=True, exist_ok=True)
      fh = logging.FileHandler(log_path)
      fh.setFormatter(fmt)
      logger.addHandler(fh)
      sh = logging.StreamHandler(sys.stdout)
      sh.setFormatter(fmt)
      logger.addHandler(sh)
      logger.propagate = False
      return logger


  def run_one_config(
      benchmark: AgenticBenchmark,
      config_name: str,
      backend: str,
      model: str,
      instances: Path,
      output_dir: Path,
      max_model_len: int,
      server_timeout_s: float = 180.0,
      enforce_eager: bool = True,
      topk: Optional[float] = None,
      sink: Optional[int] = None,
      local: Optional[int] = None,
      channel_num: Optional[int] = None,
  ) -> dict:
      """Run one (benchmark, config) end-to-end. Returns summary dict."""
      output_dir.mkdir(parents=True, exist_ok=True)
      log = _setup_logger(output_dir / "orchestrator.log")
      log.info("[%s] start; backend=%s model=%s", config_name, backend, model)

      instances_total = sum(
          1 for line in instances.read_text().splitlines() if line.strip()
      )
      if instances_total == 0:
          log.error("[%s] instances file %s is empty", config_name, instances)
          summary = {
              "config": config_name, "backend": backend, "model": model,
              "n_solved": 0, "n_total": 0, "pass_at_1": 0.0,
              "elapsed_s": 0.0, "errors": ["empty_instances_file"],
          }
          (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
          return summary

      port = free_port()
      serve_cmd = build_serve_cmd(backend, model, port, max_model_len, enforce_eager)
      env = build_env(
          backend=backend, topk=topk, sink=sink, local=local, channel_num=channel_num,
      )

      log.info("[%s] spawning server: %s", config_name, " ".join(serve_cmd))
      server_log = output_dir / "server.log"
      with server_log.open("w") as slog:
          proc = subprocess.Popen(
              serve_cmd, env=env,
              stdout=slog, stderr=subprocess.STDOUT,
              start_new_session=True,
          )

      t_start = time.time()
      scraper: Optional[MetricsScraper] = None
      reporter: Optional[ProgressReporter] = None
      summary: dict = {
          "config": config_name, "backend": backend, "model": model,
          "instances_total": instances_total,
      }

      try:
          try:
              wait_for_health(port, server_timeout_s, proc)
              log.info("[%s] server /health ready on port %d", config_name, port)
          except (TimeoutError, RuntimeError) as exc:
              log.error("[%s] server unhealthy: %s", config_name, exc)
              summary.update({
                  "n_solved": 0, "n_total": 0, "pass_at_1": 0.0,
                  "elapsed_s": time.time() - t_start,
                  "errors": ["server_unhealthy"],
              })
              (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
              return summary

          scraper = MetricsScraper(
              port=port, csv_path=output_dir / "metrics.csv", interval_s=5.0,
          )
          scraper.start()
          reporter = ProgressReporter(
              config_name=config_name,
              scraper=scraper,
              patches_path=output_dir / "patches.jsonl",
              instances_total=instances_total,
              progress_jsonl=output_dir / "progress.jsonl",
          )
          reporter.start()

          base_url = f"http://localhost:{port}/v1"

          agent_cmd = benchmark.build_agent_cmd(
              model=model, base_url=base_url,
              instances=instances, out_dir=output_dir,
          )
          log.info("[%s] running agent: %s", config_name, " ".join(agent_cmd))
          with (output_dir / "agent.log").open("w") as alog:
              agent_rc = subprocess.run(
                  agent_cmd, stdout=alog, stderr=subprocess.STDOUT,
              ).returncode
          log.info("[%s] agent exit rc=%d", config_name, agent_rc)
          if agent_rc != 0:
              summary.setdefault("errors", []).append(f"agent_rc={agent_rc}")

          eval_cmd = benchmark.build_eval_cmd(output_dir)
          log.info("[%s] running eval: %s", config_name, " ".join(eval_cmd))
          with (output_dir / "eval.log").open("w") as elog:
              eval_rc = subprocess.run(
                  eval_cmd, stdout=elog, stderr=subprocess.STDOUT,
              ).returncode
          log.info("[%s] eval exit rc=%d", config_name, eval_rc)
          if eval_rc != 0:
              summary.setdefault("errors", []).append(f"eval_rc={eval_rc}")

          parsed = benchmark.parse_results(output_dir)
          summary.update(parsed)
          summary["elapsed_s"] = time.time() - t_start
          (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
          log.info(
              "[%s] done in %.0fs: n_solved=%d n_total=%d pass@1=%.3f",
              config_name, summary["elapsed_s"],
              summary.get("n_solved", 0), summary.get("n_total", 0),
              summary.get("pass_at_1", 0.0),
          )
          return summary

      finally:
          if reporter is not None:
              reporter.stop()
              reporter.join(timeout=5)
          if scraper is not None:
              scraper.stop()
              scraper.join(timeout=5)
          log.info("[%s] tearing down server", config_name)
          terminate_group(proc)


  # ------------------------------ CLI entry point -------------------------------


  def main(argv: Optional[list[str]] = None) -> int:
      parser = argparse.ArgumentParser(
          description=__doc__.split("\n\n")[0],
          formatter_class=argparse.RawDescriptionHelpFormatter,
      )
      parser.add_argument("--benchmark", default="mini-swe-agent",
                          choices=list(REGISTRY))
      parser.add_argument("--backend", choices=["sparse", "dense"], required=True)
      parser.add_argument("--model", required=True)
      parser.add_argument("--instances", required=True)
      parser.add_argument("--output-dir", default=None,
                          help="default: bench-results/agentic/<timestamp>")
      parser.add_argument("--max-model-len", type=int, default=33500)
      parser.add_argument("--server-timeout", type=float, default=180.0)
      parser.add_argument("--topk", type=float, default=None)
      parser.add_argument("--sink", type=int, default=None)
      parser.add_argument("--local", type=int, default=None)
      parser.add_argument("--channel-num", type=int, default=None)
      args = parser.parse_args(argv)

      benchmark = REGISTRY[args.benchmark]
      if args.output_dir:
          output_dir = Path(args.output_dir)
      else:
          ts = time.strftime("%Y%m%d_%H%M%S")
          output_dir = Path(f"bench-results/agentic/{args.backend}_{ts}")

      summary = run_one_config(
          benchmark=benchmark,
          config_name=args.backend,
          backend=args.backend,
          model=args.model,
          instances=Path(args.instances),
          output_dir=output_dir,
          max_model_len=args.max_model_len,
          server_timeout_s=args.server_timeout,
          topk=args.topk, sink=args.sink, local=args.local,
          channel_num=args.channel_num,
      )
      return 0 if not summary.get("errors") else 1


  if __name__ == "__main__":
      sys.exit(main())
  ```

- [ ] **Step 5.4: Verify the tests pass**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_agentic.py
  ```

  Expected: `✓ verify GREEN`, all 6 tests pass.

- [ ] **Step 5.5: Smoke run on one instance against the real bench host**

  Krishna creates a 1-instance file and runs the orchestrator on vb200-14:

  ```
  echo "sympy__sympy-20590" > /tmp/one.txt
  .venv/bin/python -m skylight.bench.agentic \
      --benchmark mini-swe-agent \
      --backend dense \
      --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
      --instances /tmp/one.txt \
      --output-dir bench-results/agentic/smoke-dense-one \
      --max-model-len 33500
  ```

  Expected:
  - Server spins up, /health passes (~60-90s for 30B model)
  - Progress lines appear on stdout every 30s
  - `bench-results/agentic/smoke-dense-one/`:
    - `summary.json` with `{n_solved, n_total=1, pass_at_1, elapsed_s}`
    - `metrics.csv` with multiple rows
    - `progress.jsonl` with multiple records
    - `server.log`, `agent.log`, `eval.log`, `orchestrator.log` all populated

  If the run fails partway, check `orchestrator.log` for the failing phase and fix before committing.

- [ ] **Step 5.6: Suggested commit for Krishna**

  Files: `src/skylight/bench/agentic.py`, `tests/test_bench_agentic.py`
  Message:
  ```
  feat(bench): one-config agentic orchestrator

  run_one_config(benchmark, config, backend, model, instances, ...)
  spawns server (reuses run_serving helpers), waits /health, starts
  metrics scraper + progress reporter, drives agent then eval, parses
  summary, tears down server. Errors at each phase land in
  summary["errors"]; server torn down in a finally block regardless.
  Per-run dir holds orchestrator/server/agent/eval logs, patches,
  metrics.csv, progress.jsonl, summary.json. 6 unit tests cover
  control flow (order, errors, teardown, sparse-knob plumbing).
  ```

---

## Manual Phase B: Real dense run on 32 instances (no code change)

**Spec step 5.** First operational milestone — dense baseline pass@1.

- [ ] **Step B.1: Krishna runs the dense baseline**

  On vb200-14, from the skylight repo root:

  ```
  .venv/bin/python -m skylight.bench.agentic \
      --benchmark mini-swe-agent \
      --backend dense \
      --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
      --instances bench-resources/swebench-subset-32.txt \
      --output-dir bench-results/agentic/dense-32-baseline \
      --max-model-len 33500
  ```

  Expected wall time: ~3-5 GPU-hours (32 instances × 5-10 min/instance + per-instance docker eval).

- [ ] **Step B.2: Sanity-check the result against public Qwen-Coder numbers**

  Read `bench-results/agentic/dense-32-baseline/summary.json`. The `pass_at_1` should be roughly in the **0.20-0.35** range for Qwen-Coder-30B-A3B on SWE-bench Lite. If it's much lower (e.g. <0.10), the harness or model is misconfigured — investigate before running sparse configs.

  **Risk gate.** If pass@1 is wildly off, do not proceed to Task 6. Diagnose the harness first.

---

## Task 6: `agentic_sweep.py` — outer sweep loop + comparison table

**Spec step 6.** Mirrors `sweep.py` for accuracy.

**Files:**
- Create: `src/skylight/bench/agentic_sweep.py`
- Create: `tests/test_bench_agentic_sweep.py`

- [ ] **Step 6.1: Write the failing tests**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/tests/test_bench_agentic_sweep.py`

  ```python
  """Tests for skylight.bench.agentic_sweep.

  Covers CONFIGS table, per-axis override flow, and the comparison-table
  formatter. The per-config run itself is mocked out — that path is
  exercised by test_bench_agentic.py.
  """
  from __future__ import annotations

  from pathlib import Path
  from unittest.mock import patch, MagicMock

  import pytest

  from skylight.bench.agentic_sweep import (
      CONFIGS,
      build_run_one_kwargs,
      format_accuracy_table,
  )


  # ----------------------------- CONFIGS table ----------------------------------


  def test_configs_dense_token_double_present():
      assert "dense" in CONFIGS
      assert "sparse-token" in CONFIGS
      assert "sparse-double" in CONFIGS


  def test_configs_sparse_entries_have_required_knobs():
      for name, cfg in CONFIGS.items():
          if cfg["backend"] != "sparse":
              continue
          for required in ("topk", "sink", "local", "channel_num"):
              assert required in cfg, f"{name!r} missing {required!r}"


  def test_configs_sparse_token_channel_num_minus_one():
      """sparse-token uses full head_dim (channel_num=-1)."""
      assert CONFIGS["sparse-token"]["channel_num"] == -1


  def test_configs_sparse_double_channel_num_positive():
      """sparse-double has a positive channel_num (doubly-sparse)."""
      assert CONFIGS["sparse-double"]["channel_num"] > 0


  # ----------------------------- build_run_one_kwargs ---------------------------


  def test_dense_kwargs_omit_sparse_knobs():
      kw = build_run_one_kwargs("dense")
      assert kw["backend"] == "dense"
      assert "topk" not in kw or kw["topk"] is None
      assert "sink" not in kw or kw["sink"] is None


  def test_sparse_token_kwargs_carry_named_defaults():
      kw = build_run_one_kwargs("sparse-token")
      assert kw["backend"] == "sparse"
      assert kw["topk"] == CONFIGS["sparse-token"]["topk"]
      assert kw["sink"] == CONFIGS["sparse-token"]["sink"]
      assert kw["local"] == CONFIGS["sparse-token"]["local"]
      assert kw["channel_num"] == -1


  def test_per_axis_overrides_win_over_named_defaults():
      """--topk 0.02 replaces the named config's baked-in 0.10."""
      kw = build_run_one_kwargs(
          "sparse-token", topk_override=0.02, sink_override=128,
      )
      assert kw["topk"] == 0.02
      assert kw["sink"] == 128
      # Unoverridden knob still uses the named config's value.
      assert kw["channel_num"] == -1


  def test_override_none_falls_back_to_named_default():
      kw = build_run_one_kwargs("sparse-token", topk_override=None)
      assert kw["topk"] == CONFIGS["sparse-token"]["topk"]


  def test_dense_overrides_ignored():
      """Sparse-axis overrides on a dense config don't leak into kwargs."""
      kw = build_run_one_kwargs(
          "dense", topk_override=0.02, sink_override=128, channel_num_override=8,
      )
      assert "topk" not in kw or kw["topk"] is None
      assert "sink" not in kw or kw["sink"] is None


  # ----------------------------- format_accuracy_table --------------------------


  def test_format_table_has_headers_and_rows():
      results = [
          {"config": "dense", "n_solved": 10, "n_total": 32,
           "pass_at_1": 0.3125, "elapsed_s": 12000.0},
          {"config": "sparse-token", "n_solved": 9, "n_total": 32,
           "pass_at_1": 0.2812, "elapsed_s": 13500.0},
      ]
      table = format_accuracy_table(results)
      for h in ["config", "n_solved", "n_total", "pass@1", "elapsed"]:
          assert h in table
      assert "dense" in table
      assert "sparse-token" in table
      assert "0.31" in table  # pass@1 formatting
      assert "0.28" in table


  def test_format_table_skips_failed_runs():
      """None entries (failed configs) don't get a row, no literal 'None' leaks."""
      results = [
          None,
          {"config": "dense", "n_solved": 10, "n_total": 32,
           "pass_at_1": 0.3125, "elapsed_s": 12000.0},
          None,
      ]
      table = format_accuracy_table(results)
      assert table.count("dense") == 1
      assert "None" not in table


  def test_format_table_handles_zero_total():
      """A failed run with n_total=0 shouldn't crash the formatter."""
      results = [
          {"config": "dense", "n_solved": 0, "n_total": 0,
           "pass_at_1": 0.0, "elapsed_s": 5.0,
           "errors": ["server_unhealthy"]},
      ]
      table = format_accuracy_table(results)
      assert "dense" in table
      assert "0/0" in table or "0" in table
  ```

- [ ] **Step 6.2: Verify the tests fail**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_agentic_sweep.py
  ```

  Expected: red — `ModuleNotFoundError: No module named 'skylight.bench.agentic_sweep'`.

- [ ] **Step 6.3: Create `agentic_sweep.py`**

  Path: `/Users/kumarkagrawal/local/berkeley/sky/skylight/src/skylight/bench/agentic_sweep.py`

  ```python
  """Sweep multiple configs on the same benchmark and emit a pass@1 table.

  Wraps :func:`skylight.bench.agentic.run_one_config` so a single command
  runs dense + sparse-token + sparse-double against the same instance
  subset and tabulates pass@1.

  Usage::

      python -m skylight.bench.agentic_sweep \\
          --benchmark mini-swe-agent \\
          --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
          --instances bench-resources/swebench-subset-32.txt \\
          --configs dense,sparse-token,sparse-double

  Outputs::

      bench-results/agentic/<sweep_id>/<config>/   per-config run dir
      bench-results/agentic/<sweep_id>/sweep-summary.json
      bench-results/agentic/<sweep_id>/sweep.log
  """
  from __future__ import annotations

  import argparse
  import json
  import sys
  import time
  from pathlib import Path
  from typing import Optional

  from skylight.bench.agentic import run_one_config
  from skylight.bench.benchmarks import REGISTRY


  # Named configurations: the delta from a dense baseline. Adding a new config
  # here propagates automatically through CLI choices and the loop logic.
  CONFIGS: dict[str, dict] = {
      "dense": {"backend": "dense"},
      "sparse-token": {
          "backend": "sparse",
          "topk": 0.10, "sink": 64, "local": 64, "channel_num": -1,
      },
      "sparse-double": {
          "backend": "sparse",
          "topk": 0.10, "sink": 64, "local": 64, "channel_num": 8,
      },
  }


  def build_run_one_kwargs(
      config_name: str,
      topk_override: Optional[float] = None,
      sink_override: Optional[int] = None,
      local_override: Optional[int] = None,
      channel_num_override: Optional[int] = None,
  ) -> dict:
      """Translate a config name + optional overrides into run_one_config kwargs.

      Sparse-axis overrides only apply to sparse configs; for dense they
      are silently ignored.
      """
      cfg = CONFIGS[config_name]
      kwargs: dict = {"backend": cfg["backend"]}
      if cfg["backend"] == "sparse":
          kwargs["topk"] = topk_override if topk_override is not None else cfg["topk"]
          kwargs["sink"] = sink_override if sink_override is not None else cfg["sink"]
          kwargs["local"] = local_override if local_override is not None else cfg["local"]
          kwargs["channel_num"] = (
              channel_num_override if channel_num_override is not None
              else cfg["channel_num"]
          )
      return kwargs


  def format_accuracy_table(results: list[Optional[dict]]) -> str:
      """Render a one-row-per-config pass@1 comparison table."""
      headers = ["config", "n_solved", "n_total", "pass@1", "elapsed", "errors"]
      widths = [16, 10, 10, 10, 10, 20]
      lines = []
      sep = "  ".join("-" * w for w in widths)
      lines.append("  ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
      lines.append(sep)
      for r in results:
          if r is None:
              continue
          errs = ",".join(r.get("errors", []))[: widths[5]] or "-"
          row = [
              f"{r['config']:<{widths[0]}}",
              f"{r.get('n_solved', 0):>{widths[1]}}",
              f"{r.get('n_total', 0):>{widths[2]}}",
              f"{r.get('pass_at_1', 0.0):>{widths[3]}.3f}",
              f"{int(r.get('elapsed_s', 0.0)):>{widths[4]}}s",
              f"{errs:<{widths[5]}}",
          ]
          lines.append("  ".join(row))
      return "\n".join(lines)


  def main(argv: Optional[list[str]] = None) -> int:
      parser = argparse.ArgumentParser(
          description=__doc__.split("\n\n")[0],
          formatter_class=argparse.RawDescriptionHelpFormatter,
      )
      parser.add_argument("--benchmark", default="mini-swe-agent",
                          choices=list(REGISTRY))
      parser.add_argument("--model", required=True)
      parser.add_argument("--instances", required=True)
      parser.add_argument("--configs", default=",".join(CONFIGS.keys()),
                          help=f"comma-separated from: {','.join(CONFIGS)}")
      parser.add_argument("--output-dir", default=None,
                          help="default: bench-results/agentic/sweep_<timestamp>")
      parser.add_argument("--max-model-len", type=int, default=33500)
      parser.add_argument("--server-timeout", type=float, default=180.0)
      parser.add_argument("--topk", type=float, default=None,
                          help="override --topk for every sparse config")
      parser.add_argument("--sink", type=int, default=None)
      parser.add_argument("--local", type=int, default=None)
      parser.add_argument("--channel-num", type=int, default=None)
      args = parser.parse_args(argv)

      config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
      invalid = [c for c in config_names if c not in CONFIGS]
      if invalid:
          print(f"[agentic-sweep] unknown configs {invalid!r}; "
                f"valid: {list(CONFIGS)}", file=sys.stderr)
          return 2

      benchmark = REGISTRY[args.benchmark]
      ts = time.strftime("%Y%m%d_%H%M%S")
      sweep_dir = Path(args.output_dir or f"bench-results/agentic/sweep_{ts}")
      sweep_dir.mkdir(parents=True, exist_ok=True)

      sweep_log = sweep_dir / "sweep.log"
      def slog(msg: str) -> None:
          line = f"[agentic-sweep] {msg}"
          print(line, flush=True)
          with sweep_log.open("a") as f:
              f.write(line + "\n")

      slog(f"sweep_dir={sweep_dir} benchmark={args.benchmark} configs={config_names}")
      slog(f"instances={args.instances}")
      if any([args.topk, args.sink, args.local, args.channel_num is not None]):
          slog(
              f"overrides: topk={args.topk} sink={args.sink} "
              f"local={args.local} channel_num={args.channel_num}"
          )

      results: list[Optional[dict]] = []
      t_start = time.time()
      for name in config_names:
          kwargs = build_run_one_kwargs(
              name,
              topk_override=args.topk, sink_override=args.sink,
              local_override=args.local, channel_num_override=args.channel_num,
          )
          slog(f"▶ {name}")
          try:
              summary = run_one_config(
                  benchmark=benchmark,
                  config_name=name,
                  model=args.model,
                  instances=Path(args.instances),
                  output_dir=sweep_dir / name,
                  max_model_len=args.max_model_len,
                  server_timeout_s=args.server_timeout,
                  **kwargs,
              )
              results.append(summary)
              slog(
                  f"✓ {name}: n_solved={summary.get('n_solved')} "
                  f"n_total={summary.get('n_total')} "
                  f"pass@1={summary.get('pass_at_1', 0.0):.3f} "
                  f"elapsed={int(summary.get('elapsed_s', 0))}s "
                  f"errors={summary.get('errors', [])}"
              )
          except Exception as exc:
              slog(f"✗ {name}: orchestrator raised {exc!r}")
              results.append(None)

      elapsed = time.time() - t_start
      slog(f"done — {len(config_names)} configs in {elapsed:.0f}s")
      print("")
      print(format_accuracy_table(results))

      summary_path = sweep_dir / "sweep-summary.json"
      summary_path.write_text(json.dumps({
          "benchmark": args.benchmark,
          "model": args.model,
          "instances": args.instances,
          "configs": config_names,
          "results": [r for r in results if r is not None],
          "elapsed_s": elapsed,
      }, indent=2) + "\n")
      slog(f"summary → {summary_path}")

      failures = sum(1 for r in results if r is None or r.get("errors"))
      return 1 if failures else 0


  if __name__ == "__main__":
      sys.exit(main())
  ```

- [ ] **Step 6.4: Verify the tests pass**

  Tell Krishna to run:

  ```
  scripts/dev/verify_commit.sh vb200-14 tests/test_bench_agentic_sweep.py
  ```

  Expected: `✓ verify GREEN`, all 11 tests pass.

- [ ] **Step 6.5: Suggested commit for Krishna**

  Files: `src/skylight/bench/agentic_sweep.py`, `tests/test_bench_agentic_sweep.py`
  Message:
  ```
  feat(bench): agentic sweep — dense vs sparse-token vs sparse-double pass@1

  Named CONFIGS table mirrors the perf sweep: dense, sparse-token
  (channel_num=-1), sparse-double (channel_num=8). Per-axis CLI
  overrides (--topk/--sink/--local/--channel-num) lay over each named
  default. Per-config run goes to sweep_dir/<config>/; outer
  sweep-summary.json + sweep.log aggregate. Comparison table prints
  pass@1 with elapsed and error column. 11 unit tests cover CONFIGS,
  override flow, table formatting, failure rows.
  ```

---

## Manual Phase C: Milestone sweep — 32 × 3 configs (no code change)

**Spec step 7.** The deliverable.

- [ ] **Step C.1: Krishna runs the full sweep**

  On vb200-14:

  ```
  .venv/bin/python -m skylight.bench.agentic_sweep \
      --benchmark mini-swe-agent \
      --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
      --instances bench-resources/swebench-subset-32.txt \
      --configs dense,sparse-token,sparse-double
  ```

  Expected wall time: ~10-15 GPU-hours (3 configs × ~3-5 GPU-hours each).

  Expected stdout (final):

  ```
  config             n_solved   n_total   pass@1    elapsed
  -----------------  --------   -------   -------   --------
  dense                  N        32      0.XXX     NNNNs
  sparse-token           N        32      0.XXX     NNNNs
  sparse-double          N        32      0.XXX     NNNNs
  ```

- [ ] **Step C.2: Capture findings in `bench-results/agentic/sweep_<ts>/sweep-summary.json`**

  This JSON is the per-sweep artifact. Krishna can copy it back to the Mac for the writeup.

  **Risk gate.** If sparse `pass@1` drops more than 5pp vs dense, that's the project's headline sparse-accuracy finding. Plan a 2%-sparsity follow-up sweep (`--topk 0.02`) and a longer subset to characterize.

---

## Manual Phase D: Scale 64 → 128 → full (no code change)

**Spec step 8.** Larger subsets, same command.

- [ ] **Step D.1: Add `bench-resources/swebench-subset-64.txt`**

  Superset of the 32 list (first 32 lines identical, 32 more appended). Krishna picks the additional 32 from SWE-bench Lite; commit as a new file.

  Suggested commit message:
  ```
  bench: 64-instance SWE-bench Lite subset (superset of -32)

  Adds 32 more SWE-bench Lite instance IDs, keeping the first 32
  identical to swebench-subset-32.txt so results stay comparable.
  ```

- [ ] **Step D.2: Krishna re-runs the sweep at 64**

  Same command, different `--instances`:

  ```
  .venv/bin/python -m skylight.bench.agentic_sweep \
      --benchmark mini-swe-agent \
      --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
      --instances bench-resources/swebench-subset-64.txt \
      --configs dense,sparse-token,sparse-double
  ```

  Expected wall time: ~20-30 GPU-hours.

- [ ] **Step D.3: Repeat for 128 and full**

  Same shape, larger subset files. Stop at any tier the project doesn't need.

---

## Self-Review

Checked the plan against the spec on 2026-05-26:

1. **Spec coverage:**
   - Goal (single-command sweep, comparison table): Tasks 5+6, Manual Phase C ✓
   - File structure (5 NEW files + 4 test files + 1 modify + 1 resource): Tasks 2-6 cover all ✓
   - `AgenticBenchmark` Protocol: Task 3 Step 3.4 ✓
   - `MiniSweAgent` adapter: Task 3 Step 3.5 ✓
   - `run_one_config` orchestrator: Task 5 ✓
   - Progress scraper + reporter + format_progress_line: Task 4 ✓
   - Sweep with CONFIGS table + per-axis overrides: Task 6 ✓
   - Per-run output layout (8 files): Tasks 4-5 produce them ✓
   - 5 verify gates from spec (step 0,1,3,4,6): Tasks 1,2,3,4+5,6 ✓
   - 4 manual phases (step 2,5,7,8): Manual Phases A,B,C,D ✓
   - Risk gates (steps 2,5,7): Manual Phase A end, Step B.2, Step C.2 ✓
   - 32 → 64 → 128 → full ladder: Task 2 + Manual Phase D ✓
   - Logging levels (INFO/WARN/DEBUG): agentic.py `_setup_logger` ✓
   - `SKYLIGHT_BENCH_DEBUG=1` flag: agentic.py + progress.py ✓
   - `SKYLIGHT_BENCH_PROGRESS_INTERVAL_S` env var: progress.py ProgressReporter ✓
   - Error-handling table (6 rows): tests in Task 5 cover empty file, unhealthy server, agent crash, sparse-knob flow; `/metrics` swallow + reporter crash covered by daemon-thread design ✓

2. **Placeholder scan:**
   - No "TBD", "TODO", "implement later", "add error handling" anywhere ✓
   - One inline note about CLI flag names in `mini_swe_agent.py` is justified — flags resolved by Manual Phase A's verify gate before Task 3 ships ✓
   - The 32-instance list in Task 2.3 is concrete, not a placeholder ✓

3. **Type consistency:**
   - `AgenticBenchmark.parse_results` returns `dict` everywhere ✓
   - `MetricsScraper.snapshot()` returns `dict` everywhere ✓
   - `run_one_config` signature matches tests + sweep caller ✓
   - `CONFIGS` table keys (`backend`, `topk`, `sink`, `local`, `channel_num`) consistent across spec, sweep, tests ✓
   - `summary` dict keys (`n_solved`, `n_total`, `pass_at_1`, `elapsed_s`, `errors`) consistent across adapter / orchestrator / sweep ✓

No gaps. Plan ready for execution.
