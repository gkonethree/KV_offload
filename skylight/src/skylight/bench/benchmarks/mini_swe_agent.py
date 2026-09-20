"""Adapter for SWE-bench/mini-swe-agent v2 + the swebench eval harness.

mini-swe-agent v2's batch runner is ``mini-extra swebench``. It loads
the default ``config/benchmarks/swebench.yaml`` (prompts, docker
environment, the upstream's default Anthropic model). We override the
model section via ``-c key=value`` pairs to point LiteLLM at the local
skylight serve endpoint.

Phase A (agent): ``mini-extra swebench`` resolves instances from
``--subset lite --split test``, optionally filters via ``--filter
<regex>`` (we build a regex anchored to our subset's instance IDs),
runs each in a per-instance docker container, writes per-instance
trajectories + a predictions file under ``--output``.

Phase B (eval): ``swebench.harness.run_evaluation`` reads the
predictions file, runs each patch's tests inside the SWE-bench docker
images, writes a final report JSON with ``resolved_ids`` /
``submitted_ids`` lists. Despite ``--report_dir`` the harness writes
the report to CWD; the orchestrator runs eval with cwd=out_dir so the
report lands alongside other artifacts.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


# Default subset path is project-relative. Resolved by caller against CWD.
_DEFAULT_INSTANCES = Path("bench-resources/swebench-subset-32.txt")

# Predictions filename mini-extra swebench writes under --output. Confirmed
# during Phase A dry run; adjust here if mini-swe-agent v2 changes it.
_PREDS_FILENAME = "preds.json"


def _mini_extra_binary() -> str:
    """Locate the mini-extra console script alongside the active python."""
    # Console scripts installed by uv/pip land in the same bin/ as sys.executable.
    return str(Path(sys.executable).parent / "mini-extra")


def _read_instance_ids(instances_path: Path) -> list[str]:
    return [
        line.strip()
        for line in instances_path.read_text().splitlines()
        if line.strip()
    ]


def _instance_filter_regex(instance_ids: list[str]) -> str:
    """Anchored alternation matching exactly the given instance IDs."""
    escaped = [re.escape(i) for i in instance_ids]
    return "^(" + "|".join(escaped) + ")$"


def _find_report_json(out_dir: Path, run_id: str | None = None) -> Path | None:
    """Find swebench's report JSON, written under out_dir thanks to the
    eval subprocess running with cwd=out_dir. We deliberately do NOT
    fall back to ``Path.cwd()`` because that risks picking up a stale
    report from a previous run.

    When ``run_id`` is set, prefer a report whose filename contains it.
    """
    candidates: list[Path] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if (
            isinstance(data, dict)
            and "resolved_ids" in data
            and "submitted_ids" in data
        ):
            candidates.append(path)
    if not candidates:
        return None
    if run_id:
        for path in candidates:
            if run_id in path.name:
                return path
    return candidates[-1]


class MiniSweAgent:
    """v2 adapter: mini-extra swebench + swebench.harness.run_evaluation."""

    name = "mini-swe-agent"

    def default_instances_path(self) -> Path:
        return _DEFAULT_INSTANCES

    def build_agent_cmd(
        self,
        model: str,
        base_url: str,
        instances: Path,
        out_dir: Path,
        step_limit: int = 100,
        request_timeout_s: int = 300,
        wall_time_limit_seconds: int = 1800,
        max_tokens: int = 8192,
    ) -> list[str]:
        """``mini-extra swebench`` against SWE-bench Lite test split,
        filtered to our subset, pointed at the local skylight serve.

        We pass ``-c swebench.yaml`` explicitly because *any* ``-c``
        flag drops the auto-loaded default config; this re-includes it
        before our overrides.

        Three overlapping caps prevent runaway compute on any one
        instance; the first one tripped wins:

        ``step_limit`` (default 100): hard cap on agent turns. Same
        value across configs = same compute envelope = fair
        config-to-config comparison.

        ``wall_time_limit_seconds`` (default 1800 = 30 min): per
        instance wall clock. Raises ``TimeExceeded`` and submits empty
        when the agent has spent too long on a single problem; this
        bounds the worst case independent of step count or token-rate.

        ``max_tokens`` (default 8192): per-call completion-length cap.
        Sent to the model as ``model_kwargs.max_tokens``. Bounds a
        single response so the model can't loop indefinitely inside
        one turn (streaming throughput keeps the HTTP request_timeout
        from firing). With max_tokens=8192 at any reasonable decode
        rate, no one call exceeds ~10 min.

        ``request_timeout_s`` (default 300 = 5 min): per-call HTTP
        read timeout. Catches a hung server (no bytes flowing). The
        SkylightLitellmModel wrapper is ALWAYS active because it's
        what writes ``litellm_trace.jsonl``; on a Timeout it also
        aborts the instance (no retry against a hung server, since
        retry would just burn another timeout window).
        """
        instance_ids = _read_instance_ids(instances)
        instance_regex = _instance_filter_regex(instance_ids)
        return [
            _mini_extra_binary(), "swebench",
            "--subset", "verified",
            "--split", "test",
            "--filter", instance_regex,
            "--output", str(out_dir),
            "--workers", "1",
            "-c", "swebench.yaml",
            "-c", f"model.model_name=openai/{model}",
            "-c", f"model.model_kwargs.api_base={base_url}",
            "-c", "model.model_kwargs.api_key=EMPTY",
            "-c", f"model.model_kwargs.request_timeout={request_timeout_s}",
            "-c", f"model.model_kwargs.max_tokens={max_tokens}",
            # SkylightLitellmModel is REQUIRED for litellm_trace.jsonl
            # (per-call observability the cron + post-process consume).
            # On Timeout it aborts (no retry against hung server).
            "-c", "model.model_class=skylight.bench.litellm_model.SkylightLitellmModel",
            # Fixed compute envelope per instance: same step cap for
            # every config so a dense-vs-sparse pass@1 diff is real
            # quality, not different turn budgets.
            "-c", f"agent.step_limit={step_limit}",
            # Per-instance wall clock: stops runaway loops the step
            # counter alone won't catch (one turn that streams forever).
            "-c", f"agent.wall_time_limit_seconds={wall_time_limit_seconds}",
        ]

    def build_eval_cmd(
        self,
        out_dir: Path,
        predictions_path: Path | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        """``swebench.harness.run_evaluation`` on predictions in out_dir.

        Caller MUST run this with cwd=out_dir; swebench writes the
        report file to CWD ignoring ````--report_dir``.

        ``predictions_path`` defaults to ``out_dir/preds.json``; pass a
        filtered copy to retry eval on a subset only.
        ``run_id`` defaults to ``out_dir.name``; use a distinct id for
        retry passes so reports do not collide.
        """
        preds = predictions_path or (out_dir / _PREDS_FILENAME)
        rid = run_id or out_dir.name
        return [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-Bench_Verified",
            "--split", "test",
            "--predictions_path", str(preds),
            "--run_id", rid,
            "--max_workers", "4",
            "--report_dir", str(out_dir),
        ]

    def parse_results(self, out_dir: Path, run_id: str | None = None) -> dict:
        """Find the swebench report and return headline numbers + id lists.

        Returns keys:
          n_solved, n_total, pass_at_1 (existing contract)
          resolved_ids, submitted_ids, empty_patch_ids, error_ids,
          completed_ids, incomplete_ids (NEW: full lists, so callers
            can do per-instance attribution and the parse_results bug
            no longer hides the real data)
          report_path: path to the report file we used
        On failure: errors=["no_results_json"] with the same shape but
        empty lists.
        """
        report = _find_report_json(out_dir, run_id=run_id)
        if report is None:
            return {
                "n_solved": 0,
                "n_total": 0,
                "pass_at_1": 0.0,
                "resolved_ids": [],
                "submitted_ids": [],
                "empty_patch_ids": [],
                "error_ids": [],
                "completed_ids": [],
                "incomplete_ids": [],
                "report_path": None,
                "errors": ["no_results_json"],
            }
        data = json.loads(report.read_text())
        resolved = list(data.get("resolved_ids") or [])
        submitted = list(data.get("submitted_ids") or [])
        empty = list(data.get("empty_patch_ids") or [])
        errored = list(data.get("error_ids") or [])
        completed = list(data.get("completed_ids") or [])
        incomplete = list(data.get("incomplete_ids") or [])
        n_solved = len(resolved)
        n_total = len(submitted)
        return {
            "n_solved": n_solved,
            "n_total": n_total,
            "pass_at_1": (n_solved / n_total) if n_total > 0 else 0.0,
            "resolved_ids": resolved,
            "submitted_ids": submitted,
            "empty_patch_ids": empty,
            "error_ids": errored,
            "completed_ids": completed,
            "incomplete_ids": incomplete,
            "report_path": str(report),
        }
