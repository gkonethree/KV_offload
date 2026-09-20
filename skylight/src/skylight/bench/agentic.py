"""Orchestrate one (benchmark, config) agentic bench run.

Lifecycle per call:
  1. Spawn the right server via run_serving helpers
  2. Wait for /health
  3. Start metrics scraper + progress reporter (daemon threads)
  4. Run the benchmark's agent subprocess (writes patches.jsonl)
  5. Run the benchmark's eval subprocess (writes results.json), with
     cwd=output_dir so the report lands alongside other artifacts
  6. Stop scraper + reporter
  7. Parse results, compute dropped-instance set, write summary.json
     + per_instance.csv with metadata stamped (config, knobs, git shas,
     pkg versions, hostname, timestamp)
  8. Teardown server

Errors at each step are captured into summary["errors"]; the server is
always torn down in a finally block.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from importlib import metadata as _imeta
from pathlib import Path
from typing import Optional

try:
    import yaml as _yaml  # mini-swe-agent / swebench bring PyYAML transitively
except Exception:
    _yaml = None

from skylight.bench.benchmarks import REGISTRY
from skylight.bench.benchmarks.base import AgenticBenchmark
from skylight.bench.micro_metrics import get_logger
from skylight.bench.progress import MetricsScraper, ProgressReporter
from skylight.bench.run_serving import (
    build_env,
    build_serve_cmd,
    free_port,
    terminate_group,
    wait_for_health,
)


# ------------------------------ run profiles ----------------------------------

RUN_PROFILES: dict[str, dict[str, int]] = {
    "verified-official": {"step_limit": 100, "wall_time_limit_seconds": 1800},
    "verified-rescue": {"step_limit": 150, "wall_time_limit_seconds": 3600},
}

RESCUE_OUTCOMES = frozenset({
    "lost_to_walltime",
    "lost_to_step_limit",
    "lost_to_docker_start",
})

DOCKER_RETRY_EXIT_STATUSES = frozenset({"CalledProcessError"})


# ------------------------------ helpers ---------------------------------------


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


def _git_sha(path: Path) -> str:
    """`git rev-parse HEAD` at path, or '<not-a-git-repo>' / '<error>'."""
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
        return "<not-a-git-repo>"
    except Exception:
        return "<error>"


def _pkg_version(name: str) -> str:
    """importlib.metadata.version(name) or '<not-installed>'."""
    try:
        return _imeta.version(name)
    except Exception:
        return "<not-installed>"


def _collect_metadata(
    config_name: str,
    backend: str,
    model: str,
    knobs: dict,
    instances_path: Path,
) -> dict:
    """Snapshot config + reproducibility info for stamping into summary."""
    # Walk up from this file to find the skylight repo root (where pyproject.toml lives).
    here = Path(__file__).resolve()
    skylight_root = here
    for _ in range(6):
        if (skylight_root / "pyproject.toml").exists():
            break
        skylight_root = skylight_root.parent
    kernels_root = skylight_root.parent / "skylight_kernels"
    return {
        "config_name": config_name,
        "backend": backend,
        "model": model,
        "knobs": knobs,
        "instances_file": str(instances_path),
        "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "git_sha_skylight": _git_sha(skylight_root),
        "git_sha_kernels": _git_sha(kernels_root),
        "pkg_versions": {
            "vllm": _pkg_version("vllm"),
            "torch": _pkg_version("torch"),
            "skylight": _pkg_version("skylight"),
            "skylight_kernels": _pkg_version("skylight-kernels"),
            "mini_swe_agent": _pkg_version("mini-swe-agent"),
            "swebench": _pkg_version("swebench"),
            "prometheus_client": _pkg_version("prometheus-client"),
        },
    }


def _read_instance_ids(instances_path: Path) -> list[str]:
    return [
        line.strip()
        for line in instances_path.read_text().splitlines()
        if line.strip()
    ]


def _preds_instance_ids(preds_path: Path) -> list[str]:
    """Instance IDs that mini-swe-agent wrote into preds.json (any patch)."""
    if not preds_path.exists():
        return []
    try:
        d = json.loads(preds_path.read_text())
    except Exception:
        return []
    if isinstance(d, dict):
        return list(d.keys())
    return []


def _preds_patches(preds_path: Path) -> dict[str, str]:
    """{instance_id: model_patch} for every entry in preds.json."""
    if not preds_path.exists():
        return {}
    try:
        d = json.loads(preds_path.read_text())
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    return {k: (v.get("model_patch") or "") for k, v in d.items()}


def _read_exit_statuses(out_dir: Path) -> dict[str, str]:
    """{instance_id: exit_status} from mini-swe-agent's exit_statuses_*.yaml.

    Schema written by mini-swe-agent v2:
        instances_by_exit_status:
            Submitted:   [iid, ...]
            Timeout:     [iid, ...]
            TimeExceeded:[iid, ...]
            ... (more enum values)
    Returns {} if file missing or yaml unavailable.
    """
    if _yaml is None:
        return {}
    candidates = sorted(out_dir.glob("exit_statuses_*.yaml"))
    if not candidates:
        return {}
    try:
        data = _yaml.safe_load(candidates[-1].read_text()) or {}
    except Exception:
        return {}
    by_status = data.get("instances_by_exit_status") or {}
    out: dict[str, str] = {}
    for status, iids in by_status.items():
        for iid in (iids or []):
            out[iid] = str(status)
    return out


def _classify_outcome(
    iid: str,
    in_preds: bool,
    patch_nonempty: bool,
    resolved: bool,
    empty_swebench: bool,
    error_swebench: bool,
    exit_status: Optional[str],
    eval_inconsistent: bool = False,
) -> str:
    """Single canonical outcome bucket per instance.

    Priority order (first match wins):
      resolved                    -> patch passed swebench eval
      swebench_eval_inconsistent  -> pytest reported pass; swebench report
                                     said FAIL_TO_PASS failure (suspect
                                     harness phase mix-up, see
                                     _detect_eval_inconsistency)
      error_swebench              -> swebench harness errored on this instance
      lost_to_timeout             -> mini-swe-agent ended on LLM Timeout abort
      lost_to_walltime            -> wall_time_limit_seconds (TimeExceeded)
      lost_to_context_overflow    -> ContextWindowExceededError (agent grew
                                     conversation past --max-model-len)
      lost_to_step_limit          -> LimitsExceeded (agent step cap)
      lost_to_docker_start        -> CalledProcessError (SWE-bench docker
                                     container failed to start)
      agent_gave_up_empty         -> Submitted but no patch (truly empty)
      submitted_but_failed        -> Submitted with patch, eval said empty
      submitted_not_resolved      -> agent submitted; eval ran; not resolved
                                     (real model miss OR eval pending mid-run)
      dropped_by_agent            -> never made it into preds.json
      unknown                     -> shouldn't happen; check agent.log
    """
    if resolved:
        return "resolved"
    if eval_inconsistent:
        return "swebench_eval_inconsistent"
    if error_swebench:
        return "error_swebench"
    if exit_status == "Timeout":
        return "lost_to_timeout"
    if exit_status == "TimeExceeded":
        return "lost_to_walltime"
    if exit_status == "ContextWindowExceededError":
        return "lost_to_context_overflow"
    if exit_status == "LimitsExceeded":
        return "lost_to_step_limit"
    if exit_status == "CalledProcessError":
        return "lost_to_docker_start"
    if not in_preds:
        return "dropped_by_agent"
    if exit_status == "Submitted" and not patch_nonempty:
        return "agent_gave_up_empty"
    if patch_nonempty and empty_swebench:
        return "submitted_but_failed"
    if patch_nonempty:
        return "submitted_not_resolved"
    return "unknown"


def _detect_eval_inconsistency(out_dir: Path, model: str, instance_id: str) -> tuple[bool, str]:
    """Catch swebench harness false-negatives.

    Pytest output might say "1 passed" while the report's FAIL_TO_PASS
    list still flags the test as failure (likely a harness phase mix-up).

    Path is fully deterministic - no globbing, no mtime sort, no
    "pick one of several" hazard:

        <out_dir>/logs/run_evaluation/<run_id>/<model_sanitized>/<instance_id>/test_output.txt

    where ``<run_id>`` is ``out_dir.name`` (we pass that as --run_id)
    and ``<model_sanitized>`` is the openai-prefixed model name with
    every '/' replaced by '__' (what mini-swe-agent writes into
    preds.json's ``model_name_or_path`` and what swebench mirrors as
    the per-instance log dir name).

    Returns (is_inconsistent, summary_string). Missing file is not
    inconsistent (we can't tell).
    """
    model_sanitized = ("openai/" + model).replace("/", "__")
    p = (
        out_dir / "logs" / "run_evaluation"
        / out_dir.name / model_sanitized / instance_id / "test_output.txt"
    )
    if not p.exists():
        return False, "no_test_output"
    import re
    text = p.read_text(errors="replace")
    def _n(pat: str) -> int:
        m = re.search(pat, text)
        return int(m.group(1)) if m else 0
    n_passed = _n(r"(\d+)\s+passed")
    n_failed = _n(r"(\d+)\s+failed")
    n_errored = _n(r"(\d+)\s+error")
    inconsistent = n_passed > 0 and n_failed == 0 and n_errored == 0
    return inconsistent, f"pytest:passed={n_passed},failed={n_failed},errored={n_errored}"


def _merge_preds_files(base_path: Path, update_path: Path) -> None:
    """Merge update preds into base and write back to base_path."""
    base: dict = {}
    if base_path.exists():
        try:
            base = json.loads(base_path.read_text())
        except Exception:
            base = {}
    if not isinstance(base, dict):
        base = {}
    if update_path.exists():
        try:
            patch = json.loads(update_path.read_text())
            if isinstance(patch, dict):
                base.update(patch)
        except Exception:
            pass
    base_path.write_text(json.dumps(base, indent=2) + "\n")


def _filter_preds_file(
    preds_path: Path,
    instance_ids: set[str],
    out_path: Path,
) -> Path:
    """Write preds subset for a retry eval pass."""
    if not preds_path.exists():
        out_path.write_text("{}\n")
        return out_path
    try:
        data = json.loads(preds_path.read_text())
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    filtered = {k: v for k, v in data.items() if k in instance_ids}
    out_path.write_text(json.dumps(filtered, indent=2) + "\n")
    return out_path


def _merge_eval_parsed(primary: dict, retry: dict) -> dict:
    """Combine primary + retry eval reports (retry wins on overlap)."""
    def _as_set(key: str) -> set[str]:
        return set(primary.get(key) or []) | set(retry.get(key) or [])

    resolved = _as_set("resolved_ids")
    submitted = _as_set("submitted_ids")
    empty = _as_set("empty_patch_ids")
    errored = _as_set("error_ids") - resolved
    completed = _as_set("completed_ids")
    incomplete = _as_set("incomplete_ids") - completed
    n_solved = len(resolved)
    n_total = len(submitted)
    merged = dict(primary)
    merged.update({
        "n_solved": n_solved,
        "n_total": n_total,
        "pass_at_1": (n_solved / n_total) if n_total > 0 else 0.0,
        "resolved_ids": sorted(resolved),
        "submitted_ids": sorted(submitted),
        "empty_patch_ids": sorted(empty),
        "error_ids": sorted(errored),
        "completed_ids": sorted(completed),
        "incomplete_ids": sorted(incomplete),
        "eval_retry_merged": True,
    })
    return merged


def _docker_retry_instance_ids(
    instance_ids: list[str],
    exit_statuses: dict[str, str],
    preds_ids: set[str],
) -> list[str]:
    """IDs worth a post-agent docker retry pass (docker start failures only)."""
    retry: list[str] = []
    for iid in instance_ids:
        status = exit_statuses.get(iid)
        if status in DOCKER_RETRY_EXIT_STATUSES:
            retry.append(iid)
        elif iid not in preds_ids and status in DOCKER_RETRY_EXIT_STATUSES:
            retry.append(iid)
    return retry


def _failed_ids_from_prior_run(run_dir: Path) -> list[str]:
    """Collect instance IDs to rerun under verified-rescue from per_instance.csv."""
    ids: list[str] = []
    for csv_path in sorted(run_dir.glob("gpu-*/per_instance.csv")):
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("outcome") in RESCUE_OUTCOMES:
                    iid = row.get("instance_id", "").strip()
                    if iid:
                        ids.append(iid)
    return sorted(set(ids))


def _write_instance_subset(ids: list[str], path: Path) -> Path:
    path.write_text("\n".join(ids) + ("\n" if ids else ""))
    return path


def _run_subprocess_logged(
    cmd: list[str],
    log_path: Path,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> int:
    with log_path.open("w") as logf:
        return subprocess.run(
            cmd, env=env, cwd=cwd,
            stdout=logf, stderr=subprocess.STDOUT,
        ).returncode


def _write_per_instance_csv(
    out_dir: Path,
    instance_ids: list[str],
    patches: dict[str, str],
    resolved: set[str],
    empty_swebench: set[str],
    error_swebench: set[str],
    completed: set[str],
    exit_statuses: dict[str, str],
    model: str,
    submitted_swebench: Optional[set[str]] = None,
    trust_pytest_on_inconsistent: bool = False,
) -> tuple[Path, Counter]:
    """One row per chunk instance + an outcome Counter for summary rollup.

    For instances classified as "submitted_not_resolved" by swebench, we
    additionally consult ``_detect_eval_inconsistency`` and re-classify
    as ``swebench_eval_inconsistent`` if pytest output disagrees with
    the report. New ``pytest_summary`` column carries the parsed counts
    for triage.

    Columns: instance_id, in_preds, patch_len, patch_nonempty,
             eval_completed, resolved, empty_patch_swebench,
             error_swebench, dropped_by_agent, exit_status,
             pytest_summary, outcome
    """
    submitted_swebench = submitted_swebench or set()
    csv_path = out_dir / "per_instance.csv"
    cols = [
        "instance_id", "in_preds", "patch_len", "patch_nonempty",
        "eval_completed", "resolved", "empty_patch_swebench",
        "error_swebench", "dropped_by_agent", "exit_status",
        "pytest_summary", "outcome",
    ]
    outcomes: Counter = Counter()
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for iid in instance_ids:
            patch = patches.get(iid, "")
            in_preds = iid in patches
            patch_nonempty = bool(patch.strip())
            exit_status = exit_statuses.get(iid)
            in_resolved = iid in resolved
            in_empty = iid in empty_swebench
            in_error = iid in error_swebench
            # Only check pytest inconsistency for "submitted, not resolved"
            # candidates (instance was eval'd, swebench said fail, patch
            # was non-empty). Cheap; skips dropped/timeout/etc.
            pytest_msg = ""
            eval_inconsistent = False
            if (
                in_preds and patch_nonempty
                and not in_resolved
                and (iid in submitted_swebench or in_empty)
            ):
                eval_inconsistent, pytest_msg = _detect_eval_inconsistency(out_dir, model, iid)
            outcome = _classify_outcome(
                iid=iid,
                in_preds=in_preds,
                patch_nonempty=patch_nonempty,
                resolved=in_resolved,
                empty_swebench=in_empty,
                error_swebench=in_error,
                exit_status=exit_status,
                eval_inconsistent=eval_inconsistent,
            )
            if trust_pytest_on_inconsistent and eval_inconsistent:
                outcome = "resolved"
            outcomes[outcome] += 1
            w.writerow({
                "instance_id": iid,
                "in_preds": int(in_preds),
                "patch_len": len(patch),
                "patch_nonempty": int(patch_nonempty),
                "eval_completed": int(iid in completed),
                "resolved": int(in_resolved),
                "empty_patch_swebench": int(in_empty),
                "error_swebench": int(in_error),
                "dropped_by_agent": int(not in_preds),
                "exit_status": exit_status or "",
                "pytest_summary": pytest_msg,
                "outcome": outcome,
            })
    return csv_path, outcomes


# ------------------------------ main routine ----------------------------------


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
    step_limit: int = 100,
    request_timeout_s: int = 300,
    wall_time_limit_seconds: int = 1800,
    max_tokens: int = 8192,
    enable_prefix_caching: bool = False,
    max_num_seqs: Optional[int] = None,
    retry_docker_failures: bool = True,
    retry_eval_errors: bool = True,
    trust_pytest_on_inconsistent: bool = False,
    profile: Optional[str] = None,
) -> dict:
    """Run one (benchmark, config) end-to-end. Returns summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logger(output_dir / "orchestrator.log")
    log.info("[%s] start; backend=%s model=%s", config_name, backend, model)

    knobs = {
        "topk": topk, "sink": sink, "local": local, "channel_num": channel_num,
        "max_model_len": max_model_len, "enforce_eager": enforce_eager,
        "enable_prefix_caching": enable_prefix_caching,
        "max_num_seqs": max_num_seqs,
        "step_limit": step_limit, "request_timeout_s": request_timeout_s,
        "wall_time_limit_seconds": wall_time_limit_seconds,
        "max_tokens": max_tokens,
        "profile": profile,
        "retry_docker_failures": retry_docker_failures,
        "retry_eval_errors": retry_eval_errors,
    }
    instance_ids = _read_instance_ids(instances)
    instances_total = len(instance_ids)
    if instances_total == 0:
        log.error("[%s] instances file %s is empty", config_name, instances)
        summary = {
            "config_name": config_name,
            "backend": backend,
            "model": model,
            "instances_file": str(instances),
            "instances_total": 0, "instance_ids": [],
            "n_solved": 0, "n_total": 0, "pass_at_1": 0.0,
            "elapsed_s": 0.0, "errors": ["empty_instances_file"],
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    meta = _collect_metadata(config_name, backend, model, knobs, instances)

    port = free_port()
    serve_cmd = build_serve_cmd(
        backend, model, port, max_model_len, enforce_eager,
        gpu_memory_utilization=float(os.environ.get("SKYLIGHT_GPU_MEM_UTIL", "0.5")),
        tool_call_parser="qwen3_coder",
        enable_prefix_caching=enable_prefix_caching,
        max_num_seqs=max_num_seqs,
    )
    env = build_env(
        backend=backend, topk=topk, sink=sink, local=local, channel_num=channel_num,
    )
    if backend == "sparse":
        metrics_dir = output_dir / "micro_metrics"
        env["SKYLIGHT_METRICS_LOG_DIR"] = str(metrics_dir)
        env.setdefault("SKYLIGHT_METRICS_SAMPLING", os.environ.get("SKYLIGHT_METRICS_SAMPLING", "0.01"))

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
        **meta,
        "instances_total": instances_total,
        "instance_ids": instance_ids,
    }

    try:
        try:
            wait_for_health(port, server_timeout_s, proc)
            t_ready = time.time()
            log.info("[%s] server /health ready on port %d", config_name, port)
        except (TimeoutError, RuntimeError) as exc:
            log.error("[%s] server unhealthy: %s", config_name, exc)
            summary.update({
                "n_solved": 0, "n_total": 0, "pass_at_1": 0.0,
                "elapsed_s": time.time() - t_start,
                "errors": ["server_unhealthy"],
                "dropped_by_agent": instance_ids,  # nothing got run
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
            step_limit=step_limit,
            request_timeout_s=request_timeout_s,
            wall_time_limit_seconds=wall_time_limit_seconds,
            max_tokens=max_tokens,
        )
        agent_env = os.environ.copy()
        agent_env.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
        agent_env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
        agent_env.setdefault(
            "LITELLM_TRACE_JSONL", str(output_dir / "litellm_trace.jsonl"),
        )
        agent_env.setdefault(
            "LITELLM_TRACE_JSONL", str(output_dir / "litellm_trace.jsonl"),
        )
        agent_env["SKYLIGHT_DOCKER_START_LOG"] = str(output_dir / "docker_start.log")
        log.info("[%s] running agent: %s", config_name, " ".join(agent_cmd))
        agent_rc = _run_subprocess_logged(
            agent_cmd, output_dir / "agent.log", env=agent_env,
        )
        log.info("[%s] agent exit rc=%d", config_name, agent_rc)
        if agent_rc != 0:
            summary.setdefault("errors", []).append(f"agent_rc={agent_rc}")

        preds_path = output_dir / "preds.json"
        exit_statuses = _read_exit_statuses(output_dir)
        preds_ids = set(_preds_instance_ids(preds_path))

        if retry_docker_failures:
            retry_ids = _docker_retry_instance_ids(instance_ids, exit_statuses, preds_ids)
            if retry_ids:
                log.warning(
                    "[%s] docker retry pass for %d instances: %s",
                    config_name, len(retry_ids), retry_ids[:5],
                )
                retry_subset = _write_instance_subset(
                    retry_ids, output_dir / "retry_docker_instances.txt",
                )
                preds_backup = output_dir / "preds.pre_docker_retry.json"
                if preds_path.exists():
                    preds_backup.write_text(preds_path.read_text())
                retry_cmd = benchmark.build_agent_cmd(
                    model=model, base_url=base_url,
                    instances=retry_subset, out_dir=output_dir,
                    step_limit=step_limit,
                    request_timeout_s=request_timeout_s,
                    wall_time_limit_seconds=wall_time_limit_seconds,
                    max_tokens=max_tokens,
                )
                retry_rc = _run_subprocess_logged(
                    retry_cmd, output_dir / "agent.docker_retry.log", env=agent_env,
                )
                log.info("[%s] docker retry agent rc=%d", config_name, retry_rc)
                if retry_rc != 0:
                    summary.setdefault("errors", []).append(f"agent_docker_retry_rc={retry_rc}")
                if preds_backup.exists():
                    _merge_preds_files(preds_backup, preds_path)
                exit_statuses = _read_exit_statuses(output_dir)
                preds_ids = set(_preds_instance_ids(preds_path))
                summary["docker_retry_ids"] = retry_ids

        # FIX #3: track instances that mini-swe-agent silently dropped.
        dropped = [iid for iid in instance_ids if iid not in preds_ids]
        if dropped:
            log.warning(
                "[%s] %d/%d instances dropped by mini-extra (no preds entry): %s",
                config_name, len(dropped), instances_total, dropped,
            )
        summary["dropped_by_agent"] = dropped

        # FIX #1: run eval with cwd=output_dir so its report file lands here
        # (swebench writes the report to CWD ignoring --report_dir).
        eval_cmd = benchmark.build_eval_cmd(output_dir)
        log.info("[%s] running eval (cwd=%s): %s",
                 config_name, output_dir, " ".join(eval_cmd))
        eval_rc = _run_subprocess_logged(eval_cmd, output_dir / "eval.log", cwd=str(output_dir))
        log.info("[%s] eval exit rc=%d", config_name, eval_rc)
        if eval_rc != 0:
            summary.setdefault("errors", []).append(f"eval_rc={eval_rc}")

        parsed = benchmark.parse_results(output_dir)
        error_ids = set(parsed.get("error_ids") or [])
        if retry_eval_errors and error_ids:
            retry_run_id = f"{output_dir.name}-retry"
            retry_preds = _filter_preds_file(
                preds_path, error_ids, output_dir / "preds_eval_retry.json",
            )
            retry_eval_cmd = benchmark.build_eval_cmd(
                output_dir,
                predictions_path=retry_preds,
                run_id=retry_run_id,
            )
            log.warning(
                "[%s] eval retry for %d error_ids", config_name, len(error_ids),
            )
            retry_eval_rc = _run_subprocess_logged(
                retry_eval_cmd, output_dir / "eval.retry.log", cwd=str(output_dir),
            )
            log.info("[%s] eval retry rc=%d", config_name, retry_eval_rc)
            if retry_eval_rc != 0:
                summary.setdefault("errors", []).append(f"eval_retry_rc={retry_eval_rc}")
            retry_parsed = benchmark.parse_results(output_dir, run_id=retry_run_id)
            if retry_parsed.get("report_path"):
                parsed = _merge_eval_parsed(parsed, retry_parsed)
                summary["eval_retry_ids"] = sorted(error_ids)

        # FIX #2: parse_results now returns full id lists, not just counts.
        summary.update(parsed)

        # FIX #5 + #6 + #7: per-instance results CSV. Joins preds +
        # swebench report + mini-swe-agent's exit_statuses yaml. Also
        # consults pytest output to flag swebench harness false-negatives.
        patches = _preds_patches(preds_path)
        try:
            csv_path, outcomes = _write_per_instance_csv(
                out_dir=output_dir,
                instance_ids=instance_ids,
                patches=patches,
                resolved=set(parsed.get("resolved_ids") or []),
                empty_swebench=set(parsed.get("empty_patch_ids") or []),
                error_swebench=set(parsed.get("error_ids") or []),
                completed=set(parsed.get("completed_ids") or []),
                exit_statuses=exit_statuses,
                model=model,
                submitted_swebench=set(parsed.get("submitted_ids") or []),
                trust_pytest_on_inconsistent=trust_pytest_on_inconsistent,
            )
            summary["per_instance_csv"] = str(csv_path)
            summary["outcomes"] = dict(outcomes)
        except Exception as exc:
            log.error("[%s] per_instance.csv write failed: %s", config_name, exc)
            summary.setdefault("errors", []).append(f"per_instance_csv_failed:{exc}")

        summary["exit_status_counts"] = dict(Counter(exit_statuses.values()))

        # FIX #8: distinguish three denominators honestly:
        #   instances_in_chunk     = lines in the chunk's input file
        #   instances_attempted    = entries that landed in preds.json
        #                            (i.e. mini-extra found them in Lite + ran the agent)
        #   instances_resolved     = passed swebench eval
        # pass@1 over each denominator. The "over_attempted" number is
        # the honest agent-quality metric; "over_chunk" is what you
        # quote if you're committed to the chunk-as-written denominator.
        n_attempted = sum(1 for iid in instance_ids if iid in preds_ids)
        n_resolved = summary.get("n_solved", 0)
        summary["instances_in_chunk"] = instances_total
        summary["instances_attempted"] = n_attempted
        summary["instances_resolved"] = n_resolved
        summary["n_attempted"] = n_attempted              # back-compat
        summary["n_dropped"] = len(dropped)
        summary["pass_at_1_over_chunk"] = (n_resolved / instances_total) if instances_total else 0.0
        summary["pass_at_1_over_attempted"] = (n_resolved / n_attempted) if n_attempted else 0.0
        # back-compat alias (older code used pass_at_1_over_subset to
        # mean over chunk)
        summary["pass_at_1_over_subset"] = summary["pass_at_1_over_chunk"]

        summary["elapsed_s"] = time.time() - t_start
        summary["server_boot_s"] = t_ready - t_start
        summary["agent_wall_s"] = time.time() - t_ready
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log.info(
            "[%s] done in %.0fs: solved=%d submitted=%d attempted=%d dropped=%d "
            "pass@1(submitted)=%.3f pass@1(subset)=%.3f boot=%.0fs agent=%.0fs",
            config_name, summary["elapsed_s"],
            summary.get("n_solved", 0), summary.get("n_total", 0),
            summary.get("n_attempted", 0), summary.get("n_dropped", 0),
            summary.get("pass_at_1", 0.0),
            summary.get("pass_at_1_over_subset", 0.0),
            summary["server_boot_s"], summary["agent_wall_s"],
        )
        return summary

    finally:
        if reporter is not None:
            reporter.stop()
            reporter.join(timeout=5)
        if scraper is not None:
            scraper.stop()
            scraper.join(timeout=5)
        try:
            get_logger().flush()
        except Exception:
            pass
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
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--server-timeout", type=float, default=180.0)
    parser.add_argument("--topk", type=float, default=None)
    parser.add_argument("--sink", type=int, default=None)
    parser.add_argument("--local", type=int, default=None)
    parser.add_argument("--channel-num", type=int, default=None)
    parser.add_argument("--config-name", default=None,
                        help="logical config name (default: --backend)")
    parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES),
        default=None,
        help="run profile: verified-official (100 steps, 1800s wall) or "
             "verified-rescue (150 steps, 3600s wall)",
    )
    parser.add_argument(
        "--retry-from-run",
        type=Path,
        default=None,
        help="prior multi-GPU run directory; rerun failed instance IDs "
             "(lost_to_walltime, lost_to_step_limit, lost_to_docker_start)",
    )
    parser.add_argument("--step-limit", type=int, default=None,
                        help="agent turn cap per instance (profile default if unset)")
    parser.add_argument("--request-timeout-s", type=int, default=900,
                        help="LiteLLM HTTP read timeout per call (no-bytes-flowing detector)")
    parser.add_argument("--wall-time-limit-seconds", type=int, default=None,
                        help="hard wall-clock cap per instance; raises TimeExceeded and submits empty")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="per-call completion token cap; bounds runaway generations")
    parser.add_argument("--no-retry-docker", action="store_true",
                        help="disable post-agent docker failure retry pass")
    parser.add_argument("--no-retry-eval", action="store_true",
                        help="disable eval retry for error_ids")
    parser.add_argument(
        "--trust-pytest-on-inconsistent",
        action="store_true",
        help="reclassify swebench_eval_inconsistent as resolved when pytest passed",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="disable CUDA graph capture (default ON; faster startup)")
    parser.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager",
                        help="enable CUDA graph capture (variant B)")
    parser.add_argument("--enable-prefix-caching", action="store_true", default=False,
                        help="enable vLLM prefix caching (variant A)")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="cap scheduler max sequences; needed for CUDA graphs on the hybrid Mamba model to fit available Mamba cache blocks (else capture crashes)")
    args = parser.parse_args(argv)

    step_limit = args.step_limit
    wall_time_limit_seconds = args.wall_time_limit_seconds
    if args.profile:
        prof = RUN_PROFILES[args.profile]
        if step_limit is None:
            step_limit = prof["step_limit"]
        if wall_time_limit_seconds is None:
            wall_time_limit_seconds = prof["wall_time_limit_seconds"]
    if step_limit is None:
        step_limit = 100
    if wall_time_limit_seconds is None:
        wall_time_limit_seconds = 1800

    instances_path = Path(args.instances)
    if args.retry_from_run:
        failed = _failed_ids_from_prior_run(args.retry_from_run)
        if not failed:
            print(
                f"No rescue-eligible instances under {args.retry_from_run}",
                file=sys.stderr,
            )
            return 1
        if args.output_dir:
            retry_inst = Path(args.output_dir) / "retry_from_run_instances.txt"
        else:
            retry_inst = Path(f"/tmp/skylight-retry-{time.strftime('%Y%m%d_%H%M%S')}.txt")
        instances_path = _write_instance_subset(failed, retry_inst)
        if args.profile is None:
            args.profile = "verified-rescue"
            prof = RUN_PROFILES[args.profile]
            step_limit = args.step_limit if args.step_limit is not None else prof["step_limit"]
            wall_time_limit_seconds = (
                args.wall_time_limit_seconds
                if args.wall_time_limit_seconds is not None
                else prof["wall_time_limit_seconds"]
            )

    benchmark = REGISTRY[args.benchmark]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"bench-results/agentic/{args.backend}_{ts}")

    summary = run_one_config(
        benchmark=benchmark,
        config_name=args.config_name or args.backend,
        backend=args.backend,
        model=args.model,
        instances=instances_path,
        output_dir=output_dir,
        max_model_len=args.max_model_len,
        server_timeout_s=args.server_timeout,
        topk=args.topk, sink=args.sink, local=args.local,
        channel_num=args.channel_num,
        step_limit=step_limit,
        request_timeout_s=args.request_timeout_s,
        wall_time_limit_seconds=wall_time_limit_seconds,
        max_tokens=args.max_tokens,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_seqs=args.max_num_seqs,
        retry_docker_failures=not args.no_retry_docker,
        retry_eval_errors=not args.no_retry_eval,
        trust_pytest_on_inconsistent=args.trust_pytest_on_inconsistent,
        profile=args.profile,
    )
    return 0 if not summary.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
