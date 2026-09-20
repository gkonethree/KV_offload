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

from skylight.bench.agentic import _classify_outcome, run_one_config


def _subprocess_ok():
    return MagicMock(returncode=0, stdout="", stderr="")
class FakeBenchmark:
    name = "fake-bench"

    def __init__(self, n_solved: int = 2, n_total: int = 3) -> None:
        self.n_solved = n_solved
        self.n_total = n_total
        self.calls: list[tuple[str, dict]] = []

    def default_instances_path(self) -> Path:
        return Path("never-used.txt")

    def build_agent_cmd(self, model, base_url, instances, out_dir, **kwargs):
        self.calls.append(("agent", {
            "model": model, "base_url": base_url,
            "instances": str(instances), "out_dir": str(out_dir),
        }))
        return ["/bin/true"]

    def build_eval_cmd(self, out_dir, **kwargs):
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
        mock_run.return_value = _subprocess_ok()
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

    assert summary["config_name"] == "dense"
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
        mock_run.return_value = _subprocess_ok()

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
               return_value=_subprocess_ok()), \
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


# --------------------------- server launch config -----------------------------


def test_agentic_orchestrator_enables_tool_call_parsing(
    tmp_path: Path, small_instances: Path,
):
    """run_one_config must pass tool_call_parser to build_serve_cmd —
    without --enable-auto-tool-choice + --tool-call-parser, vLLM
    rejects every mini-swe-agent request (which always carries
    ``tools=[BASH_TOOL]``) with BadRequestError."""
    bench = FakeBenchmark()
    with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
         patch("skylight.bench.agentic.subprocess.run",
               return_value=_subprocess_ok()), \
         patch("skylight.bench.agentic.wait_for_health"), \
         patch("skylight.bench.agentic.terminate_group"), \
         patch("skylight.bench.agentic.free_port", return_value=12345), \
         patch("skylight.bench.agentic.MetricsScraper"), \
         patch("skylight.bench.agentic.ProgressReporter"), \
         patch("skylight.bench.agentic.build_serve_cmd",
               return_value=["/bin/true"]) as mock_build_serve:
        mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))

        run_one_config(
            benchmark=bench, config_name="dense", backend="dense",
            model="Qwen/Qwen3.5-27B", instances=small_instances,
            output_dir=tmp_path / "run", max_model_len=33500,
        )

    mock_build_serve.assert_called_once()
    kwargs = mock_build_serve.call_args.kwargs
    assert kwargs.get("tool_call_parser") == "qwen3_coder"


# --------------------------- outcome classification -----------------------------


@pytest.mark.parametrize("exit_status,expected", [
    ("LimitsExceeded", "lost_to_step_limit"),
    ("CalledProcessError", "lost_to_docker_start"),
    ("ContextWindowExceededError", "lost_to_context_overflow"),
    ("TimeExceeded", "lost_to_walltime"),
    ("Timeout", "lost_to_timeout"),
])
def test_classify_outcome_maps_exit_statuses(exit_status, expected):
    got = _classify_outcome(
        iid="x",
        in_preds=False,
        patch_nonempty=False,
        resolved=False,
        empty_swebench=False,
        error_swebench=False,
        exit_status=exit_status,
    )
    assert got == expected


def test_classify_outcome_limits_exceeded_with_empty_patch():
    assert _classify_outcome(
        iid="x", in_preds=True, patch_nonempty=False,
        resolved=False, empty_swebench=False, error_swebench=False,
        exit_status="LimitsExceeded",
    ) == "lost_to_step_limit"


# --------------------------- agent subprocess env -----------------------------


def test_agent_subprocess_gets_fail_fast_retry_and_trace_env(
    tmp_path: Path, small_instances: Path,
):
    """run_one_config must inject MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT
    and LITELLM_TRACE_JSONL into the agent subprocess env. These pair
    with the LitellmModel subclass + the -c overrides to deliver the
    fail-fast retry budget and per-attempt categorized trace."""
    bench = FakeBenchmark()
    with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
         patch("skylight.bench.agentic.subprocess.run",
               return_value=_subprocess_ok()) as mock_run, \
         patch("skylight.bench.agentic.wait_for_health"), \
         patch("skylight.bench.agentic.terminate_group"), \
         patch("skylight.bench.agentic.free_port", return_value=12345), \
         patch("skylight.bench.agentic.MetricsScraper"), \
         patch("skylight.bench.agentic.ProgressReporter"):
        mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))

        out_dir = tmp_path / "run"
        run_one_config(
            benchmark=bench, config_name="dense", backend="dense",
            model="X", instances=small_instances,
            output_dir=out_dir, max_model_len=33500,
        )

    # Agent subprocess is not the first subprocess.run call (_git_sha runs first).
    agent_calls = [
        c for c in mock_run.call_args_list
        if (c.kwargs.get("env") or {}).get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT") is not None
    ]
    assert agent_calls, "agent subprocess must receive an explicit env"
    env = agent_calls[0].kwargs.get("env")
    assert env is not None, "agent subprocess must receive an explicit env"
    assert env.get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT") == "3"
    assert env.get("LITELLM_TRACE_JSONL") == str(out_dir / "litellm_trace.jsonl")
    # Custom/local models aren't in litellm's price registry, so cost
    # calc would raise RuntimeError after the first successful call
    # and kill the agent. Disable it by default — matches the
    # handoff's hand-run defaults.
    assert env.get("MSWEA_COST_TRACKING") == "ignore_errors"


def test_agent_subprocess_env_respects_caller_overrides(
    tmp_path: Path, small_instances: Path, monkeypatch,
):
    """Caller-set env vars take precedence over the defaults — useful
    when debugging (e.g. point LITELLM_TRACE_JSONL at /tmp)."""
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
    monkeypatch.setenv("LITELLM_TRACE_JSONL", "/tmp/custom_trace.jsonl")
    bench = FakeBenchmark()
    with patch("skylight.bench.agentic.subprocess.Popen") as mock_popen, \
         patch("skylight.bench.agentic.subprocess.run",
               return_value=_subprocess_ok()) as mock_run, \
         patch("skylight.bench.agentic.wait_for_health"), \
         patch("skylight.bench.agentic.terminate_group"), \
         patch("skylight.bench.agentic.free_port", return_value=12345), \
         patch("skylight.bench.agentic.MetricsScraper"), \
         patch("skylight.bench.agentic.ProgressReporter"):
        mock_popen.return_value = MagicMock(pid=1, poll=MagicMock(return_value=None))

        run_one_config(
            benchmark=bench, config_name="dense", backend="dense",
            model="X", instances=small_instances,
            output_dir=tmp_path / "run", max_model_len=33500,
        )

    agent_calls = [
        c for c in mock_run.call_args_list
        if (c.kwargs.get("env") or {}).get("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT") is not None
    ]
    env = agent_calls[0].kwargs.get("env")
    assert env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "1"
    assert env["LITELLM_TRACE_JSONL"] == "/tmp/custom_trace.jsonl"
