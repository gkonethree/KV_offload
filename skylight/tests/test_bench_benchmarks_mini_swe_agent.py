"""Tests for skylight.bench.benchmarks.mini_swe_agent.

Pure-Python: covers cmd-building (build_agent_cmd, build_eval_cmd) and
result-parsing (parse_results) with no subprocess execution. Targets
the v2 mini-swe-agent CLI shape:

  * Agent: `mini-extra swebench --subset lite --split test --filter ...
    --output ... -c swebench.yaml -c model.model_name=... -c
    model.model_kwargs.api_base=... -c model.model_kwargs.api_key=...`
  * Eval:  `python -m swebench.harness.run_evaluation
    --dataset_name SWE-bench/SWE-bench_Lite --split test
    --predictions_path <out>/preds.json --run_id <out.name>
    --max_workers 4 --report_dir <out>`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from skylight.bench.benchmarks.mini_swe_agent import MiniSweAgent


@pytest.fixture
def tiny_instances(tmp_path: Path) -> Path:
    p = tmp_path / "subset.txt"
    p.write_text(
        "django__django-12915\n"
        "sympy__sympy-20590\n"
        "flask__flask-4992\n"
    )
    return p


# ----------------------------- build_agent_cmd --------------------------------


def test_build_agent_cmd_includes_model_and_base_url(tiny_instances: Path):
    cmd = MiniSweAgent().build_agent_cmd(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        base_url="http://localhost:8000/v1",
        instances=tiny_instances,
        out_dir=Path("/tmp/out"),
    )
    joined = " ".join(cmd)
    # Model name flows in via the model.model_name override (with litellm
    # openai/ prefix), base URL via model.model_kwargs.api_base.
    assert "model.model_name=openai/Qwen/Qwen3-Coder-30B-A3B-Instruct" in joined
    assert "model.model_kwargs.api_base=http://localhost:8000/v1" in joined
    assert "model.model_kwargs.api_key=EMPTY" in joined


def test_build_agent_cmd_includes_swebench_default_config(tiny_instances: Path):
    """`-c swebench.yaml` must be re-included because any -c drops the default."""
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    # The default config is re-injected explicitly before our overrides.
    assert "swebench.yaml" in cmd
    # And it appears as a -c arg, not just stray text.
    swebench_idx = cmd.index("swebench.yaml")
    assert cmd[swebench_idx - 1] == "-c"


def test_build_agent_cmd_filter_regex_contains_all_instance_ids(
    tiny_instances: Path,
):
    """--filter regex restricts the run to exactly our subset's IDs."""
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    assert "--filter" in cmd
    regex = cmd[cmd.index("--filter") + 1]
    assert "django__django\\-12915" in regex or "django__django-12915" in regex
    assert "sympy__sympy\\-20590" in regex or "sympy__sympy-20590" in regex
    assert "flask__flask\\-4992" in regex or "flask__flask-4992" in regex
    # Anchored so partial matches don't sneak in.
    assert regex.startswith("^(")
    assert regex.endswith(")$")


def test_build_agent_cmd_targets_swebench_lite_test_split(tiny_instances: Path):
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    assert cmd[cmd.index("--subset") + 1] == "lite"
    # Default --split is dev; we explicitly request test (the 300-instance set).
    assert cmd[cmd.index("--split") + 1] == "test"


def test_build_agent_cmd_writes_to_out_dir(tiny_instances: Path):
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/data/run42"),
    )
    assert cmd[cmd.index("--output") + 1] == "/data/run42"


def test_build_agent_cmd_invokes_mini_extra_binary(tiny_instances: Path):
    """v2 uses the `mini-extra swebench` console script, not python -m."""
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    # The first arg is the mini-extra binary sibling to the active python;
    # second arg is the subcommand.
    assert cmd[0].endswith("/mini-extra")
    assert cmd[1] == "swebench"


def test_build_agent_cmd_runs_single_worker_by_default(tiny_instances: Path):
    """Sequential within a config — we don't add another layer of parallelism."""
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    assert cmd[cmd.index("--workers") + 1] == "1"


# ----------------------------- fail-fast bounds -------------------------------


def test_build_agent_cmd_sets_per_request_timeout(tiny_instances: Path):
    """request_timeout=60s caps the per-HTTP-call deadline; healthy p99
    is ~31s in the captured trace, so 60s leaves comfortable headroom
    while still failing in under a minute on a hung server."""
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    assert "model.model_kwargs.request_timeout=60" in cmd


def test_build_agent_cmd_registers_skylight_litellm_model(tiny_instances: Path):
    """Our subclass adds Timeout to abort_exceptions and writes the
    per-attempt categorized trace."""
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    assert (
        "model.model_class=skylight.bench.litellm_model.SkylightLitellmModel"
        in cmd
    )


def test_build_agent_cmd_sets_wall_time_limit(tiny_instances: Path):
    """Per-instance wall-clock cap as a defensive ceiling on stuck loops."""
    cmd = MiniSweAgent().build_agent_cmd(
        model="X", base_url="http://h/v1",
        instances=tiny_instances, out_dir=Path("/tmp/o"),
    )
    assert "agent.wall_time_limit_seconds=600" in cmd


# ----------------------------- build_eval_cmd ---------------------------------


def test_build_eval_cmd_points_at_preds_json_in_out_dir():
    cmd = MiniSweAgent().build_eval_cmd(Path("/tmp/run42"))
    assert "--predictions_path" in cmd
    assert cmd[cmd.index("--predictions_path") + 1] == "/tmp/run42/preds.json"


def test_build_eval_cmd_invokes_swebench_harness_module():
    cmd = MiniSweAgent().build_eval_cmd(Path("/tmp/o"))
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert any("swebench.harness.run_evaluation" in part for part in cmd)


def test_build_eval_cmd_uses_swe_bench_lite_test_split():
    cmd = MiniSweAgent().build_eval_cmd(Path("/tmp/o"))
    assert cmd[cmd.index("--dataset_name") + 1] == "SWE-bench/SWE-bench_Lite"
    assert cmd[cmd.index("--split") + 1] == "test"


def test_build_eval_cmd_uses_out_dir_name_as_run_id():
    """--run_id is the out_dir basename so the swebench report file is namespaced."""
    cmd = MiniSweAgent().build_eval_cmd(Path("/tmp/run-2026-05-26"))
    assert "--run_id" in cmd
    assert cmd[cmd.index("--run_id") + 1] == "run-2026-05-26"


def test_build_eval_cmd_writes_report_to_out_dir():
    cmd = MiniSweAgent().build_eval_cmd(Path("/tmp/r"))
    assert cmd[cmd.index("--report_dir") + 1] == "/tmp/r"


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
