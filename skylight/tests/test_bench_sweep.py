"""Tests for skylight.bench.sweep — orchestrator helpers.

Exercises the pure-Python pieces (cmd building, JSON parsing, table
formatting). The actual sub-process invocations are covered by the
underlying ``skylight.bench.run_serving`` tests and by manual runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from skylight.bench.sweep import (
    CONFIGS,
    build_run_serving_cmd,
    extract_summary,
    format_table,
)


# -------------------------------- build_run_serving_cmd ----------------------


def test_dense_config_omits_sparse_flags():
    """dense config translates to a `--backend dense` argv with no sparse knobs."""
    cmd = build_run_serving_cmd(
        "Qwen/Qwen3-0.6B", "dense", input_len=1024, output_len=64,
        num_prompts=50, max_model_len=2048, output_json="out.json",
    )
    assert "--backend" in cmd
    assert cmd[cmd.index("--backend") + 1] == "dense"
    assert "--topk" not in cmd
    assert "--sink" not in cmd
    assert "--local" not in cmd
    assert "--channel-num" not in cmd


def test_sparse_token_config_sets_topk_sink_local_and_full_head_dim():
    """sparse-token uses channel_num=-1 (full head_dim → token-sparse only)."""
    cmd = build_run_serving_cmd(
        "Qwen/Qwen3-0.6B", "sparse-token", input_len=4096, output_len=64,
        num_prompts=50, max_model_len=8192, output_json="out.json",
    )
    assert cmd[cmd.index("--backend") + 1] == "sparse"
    assert cmd[cmd.index("--topk") + 1] == "0.1"
    assert cmd[cmd.index("--sink") + 1] == "64"
    assert cmd[cmd.index("--local") + 1] == "64"
    assert cmd[cmd.index("--channel-num") + 1] == "-1"


def test_sparse_double_config_sets_channel_num_8():
    """sparse-double uses a small channel_num for the score kernel."""
    cmd = build_run_serving_cmd(
        "Qwen/Qwen3-0.6B", "sparse-double", input_len=4096, output_len=64,
        num_prompts=50, max_model_len=8192, output_json="out.json",
    )
    assert cmd[cmd.index("--channel-num") + 1] == "8"


def test_cmd_passes_input_output_and_max_model_lens():
    cmd = build_run_serving_cmd(
        "X", "dense", input_len=16384, output_len=128,
        num_prompts=20, max_model_len=20000, output_json="r.json",
    )
    assert cmd[cmd.index("--random-input-len") + 1] == "16384"
    assert cmd[cmd.index("--random-output-len") + 1] == "128"
    assert cmd[cmd.index("--max-model-len") + 1] == "20000"
    assert cmd[cmd.index("--num-prompts") + 1] == "20"
    assert cmd[cmd.index("--output") + 1] == "r.json"


def test_cmd_invokes_run_serving_module():
    cmd = build_run_serving_cmd(
        "X", "dense", input_len=1024, output_len=64,
        num_prompts=10, max_model_len=2048, output_json="o.json",
    )
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "skylight.bench.run_serving" in cmd


# -------------------------------- overrides ----------------------------------


def test_topk_override_wins_over_named_config_default():
    """--topk 0.02 on sparse-token replaces its baked-in 0.10."""
    cmd = build_run_serving_cmd(
        "X", "sparse-token", input_len=4096, output_len=64,
        num_prompts=10, max_model_len=8192, output_json="o.json",
        topk_override=0.02,
    )
    assert cmd[cmd.index("--topk") + 1] == "0.02"
    # The other knobs from the named config are still applied (no override here).
    assert cmd[cmd.index("--sink") + 1] == "64"
    assert cmd[cmd.index("--channel-num") + 1] == "-1"


def test_all_axis_overrides_win():
    cmd = build_run_serving_cmd(
        "X", "sparse-double", input_len=4096, output_len=64,
        num_prompts=10, max_model_len=8192, output_json="o.json",
        topk_override=0.05, sink_override=128, local_override=128,
        channel_num_override=16,
    )
    assert cmd[cmd.index("--topk") + 1] == "0.05"
    assert cmd[cmd.index("--sink") + 1] == "128"
    assert cmd[cmd.index("--local") + 1] == "128"
    assert cmd[cmd.index("--channel-num") + 1] == "16"


def test_overrides_dont_change_dense_cmd():
    """Dense configs ignore sparse-axis overrides (no --topk etc. emitted)."""
    cmd = build_run_serving_cmd(
        "X", "dense", input_len=1024, output_len=64,
        num_prompts=10, max_model_len=2048, output_json="o.json",
        topk_override=0.02, sink_override=128, channel_num_override=16,
    )
    assert "--topk" not in cmd
    assert "--sink" not in cmd
    assert "--channel-num" not in cmd


def test_override_none_falls_back_to_named_config_default():
    """topk_override=None preserves the config's baked-in topk."""
    cmd = build_run_serving_cmd(
        "X", "sparse-token", input_len=1024, output_len=64,
        num_prompts=10, max_model_len=2048, output_json="o.json",
        topk_override=None,
    )
    # CONFIGS["sparse-token"]["topk"] == 0.10
    assert cmd[cmd.index("--topk") + 1] == "0.1"


# -------------------------------- extract_summary ----------------------------


def test_extract_summary_reads_expected_fields(tmp_path: Path):
    """Reads all headline metrics from a benchmark_serving-shaped JSON."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "request_throughput": 15.23,
        "output_throughput": 1950.0,
        "median_ttft_ms": 543.0,
        "p99_ttft_ms": 699.0,
        "median_tpot_ms": 20.77,
        "p99_tpot_ms": 21.12,
    }))
    s = extract_summary(str(p), "sparse-token", 4096, 73.5)
    assert s == {
        "config": "sparse-token",
        "input_len": 4096,
        "req_per_sec": 15.23,
        "out_tok_per_sec": 1950.0,
        "ttft_p50_ms": 543.0,
        "ttft_p99_ms": 699.0,
        "tpot_p50_ms": 20.77,
        "tpot_p99_ms": 21.12,
        "elapsed_s": 73.5,
    }


def test_extract_summary_missing_file_returns_none(tmp_path: Path):
    s = extract_summary(str(tmp_path / "does-not-exist.json"), "dense", 1024, 0.0)
    assert s is None


def test_extract_summary_missing_key_returns_none_and_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
):
    """If the result JSON is missing an expected key, log and return None."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"request_throughput": 1.0}))  # missing the rest
    s = extract_summary(str(p), "dense", 1024, 0.0)
    assert s is None
    err = capsys.readouterr().err
    assert "missing key" in err
    assert str(p) in err


# -------------------------------- format_table -------------------------------


def test_format_table_includes_headers_and_data_rows():
    results = [
        {
            "config": "dense", "input_len": 1024,
            "req_per_sec": 32.51, "out_tok_per_sec": 4160.70,
            "ttft_p50_ms": 264.52, "ttft_p99_ms": 314.34,
            "tpot_p50_ms": 9.79, "tpot_p99_ms": 9.99,
            "elapsed_s": 50.0,
        },
        {
            "config": "sparse-token", "input_len": 1024,
            "req_per_sec": 15.23, "out_tok_per_sec": 1950.08,
            "ttft_p50_ms": 543.11, "ttft_p99_ms": 699.30,
            "tpot_p50_ms": 20.77, "tpot_p99_ms": 21.12,
            "elapsed_s": 80.0,
        },
    ]
    table = format_table(results)
    # Headers
    for h in ["config", "L", "req/s", "out_tok/s", "ttft_p50", "tpot_p50"]:
        assert h in table
    # Both rows show up
    assert "dense" in table
    assert "sparse-token" in table
    assert "32.51" in table
    assert "15.23" in table


def test_format_table_skips_failed_runs():
    """None entries (failed runs) don't get a row."""
    results = [
        None,
        {
            "config": "dense", "input_len": 1024,
            "req_per_sec": 1.0, "out_tok_per_sec": 1.0,
            "ttft_p50_ms": 1.0, "ttft_p99_ms": 1.0,
            "tpot_p50_ms": 1.0, "tpot_p99_ms": 1.0,
            "elapsed_s": 0.0,
        },
        None,
    ]
    table = format_table(results)
    # Should have header + separator + 1 data row, but content for the dense row only.
    assert table.count("dense") == 1
    # No literal "None" leaked in.
    assert "None" not in table


# -------------------------------- CONFIGS ------------------------------------


def test_configs_dense_token_double_present():
    """The three named configs the team uses for A/B/C exist."""
    assert "dense" in CONFIGS
    assert "sparse-token" in CONFIGS
    assert "sparse-double" in CONFIGS


def test_configs_sparse_entries_carry_required_keys():
    """Every sparse config defines topk, sink, local, channel_num."""
    for name, cfg in CONFIGS.items():
        if cfg["backend"] != "sparse":
            continue
        for required in ("topk", "sink", "local", "channel_num"):
            assert required in cfg, f"{name!r} missing {required!r}"


def test_configs_channel_num_minus_one_for_token_only():
    """sparse-token explicitly disables channel sparsity (channel_num=-1)."""
    assert CONFIGS["sparse-token"]["channel_num"] == -1


def test_configs_channel_num_positive_for_double():
    """sparse-double uses a positive channel_num (channel sparsity ON)."""
    assert CONFIGS["sparse-double"]["channel_num"] > 0
