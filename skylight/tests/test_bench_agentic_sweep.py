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
    # Zero counts visible without crashing
    assert "0" in table
