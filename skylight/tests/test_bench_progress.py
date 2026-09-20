"""Tests for skylight.bench.progress.

Covers the pure-Python pieces: Prometheus text parsing, progress-line
formatting, and ProgressReporter's tick output (driven via a
controllable clock, not real time).
"""
from __future__ import annotations

import json
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
skylight_effective_sparsity_fraction 0.0925
skylight_observed_sparsity_fraction 0.1111
some_unrelated_metric 99.0
"""


def test_parse_extracts_tracked_metrics():
    out = _parse_prometheus_text(PROM_PAYLOAD)
    assert out["vllm:generation_tokens_total"] == 12345.0
    assert out["vllm:request_success_total"] == 7.0
    assert out["skylight_effective_sparsity_fraction"] == 0.0925


def test_parse_aliases_legacy_sparsity_metric():
    payload = "skylight_observed_sparsity_fraction 0.42\n"
    out = _parse_prometheus_text(payload)
    assert out["skylight_effective_sparsity_fraction"] == 0.42


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
        "skylight_effective_sparsity_fraction": 0.0925,
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
    assert "ETA --:--" in line


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
