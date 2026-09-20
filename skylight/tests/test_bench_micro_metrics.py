"""Tests for skylight.bench.micro_metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skylight.bench.micro_metrics import MicroMetricLogger


def test_micro_metrics_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SKYLIGHT_METRICS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SKYLIGHT_METRICS_SAMPLING", "1.0")
    MicroMetricLogger._instance = None
    MicroMetricLogger._lock = __import__("threading").Lock()

    logger = MicroMetricLogger.from_env()
    assert logger.enabled
    logger.log(
        "skylight_sparsity_fraction",
        0.21,
        metadata={"layer_idx": 3, "layer_name": "layers.3.attn"},
        location="test",
    )
    logger.flush()

    lines = (tmp_path / "micro_metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["metric"] == "skylight_sparsity_fraction"
    assert row["value"] == 0.21
    assert row["metadata"]["layer_idx"] == 3
