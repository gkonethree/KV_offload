"""Tests for skylight.config.SkylightSparseConfig."""
from __future__ import annotations

import os
from dataclasses import FrozenInstanceError

import pytest

from skylight.config import SkylightSparseConfig


SPARSE_ENV_VARS = (
    "SKYLIGHT_SPARSE_TOPK",
    "SKYLIGHT_SPARSE_SINK",
    "SKYLIGHT_SPARSE_LOCAL",
    "SKYLIGHT_SPARSE_CHANNEL_NUM",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove SKYLIGHT_SPARSE_* env vars before each test; monkeypatch reverts after."""
    for v in SPARSE_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    yield


# -------------------------------- happy paths --------------------------------


def test_topk_only(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.10")
    cfg = SkylightSparseConfig.from_env()
    assert cfg.topk == 0.10
    assert cfg.sink_size == 0
    assert cfg.local_size == 0
    assert cfg.channel_num == -1  # default


def test_sink_local_only(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "64")
    monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "128")
    cfg = SkylightSparseConfig.from_env()
    assert cfg.topk == 0.0
    assert cfg.sink_size == 64
    assert cfg.local_size == 128
    assert cfg.channel_num == -1


def test_all_three_knobs(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.05")
    monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "32")
    monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "256")
    monkeypatch.setenv("SKYLIGHT_SPARSE_CHANNEL_NUM", "8")
    cfg = SkylightSparseConfig.from_env()
    assert cfg.topk == 0.05
    assert cfg.sink_size == 32
    assert cfg.local_size == 256
    assert cfg.channel_num == 8


def test_topk_boundaries(monkeypatch):
    """topk=0 and topk=1 are both accepted (boundary values)."""
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0")
    monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "1")  # to satisfy all-zero check
    cfg = SkylightSparseConfig.from_env()
    assert cfg.topk == 0.0

    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "1.0")
    cfg = SkylightSparseConfig.from_env()
    assert cfg.topk == 1.0


def test_channel_num_minus_one_sentinel(monkeypatch):
    """channel_num=-1 explicitly is preserved (sentinel for full head_dim)."""
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
    monkeypatch.setenv("SKYLIGHT_SPARSE_CHANNEL_NUM", "-1")
    cfg = SkylightSparseConfig.from_env()
    assert cfg.channel_num == -1


# -------------------------------- validation ---------------------------------


def test_all_zero_raises():
    """All of TOPK/SINK/LOCAL absent → must raise (degenerate config)."""
    with pytest.raises(ValueError, match="at least one of SKYLIGHT_SPARSE_"):
        SkylightSparseConfig.from_env()


def test_all_zero_explicit_raises(monkeypatch):
    """Explicit 0/0/0 also raises (same path as missing)."""
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0")
    monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "0")
    monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "0")
    with pytest.raises(ValueError, match="at least one of SKYLIGHT_SPARSE_"):
        SkylightSparseConfig.from_env()


def test_topk_out_of_range_high(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "1.5")
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        SkylightSparseConfig.from_env()


def test_topk_negative(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "-0.1")
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        SkylightSparseConfig.from_env()


def test_topk_malformed(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "not-a-number")
    with pytest.raises(ValueError, match="must be a float"):
        SkylightSparseConfig.from_env()


def test_sink_negative(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
    monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "-5")
    with pytest.raises(ValueError, match="must be >= 0"):
        SkylightSparseConfig.from_env()


def test_local_malformed(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
    monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "garbage")
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        SkylightSparseConfig.from_env()


def test_channel_num_zero_raises(monkeypatch):
    """channel_num=0 is invalid (would zero-out scoring)."""
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
    monkeypatch.setenv("SKYLIGHT_SPARSE_CHANNEL_NUM", "0")
    with pytest.raises(ValueError, match="must be -1"):
        SkylightSparseConfig.from_env()


def test_channel_num_minus_two_raises(monkeypatch):
    """Only -1 is a valid sentinel; -2 is invalid."""
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
    monkeypatch.setenv("SKYLIGHT_SPARSE_CHANNEL_NUM", "-2")
    with pytest.raises(ValueError, match="must be -1"):
        SkylightSparseConfig.from_env()


def test_channel_num_malformed(monkeypatch):
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
    monkeypatch.setenv("SKYLIGHT_SPARSE_CHANNEL_NUM", "eight")
    with pytest.raises(ValueError, match="must be -1 or a positive int"):
        SkylightSparseConfig.from_env()


# -------------------------------- immutability ------------------------------


def test_config_is_frozen(monkeypatch):
    """The dataclass is frozen — mutation must raise FrozenInstanceError."""
    monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
    cfg = SkylightSparseConfig.from_env()
    with pytest.raises(FrozenInstanceError):
        cfg.topk = 0.2  # type: ignore[misc]
