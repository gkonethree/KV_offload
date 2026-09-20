"""Tests for skylight.blackwell: env defaults + state tracking."""
from __future__ import annotations

import os

import pytest

import skylight.blackwell as blackwell


@pytest.fixture(autouse=True)
def _reset_blackwell():
    """Reset blackwell module state before each test."""
    blackwell._APPLIED = False
    blackwell.state.clear()
    yield


def test_set_env_default_new_key():
    """If env var is absent, _set_env_default sets it + records 'default'."""
    os.environ.pop("SKYLIGHT_TEST_KEY", None)
    blackwell._set_env_default("SKYLIGHT_TEST_KEY", "abc", "test reason")

    assert os.environ["SKYLIGHT_TEST_KEY"] == "abc"
    assert blackwell.state == [
        ("SKYLIGHT_TEST_KEY", "abc", "default", "test reason"),
    ]
    del os.environ["SKYLIGHT_TEST_KEY"]


def test_set_env_default_user_set_preserves():
    """If env var is already set, _set_env_default preserves it + records 'user-set'."""
    os.environ["SKYLIGHT_TEST_KEY"] = "user_value"
    blackwell._set_env_default("SKYLIGHT_TEST_KEY", "default_value", "test reason")

    assert os.environ["SKYLIGHT_TEST_KEY"] == "user_value"  # NOT overwritten
    assert blackwell.state == [
        ("SKYLIGHT_TEST_KEY", "user_value", "user-set", "test reason"),
    ]
    del os.environ["SKYLIGHT_TEST_KEY"]


def test_apply_sets_expected_keys():
    """apply() sets FLASHINFER_TOPK_ALGO and VLLM_USE_DEEP_GEMM when absent."""
    os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    os.environ.pop("VLLM_USE_DEEP_GEMM", None)

    blackwell.apply()

    names = [name for name, *_ in blackwell.state]
    assert "FLASHINFER_TOPK_ALGO" in names
    assert "VLLM_USE_DEEP_GEMM" in names
    assert os.environ.get("FLASHINFER_TOPK_ALGO") == "radix"
    assert os.environ.get("VLLM_USE_DEEP_GEMM") == "0"


def test_apply_records_user_set_when_env_already_set():
    """apply() preserves user-set FLASHINFER_TOPK_ALGO and records 'user-set'."""
    os.environ["FLASHINFER_TOPK_ALGO"] = "clusters"  # operator override

    blackwell.apply()

    assert os.environ["FLASHINFER_TOPK_ALGO"] == "clusters"  # preserved
    entry = next(e for e in blackwell.state if e[0] == "FLASHINFER_TOPK_ALGO")
    assert entry[2] == "user-set"
    del os.environ["FLASHINFER_TOPK_ALGO"]


def test_apply_idempotent():
    """Calling apply() twice must not duplicate state entries."""
    os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    os.environ.pop("VLLM_USE_DEEP_GEMM", None)

    blackwell.apply()
    first_len = len(blackwell.state)
    assert first_len > 0

    blackwell.apply()
    assert len(blackwell.state) == first_len  # no duplicates from second call


def test_apply_adds_venv_bin_to_path(monkeypatch):
    """apply() puts sys.executable's dir on PATH (for ninja / venv tooling)."""
    import sys
    venv_bin = os.path.dirname(sys.executable)
    # Start from a PATH that explicitly excludes venv_bin.
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    blackwell.apply()

    assert venv_bin in os.environ["PATH"].split(":")


def test_apply_records_cudnn_sdp_disable():
    """apply() should record the torch.backends.cuda.enable_cudnn_sdp(False) action."""
    blackwell.apply()

    entry = next(
        (e for e in blackwell.state if e[0] == "torch.backends.cuda.enable_cudnn_sdp"),
        None,
    )
    assert entry is not None, "cudnn SDP disable action not recorded"
    # Either "applied" (torch present, normal path) or "FAILED" (torch missing).
    # On the bench host torch is always present, so prefer "applied".
    assert entry[2] in {"applied", "FAILED"}


def test_print_banner_smoke(capsys):
    """print_banner() emits at least the header even with empty state."""
    blackwell.print_banner()  # state empty here due to fixture
    captured = capsys.readouterr()
    assert "[skylight.blackwell] B200 base prereqs:" in captured.out
