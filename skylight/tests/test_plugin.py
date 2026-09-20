"""Tests for skylight.plugin: entry-point registration + idempotency.

The vllm.general_plugins entry point is the mechanism vllm uses to discover
skylight in each worker process; this file pins the contract that:

  1. The entry-point is declared in our pyproject and resolvable via
     importlib.metadata.entry_points (would silently break otherwise).
  2. install_plugin() is idempotent (vllm may call it in parent + workers).
  3. install_plugin() applies blackwell prereqs as a side effect.
"""
from __future__ import annotations

import os
from importlib.metadata import entry_points
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

import skylight.blackwell as blackwell
import skylight.plugin as plugin
from skylight.runtime import Platform


@pytest.fixture(autouse=True)
def _reset_plugin_state(monkeypatch):
    """Reset both module-level guards before each test."""
    plugin._INSTALLED = False
    blackwell._APPLIED = False
    blackwell.state.clear()
    monkeypatch.setattr(
        plugin,
        "configure_runtime",
        lambda: Platform("blackwell", (10, 0), "sm100", "10.0"),
    )

    # Keep unit tests independent of the heavyweight vllm/prometheus install.
    monkeypatch.setitem(sys.modules, "skylight.metrics", ModuleType("skylight.metrics"))
    for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = type("AttentionBackendEnum", (), {"CUSTOM": "CUSTOM"})
    registry.register_backend = Mock()
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    yield


def test_plugin_module_importable():
    """Module imports cleanly and exposes install_plugin."""
    assert hasattr(plugin, "install_plugin")
    assert callable(plugin.install_plugin)


def test_entry_point_discoverable():
    """vllm.general_plugins must include 'skylight' resolving to our install_plugin."""
    eps = entry_points(group="vllm.general_plugins")
    names = [e.name for e in eps]
    assert "skylight" in names, (
        f"'skylight' not found under vllm.general_plugins; saw {names!r}. "
        f"Has the workspace pyproject's [project.entry-points.\"vllm.general_plugins\"] "
        f"section been applied and `uv sync` rerun?"
    )

    skylight_ep = next(e for e in eps if e.name == "skylight")
    loaded = skylight_ep.load()
    assert callable(loaded)
    assert loaded.__module__ == "skylight.plugin"
    assert loaded.__name__ == "install_plugin"


def test_install_plugin_idempotent():
    """Calling install_plugin twice does not raise and does not duplicate state."""
    plugin.install_plugin()
    assert plugin._INSTALLED is True
    state_after_first = list(blackwell.state)

    plugin.install_plugin()
    assert plugin._INSTALLED is True
    assert list(blackwell.state) == state_after_first  # no duplicates


def test_install_plugin_applies_blackwell():
    """install_plugin() should trigger blackwell.apply() side effects."""
    os.environ.pop("FLASHINFER_TOPK_ALGO", None)

    plugin.install_plugin()

    assert blackwell._APPLIED is True
    names = [name for name, *_ in blackwell.state]
    assert "FLASHINFER_TOPK_ALGO" in names
    assert os.environ.get("FLASHINFER_TOPK_ALGO") == "radix"


def test_install_plugin_does_not_apply_blackwell_on_hopper(monkeypatch):
    configure = Mock(
        return_value=Platform("hopper", (9, 0), "sm90", "9.0")
    )
    apply = Mock()
    monkeypatch.setattr(plugin, "configure_runtime", configure)
    monkeypatch.setattr(blackwell, "apply", apply)

    plugin.install_plugin()

    configure.assert_called_once_with()
    apply.assert_not_called()


def test_install_plugin_applies_blackwell_on_sm100(monkeypatch):
    configure = Mock(
        return_value=Platform("blackwell", (10, 0), "sm100", "10.0")
    )
    apply = Mock()
    monkeypatch.setattr(plugin, "configure_runtime", configure)
    monkeypatch.setattr(blackwell, "apply", apply)

    plugin.install_plugin()

    configure.assert_called_once_with()
    apply.assert_called_once_with()
