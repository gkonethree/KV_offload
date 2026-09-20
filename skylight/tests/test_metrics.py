"""Tests for skylight.metrics + the _NoFastPlanDecodeWrapper Gauge update.

The Gauge is registered in :data:`prometheus_client.REGISTRY` at module
import, which is the same default registry vllm's ``/metrics`` HTTP endpoint
scrapes from. ``_NoFastPlanDecodeWrapper.run()`` updates the Gauge after
each decode batch using the inner kernel wrapper's ``get_sparsity_stats``
accessor.

We pin:
  1. The metric appears in the registry by name.
  2. Documentation mentions sparse + dense (so operators understand 1.0).
  3. Calling outer.run() with an inner that exposes get_sparsity_stats
     updates the Gauge to that inner's last_fraction.
  4. Calling outer.run() with an inner that DOES NOT expose
     get_sparsity_stats does not crash (defensive).
"""
from __future__ import annotations

import pytest
from prometheus_client import REGISTRY


# Importing the module at test-collection time registers the Gauge.
import skylight.metrics  # noqa: F401, E402


METRIC_NAME = "skylight_effective_sparsity_fraction"


def test_gauge_registered_in_default_registry():
    """Importing skylight.metrics adds the gauge to prometheus_client's REGISTRY."""
    names = [m.name for m in REGISTRY.collect()]
    assert METRIC_NAME in names, (
        f"{METRIC_NAME!r} not in REGISTRY; saw {sorted(names)!r}"
    )


def test_gauge_has_expected_documentation():
    """Gauge's help text mentions sparse + dense (so operators understand 1.0)."""
    metric = next(
        (m for m in REGISTRY.collect() if m.name == METRIC_NAME),
        None,
    )
    assert metric is not None
    doc = metric.documentation.lower()
    assert "sparse" in doc, f"help text missing 'sparse': {metric.documentation!r}"
    assert "dense" in doc, f"help text missing 'dense': {metric.documentation!r}"


def test_wrapper_run_updates_gauge_when_inner_exposes_stats():
    """When inner has get_sparsity_stats, outer.run() sets the Gauge to last_fraction."""
    from skylight.backend import _NoFastPlanDecodeWrapper

    class _SparsityStats:
        def __init__(self, fraction: float) -> None:
            self.last_fraction = fraction

    class _InnerWithStats:
        def __init__(self) -> None:
            self.last_run = None

        def run(self, q: str, kv: str) -> str:
            self.last_run = (q, kv)
            return "ok"

        def get_sparsity_stats(self) -> _SparsityStats:
            return _SparsityStats(fraction=0.42)

    inner = _InnerWithStats()
    outer = _NoFastPlanDecodeWrapper(inner)

    result = outer.run("q", "kv")

    assert result == "ok"
    assert inner.last_run == ("q", "kv")
    sampled = REGISTRY.get_sample_value(METRIC_NAME)
    assert sampled == pytest.approx(0.42)


def test_wrapper_run_without_get_sparsity_stats_does_not_crash():
    """Inner wrappers without get_sparsity_stats are tolerated (no AttributeError)."""
    from skylight.backend import _NoFastPlanDecodeWrapper

    class _InnerWithoutStats:
        def run(self, q: str, kv: str) -> str:
            return "ok"

    inner = _InnerWithoutStats()
    outer = _NoFastPlanDecodeWrapper(inner)

    # Must not raise; gauge update silently skipped.
    assert outer.run("q", "kv") == "ok"
