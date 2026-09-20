"""Prometheus metrics exposed by SkylightSparseBackend.

vllm already exposes a ``/metrics`` endpoint that scrapes from
:data:`prometheus_client.REGISTRY`. Importing this module registers our
metrics in that same default registry, so they appear on vllm's endpoint
automatically — no new HTTP route, no separate scraper.

Multi-process aggregation
-------------------------

vllm runs the API server and each EngineCore worker in separate processes.
Our gauge is updated from the WORKER (where ``_NoFastPlanDecodeWrapper.run``
fires), but ``/metrics`` is served by the parent. Without multi-process
mode, the parent reads its own (untouched) Gauge → ``0.0``.

prometheus_client's multi-process mode aggregates per-process state via
files in :envvar:`PROMETHEUS_MULTIPROC_DIR`. vllm sets that env var when
the server starts so its own metrics aggregate; we ride the same machinery
by declaring ``multiprocess_mode`` on the Gauge. But the kwarg is only
valid when the env var is actually set — outside the server (e.g. unit
tests in a single process), we must omit it.

Metrics declared here are updated by hot-path code in :mod:`skylight.backend`
(per-decode-batch ``run()`` callback). The values come from the kernel
wrapper's sync-free :meth:`get_sparsity_stats` accessor — no GPU readback,
~µs per update.
"""
from __future__ import annotations

import os

from prometheus_client import Gauge


_gauge_kwargs: dict = {}
if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    # ``livemostrecent`` exposes the most recent set() from any LIVE process;
    # right semantic for "what fraction did the latest decode use".
    _gauge_kwargs["multiprocess_mode"] = "livemostrecent"


SPARSE_EFFECTIVE_FRACTION: Gauge = Gauge(
    "skylight_effective_sparsity_fraction",
    "Fraction of per-batch max KV positions actually selected by the "
    "sparse decode kernel (sink + topk_middle + local, capped at n_keys). "
    "1.0 indicates dense behaviour — either the operator configured it "
    "that way (topk=1.0) or a regression silently fell back to dense.",
    **_gauge_kwargs,
)
