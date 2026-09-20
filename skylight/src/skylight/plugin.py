"""vllm.general_plugins entry point.

Invoked by vllm once per process at startup — including in every
spawn-launched EngineCore worker, which doesn't inherit module-level
state from the API server. That's the only way plugin actions
(env-var defaults, backend registration) reach every worker.

Two responsibilities:
  1. Apply Blackwell base env prereqs (FLASHINFER_TOPK_ALGO=radix, etc.).
  2. Register :class:`SkylightSparseBackend` under
     :attr:`AttentionBackendEnum.CUSTOM`, so ``VLLM_ATTENTION_BACKEND=CUSTOM``
     resolves to our backend.

Idempotent — safe to invoke multiple times.
"""
from __future__ import annotations

import logging

from skylight.runtime import configure_runtime

logger = logging.getLogger(__name__)

_INSTALLED = False


def install_plugin() -> None:
    """Apply skylight's vllm-side prerequisites once per process. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    platform = configure_runtime()
    if platform.name == "blackwell":
        from skylight import blackwell
        blackwell.apply()

    # Register our Prometheus metrics with the default REGISTRY so vllm's
    # /metrics endpoint scrapes them. Side-effect import; the Gauge is
    # declared at module scope.
    from skylight import metrics  # noqa: F401

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "skylight.backend.SkylightSparseBackend",
    )

    logger.info(
        "skylight plugin loaded: SkylightSparseBackend registered under "
        "AttentionBackendEnum.CUSTOM (activate via VLLM_ATTENTION_BACKEND=CUSTOM)"
    )
