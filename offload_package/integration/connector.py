# SPDX-License-Identifier: Apache-2.0
"""Main OffloadPackageConnector - vLLM KV Connector entry point."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Optional

import sys
import os as _os
import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.core.sched.output import SchedulerOutput

from offload_package.config import OffloadConfig
from offload_package.debug import dbg as _dbg
# Lazy imports for scheduler_connector/worker_connector/manager: importing
# them at module level creates a cycle (vllm's kv_transfer v1 __init__
# re-exports this module, which would re-enter offload_package.integration
# while manager is still mid-import).

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


_PQ_METHODS = ("pq", "pq_optimized", "pqrs", "pqrs_optimized", "pqis", "pqis_optimized")


def _union_selector_count(vllm_config, kv_cache_config, share: bool) -> int:
    """Number of independently-selecting heads (G_eff) for staging sizing.

    The fetch path stages the UNION of every scoring head's selection, so the
    pool must cover the union, not one head's picks: a mid-range block is
    picked by >=1 of G_eff selectors each choosing a topk fraction ->
    covered fraction 1-(1-topk)^G_eff. With GQA-share the selectors are the
    TP-local kv-heads (H_kv); without, every q-head picks independently
    (H_q). H_kv comes from the AttentionSpec (already TP-local); H_q from
    hf_config (divided by TP), falling back to H_kv + a warning when the
    model config does not expose attention-head counts.
    """
    spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    h_kv = int(getattr(spec, "num_kv_heads", 0) or 0)
    if share:
        return max(h_kv, 1)
    h_q = 0
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    if hf_config is not None:
        h_q = int(getattr(hf_config, "num_attention_heads", 0) or 0)
        tp = int(getattr(
            getattr(vllm_config, "parallel_config", None),
            "tensor_parallel_size", 1) or 1)
        h_q = max(h_q // tp, 0)
    if h_q <= 0:
        logger.warning(
            "staging sizing: q-head count unavailable; falling back to "
            "H_kv=%d (underestimates the union without GQA-share)", h_kv,
        )
        return max(h_kv, 1)
    return h_q


def _auto_staging_slots_for(vllm_config, kv_cache_config, extra_config=None) -> int:
    """Auto-size staging: cover the UNION of all scoring heads' selections per
    request, times the steady-state max batch, plus margin, clamped to a
    fraction of the KV pool (never crashes on small pools).

    Knobs come from kv_connector_extra_config first (staging_safety_margin,
    staging_max_pool_frac), then the OFFLOAD_PACKAGE_* env vars.
    """
    extra_config = dict(extra_config or {})
    topk = float(_os.getenv("SKYLIGHT_SPARSE_TOPK", "0.1"))
    sink = int(_os.getenv("SKYLIGHT_SPARSE_SINK", "0"))
    local = int(_os.getenv("SKYLIGHT_SPARSE_LOCAL", "0"))
    max_len = int(vllm_config.model_config.max_model_len)
    max_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    page_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
    mid = max(0, max_len - sink - local)

    method = _os.getenv("SKYLIGHT_SPARSE_METHOD", "oracle").strip().lower()
    share = (
        method in _PQ_METHODS
        and _os.getenv("SKYLIGHT_PQ_GQA_SHARE", "0") == "1"
    )
    g_eff = _union_selector_count(vllm_config, kv_cache_config, share)
    union_factor = 1.0 - (1.0 - min(topk, 1.0)) ** g_eff
    pages_per_req = ((sink + local) // page_size
                     + int(union_factor * mid / page_size + 0.999))
    safety = float(extra_config.get(
        "staging_safety_margin",
        _os.getenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10")))
    frac = float(extra_config.get(
        "staging_max_pool_frac",
        _os.getenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25")))
    num_blocks = int(kv_cache_config.num_blocks)
    est = int(max_seqs * pages_per_req * safety)
    cap = max(1, int(num_blocks * frac))
    result = max(64, min(est, cap))
    logger.info(
        "OffloadPackageConnector: auto staging policy -> %d slots "
        "(est=%d, cap=%d, max_len=%d, max_seqs=%d, topk=%.4f, G_eff=%d, "
        "share=%s, union=%.3f, sink=%d, local=%d)",
        result, est, cap, max_len, max_seqs, topk, g_eff, share,
        union_factor, sink, local,
    )
    return result


class OffloadPackageConnector(KVConnectorBase_V1):
    """Main OffloadPackageConnector - vLLM KV Connector entry point."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        enabled = extra_config.get("enabled", True)
        num_staging_slots = int(extra_config.get("num_staging_slots", 64))

        policy = extra_config.get(
            "staging_policy",
            _os.getenv("OFFLOAD_PACKAGE_STAGING_POLICY", "fixed"),
        )
        if policy == "auto":
            num_staging_slots = _auto_staging_slots_for(
                vllm_config, kv_cache_config, extra_config,
            )

        self.offload_config = OffloadConfig(
            enabled=enabled,
            offload_fraction=extra_config.get("offload_fraction", 0.2),
            num_cpu_slots=extra_config.get("num_cpu_slots", 4096),
            num_staging_slots=num_staging_slots,
            interval_tokens=extra_config.get("interval_tokens", 32),
            scorer_decay=extra_config.get("scorer_decay", 1.0),
            scorer_head_reduction=extra_config.get("scorer_head_reduction", "mean"),
            true_free=extra_config.get(
                "true_free",
                _os.getenv("OFFLOAD_PACKAGE_TRUE_FREE", "1").lower()
                in {"1", "true", "yes", "on"},
            ),
            staging_policy=policy,
            staging_max_pool_frac=float(extra_config.get(
                "staging_max_pool_frac",
                _os.getenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25"))),
            staging_safety_margin=float(extra_config.get(
                "staging_safety_margin",
                _os.getenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10"))),
            scorer_type=extra_config.get(
                "scorer_type",
                _os.getenv("OFFLOAD_PACKAGE_SCORER_TYPE", "paged_eviction")),
            staging_eviction=extra_config.get(
                "staging_eviction",
                _os.getenv("OFFLOAD_PACKAGE_STAGING_EVICTION", "lru")),
        )
        
        self.kv_cache_config = kv_cache_config
        self._orchestrator = None
        self._model_runner = None
        self._worker_handler = None
        self._scheduler_manager = None
        
        # Create BOTH handlers for uni-process mode where scheduler and worker
        # share the same connector instance. The worker initializes first (role=WORKER),
        # then scheduler reuses the connector but needs scheduler_manager.
        from offload_package.integration.worker_connector import OffloadPackageWorker
        from offload_package.integration.scheduler_connector import OffloadPackageScheduler
        
        self._worker_handler = OffloadPackageWorker(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
            offload_config=self.offload_config,
        )
        
        # Create scheduler manager - will be bound to GPU block pool later
        self._scheduler_manager = OffloadPackageScheduler(
            vllm_config,
            kv_cache_config,
            self.offload_config,
            worker_handler=self._worker_handler,
        )
        
        logger.info("OffloadPackageConnector initialized with both worker and scheduler handlers")

    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        """Initialize worker handler with KV cache config."""
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool", kv_cache_manager=None) -> None:
        """Bind GPU BlockPool to scheduler manager."""
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(
                gpu_block_pool, kv_cache_manager=kv_cache_manager
            )

    def set_runner(self, runner) -> None:
        _dbg(f"[CONNECTOR DEBUG] set_runner called, runner={type(runner)}, worker_handler={id(self._worker_handler) if self._worker_handler else None}")
        self._model_runner = runner
        # Delegate to worker handler which builds the orchestrator
        if self._worker_handler is not None:
            self._worker_handler.set_runner(runner)

    # Abstract methods from KVConnectorBase_V1
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        return 0, False
    
    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        pass
    
    def request_finished(self, request, block_ids):
        if self._scheduler_manager is not None:
            return self._scheduler_manager.request_finished(request, block_ids)
        return False, None
    
    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids=[])
    
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass
    
    def wait_for_save(self):
        # Delegate: the worker records store-event completion so
        # get_finished() -> build_connector_worker_meta() reports it to the
        # scheduler, which then frees the touched GPU blocks.
        if self._worker_handler is not None:
            self._worker_handler.wait_for_save()

    def build_connector_worker_meta(self):
        if self._worker_handler is not None:
            return self._worker_handler.build_worker_meta()
        return None
    
    def get_finished(self, finished_req_ids):
        if self._worker_handler is not None:
            return self._worker_handler.get_finished(finished_req_ids)
        return None, None
    
    def shutdown(self):
        if self._worker_handler is not None:
            self._worker_handler.shutdown()
    
    def get_kv_connector_stats(self):
        return None
    
    def get_kv_connector_kv_cache_events(self):
        return None
    
    def get_handshake_metadata(self):
        if self._worker_handler is not None:
            return self._worker_handler.get_handshake_metadata()
        return None
    
    def get_block_ids_with_load_errors(self):
        return set()
    
    def register_kv_caches(self, kv_caches):
        # Register KV caches with worker handler
        if self._worker_handler is not None:
            self._worker_handler.register_kv_caches(kv_caches)
    
    def bind_connector_metadata(self, connector_metadata):
        if self._worker_handler is not None:
            self._worker_handler.bind_connector_metadata(connector_metadata)
    
    def clear_connector_metadata(self):
        if self._worker_handler is not None:
            self._worker_handler.clear_connector_metadata()
    
    def handle_preemptions(self, kv_connector_metadata):
        if self._worker_handler is not None:
            self._worker_handler.handle_preemptions(kv_connector_metadata)
    
    def start_load_kv(self, forward_context):
        if self._worker_handler is not None:
            self._worker_handler.start_load_kv(forward_context)
    
    def wait_for_layer_load(self, layer_name):
        if self._worker_handler is not None:
            self._worker_handler.wait_for_layer_load(layer_name)
    
    def build_connector_meta(self, scheduler_output):
        """Build metadata for this step's offload/staging operations.
        
        Guarantees a valid OffloadPackageMetadata is returned (never None)
        to satisfy the vLLM worker's assertion.
        """
        try:
            if self._scheduler_manager is not None:
                result = self._scheduler_manager.build_connector_meta(scheduler_output)
                if result is not None:
                    return result
        except Exception as e:
            import logging
            logging.warning(f"build_connector_meta scheduler_manager failed: {e}")
        
        # Fallback: always return a valid (possibly empty) metadata object
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        if self._scheduler_manager is not None:
            self._scheduler_manager.update_connector_output(connector_output)

# Runtime patch for gpu_model_runner to handle None metadata gracefully.
# This runs when the module is imported in the worker process.
try:
    import vllm.v1.worker.gpu_model_runner as _gmr
    _orig_execute_model = _gmr.GPUModelRunner.execute_model
    
    def _patched_execute_model(self, scheduler_output, *args, **kwargs):
        # Ensure kv_connector_metadata is never None when has_kv_transfer_group() is True
        from vllm.distributed.kv_transfer import has_kv_transfer_group
        from offload_package.integration.metadata import OffloadPackageMetadata
        if has_kv_transfer_group():
            if scheduler_output.kv_connector_metadata is None:
                import logging
                #logging.warning("kv_connector_metadata was None, substituting empty OffloadPackageMetadata")
                scheduler_output.kv_connector_metadata = OffloadPackageMetadata()
        return _orig_execute_model(self, scheduler_output, *args, **kwargs)
    
    _gmr.GPUModelRunner.execute_model = _patched_execute_model
except Exception:
    pass  # Patch failed silently, original behavior preserved    