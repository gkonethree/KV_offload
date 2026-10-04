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
        self.offload_config = OffloadConfig(
            enabled=extra_config.get("enabled", True),
            offload_fraction=extra_config.get("offload_fraction", 0.2),
            num_cpu_slots=extra_config.get("num_cpu_slots", 4096),
            num_staging_slots=extra_config.get("num_staging_slots", 64),
            interval_tokens=extra_config.get("interval_tokens", 32),
            scorer_decay=extra_config.get("scorer_decay", 1.0),
            scorer_head_reduction=extra_config.get("scorer_head_reduction", "mean"),
            true_free=extra_config.get(
                "true_free",
                _os.getenv("OFFLOAD_PACKAGE_TRUE_FREE", "1").lower()
                in {"1", "true", "yes", "on"},
            ),
            promote_per_step=int(extra_config.get(
                "promote_per_step",
                _os.getenv("OFFLOAD_PACKAGE_PROMOTE_PER_STEP", "4"),
            )),
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
        if self._scheduler_manager is not None:
            return self._scheduler_manager.build_connector_meta(scheduler_output)
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        if self._scheduler_manager is not None:
            self._scheduler_manager.update_connector_output(connector_output)    