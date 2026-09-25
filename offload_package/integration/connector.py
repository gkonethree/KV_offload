# SPDX-License-Identifier: Apache-2.0
"""Main OffloadPackageConnector - vLLM KV Connector entry point."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Optional

import sys
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
from offload_package.integration.scheduler_connector import OffloadPackageScheduler
from offload_package.integration.worker_connector import OffloadPackageWorker
from offload_package.integration.metadata import OffloadPackageMetadata
from offload_package.integration.manager import OffloadOrchestrator

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
        )
        
        self.kv_cache_config = kv_cache_config
        self._orchestrator = None
        self._worker_handler = None
        self._scheduler_manager = None
        self._model_runner = None
        
        logger.info("OffloadPackageConnector initialized")

    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        """Initialize worker handler with KV cache config."""
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(kv_cache_config)

    def _build_orchestrator(self) -> None:
        """Build the offload orchestrator with all components."""
        from offload_package.vllm_adapter import normalize_runner_kv_caches
        
        if self._model_runner is None:
            return
            
        kv_cache_view = normalize_runner_kv_caches(
            self._model_runner.kv_caches,
            self.kv_cache_config,
            staging_slots=self._staging_slots,
        )

        from offload_package.shared.types import OrchestratorConfig, build_orchestrator
        
        cfg = OrchestratorConfig(
            offload_fraction=self.offload_config.offload_fraction,
            num_cpu_slots=self.offload_config.num_cpu_slots,
            num_staging_slots=self.offload_config.num_staging_slots,
            W=self.offload_config.interval_tokens,
            page_size=self.kv_cache_spec.block_size,
            dtype=kv_cache_view.dtype,
            device=kv_cache_view.device,
            gpu_budget_blocks=kv_cache_view.staging_base_page,
            kv_layout=kv_cache_view.kv_layout,
            scorer_decay=self.offload_config.scorer_decay,
            scorer_head_reduction=self.offload_config.scorer_head_reduction,
        )

        from offload_package.shared.types import build_orchestrator
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

    def initialize_from_config(self, kv_cache_config) -> None:
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(kv_cache_config)

    def _build_orchestrator(self) -> None:
        from offload_package.vllm_adapter import normalize_runner_kv_caches
        
        if self._model_runner is None:
            return
            
        kv_cache_view = normalize_runner_kv_caches(
            self._model_runner.kv_caches,
            self.kv_cache_config,
            staging_slots=self._staging_slots,
        )

        from offload_package.shared.types import OrchestratorConfig, build_orchestrator
        
        cfg = OrchestratorConfig(
            offload_fraction=self.offload_config.offload_fraction,
            num_cpu_slots=self.offload_config.num_cpu_slots,
            num_staging_slots=self.offload_config.num_staging_slots,
            W=self.offload_config.interval_tokens,
            page_size=self.kv_cache_spec.block_size,
            dtype=kv_cache_view.dtype,
            device=kv_cache_view.device,
            gpu_budget_blocks=kv_cache_view.staging_base_page,
            kv_layout=kv_cache_view.kv_layout,
            scorer_decay=self.offload_config.scorer_decay,
            scorer_head_reduction=self.offload_config.scorer_head_reduction,
        )

        from offload_package.shared.types import build_orchestrator
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

    def initialize_from_config(self, kv_cache_config) -> None:
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(kv_cache_config)

    def _build_orchestrator(self) -> None:
        from offload_package.vllm_adapter import normalize_runner_kv_caches
        
        if self._model_runner is None:
            return
            
        kv_cache_view = normalize_runner_kv_caches(
            self._model_runner.kv_caches,
            self.kv_cache_config,
            staging_slots=self._staging_slots,
        )

        from offload_package.shared.types import OrchestratorConfig, build_orchestrator
        
        cfg = OrchestratorConfig(
            offload_fraction=self.offload_config.offload_fraction,
            num_cpu_slots=self.offload_config.num_cpu_slots,
            num_staging_slots=self.offload_config.num_staging_slots,
            W=self.offload_config.interval_tokens,
            page_size=self.kv_cache_spec.block_size,
            dtype=kv_cache_view.dtype,
            device=kv_cache_view.device,
            gpu_budget_blocks=kv_cache_view.staging_base_page,
            kv_layout=kv_cache_view.kv_layout,
            scorer_decay=self.offload_config.scorer_decay,
            scorer_head_reduction=self.offload_config.scorer_head_reduction,
        )

        from offload_package.shared.types import build_orchestrator
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

    def initialize_from_config(self, kv_cache_config) -> None:
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(kv_cache_config)

    def _build_orchestrator(self) -> None:
        from offload_package.vllm_adapter import normalize_runner_kv_caches
        
        if self._model_runner is None:
            return
            
        kv_cache_view = normalize_runner_kv_caches(
            self._model_runner.kv_caches,
            self.kv_cache_config,
            staging_slots=self._staging_slots,
        )

        from offload_package.shared.types import OrchestratorConfig, build_orchestrator
        
        cfg = OrchestratorConfig(
            offload_fraction=self.offload_config.offload_fraction,
            num_cpu_slots=self.offload_config.num_cpu_slots,
            num_staging_slots=self.offload_config.num_staging_slots,
            W=self.offload_config.interval_tokens,
            page_size=self.kv_cache_spec.block_size,
            dtype=kv_cache_view.dtype,
            device=kv_cache_view.device,
            gpu_budget_blocks=kv_cache_view.staging_base_page,
            kv_layout=kv_cache_view.kv_layout,
            scorer_decay=self.offload_config.scorer_decay,
            scorer_head_reduction=self.offload_config.scorer_head_reduction,
        )

        from offload_package.shared.types import build_orchestrator
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

    def initialize_from_config(self, kv_cache_config) -> None:
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(kv_cache_config)

    def _build_orchestrator(self) -> None:
        from offload_package.vllm_adapter import normalize_runner_kv_caches
        
        if self._model_runner is None:
            return
            
        kv_cache_view = normalize_runner_kv_caches(
            self._model_runner.kv_caches,
            self.kv_cache_config,
            staging_slots=self._staging_slots,
        )

        from offload_package.shared.types import OrchestratorConfig, build_orchestrator
        
        cfg = OrchestratorConfig(
            offload_fraction=self.offload_config.offload_fraction,
            num_cpu_slots=self.offload_config.num_cpu_slots,
            num_staging_slots=self.offload_config.num_staging_slots,
            W=self.offload_config.interval_tokens,
            page_size=self.kv_cache_spec.block_size,
            dtype=kv_cache_view.dtype,
            device=kv_cache_view.device,
            gpu_budget_blocks=kv_cache_view.staging_base_page,
            kv_layout=kv_cache_view.kv_layout,
            scorer_decay=self.offload_config.scorer_decay,
            scorer_head_reduction=self.offload_config.scorer_head_reduction,
        )

        from offload_package.shared.types import build_orchestrator
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

    def initialize_from_config(self, kv_cache_config) -> None:
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(kv_cache_config)

    def _build_orchestrator(self) -> None:
        from offload_package.vllm_adapter import normalize_runner_kv_caches
        
        if self._model_runner is None:
            return
            
        kv_cache_view = normalize_runner_kv_caches(
            self._model_runner.kv_caches,
            self.kv_cache_config,
            staging_slots=self._staging_slots,
        )

        from offload_package.shared.types import OrchestratorConfig, build_orchestrator
        
        cfg = OrchestratorConfig(
            offload_fraction=self.offload_config.offload_fraction,
            num_cpu_slots=self.offload_config.num_cpu_slots,
            num_staging_slots=self.offload_config.num_staging_slots,
            W=self.offload_config.interval_tokens,
            page_size=self.kv_cache_spec.block_size,
            dtype=kv_cache_view.dtype,
            device=kv_cache_view.device,
            gpu_budget_blocks=kv_cache_view.staging_base_page,
            kv_layout=kv_cache_view.kv_layout,
            scorer_decay=self.offload_config.scorer_decay,
            scorer_head_reduction=self.offload_config.scorer_head_reduction,
        )

        from offload_package.shared.types import build_orchestrator
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

    def initialize_from_config(self, kv_cache_config) -> None:
        if self._worker_handler is not None:
            self._worker_handler.initialize_from_config(kv_cache_config)
        if self._scheduler_manager is not None:
            self._scheduler_manager.bind_gpu_block_pool(kv_cache_config)

    def set_runner(self, runner) -> None:
        self._model_runner = runner
        if self._orchestrator is None:
            self._build_orchestrator()

    # Abstract methods from KVConnectorBase_V1
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int):
        return 0, False
    
    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        pass
    
    def request_finished(self, request, block_ids):
        return False, None
    
    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids=[])
    
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass
    
    def wait_for_save(self):
        pass
    
    def get_finished(self, finished_req_ids):
        return None, None
    
    def shutdown(self):
        pass
    
    def get_kv_connector_stats(self):
        return None
    
    def get_kv_connector_kv_cache_events(self):
        return None
    
    def get_handshake_metadata(self):
        return None
    
    def get_block_ids_with_load_errors(self):
        return set()
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None
    
    def register_kv_caches(self, kv_caches):
        pass
    
    def bind_connector_metadata(self, connector_metadata):
        pass
    
    def clear_connector_metadata(self):
        pass
    
    def handle_preemptions(self, kv_connector_metadata):
        pass
    
    def start_load_kv(self, forward_context):
        pass
    
    def wait_for_layer_load(self, layer_name):
        pass
    
    def build_connector_meta(self, scheduler_output):
        from offload_package.integration.metadata import OffloadPackageMetadata
        return OffloadPackageMetadata()
    
    def update_connector_output(self, connector_output):
        pass
    
    def get_handshake_metadata(self):
        return None

