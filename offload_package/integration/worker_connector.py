# SPDX-License-Identifier: Apache-2.0
"""Worker-side handler for async GPU↔CPU transfers and sparse attention staging."""

from __future__ import annotations

import os
import logging
from typing import Optional

import torch
from collections import defaultdict

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.forward_context import ForwardContext

from offload_package.config import OffloadConfig
from offload_package.controller import KVCompressionController
from offload_package.integration.manager import OffloadOrchestrator
from offload_package.scoring.base import BlockRef
from offload_package.integration.metadata import (
    OffloadPackageMetadata,
    OffloadPackageWorkerMetadata,
)
from offload_package.offload.manager import KVOffloadManager
from offload_package.selection.adapter import (
    collect_needed_block_refs,
    patch_paged_kv_indices_with_staging,
)
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable
from offload_package.shared.types import (
    OrchestratorConfig,
    build_orchestrator,
    normalize_runner_kv_caches,
)

logger = init_logger(__name__)

INVALID_JOB_ID = -1


class OffloadPackageWorker(KVConnectorBase_V1):
    """Worker-side handler for async GPU↔CPU transfers and sparse attention staging."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
        offload_config: OffloadConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self.offload_config = offload_config
        self._vllm_config = vllm_config

        # Will be initialized from vLLM runtime
        self.kv_cache_config: Optional[KVCacheConfig] = None
        self.kv_cache_spec = None
        self._orchestrator: Optional[OffloadOrchestrator] = None

        # Worker reference for getting request IDs
        self._worker = None

        # Stream for async copies
        self._load_stream = torch.cuda.Stream()
        self._store_stream = torch.cuda.Stream()

        # Transfer tracking
        self._load_events: list[tuple[int, torch.cuda.Event]] = []
        self._store_events: list[tuple[int, torch.cuda.Event]] = []

        # Model runner reference (set by set_runner)
        self._model_runner = None

        # Completion watermarks
        self._load_hwm = -1
        self._store_hwm = -1

        # Pending event tracking
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()

        # Completed event counts (for multi-worker aggregation)
        self._completed_load_events: dict[int, int] = defaultdict(int)
        self._completed_store_events: dict[int, int] = defaultdict(int)

        # Metadata from scheduler
        self._metadata: Optional[OffloadPackageMetadata] = None

        # Staging pool config
        self._staging_slots = offload_config.num_staging_slots
        self._staging_base_page = 0

        logger.info("OffloadPackageWorker initialized")

    def initialize_from_config(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Initialize worker after KV cache config is known."""
        self.kv_cache_config = kv_cache_config
        # Extract AttentionSpec from KVCacheConfig
        from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs
        spec = None
        for group in kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                for layer_name in group.layer_names:
                    layer_spec = group_spec.kv_cache_specs[layer_name]
                    if isinstance(layer_spec, AttentionSpec):
                        spec = layer_spec
                        break
            elif hasattr(group_spec, 'block_size') and hasattr(group_spec, 'num_kv_heads'):
                spec = group_spec
            if spec:
                break
        if spec is None:
            raise ValueError("No AttentionSpec found in KVCacheConfig")
        self.kv_cache_spec = spec
        self._staging_base_page = self.kv_cache_config.num_blocks - self._staging_slots

    def _build_orchestrator(self) -> None:
        """Build the offload orchestrator with all components."""
        # Normalize KV caches to get the view
        kv_cache_view = normalize_runner_kv_caches(
            self._model_runner.kv_caches,
            self.kv_cache_config,
            staging_slots=self._staging_slots,
        )

        # Build orchestrator config
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

        # Build orchestrator using the factory function
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

        # Inject orchestrator into metadata builders
        if self._orchestrator is not None:
            # Get layer names from kv_cache_config
            for group in self.kv_cache_config.kv_cache_groups:
                for layer_name in group.layer_names:
                    layer_spec = self.kv_cache_spec
                    if hasattr(layer_spec, 'metadata_builder') and layer_spec.metadata_builder is not None:
                        builder_cls = type(layer_spec.metadata_builder)
                        if hasattr(builder_cls, '_offload_orchestrator'):
                            builder_cls._offload_orchestrator = self._orchestrator
                            layer_spec.metadata_builder._offload_orchestrator = self._orchestrator

        logger.info("OffloadPackageWorker orchestrator built successfully")

    def bind_worker(self, worker) -> None:
        """Bind to the GPU worker for accessing model runner."""
        self._worker = worker

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler."""
        assert isinstance(connector_metadata, OffloadPackageMetadata)
        self._metadata = connector_metadata
        print(f"[WORKER DEBUG] bind_connector_metadata: load_event={connector_metadata.load_event}, store_event={connector_metadata.store_event}, request_ids={connector_metadata.request_ids}, store_gpu_blocks={connector_metadata.store_gpu_blocks}", flush=True)

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata."""
        self._metadata = None
        print(f"[WORKER DEBUG] clear_connector_metadata", flush=True)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Handle preempted requests."""
        pass

    def start_load_kv(self, forward_context: ForwardContext) -> None:
        """Launch async CPU→GPU copies for staging (sparse attention fetch)."""
        # Try to sync blocks before staging
        self._try_sync_blocks()
        print(f"[WORKER DEBUG] start_load_kv called: metadata={self._metadata is not None}", flush=True)
        if self._metadata is None:
            print(f"[WORKER DEBUG] start_load_kv: early return - no metadata", flush=True)
            return
        # Use a dummy load_event if not set by scheduler
        load_event = self._metadata.load_event if self._metadata.load_event != -1 else 0

        if self._orchestrator is None:
            print(f"[WORKER DEBUG] start_load_kv: no orchestrator", flush=True)
            return

        # Get cached sparse pattern from Skylight backend (from previous step)
        sparse_idx, sparse_len, request_ids = self._get_cached_sparse_pattern()
        # Fallback to metadata's request_ids if not in cache
        if (request_ids is None or len(request_ids) == 0) and self._metadata is not None:
            request_ids = self._metadata.request_ids
        print(f"[WORKER DEBUG] cached pattern: sparse_idx={sparse_idx is not None}, sparse_len={sparse_len is not None}, request_ids={request_ids}", flush=True)
        if sparse_idx is None or sparse_len is None or request_ids is None or len(request_ids) == 0:
            # No cached pattern yet (first step) - skip staging for this step
            print(f"[WORKER DEBUG] start_load_kv: no cached pattern, skipping", flush=True)
            return

        print(f"[WORKER DEBUG] sparse_idx shape={sparse_idx.shape}, sparse_len shape={sparse_len.shape}", flush=True)

        # Use orchestrator's staging pool to fetch blocks
        print(f"[WORKER DEBUG] prepare_staging: request_ids={request_ids}, sparse_idx.shape={sparse_idx.shape}, sparse_len.shape={sparse_len.shape}", flush=True)
        staging_map = self._orchestrator.prepare_staging(
            request_ids,
            sparse_idx.to(self._orchestrator.staging_pool.gpu_k_caches[0].device),
            sparse_len.to(self._orchestrator.staging_pool.gpu_k_caches[0].device),
        )
        print(f"[WORKER DEBUG] staging_map: {staging_map}", flush=True)

        # Launch async copies for each block
        for ref, staging_slot in staging_map.items():
            loc = self._orchestrator.residency.get(ref)
            if loc and loc.cpu_slot is not None:
                print(f"[WORKER DEBUG] Fetching block {ref} (cpu_slot={loc.cpu_slot}) to staging_slot={staging_slot}", flush=True)
                self._launch_cpu_to_staging_copy(loc.cpu_slot, staging_slot)
            else:
                print(f"[WORKER DEBUG] Block {ref} not on CPU or no cpu_slot: loc={loc}", flush=True)

        # Record completion event
        event = torch.cuda.Event()
        event.record(self._load_stream)
        self._load_events.append((load_event, event))

    def _get_cached_sparse_pattern(self):
        """Get cached sparse pattern from Skylight backend."""
        import os
        try:
            # Import the module to access module-level cached pattern
            import skylight.backend as skylight_backend
            sparse_idx = skylight_backend._sparse_pattern_cache.get('sparse_idx', None)
            sparse_len = skylight_backend._sparse_pattern_cache.get('sparse_len', None)
            request_ids = skylight_backend._sparse_pattern_cache.get('request_ids', None)

            print(f"[WORKER DEBUG] pid={os.getpid()}, _get_cached_sparse_pattern: skylight_backend id={id(skylight_backend)}, sparse_idx id={id(sparse_idx)}", flush=True)
            print(f"[WORKER DEBUG] _cached_sparse_idx={sparse_idx is not None}, _cached_sparse_len={sparse_len is not None}, _cached_request_ids={request_ids}", flush=True)

            # If cached request_ids not available, try to get from model runner
            if request_ids is None and self._worker is not None:
                if hasattr(self._worker, 'model_runner') and self._model_runner is not None:
                    runner = self._model_runner
                    if hasattr(runner, 'execute_model_state') and runner.execute_model_state is not None:
                        input_batch = runner.execute_model_state.input_batch
                        if input_batch is not None and hasattr(input_batch, 'req_ids'):
                            request_ids = input_batch.req_ids
                            print(f"[WORKER DEBUG] Got request_ids from model runner: {request_ids}", flush=True)
                            # Store in cache for next time
                            skylight_backend._sparse_pattern_cache['request_ids'] = request_ids

            return sparse_idx, sparse_len, request_ids
        except Exception as e:
            logger.warning(f"Could not get cached sparse pattern: {e}")
            return None, None, None

    def _launch_cpu_to_staging_copy(self, cpu_slot: int, staging_slot: int):
        """Launch async copy from CPU cache to GPU staging slot."""
        if self._orchestrator is None:
            return
        # Access staging pool and copy
        staging_pool = self._orchestrator.staging_pool
        with torch.cuda.stream(self._load_stream):
            for layer, (k_cache, v_cache) in enumerate(zip(
                staging_pool.gpu_k_caches, staging_pool.gpu_v_caches
            )):
                # CPU cache: [page_size, num_kv_heads, head_dim] -> GPU: [num_kv_heads, page_size, 2*head_dim]
                k_src = staging_pool.cpu_cache[layer, cpu_slot, 0]  # [page_size, num_kv_heads, head_dim]
                v_src = staging_pool.cpu_cache[layer, cpu_slot, 1]  # [page_size, num_kv_heads, head_dim]
                
                # Transpose from [page_size, num_kv_heads, head_dim] to [num_kv_heads, page_size, head_dim]
                k_src = k_src.permute(1, 0, 2).contiguous()
                v_src = v_src.permute(1, 0, 2).contiguous()
                
                # Combine K and V in last dimension: [num_kv_heads, page_size, 2*head_dim]
                combined = torch.cat([k_src, v_src], dim=-1)
                
                # Copy to GPU staging (K and V are combined in the same tensor)
                k_cache[staging_pool.base + staging_slot].copy_(combined, non_blocking=True)
                # v_cache is not used since K and V are combined
        # Ensure visibility to default stream
        torch.cuda.current_stream().wait_stream(self._load_stream)

    def wait_for_save(self) -> None:
        """Launch async GPU→CPU copies for offloaded blocks."""
        print(f"[WORKER DEBUG] wait_for_save called: store_event={self._metadata.store_event if self._metadata else None}, store_gpu_blocks={self._metadata.store_gpu_blocks if self._metadata else None}", flush=True)
        if self._metadata is None or self._metadata.store_event == -1:
            return
        if not self._metadata.store_gpu_blocks:
            return

        if self._orchestrator is None:
            return

        # Wait for compute to finish before reading KV cache
        compute_done = torch.cuda.Event()
        compute_done.record(torch.cuda.current_stream())
        self._store_stream.wait_event(compute_done)

        # Launch copies for each block
        for gpu_block_id, cpu_block_id in zip(
            self._metadata.store_gpu_blocks, self._metadata.store_cpu_blocks
        ):
            print(f"[WORKER DEBUG] Copying gpu_block={gpu_block_id} to cpu_block", flush=True)
            self._orchestrator.offload_manager._copy_gpu_to_cpu(gpu_block_id)

        # Record completion event
        event = torch.cuda.Event()
        event.record(self._store_stream)
        self._store_events.append((self._metadata.store_event, event))
        self._pending_store_event_indices.add(self._metadata.store_event)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Poll for completed transfers and return finished request IDs."""
        finished_recving: set[str] = set()

        # Check load events (staging)
        if self._pending_load_event_indices:
            load_wm = self._poll_events(is_store=False)
            for event_idx in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(event_idx)
                self._completed_load_events[event_idx] = 1
                # Map event to request IDs
                if self._metadata and event_idx in self._metadata.load_event_to_reqs:
                    finished_recving.update(self._metadata.load_event_to_reqs[event_idx])

        # Check store events (offload)
        if self._pending_store_event_indices:
            store_wm = self._poll_events(is_store=True)
            for event_idx in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(event_idx)
                self._completed_store_events[event_idx] = 1

        return None, finished_recving or None

    def _poll_events(self, is_store: bool) -> int:
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm

    def build_worker_meta(self) -> KVConnectorWorkerMetadata:
        """Build metadata to send back to scheduler."""
        if not self._completed_store_events and not self._completed_load_events:
            return None
        meta = OffloadPackageWorkerMetadata(
            completed_store_events=self._completed_store_events,
            completed_load_events=self._completed_load_events,
        )
        self._completed_store_events = {}
        self._completed_load_events = {}
        return meta

    def flush_and_sync(self) -> None:
        """Wait for all pending transfers to complete."""
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()
        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    # Abstract methods from KVConnectorBase_V1 (minimal implementations for testing)
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the KV for a specific layer is loaded."""
        # For testing, we don't do layer-by-layer pipelining
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Start saving a layer of KV cache to the connector."""
        # For testing, we don't do layer-by-layer saves
        pass

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Get number of new matched tokens for prefix caching."""
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Update state after block allocation."""
        pass

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes."""
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request finishes in all KV cache groups."""
        return self.request_finished(request, block_ids=[])

    def shutdown(self) -> None:
        """Shutdown the connector."""
        self.flush_and_sync()

    def get_handshake_metadata(self):
        """Get handshake metadata for P/D workers."""
        return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def build_connector_meta(self) -> KVConnectorWorkerMetadata | None:
        """Build worker metadata for the scheduler."""
        return self.build_worker_meta()

    def _sync_blocks_from_runner(self) -> None:
        """Sync block IDs from model runner's block table to orchestrator."""
        print(f"[SYNC DEBUG] _sync_blocks_from_runner called, id={id(self)}", flush=True)
        if self._model_runner is None or self._orchestrator is None:
            print(f"[SYNC DEBUG] model_runner or orchestrator is None", flush=True)
            return
        runner = self._model_runner
        # runner IS the model_runner
        if not hasattr(runner, 'block_tables') or not runner.block_tables:
            print(f"[SYNC DEBUG] no block_tables", flush=True)
            return
        
        block_tables = runner.block_tables
        if not hasattr(block_tables, 'block_tables') or not block_tables.block_tables:
            print(f"[SYNC DEBUG] no block_tables.block_tables", flush=True)
            return
        
        # Get block tables for the first KV cache group (group 0)
        group_0_table = block_tables.block_tables[0]
        num_blocks_tensor = block_tables.num_blocks
        
        # Get request IDs from the model runner's execute_model_state
        if not hasattr(runner, 'execute_model_state') or runner.execute_model_state is None:
            print(f"[SYNC DEBUG] no execute_model_state", flush=True)
            return
        input_batch = runner.execute_model_state.input_batch
        if not hasattr(input_batch, 'req_ids'):
            print(f"[SYNC DEBUG] no req_ids", flush=True)
            return
        
        req_ids = input_batch.req_ids
        print(f"[SYNC DEBUG] req_ids={req_ids}", flush=True)
        for req_idx, req_id in enumerate(req_ids):
            # Get number of blocks for this request in group 0
            num_blocks = int(num_blocks_tensor.np[0, req_idx]) if num_blocks_tensor.np.ndim == 2 else 0
            print(f"[SYNC DEBUG] req_idx={req_idx}, req_id={req_id}, num_blocks={num_blocks}", flush=True)
            if num_blocks == 0:
                continue
            
            # Get block IDs for this request
            block_ids = group_0_table.gpu[req_idx, :num_blocks].tolist()
            print(f"[SYNC DEBUG] block_ids={block_ids}", flush=True)
            
            # Sync with orchestrator
            self._orchestrator.sync_request_blocks(req_id, block_ids, 0)
            # Mark as complete since we're in decode phase
            for block_idx, gpu_block_id in enumerate(block_ids):
                ref = BlockRef(str(req_id), block_idx)
                loc = self._orchestrator.residency.get(ref)
                if loc and not loc.complete:
                    self._orchestrator.residency.mark_complete(ref)
        
        req_ids = input_batch.req_ids
        print(f"[SYNC DEBUG] req_ids={req_ids}", flush=True)
        for req_idx, req_id in enumerate(req_ids):
            # Get number of blocks for this request in group 0
            num_blocks = int(num_blocks_tensor.np[0, req_idx]) if num_blocks_tensor.np.ndim == 2 else 0
            print(f"[SYNC DEBUG] req_idx={req_idx}, req_id={req_id}, num_blocks={num_blocks}", flush=True)
            if num_blocks == 0:
                continue
            
            # Get block IDs for this request
            block_ids = group_0_table.gpu[req_idx, :num_blocks].tolist()
            print(f"[SYNC DEBUG] block_ids={block_ids}", flush=True)
            
            # Sync with orchestrator
            self._orchestrator.sync_request_blocks(req_id, block_ids, 0)
            # Mark as complete since we're in decode phase
            for block_idx, gpu_block_id in enumerate(block_ids):
                ref = BlockRef(str(req_id), block_idx)
                loc = self._orchestrator.residency.get(ref)
                if loc and not loc.complete:
                    self._orchestrator.residency.mark_complete(ref)

    def _try_sync_blocks(self) -> None:
        """Try to sync blocks, retrying if model_runner not ready."""
        print(f"[TRY SYNC DEBUG] _try_sync_blocks called, id={id(self)}, _model_runner={id(self._model_runner) if self._model_runner else None}", flush=True)
        if self._model_runner is None or self._orchestrator is None:
            print(f"[TRY SYNC DEBUG] model_runner or orchestrator is None", flush=True)
            return
        runner = self._model_runner
        # runner IS the model_runner, no need to check for model_runner attribute
        if not hasattr(runner, 'block_tables') or not runner.block_tables:
            print(f"[TRY SYNC DEBUG] no block_tables", flush=True)
            return
        block_tables = runner.block_tables
        print(f"[TRY SYNC DEBUG] block_tables={block_tables}, block_tables.block_tables={block_tables.block_tables}", flush=True)
        if not hasattr(block_tables, 'block_tables') or not block_tables.block_tables:
            print(f"[TRY SYNC DEBUG] no block_tables.block_tables", flush=True)
            return
        ems = getattr(runner, 'execute_model_state', None)
        print(f"[TRY SYNC DEBUG] execute_model_state={ems}", flush=True)
        if not hasattr(runner, 'execute_model_state') or runner.execute_model_state is None:
            # execute_model_state not ready yet (e.g., during warmup), skip this sync
            print(f"[TRY SYNC DEBUG] execute_model_state not ready, skipping sync", flush=True)
            return
        # All checks passed, do the sync
        self._sync_blocks_from_runner()
        input_batch = runner.execute_model_state.input_batch
        if not hasattr(input_batch, 'req_ids') or not input_batch.req_ids:
            print(f"[TRY SYNC DEBUG] no req_ids", flush=True)
            return
        # All checks passed, do the sync
        print(f"[TRY SYNC DEBUG] All checks passed, calling _sync_blocks_from_runner", flush=True)
        self._sync_blocks_from_runner()

    def set_runner(self, runner) -> None:
        """Bind the GPU model runner for accessing request IDs and block tables."""
        print(f"[SET RUNNER DEBUG] set_runner called, id={id(self)}, runner={type(runner)}", flush=True)
        # vLLM calls this with the model_runner (self), not the worker
        self._model_runner = runner
        print(f"[SET RUNNER DEBUG] _model_runner set to {type(self._model_runner)}, id={id(self._model_runner)}", flush=True)
        if self._orchestrator is None:
            self._build_orchestrator()
        # Try to sync blocks (will succeed when model_runner is ready)
        self._try_sync_blocks()
