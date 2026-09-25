# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side manager for OffloadPackageConnector."""

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    AttentionSpec,
)
from offload_package.integration.metadata import INVALID_JOB_ID
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from offload_package.config import OffloadConfig
from offload_package.controller import KVCompressionController
from offload_package.integration.manager import OffloadOrchestrator
from offload_package.integration.vllm_bridge import initialize_for_runner
from offload_package.offload.manager import KVOffloadManager
from offload_package.selection.adapter import (
    collect_needed_block_refs,
    patch_paged_kv_indices_with_staging,
)
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable
from offload_package.vllm_adapter import KVCacheView
from offload_package.shared.types import (
    OrchestratorConfig,
    build_orchestrator,
    normalize_runner_kv_caches,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


class OffloadPackageScheduler:
    """Scheduler-side offloading manager. Binds GPU BlockPool for memory savings."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        offload_config: OffloadConfig,
        worker_handler: "OffloadPackageWorker | None" = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.offload_config = offload_config
        self._worker_handler = worker_handler
        
        # GPU block pool (bound later via bind_gpu_block_pool)
        self._gpu_block_pool: BlockPool | None = None
        
        # CPU-side coordinator and block pool for receiving offloaded blocks
        cpu_kv_cache_config = self._derive_cpu_config(kv_cache_config)
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=vllm_config.parallel_config.decode_context_parallel_size,
            pcp_world_size=1,
            scheduler_block_size=self._get_scheduler_block_size(kv_cache_config),
            hash_block_size=self._get_hash_block_size(kv_cache_config),
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        
        # Orchestrator for scoring/offloading logic (initialized from worker)
        self._orchestrator: OffloadOrchestrator | None = None
        
        # Per-request decode token tracking for interval-based scoring
        self._controller = KVCompressionController(
            W=offload_config.interval_tokens,
            page_size=self._get_page_size(kv_cache_config),
        )
        
        # Event counters
        self._store_event_counter = 0
        self._load_event_counter = 0
        
        # Pending store events: event_idx -> (gpu_block_ids, cpu_block_ids, req_ids)
        self._pending_store_events: dict[int, tuple[list[int], list[int], list[str]]] = {}
        # Pending load events: event_idx -> (staging_slot_ids, cpu_block_ids, req_ids)
        self._pending_load_events: dict[int, tuple[list[int], list[int], list[str]]] = {}
        
        # World size for multi-worker completion tracking
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = defaultdict(int)
        
        # Track prefill completion per request
        self._prefill_completed: set[str] = set()
        # Track prompt lengths for prefill completion detection
        self._prompt_lens: dict[str, int] = {}
        # Pending store metadata to include in next metadata
        self._pending_store_metadata: dict = {}
        
    def _derive_cpu_config(self, gpu_config) -> "KVCacheConfig":
        """Derive CPU KVCacheConfig from GPU config, scaling num_blocks by CPU/GPU ratio."""
        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
        
        gpu_total_bytes = gpu_config.kv_cache_tensors[0].size
        num_gpu_blocks = gpu_config.num_blocks
        bytes_per_block = gpu_total_bytes // num_gpu_blocks
        
        cpu_num_blocks = self.offload_config.num_cpu_slots
        cpu_total_bytes = cpu_num_blocks * bytes_per_block
        
        # Create new KVCacheConfig with CPU capacity
        cpu_kv_cache_tensors = [
            KVCacheTensor(
                size=cpu_total_bytes,
                layers=t.layers,
                layer_stride=t.layer_stride,
                block_stride=t.block_stride,
                offset=t.offset,
            )
            for t in gpu_config.kv_cache_tensors
        ]
        
        return KVCacheConfig(
            num_blocks=cpu_num_blocks,
            kv_cache_tensors=cpu_kv_cache_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )
        
    def _get_scheduler_block_size(self, kv_cache_config) -> int:
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
        return resolve_kv_cache_block_sizes(kv_cache_config, self.vllm_config)[0]
        
    def _get_hash_block_size(self, kv_cache_config) -> int:
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
        return resolve_kv_cache_block_sizes(kv_cache_config, self.vllm_config)[1]
        
    def _get_page_size(self, kv_cache_config) -> int:
        return kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        
    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU BlockPool - enables freeing blocks after offload."""
        self._gpu_block_pool = gpu_block_pool
        logger.info("OffloadPackageScheduler: Bound GPU BlockPool (%d blocks)",
                    gpu_block_pool.num_gpu_blocks)
        
    def initialize_orchestrator(self, runner) -> None:
        """Called from worker after KV caches initialized to set up orchestrator."""
        self._orchestrator = initialize_for_runner(
            runner,
            normal_gpu_pages=self.kv_cache_config.num_blocks - self.offload_config.num_staging_slots,
            block_size=self._get_page_size(self.kv_cache_config),
            offload_config=self.offload_config,
        )
        logger.info("OffloadPackageScheduler: Orchestrator initialized")
        
    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """Build metadata for this step's offload/staging operations."""
        print(f"[SCHEDULER DEBUG] build_connector_meta called, orchestrator={self._orchestrator is not None}, gpu_block_pool={self._gpu_block_pool is not None}", flush=True)
        # Get orchestrator from worker handler
        if self._worker_handler is not None:
            self._orchestrator = self._worker_handler._orchestrator
        if self._orchestrator is None or self._gpu_block_pool is None:
            from offload_package.integration.metadata import OffloadPackageMetadata
            return OffloadPackageMetadata()
        
        from offload_package.integration.metadata import OffloadPackageMetadata
        metadata = OffloadPackageMetadata()
        
        # Include pending store metadata from previous step
        if self._pending_store_metadata:
            metadata.store_event = self._pending_store_metadata.get('store_event', INVALID_JOB_ID)
            metadata.store_gpu_blocks = self._pending_store_metadata.get('store_gpu_blocks', [])
            metadata.store_cpu_blocks = self._pending_store_metadata.get('store_cpu_blocks', [])
            metadata.store_event_to_reqs = self._pending_store_metadata.get('store_event_to_reqs', {})
            print(f"[SCHEDULER DEBUG] Included pending store metadata: store_event={metadata.store_event}, store_gpu_blocks={metadata.store_gpu_blocks}", flush=True)
        
        # 1. Track new requests and update block mappings
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            block_ids = new_req_data.block_ids[0] if new_req_data.block_ids else []
            self._orchestrator.sync_request_blocks(req_id, block_ids, 0)
            if req_id not in self._prefill_completed:
                self._controller.on_prefill_complete(req_id)
            # Store prompt length for prefill completion detection
            self._prompt_lens[req_id] = new_req_data.prompt_len
        
        # 2. Update existing requests
        for req_id, num_computed, new_block_ids in zip(
            scheduler_output.scheduled_cached_reqs.req_ids,
            scheduler_output.scheduled_cached_reqs.num_computed_tokens,
            scheduler_output.scheduled_cached_reqs.new_block_ids,
        ):
            block_ids = new_block_ids[0] if new_block_ids else []
            self._orchestrator.sync_request_blocks(req_id, block_ids, num_computed)
        
            # Check for prefill completion (transition from prefill to decode)
            prompt_len = self._prompt_lens.get(req_id, 0)
            if prompt_len > 0 and num_computed >= prompt_len and req_id not in self._prefill_completed:
                self._prefill_completed.add(req_id)
                self._orchestrator.on_prefill_complete(req_id)
                self._trigger_global_offload(metadata)
        
            # Track decode tokens for interval-based scoring
            # Use actual scheduled tokens for this request
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_scheduled > 0:
                should_offload = self._controller.on_decode(req_id, num_scheduled)
                print(f"[SCHEDULER DEBUG] on_decode req_id={req_id}, num_scheduled={num_scheduled}, should_offload={should_offload}", flush=True)
                if should_offload:
                    print(f"[SCHEDULER DEBUG] Calling _trigger_global_offload", flush=True)
                    try:
                        self._trigger_global_offload(metadata)
                        print(f"[SCHEDULER DEBUG] _trigger_global_offload returned normally", flush=True)
                    except Exception as e:
                        print(f"[SCHEDULER DEBUG] _trigger_global_offload failed: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
        
        # 3. Handle prefill completions (detect when prefill done)
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            prompt_len = new_req_data.prompt_len
            num_computed = new_req_data.num_computed_tokens
            if num_computed >= prompt_len and req_id not in self._prefill_completed:
                self._prefill_completed.add(req_id)
                self._orchestrator.on_prefill_complete(req_id)
                # Also trigger offload after prefill
                self._trigger_global_offload(metadata)
        
        # 4. Handle staging loads for sparse attention
        active_req_ids = self._get_active_decode_requests(scheduler_output)
        if active_req_ids:
            self._prepare_staging_loads(metadata, active_req_ids)
        
        print(f"[SCHEDULER DEBUG] build_connector_meta returning: load_event={metadata.load_event}, store_event={metadata.store_event}, store_gpu_blocks={metadata.store_gpu_blocks}", flush=True)
        return metadata
        
        # 4. Handle staging loads for sparse attention
        active_req_ids = self._get_active_decode_requests(scheduler_output)
        if active_req_ids:
            self._prepare_staging_loads(metadata, active_req_ids)
        
        print(f"[SCHEDULER DEBUG] build_connector_meta returning: load_event={metadata.load_event}, store_event={metadata.store_event}, store_gpu_blocks={metadata.store_gpu_blocks}", flush=True)
        return metadata
        
    def _get_active_decode_requests(self, scheduler_output: SchedulerOutput) -> list[str]:
        """Get request IDs that are in decode phase."""
        active = []
        print(f"[SCHEDULER DEBUG] _get_active_decode_requests: scheduled_cached_reqs={scheduler_output.scheduled_cached_reqs.req_ids}, prefill_completed={self._prefill_completed}", flush=True)
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            if req_id in self._prefill_completed:
                active.append(req_id)
        print(f"[SCHEDULER DEBUG] _get_active_decode_requests returning: {active}", flush=True)
        return active
        
    def _trigger_global_offload(self, metadata):
        """Score all complete GPU blocks globally and offload fraction.
        
        Re-scores ALL complete blocks every W steps to capture changing importance.
        """
        print(f"[SCHEDULER DEBUG] _trigger_global_offload called, orchestrator={self._orchestrator is not None}", flush=True)
        if self._orchestrator is None:
            # Try to get from worker handler
            if self._worker_handler is not None:
                self._orchestrator = self._worker_handler._orchestrator
                print(f"[SCHEDULER DEBUG] Got orchestrator from worker handler: {self._orchestrator is not None}", flush=True)
            if self._orchestrator is None:
                print(f"[SCHEDULER DEBUG] _trigger_global_offload: orchestrator is None", flush=True)
                return
        
        # Get all complete GPU blocks across all requests
        all_gpu_complete = self._orchestrator.residency.all_gpu_blocks(complete_only=True)
        print(f"[SCHEDULER DEBUG] _trigger_global_offload: all_gpu_complete={len(all_gpu_complete)}, _scored={len(self._orchestrator._scored)}", flush=True)
        
        # Re-score ALL complete blocks (not just unscored) to capture changing importance
        for ref in all_gpu_complete:
            loc = self._orchestrator.residency.get(ref)
            if loc and loc.gpu_block_id is not None:
                self._orchestrator.offload_manager.on_block_full(ref)
                # Mark as scored for this interval (but we'll re-score next interval)
                self._orchestrator._scored.add(ref)
        
        # Offload fraction globally
        offloaded = self._orchestrator.offload_manager.offload_fraction(
            all_gpu_complete,
            self.offload_config.offload_fraction,
            self._orchestrator._scored
        )
        
        if not offloaded:
            return
        
        # Allocate CPU blocks
        cpu_blocks = self.cpu_block_pool.get_new_blocks(len(offloaded))
        cpu_block_ids = [blk.block_id for blk in cpu_blocks]
        gpu_block_ids = []
        for ref in offloaded:
            loc = self._orchestrator.residency.get(ref)
            if loc and loc.gpu_block_id is not None:
                gpu_block_ids.append(loc.gpu_block_id)
        
        if len(gpu_block_ids) != len(offloaded):
            logger.warning("Some offloaded blocks missing GPU mapping")
            return
        
        # Stamp block hashes on CPU blocks for prefix caching
        for cpu_blk, ref in zip(cpu_blocks, offloaded):
            loc = self._orchestrator.residency.get(ref)
            if loc and loc.gpu_block_id is not None:
                gpu_blk = self._gpu_block_pool.blocks[loc.gpu_block_id]
                if gpu_blk.block_hash:
                    cpu_blk._block_hash = gpu_blk.block_hash
        
        # Record store event
        event_idx = self._store_event_counter
        self._store_event_counter += 1
        
        metadata.store_event = event_idx
        metadata.store_gpu_blocks = gpu_block_ids
        metadata.store_cpu_blocks = cpu_block_ids
        metadata.store_event_to_reqs[event_idx] = list(set(r.request_id for r in offloaded))
        print(f"[SCHEDULER DEBUG] _trigger_global_offload set store_event={metadata.store_event}, store_gpu_blocks={metadata.store_gpu_blocks}", flush=True)
        
        # Track for completion
        self._pending_store_events[event_idx] = (gpu_block_ids, cpu_block_ids,
                                                  metadata.store_event_to_reqs[event_idx])
        self._store_event_pending_counts[event_idx] = 0
        
        # Touch GPU blocks to prevent premature freeing during async copy
        self._gpu_block_pool.touch([self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids])
        
        logger.info("OffloadPackageScheduler: Offloaded %d blocks (event %d)",
                    len(offloaded), event_idx)
        
        # Store metadata for next step's worker
        self._pending_store_metadata = {
            'store_event': event_idx,
            'store_gpu_blocks': gpu_block_ids,
            'store_cpu_blocks': cpu_block_ids,
            'store_event_to_reqs': dict(metadata.store_event_to_reqs),
        }
        
    def _prepare_staging_loads(self, metadata, active_req_ids):
        """Prepare CPU->GPU staging loads for sparse attention."""
        # This will be populated by the worker after sparse kernel determines
        # which tokens to attend to. For now, we set up the structure.
        metadata.request_ids = active_req_ids
        # Assign a load event ID for tracking
        event_idx = self._load_event_counter
        self._load_event_counter += 1
        metadata.load_event = event_idx
        metadata.load_event_to_reqs[event_idx] = active_req_ids
        print(f"[SCHEDULER DEBUG] _prepare_staging_loads: active_req_ids={active_req_ids}, load_event={event_idx}, metadata_id={id(metadata)}", flush=True)
        print(f"[SCHEDULER DEBUG] metadata after: load_event={metadata.load_event}, request_ids={metadata.request_ids}", flush=True)
        
    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Process completed transfers from worker."""
        # Handle load completions (staging)
        for req_id in connector_output.finished_recving or []:
            # Find and clean up load event
            for event_idx, reqs in list(self._pending_load_events.items()):
                if req_id in reqs[2]:
                    # Clean up staging slot
                    staging_slot_ids = reqs[0]
                    self._orchestrator.staging_pool.release(req_id)  # Simplified
                    del self._pending_load_events[event_idx]
                    break
        
        # Handle store completions (offload)
        worker_meta = connector_output.kv_connector_worker_meta
        if worker_meta and hasattr(worker_meta, 'completed_store_events'):
            for event_idx, count in worker_meta.completed_store_events.items():
                self._store_event_pending_counts[event_idx] += count
                if self._store_event_pending_counts[event_idx] >= self._expected_worker_count:
                    self._process_store_completion(event_idx)
                    del self._store_event_pending_counts[event_idx]
        
    def _process_store_completion(self, event_idx: int):
        """Called when all workers report store completion - FREE GPU blocks."""
        if event_idx not in self._pending_store_events:
            return
        
        gpu_block_ids, cpu_block_ids, req_ids = self._pending_store_events.pop(event_idx)
        
        # Register CPU blocks in CPU pool cache (for prefix caching)
        for cpu_blk_id in cpu_block_ids:
            cpu_blk = self.cpu_block_pool.blocks[cpu_blk_id]
            if cpu_blk.block_hash:
                self.cpu_block_pool.cached_block_hash_to_block.insert(cpu_blk.block_hash, cpu_blk)
        self.cpu_block_pool.free_blocks([self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids])
        
        # CRITICAL: Free GPU blocks back to BlockPool - THIS SAVES MEMORY
        if self._gpu_block_pool:
            gpu_blocks = [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
            self._gpu_block_pool.free_blocks(gpu_blocks)
            logger.debug("Freed %d GPU blocks to pool after offload", len(gpu_block_ids))
        
    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int):
        # Not used for our offload pattern
        return 0, False
        
    def update_state_after_alloc(self, request: Request, blocks: "KVCacheBlocks", num_external_tokens: int):
        # Not used
        pass
        
    def request_finished(self, request: Request, block_ids: list[int]):
        req_id = request.request_id
        # Clean up orchestrator state
        if self._orchestrator:
            self._orchestrator.on_request_done(req_id)
        # Clean up controller
        self._controller.remove(req_id)
        self._prefill_completed.discard(req_id)
        return False, None
        
    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ):
        return self.request_finished(request, block_ids=[])
        
    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()
        
    def reset(self) -> bool:
        # Clean up pending transfers
        return self.cpu_block_pool.reset_prefix_cache() 
        