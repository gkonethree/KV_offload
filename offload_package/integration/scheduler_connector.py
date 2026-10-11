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
from offload_package import timing as _timing
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from offload_package.config import OffloadConfig
from offload_package.debug import dbg as _dbg
from offload_package.controller import KVCompressionController
from offload_package.integration.manager import OffloadOrchestrator
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

        # World size for multi-worker completion tracking
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = defaultdict(int)
        
        # Track prefill completion per request
        self._prefill_completed: set[str] = set()
        # Track prompt lengths for prefill completion detection
        self._prompt_lens: dict[str, int] = {}
        # Per-step accumulator for offload batches: each entry is
        # (refs, gpu_block_ids, cpu_block_ids, req_ids). Multiple triggers within
        # one build_connector_meta call are flushed into a SINGLE store event
        # at the end of the step so the worker sees all of them.
        self._step_store_acc: list[tuple[list, list[int], list[int], list[str]]] = []
        # Pending store events: event_idx -> (refs, gpu_block_ids, cpu_block_ids, req_ids)
        self._pending_store_events: dict[int, tuple[list, list[int], list[int], list[str]]] = {}
        # Pending load events: event_idx -> (staging_slot_ids, cpu_block_ids, req_ids)
        self._pending_load_events: dict[int, tuple[list[int], list[int], list[str]]] = {}
        # Full cumulative GPU block-id list per request (mirrors the model
        # runner's block table: append new_block_ids, replace on resume).
        self._req_block_table: dict[str, list[int]] = {}

        # Scheduler-side kv_cache_manager (bound via bind_gpu_block_pool);
        # needed for req_to_blocks surgery when truly freeing GPU blocks.
        self._kv_cache_manager = None

        # True freeing recycles GPU pages after offload. Only safe in
        # uni-process, single-worker, synchronous scheduling: with async
        # scheduling the next step may already be reading those pages when the
        # completion event is processed. With CUDA graphs the in-run fetch can't
        # run during replay, so freed pages must stay readable (shadow tier).
        import os as _os
        _async = bool(getattr(vllm_config.scheduler_config, "async_scheduling", False))
        _mp = _os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") not in ("0", "false", "False")
        _world = vllm_config.parallel_config.world_size
        _cg_mode = getattr(
            getattr(vllm_config, "compilation_config", None), "cudagraph_mode", None
        )
        _graphs = getattr(_cg_mode, "name", str(_cg_mode)).upper() not in (
            None, "NONE", "OFF",
        ) if _cg_mode is not None else False
        self._graphs_active = _graphs
        self._true_free_enabled = (
            offload_config.enabled
            and offload_config.true_free
            and not _async
            and not _mp
            and _world == 1
            and not _graphs
        )
        if not self._true_free_enabled:
            logger.info(
                "OffloadPackageScheduler: true GPU freeing disabled "
                "(async_scheduling=%s, multiprocess=%s, world_size=%s, graphs=%s); "
                "running in shadow-tier mode (no memory reclamation)",
                _async, _mp, _world, _graphs,
            )
        
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
        
    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool, kv_cache_manager=None) -> None:
        """Bind GPU BlockPool - enables freeing blocks after offload."""
        self._gpu_block_pool = gpu_block_pool
        self._kv_cache_manager = kv_cache_manager
        if self._true_free_enabled and kv_cache_manager is None:
            logger.warning(
                "OffloadPackageScheduler: kv_cache_manager not provided; "
                "true GPU freeing needs block-table surgery and is disabled"
            )
            self._true_free_enabled = False
        logger.info("OffloadPackageScheduler: Bound GPU BlockPool (%d blocks)",
                    gpu_block_pool.num_gpu_blocks)
        # Permanently reserve the tail pages for the staging pool so the
        # allocator never hands them to a request. Without this, staging slots
        # ([num_blocks - S, num_blocks)) overlap allocatable pages and fetch
        # writes corrupt live KV data (and vice versa).
        if self.offload_config.enabled and self.offload_config.num_staging_slots > 0:
            from offload_package.cache_allocator import reserve_staging_tail
            num_slots = self.offload_config.num_staging_slots
            reserved = reserve_staging_tail(gpu_block_pool, num_slots)
            # Cross-check against the worker-side staging base derived from the
            # block pool's actual capacity. Use the pool's block count (which
            # includes the staging tail) for both reservation and verification.
            pool_blocks = gpu_block_pool.num_gpu_blocks
            expected_base = pool_blocks - num_slots
            assert reserved[0] == expected_base, (
                f"staging tail mismatch: reserved starts at {reserved[0]}, "
                f"connector expects staging base {expected_base} "
                f"(num_slots={num_slots}, pool_blocks={pool_blocks})"
            )
            self._reserved_staging_ids = reserved
            logger.info(
                "OffloadPackageScheduler: reserved staging tail [%d, %d] "
                "(%d allocatable blocks remain)",
                reserved[0], reserved[-1], gpu_block_pool.get_num_free_blocks(),
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """Build metadata for this step's offload/staging operations."""
        try:
            _dbg(f"[SCHEDULER DEBUG] build_connector_meta called, orchestrator={self._orchestrator is not None}, gpu_block_pool={self._gpu_block_pool is not None}")
            # Get orchestrator from worker handler
            if self._worker_handler is not None:
                self._orchestrator = self._worker_handler._orchestrator
            if self._orchestrator is not None:
                # The Skylight builder null-fills CPU-resident paged indices only
                # when their original pages may have been freed.
                self._orchestrator.true_freeing = self._true_free_enabled
            if self._orchestrator is None or self._gpu_block_pool is None:
                from offload_package.integration.metadata import OffloadPackageMetadata
                return OffloadPackageMetadata()
            
            from offload_package.integration.metadata import OffloadPackageMetadata
            metadata = OffloadPackageMetadata()
        except Exception as e:
            import logging
            logging.warning(f"build_connector_meta early setup failed: {e}")
            from offload_package.integration.metadata import OffloadPackageMetadata
            return OffloadPackageMetadata()

        # 1. Track new requests and update block mappings
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            block_ids = new_req_data.block_ids[0] if new_req_data.block_ids else []
            self._req_block_table[req_id] = list(block_ids)
            self._orchestrator.sync_request_blocks(req_id, block_ids, new_req_data.num_computed_tokens)
            if req_id not in self._prefill_completed:
                self._controller.on_prefill_complete(req_id)
            # Store prompt length for prefill completion detection
            self._prompt_lens[req_id] = new_req_data.prompt_len
        
        # 2. Update existing requests
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, num_computed, new_block_ids in zip(
            cached.req_ids,
            cached.num_computed_tokens,
            cached.new_block_ids,
        ):
            # Reconstruct the FULL block list, mirroring the model runner's
            # block-table update: append new tail blocks, replace on resume
            # from preemption. new_block_ids is append-only in vLLM 0.29, so
            # syncing it directly would re-enumerate logical block_idx from 0
            # and never re-mark prefill blocks complete.
            new_ids = new_block_ids[0] if new_block_ids else None
            if req_id in cached.resumed_req_ids:
                full = list(new_ids) if new_ids is not None else []
            elif new_ids is not None:
                full = self._req_block_table.get(req_id, []) + list(new_ids)
            else:
                full = self._req_block_table.get(req_id, [])
            self._req_block_table[req_id] = full
            self._orchestrator.sync_request_blocks(req_id, full, num_computed)
        
            # Check for prefill completion (transition from prefill to decode)
            prompt_len = self._prompt_lens.get(req_id, 0)
            if prompt_len > 0 and num_computed >= prompt_len and req_id not in self._prefill_completed:
                self._prefill_completed.add(req_id)
                self._orchestrator.on_prefill_complete(req_id, num_computed)
                self._trigger_global_offload(metadata, scheduler_output)
        
            # Track decode tokens for interval-based scoring - ONLY for requests that completed prefill
            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_scheduled > 0 and req_id in self._prefill_completed:
                should_offload = self._controller.on_decode(req_id, num_scheduled)
                _dbg(f"[SCHEDULER DEBUG] on_decode req_id={req_id}, num_scheduled={num_scheduled}, should_offload={should_offload}")
                if should_offload:
                    _dbg(f"[SCHEDULER DEBUG] >>> TRIGGER OFFLOAD <<< req_id={req_id}")
                    self._trigger_global_offload(metadata, scheduler_output)
                    _dbg(f"[SCHEDULER DEBUG] _trigger_global_offload returned")
        
        # 3. Handle prefill completions (detect when prefill done)
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            prompt_len = new_req_data.prompt_len
            num_computed = new_req_data.num_computed_tokens
            if num_computed >= prompt_len and req_id not in self._prefill_completed:
                self._prefill_completed.add(req_id)
                self._orchestrator.on_prefill_complete(req_id, num_computed)
                # Also trigger offload after prefill
                self._trigger_global_offload(metadata, scheduler_output)
        
        # 4. Handle staging loads for sparse attention
        active_req_ids = self._get_active_decode_requests(scheduler_output)
        if active_req_ids:
            self._prepare_staging_loads(metadata, active_req_ids)
            metadata.request_ids = active_req_ids
            _dbg(f"[SCHEDULER DEBUG] Set metadata.request_ids = {active_req_ids}")

        # 5. Flush ALL of this step's offload triggers into a single store
        # event so the worker (and the completion free) sees every batch.
        if self._step_store_acc:
            refs_all = []
            gpu_ids: list[int] = []
            cpu_ids: list[int] = []
            req_ids: list[str] = []
            for r, g, c, q in self._step_store_acc:
                refs_all.extend(r)
                gpu_ids.extend(g)
                cpu_ids.extend(c)
                req_ids.extend(q)
            self._step_store_acc = []

            event_idx = self._store_event_counter
            self._store_event_counter += 1
            metadata.store_event = event_idx
            metadata.store_gpu_blocks = gpu_ids
            metadata.store_cpu_blocks = cpu_ids
            metadata.store_event_to_reqs[event_idx] = sorted(set(req_ids))
            self._pending_store_events[event_idx] = (
                refs_all, gpu_ids, cpu_ids, metadata.store_event_to_reqs[event_idx],
            )
            self._store_event_pending_counts[event_idx] = 0
            _dbg(f"[SCHEDULER DEBUG] Flushed store event {event_idx}: "
                 f"{len(gpu_ids)} gpu blocks, {len(cpu_ids)} cpu blocks, "
                 f"reqs={metadata.store_event_to_reqs[event_idx]}")

        _dbg(f"[SCHEDULER DEBUG] build_connector_meta returning: load_event={metadata.load_event}, store_event={metadata.store_event}, n_store_gpu_blocks={len(metadata.store_gpu_blocks)}")
        _timing.step()
        return metadata
        
    def _get_active_decode_requests(self, scheduler_output: Optional[SchedulerOutput] = None) -> list[str]:
        """Get request IDs that are in decode phase."""
        if scheduler_output is not None:
            _dbg(f"[SCHEDULER DEBUG] _get_active_decode_requests: scheduled_cached_reqs={scheduler_output.scheduled_cached_reqs.req_ids}, prefill_completed={self._prefill_completed}")
            active = []
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self._prefill_completed:
                    active.append(req_id)
            _dbg(f"[SCHEDULER DEBUG] _get_active_decode_requests returning: {active}")
            return active
        else:
            # Fallback: return prefill_completed requests
            active = list(self._prefill_completed)
            _dbg(f"[SCHEDULER DEBUG] _get_active_decode_requests (fallback): returning {active}")
            return active
        
    def _trigger_global_offload(self, metadata, scheduler_output=None):
        """Score all complete blocks globally and offload fraction.
        
        Re-scores ALL complete blocks every W steps to capture changing importance.
        For PQ mode, scores ALL blocks (including CPU-resident) using PQ codes
        which are permanently on GPU, so offload decisions match sparse attention.
        """
        _dbg(f"[SCHEDULER DEBUG] _trigger_global_offload called, orchestrator={self._orchestrator is not None}")
        if self._orchestrator is None:
            # Try to get from worker handler
            if self._worker_handler is not None:
                self._orchestrator = self._worker_handler._orchestrator
                _dbg(f"[SCHEDULER DEBUG] Got orchestrator from worker handler: {self._orchestrator is not None}")
            if self._orchestrator is None:
                _dbg(f"[SCHEDULER DEBUG] _trigger_global_offload: orchestrator is None")
                return
        
        all_gpu_complete = self._orchestrator.residency.all_gpu_blocks(complete_only=True)
        _dbg(f"[SCHEDULER DEBUG] _trigger_global_offload: all_gpu_complete={len(all_gpu_complete)}, _scored={len(self._orchestrator._scored)}")

        # Only sole-owned blocks are eligible: prefix-shared blocks (ref_cnt>1)
        # belong to several requests and must never be copied/freed under one
        # request's BlockRef.
        if self._true_free_enabled:
            allowed = []
            for ref in all_gpu_complete:
                loc = self._orchestrator.residency.get(ref)
                if loc is None or loc.gpu_block_id is None:
                    continue
                if self._gpu_block_pool.blocks[loc.gpu_block_id].ref_cnt == 1:
                    allowed.append(ref)
        else:
            allowed = all_gpu_complete

        # Re-score ALL complete blocks (not just unscored) to capture changing
        # importance — batched: one index_select per layer over ALL blocks and
        # one host sync total, instead of per-block*per-layer syncs.
        _score_refs, _score_gids = [], []
        for ref in allowed:
            loc = self._orchestrator.residency.get(ref)
            if loc and loc.gpu_block_id is not None:
                _score_refs.append(ref)
                _score_gids.append(loc.gpu_block_id)
        if _score_refs:
            _timing.start("sched.score")
            for ref in _score_refs:
                self._orchestrator.residency.mark_complete(ref)
            self._orchestrator.offload_manager.score_blocks(_score_refs, _score_gids)
            self._orchestrator._scored.update(_score_refs)
            _timing.end("sched.score")

        # Offload fraction of scored blocks
        _timing.start("sched.offload")
        offloaded = self._orchestrator.offload_manager.offload_fraction(
            allowed, self.offload_config.offload_fraction, self._orchestrator._scored
        )
        _timing.end("sched.offload")
    
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
        
        # Batch into this step's single store event, which is flushed at the
        # end of build_connector_meta — setting the event here directly would
        # lose earlier triggers' events when multiple fire in one step.
        # NOTE: no touch() here. The GPU→CPU copies inside offload_fraction
        # are event-synchronized, and with true freeing the request's own
        # reference is replaced by null_block at store completion; touching
        # would leave ref_cnt at 1 and blocks would never be reclaimed.
        self._step_store_acc.append((
            offloaded,
            gpu_block_ids,
            cpu_block_ids,
            list(set(r.request_id for r in offloaded)),
        ))

        logger.info("OffloadPackageScheduler: offloaded %d blocks (batched this step)",
                    len(offloaded))
        
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
        _dbg(f"[SCHEDULER DEBUG] _prepare_staging_loads: active_req_ids={active_req_ids}, load_event={event_idx}, metadata_id={id(metadata)}")
        _dbg(f"[SCHEDULER DEBUG] metadata after: load_event={metadata.load_event}, request_ids={metadata.request_ids}")
        
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
        """Called when the worker reports the store event complete.

        With true freeing enabled, the request's block-table entries for the
        offloaded pages are replaced with the pool's null_block (so vLLM never
        touches those page ids again), prefix-cache hashes are evicted, and the
        pages are finally returned to the free queue by ``free_blocks`` (their
        only remaining reference was the request's block table entry).
        """
        if event_idx not in self._pending_store_events:
            return
        _timing.start("sched.surgery")

        refs, gpu_block_ids, cpu_block_ids, req_ids = self._pending_store_events.pop(event_idx)
        _dbg(f"[SCHED DEBUG] _process_store_completion event={event_idx}, gpu_blocks={len(gpu_block_ids)}, cpu_blocks={len(cpu_block_ids)}")

        # Register CPU blocks in CPU pool cache (for prefix caching)
        for cpu_blk_id in cpu_block_ids:
            cpu_blk = self.cpu_block_pool.blocks[cpu_blk_id]
            if cpu_blk.block_hash:
                self.cpu_block_pool.cached_block_hash_to_block.insert(cpu_blk.block_hash, cpu_blk)
        self.cpu_block_pool.free_blocks([self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids])

        if not self._gpu_block_pool:
            return

        gpu_blocks = [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        if not self._true_free_enabled:
            # Shadow-tier mode: pages stay pinned to the request (no reclaim).
            _dbg(f"[SCHED DEBUG] true freeing disabled; {len(gpu_block_ids)} blocks keep GPU residency")
            return

        mgr = None
        if self._kv_cache_manager is not None:
            mgrs = self._kv_cache_manager.coordinator.single_type_managers
            mgr = mgrs[0] if mgrs else None
        if mgr is None:
            logger.warning("true freeing enabled but no kv_cache_manager; skipping free")
            return

        freed: list = []
        evict_ids: set[int] = set()
        for ref, gpu_id in zip(refs, gpu_block_ids):
            blocks = mgr.req_to_blocks.get(ref.request_id)
            if blocks is None:
                # Request already finished; vLLM recycled all its blocks.
                continue
            idx = ref.block_idx
            if idx >= len(blocks) or blocks[idx].block_id != gpu_id:
                # Table entry no longer points at the offloaded page (e.g.
                # preemption rewrote it); never free a mismatched block.
                logger.warning(
                    "offload table surgery mismatch for %s block %d (table=%s, gpu=%s); skipping",
                    ref.request_id, idx,
                    -1 if idx >= len(blocks) else blocks[idx].block_id, gpu_id,
                )
                continue
            blocks[idx] = mgr._null_block
            if idx < len(self._req_block_table.get(ref.request_id, [])):
                self._req_block_table[ref.request_id][idx] = mgr._null_block.block_id
            blk = self._gpu_block_pool.blocks[gpu_id]
            if blk.block_hash is not None:
                evict_ids.add(gpu_id)
            freed.append(blk)

        if evict_ids:
            # Evict hash entries so prefix lookups never resolve to dead pages.
            self._gpu_block_pool.evict_blocks(evict_ids)
        if freed:
            self._gpu_block_pool.free_blocks(freed)
            _dbg(f"[SCHED DEBUG] True-freed {len(freed)} GPU blocks: free_gpu_blocks={self._gpu_block_pool.get_num_free_blocks()}")
            logger.debug("True-freed %d GPU blocks to pool after offload", len(freed))
        _timing.end("sched.surgery")

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
        # Clean up controller and block-table bookkeeping
        self._controller.remove(req_id)
        self._prefill_completed.discard(req_id)
        self._req_block_table.pop(req_id, None)
        self._prompt_lens.pop(req_id, None)
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