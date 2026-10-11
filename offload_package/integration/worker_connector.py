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
from offload_package.debug import dbg as _dbg
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
            scorer_type=self.offload_config.scorer_type,
            staging_eviction=self.offload_config.staging_eviction,
        )

        # Get the PQ wrapper from the model runner's attention groups
        pq_wrapper = None
        if self._model_runner is not None:
            for attn_groups in getattr(self._model_runner, 'attn_groups', []) or []:
                for attn_group in attn_groups:
                    try:
                        builder = attn_group.get_metadata_builder()
                    except Exception:
                        continue
                    if builder is not None and hasattr(builder, '_maintainer'):
                        pq_wrapper = builder
                        break
                if pq_wrapper is not None:
                    break

        # Build orchestrator using the factory function
        self._orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)

        # Hand the runner to the skylight builder so build() can derive the
        # CURRENT step's decode request ids from the (post-reorder) input
        # batch. The scheduler-bound ids arrive one step late and their order
        # is not guaranteed to match decode row order in mixed batches.
        if self._model_runner is not None:
            try:
                import skylight.backend as skylight_backend
                skylight_backend.SkylightSparseMetadataBuilder._offload_runner = (
                    self._model_runner
                )
                logger.info("Injected model runner into SkylightSparseMetadataBuilder")
            except ImportError:
                logger.info("skylight not importable; runner id derivation disabled")

        # Config sanity: with true freeing the original GPU pages of offloaded
        # blocks are freed, so the same-step in-run fetch is the only correct
        # read path for selected cold blocks. Running without it means silent
        # reads of freed pages.
        if (
            self.offload_config.enabled
            and self.offload_config.true_free
            and os.environ.get("SKYLIGHT_SPARSE_METHOD", "oracle").strip().lower()
            in ("pq", "pq_optimized", "pqrs", "pqrs_optimized", "pqis", "pqis_optimized")
            and os.environ.get("SKYLIGHT_PQ_INRUN_FETCH", "1") != "1"
        ):
            logger.warning(
                "OffloadPackageWorker: true_free is enabled but "
                "SKYLIGHT_PQ_INRUN_FETCH is disabled -- selected offloaded "
                "blocks will be read from freed/null pages. This "
                "configuration is unsound; the in-run fetch will raise "
                "instead of corrupting silently."
            )

        # Inject orchestrator into attention metadata builders. The Skylight
        # sparse builder reads the class-level _offload_orchestrator in build()
        # to stage CPU blocks and patch paged indices before the decode plan.
        if self._orchestrator is not None and self._model_runner is not None:
            injected = 0
            for attn_groups in getattr(self._model_runner, 'attn_groups', []) or []:
                for attn_group in attn_groups:
                    try:
                        builder = attn_group.get_metadata_builder()
                    except Exception as e:
                        logger.warning("get_metadata_builder failed: %s", e)
                        continue
                    if builder is not None and hasattr(type(builder), '_offload_orchestrator'):
                        type(builder)._offload_orchestrator = self._orchestrator
                        injected += 1
            try:
                import skylight.backend as skylight_backend
                armed = (skylight_backend.SkylightSparseMetadataBuilder
                         ._offload_orchestrator is self._orchestrator)
                assert not injected or armed, (
                    "orchestrator injection failed: builders were found but the "
                    "Skylight builder class does not see the orchestrator"
                )
                logger.info("Orchestrator injected into %d attention builders (armed=%s)",
                            injected, armed)
            except ImportError:
                logger.info("Orchestrator injected into %d attention builders (no skylight)",
                            injected)

        logger.info("OffloadPackageWorker orchestrator built successfully")

    def bind_worker(self, worker) -> None:
        """Bind to the GPU worker for accessing model runner."""
        self._worker = worker

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        Request ids are NOT stashed here: the skylight builder derives the
        CURRENT step's decode order from the runner's input batch at build
        time (F1-A), which runs before this bind. Stashing here would only
        write a one-step-stale list that build() immediately overwrites.
        """
        assert isinstance(connector_metadata, OffloadPackageMetadata)
        self._metadata = connector_metadata
        _dbg(f"[WORKER DEBUG] bind_connector_metadata: load_event={connector_metadata.load_event}, store_event={connector_metadata.store_event}, n_request_ids={len(connector_metadata.request_ids)}, n_store_gpu_blocks={len(connector_metadata.store_gpu_blocks)}")

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata."""
        self._metadata = None
        _dbg(f"[WORKER DEBUG] clear_connector_metadata")

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Handle preempted requests."""
        pass

    def start_load_kv(self, forward_context: ForwardContext) -> None:
        """No-op: the fetch is owned by the attention path.

        PQ fetches at selection time (in-run, 0-lag) inside the decode
        wrapper; the oracle family prefetches at build time via the skylight
        builder's index patcher. There is no remaining worker-side fetch.
        """
        pass

    def wait_for_save(self) -> None:
        """Record store-event completion for the scheduler.

        The GPU→CPU copies were already launched AND synchronized on the
        scheduler side by offload_manager.offload_fraction() (which also
        marks residency CPU). Do NOT copy here again: that would allocate
        duplicate CPU slots and desync the residency table. This hook only
        tracks the store event so get_finished() can report completion.
        """
        if self._metadata is None or self._metadata.store_event == -1:
            return
        if not self._metadata.store_gpu_blocks:
            return

        if self._orchestrator is None:
            return

        # Record completion event (copies are already complete)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
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

    def set_runner(self, runner) -> None:
        """Bind the GPU model runner: the skylight builder reads it at build
        time to derive the current step's decode ids (F1-A); the orchestrator
        is built here once. Block syncing is scheduler-side (sync_request_blocks
        per step with num_computed); the worker-side sync path was dead."""
        _dbg(f"[SET RUNNER DEBUG] set_runner called, id={id(self)}, runner={type(runner)}")
        self._model_runner = runner
        if self._orchestrator is None:
            self._build_orchestrator()
