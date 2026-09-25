from __future__ import annotations

from collections.abc import Sequence

import torch

from ..controller import KVCompressionController
from ..offload.manager import KVOffloadManager
from ..selection.adapter import (
    collect_needed_block_refs,
    patch_paged_kv_indices_with_staging,
)
from ..scoring.base import BlockRef
from ..staging.pool import KVStagingPool
from ..staging.residency import Residency, ResidencyTable


class OffloadOrchestrator:
    """Runtime coordinator between vLLM, scorer, offloader and sparse attention."""

    def __init__(
        self,
        *,
        offload_manager: KVOffloadManager,
        staging_pool: KVStagingPool,
        residency: ResidencyTable,
        controller: KVCompressionController,
        page_size: int,
        staging_base_page_idx: int,
        offload_fraction: float,
    ) -> None:
        self.offload_fraction = float(offload_fraction)
        self.offload_manager = offload_manager
        self.staging_pool = staging_pool
        self.residency = residency
        self.controller = controller
        self.page_size = int(page_size)
        self.staging_base_page_idx = int(staging_base_page_idx)
        self._request_blocks: dict[str, list[BlockRef]] = {}
        self._scored: set[BlockRef] = set()
        self._sparse_mode = True  # Enable offloading by default

    def _enable_sparse_mode(self) -> None:
        """Enable sparse attention offloading mode."""
        self._sparse_mode = True

    # FIX — track last seen block count per request and only process new tail blocks:
    def sync_request_blocks(self, request_id, gpu_block_ids, num_computed_tokens: int):
        # Skip warmup requests
        if str(request_id).startswith("_warmup_"):
            return []
            
        prev_count = len(self._request_blocks.get(request_id, []))
        refs = []
        for block_idx, gpu_id in enumerate(gpu_block_ids):
            ref = BlockRef(str(request_id), int(block_idx))
            complete = (block_idx + 1) * self.page_size <= int(num_computed_tokens)
            old = self.residency.get(ref)
            if old is None or old.gpu_block_id != int(gpu_id):
                self.residency.mark_gpu(ref, int(gpu_id), complete=complete)
            elif complete and not old.complete:
                self.residency.mark_complete(ref)
            refs.append(ref)

        # Score newly added blocks
        for ref in refs[prev_count:]:
            loc = self.residency.get(ref)
            if loc is not None and loc.complete and ref not in self._scored:
                if loc.residency == Residency.GPU:
                    self.offload_manager.on_block_full(ref)
                    self._scored.add(ref)
        
        # Also score previously tracked blocks that just became complete
        for ref in refs[:prev_count]:
            loc = self.residency.get(ref)
            if loc is not None and loc.complete and ref not in self._scored:
                if loc.residency == Residency.GPU:
                    self.offload_manager.on_block_full(ref)
                    self._scored.add(ref)
        
        self._request_blocks[request_id] = refs
        return refs

    def on_prefill_complete(self, request_id: str) -> list[BlockRef]:
        """Offload globally lowest-scoring complete GPU blocks across all requests."""
        # Skip warmup requests
        if str(request_id).startswith("_warmup_"):
            return []
            
        if not self._sparse_mode:
            return []
        
        # Mark all blocks for this request as complete (prefill is done)
        refs = self._request_blocks.get(request_id, [])
        for ref in refs:
            loc = self.residency.get(ref)
            if loc is not None and not loc.complete:
                self.residency.mark_complete(ref)
            # Score blocks that became complete now
            if loc is not None and loc.complete and ref not in self._scored:
                if loc.residency == Residency.GPU:
                    self.offload_manager.on_block_full(ref)
                    self._scored.add(ref)
        
        # Global candidate pool: all complete GPU blocks across all requests
        all_gpu_complete = self.residency.all_gpu_blocks(complete_only=True)
        result = self.offload_manager.offload_fraction(all_gpu_complete, self.offload_fraction, self._scored)
        return result

    def on_decode_step(
        self,
        request_ids: Sequence[str],
        num_tokens: int = 1,
    ) -> list[BlockRef]:

        triggered_requests = []

        for request_id in request_ids:
            if self.controller.on_decode(
                request_id,
                num_tokens,
            ):
                triggered_requests.append(request_id)

        if not triggered_requests:
            return []

        # Debug: Print block residency for each active request
        for request_id in request_ids:
            refs = self.residency.refs_for_request(request_id)
            if refs:
                print(f"[ORCH DECODE] req={request_id}:")
                for bref in refs:
                    loc = self.residency.get(bref)
                    if loc:
                        print(f"  block_idx={bref.block_idx}, gpu_block_id={loc.gpu_block_id}, residency={loc.residency.name}, complete={loc.complete}")

        # Global candidate pool for all triggered requests
        all_gpu_complete = self.residency.all_gpu_blocks(complete_only=True)
        return self.offload_manager.offload_fraction(all_gpu_complete, self.offload_fraction)

    def prepare_staging(
        self,
        request_ids: Sequence[str],
        sparse_idx: torch.Tensor,
        sparse_len: torch.Tensor,
    ) -> dict[BlockRef, int]:
        print(f"[ORCHESTRATOR DEBUG] prepare_staging called: request_ids={request_ids}, sparse_idx.shape={sparse_idx.shape}", flush=True)
        needed = collect_needed_block_refs(
            request_ids, sparse_idx, sparse_len, self.page_size
        )
        print(f"[ORCHESTRATOR DEBUG] needed blocks: {[(ref.request_id, ref.block_idx) for ref in needed]}", flush=True)
        cpu_needed = [
            ref for ref in needed
            if (loc := self.residency.get(ref)) is not None
            and loc.residency == Residency.CPU
        ]
        print(f"[ORCHESTRATOR DEBUG] cpu_needed: {[(ref.request_id, ref.block_idx) for ref in cpu_needed]}", flush=True)
        for ref in needed:
            loc = self.residency.get(ref)
            if loc:
                print(f"[ORCHESTRATOR DEBUG]   Block {ref}: residency={loc.residency.name}, gpu_block={loc.gpu_block_id}, cpu_slot={loc.cpu_slot}, staging_slot={loc.staging_slot}", flush=True)
        result = self.staging_pool.ensure_resident(cpu_needed)
        print(f"[ORCHESTRATOR DEBUG] ensure_resident result: {result}", flush=True)
        return result

    def patch_paged_kv_indices(
        self,
        paged_kv_indices: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        request_ids: Sequence[str],
        sparse_idx: torch.Tensor,
        sparse_len: torch.Tensor,
        staging_map: dict[BlockRef, int],
    ) -> torch.Tensor:
        return patch_paged_kv_indices_with_staging(
            paged_kv_indices,
            paged_kv_indptr,
            request_ids,
            sparse_idx,
            sparse_len,
            self.page_size,
            self.staging_base_page_idx,
            self.residency,
            staging_map,
        )

    def on_request_done(self, request_id: str) -> None:
        self._cleanup_request(request_id)

    def on_request_preempted(self, request_id: str) -> None:
        """Handle preempted request - same cleanup as finished request."""
        self._cleanup_request(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        refs = self._request_blocks.pop(request_id, [])
        for ref in refs:
            self.staging_pool.release(ref)
            self.offload_manager.free(ref)
            self._scored.discard(ref)
        self.controller.remove(request_id)
