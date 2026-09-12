from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
from ..staging.sparse_page_table import SparsePageTable


@dataclass(frozen=True)
class OffloadScheduleConfig:
    interval_tokens: int
    page_size: int


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
    ) -> None:
        self.offload_manager = offload_manager
        self.staging_pool = staging_pool
        self.residency = residency
        self.controller = controller
        self.page_size = int(page_size)
        self.page_table = SparsePageTable(residency, staging_base_page_idx)
        self.staging_base_page_idx = int(staging_base_page_idx)
        self._request_blocks: dict[str, list[BlockRef]] = {}
        self._scored: set[BlockRef] = set()

    # FIX — track last seen block count per request and only process new tail blocks:
    def sync_request_blocks(self, request_id, gpu_block_ids, seq_len):
        prev_count = len(self._request_blocks.get(request_id, []))
        refs = []
        for block_idx, gpu_id in enumerate(gpu_block_ids):
            ref = BlockRef(str(request_id), int(block_idx))
            complete = (block_idx + 1) * self.page_size <= int(seq_len)
            old = self.residency.get(ref)
            if old is None or old.gpu_block_id != int(gpu_id):
                self.residency.mark_gpu(ref, int(gpu_id), complete=complete)
            elif complete and not old.complete:
                self.residency.mark_complete(ref)
            refs.append(ref)

        # Only score blocks that are new since last call.
        for ref in refs[prev_count:]:
            loc = self.residency.get(ref)
            if loc is not None and loc.complete and ref not in self._scored:
                if loc.residency == Residency.GPU:
                    self.offload_manager.on_block_full(ref)
                    self._scored.add(ref)
        self._request_blocks[request_id] = refs
        return refs

    def on_prefill_complete(self, request_id: str) -> None:
        self.controller.on_prefill_complete(request_id)

    def on_decode_step(self, request_ids: Sequence[str], num_tokens: int = 1) -> list[BlockRef]:
        """Advance per-request counters; any boundary triggers one global pass."""
        triggered = False
        for request_id in request_ids:
            if self.controller.on_decode(request_id, num_tokens):
                triggered = True
        if not triggered:
            return []
        return self.offload_manager.maybe_offload(self.residency.all_gpu_blocks(complete_only=True))

    def prepare_staging(
        self,
        request_ids: Sequence[str],
        sparse_idx: torch.Tensor,
        sparse_len: torch.Tensor,
    ) -> dict[BlockRef, int]:
        needed = collect_needed_block_refs(
            request_ids, sparse_idx, sparse_len, self.page_size
        )
        cpu_needed = [
            ref for ref in needed
            if (loc := self.residency.get(ref)) is not None
            and loc.residency in (Residency.CPU, Residency.IN_FLIGHT)
        ]
        return self.staging_pool.ensure_resident(cpu_needed)

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
        refs = self._request_blocks.pop(request_id, [])
        for ref in refs:
            self.staging_pool.release(ref)
            self.offload_manager.free(ref)
            self._scored.discard(ref)
        self.controller.remove(request_id)
