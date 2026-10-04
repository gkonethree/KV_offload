# SPDX-License-Identifier: Apache-2.0
"""GPU staging pool for offloaded KV pages (tail pages of the KV tensors)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch

from vllm.logger import init_logger

from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.base import BlockRef
from offload_package.staging.residency import Residency, ResidencyTable

logger = init_logger(__name__)


class KVStagingPool:
    """Temporary GPU pages used to satisfy sparse attention requests.

    The staging pages are expected to be the tail pages of the same per-layer
    KV cache tensors passed to the attention kernel.  This is important:
    without a shared address space, the existing sparse kernel cannot address
    staging pages without a CUDA-kernel change.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        staging_base_page: int,
        gpu_k_caches: Sequence[torch.Tensor],
        gpu_v_caches: Sequence[torch.Tensor],
        cpu_cache: torch.Tensor,
        residency: ResidencyTable,
        offload_manager: KVOffloadManager,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        if len(gpu_k_caches) != len(gpu_v_caches):
            raise ValueError("GPU K/V layer counts must match")
        if staging_base_page < 0:
            raise ValueError("staging_base_page must be >= 0")
        if staging_base_page + num_slots > gpu_k_caches[0].shape[0]:
            raise ValueError("staging pages do not fit in GPU cache")

        self.num_slots = int(num_slots)
        self.base = int(staging_base_page)
        self.gpu_k_caches = list(gpu_k_caches)
        self.gpu_v_caches = list(gpu_v_caches)
        self.cpu_cache = cpu_cache
        self.residency = residency
        self.offload_manager = offload_manager
        self._copy_stream: torch.cuda.Stream | None = None
        self._lru: OrderedDict[BlockRef, int] = OrderedDict()
        self._free = list(range(num_slots - 1, -1, -1))

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        """Lazily initialize the copy stream."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.gpu_k_caches[0].device)
        return self._copy_stream

    def ensure_resident(self, refs: Sequence[BlockRef]) -> dict[BlockRef, int]:
        result: dict[BlockRef, int] = {}
        to_fetch: list[tuple[BlockRef, int]] = []
        for ref in dict.fromkeys(refs):
            loc = self.residency.get(ref)
            if loc is None:
                raise KeyError(f"Unknown logical block {ref}")
            if loc.residency == Residency.GPU:
                continue
            if loc.residency == Residency.STAGING and loc.staging_slot is not None:
                self._lru.move_to_end(ref)
                result[ref] = loc.staging_slot
                continue
            if loc.cpu_slot is None:
                raise RuntimeError(f"{ref} is not backed by a CPU slot")

            self.offload_manager.wait_cpu_slot(loc.cpu_slot)
            to_fetch.append((ref, loc.cpu_slot))
        if not to_fetch:
            return result

        # Phase 1: allocate/evict staging slots for the whole incoming batch
        # (eviction choice identical to the previous per-ref loop, except
        # slots already assigned within this batch are protected: their page
        # content is written by the batched copy in phase 2).
        allocs: list[tuple[BlockRef, int, int]] = []
        protected: set = set()
        for ref, cpu_slot in to_fetch:
            slot = self._allocate_slot(ref, exclude=protected)
            protected.add(slot)
            allocs.append((ref, cpu_slot, slot))
            result[ref] = slot

        # Phase 2: batched copies — per layer, ONE gather on CPU, one H2D
        # index_copy_ into the staging pages. Same bytes as before, ~L launches
        # per batch instead of L per block.
        device = self.gpu_k_caches[0].device
        cpu_slots_t = torch.tensor(
            [cs for _, cs, _ in allocs], dtype=torch.long
        )
        base_ids = (
            torch.tensor([s for _, _, s in allocs], dtype=torch.long, device=device)
            + self.base
        )
        with torch.cuda.stream(self.copy_stream):
            for layer, (k_cache, _v_cache) in enumerate(
                zip(self.gpu_k_caches, self.gpu_v_caches)
            ):
                # cpu_cache layer: [S, 2, P, H, C] -> gather slots -> split KV ->
                # transpose to [n, H, P, C] each -> cat -> [n, H, P, 2C]
                g = self.cpu_cache[layer].index_select(0, cpu_slots_t)
                k = g[:, 0].permute(0, 2, 1, 3)
                v = g[:, 1].permute(0, 2, 1, 3)
                combined = torch.cat([k, v], dim=-1).contiguous()
                combined_gpu = combined.to(device, non_blocking=True)
                k_cache.index_copy_(0, base_ids, combined_gpu)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        # One ordering point for the batch: attention reads staged pages only
        # after this.
        torch.cuda.current_stream(device=device).wait_stream(self.copy_stream)
        for ref, _cs, slot in allocs:
            self.residency.mark_staging(ref, slot)
        return result

    def _allocate_slot(self, incoming: BlockRef, exclude: set | None = None) -> int:
        if self._free:
            slot = self._free.pop()
        else:
            # Score-aware eviction: evict lowest-scored staging block.
            # `exclude` protects slots already assigned to the current fetch
            # batch, whose page content is written by the batched copy later.
            lru_candidates = [
                ref for ref in self._lru.keys()
                if exclude is None or ref not in exclude
            ]
            if not lru_candidates:
                raise RuntimeError("No staging slots available and no candidates to evict")
            scores = self.offload_manager.scorer.scores(lru_candidates)
            evict_idx = int(torch.argmin(scores).item())
            evicted = lru_candidates[evict_idx]
            old = self.residency.get(evicted)
            if old is None or old.cpu_slot is None:
                raise RuntimeError(f"Cannot evict staging block {evicted}")
            self.residency.mark_cpu(evicted, old.cpu_slot)
            slot = self._lru.pop(evicted)
        self._lru[incoming] = slot
        return slot

    def release(self, ref: BlockRef) -> None:
        slot = self._lru.pop(ref, None)
        if slot is not None:
            self._free.append(slot)


