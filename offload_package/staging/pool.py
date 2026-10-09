# SPDX-License-Identifier: Apache-2.0
"""GPU staging pool for offloaded KV pages (tail pages of the KV tensors)."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch

from vllm.logger import init_logger

from offload_package import timing as _timing
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.base import BlockRef
from offload_package.staging.residency import Residency, ResidencyTable

logger = init_logger(__name__)

# One-shot warning flag for occupancy-LRU/residency desync observations.
_DESYNC_WARNED: dict = {}


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

    def ensure_resident(self, refs: Sequence[BlockRef]) -> tuple[dict[BlockRef, int], list[BlockRef]]:
        """Stage `refs` into GPU staging pages.

        Returns ``(staged, skipped)``: `staged` maps every ref that is
        GPU- or staging-resident afterwards to its staging slot; `skipped`
        lists refs that could not be allocated (staging exhausted with no
        evictable candidate). Callers must treat a non-empty `skipped` as a
        partial failure: with true freeing the un-staged refs' index entries
        point at null pages that the kernel reads with weight 1.
        """
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
            return result, []

        # Phase 1: allocate/evict staging slots for the batch. Slots are
        # RESERVED here but only entered into the occupancy LRU after the
        # phase-2 copies succeed, so a failed batch cannot leak slots or
        # leave the LRU disagreeing with the residency table. If a block
        # exceeds capacity (no evictable candidate) SKIP it and report it in
        # `skipped` -- the caller decides between chunked decode (exact) or
        # failing the request.
        allocs: list[tuple[BlockRef, int, int]] = []
        skipped: list[BlockRef] = []
        protected: set[int] = set()
        for ref, cpu_slot in to_fetch:
            try:
                slot = self._allocate_slot(ref, exclude=protected)
            except RuntimeError as exc:
                try:
                    from offload_package.debug import dbg as _dbg_pool
                    _dbg_pool(f"[POOL DEBUG] skip {ref}: {exc}")
                except Exception:
                    pass
                skipped.append(ref)
                continue
            protected.add(slot)
            allocs.append((ref, cpu_slot, slot))

        # Phase 2: batched copies — per layer, ONE gather on CPU, one H2D
        # index_copy_ into the staging pages. Same bytes as before, ~L launches
        # per batch instead of L per block.
        device = self.gpu_k_caches[0].device
        cpu_slots_t = torch.tensor(
            [cs for _, cs, _ in allocs], dtype=torch.long
        )
        dbg = None
        try:
            from offload_package.debug import dbg as _dbg_fetch
            dbg = _dbg_fetch
        except Exception:
            pass
        # Host and device temporaries must outlive the async H2D copy issued on
        # copy_stream; dropping them early lets the caching allocator recycle
        # staging tensors mid-write. `base_ids` must also be created inside the
        # copy_stream context: it is consumed by kernels on copy_stream, and a
        # default-stream allocation would be recycled by the default-stream
        # allocator while the copy-stream index_copy_ is still reading it
        # (cross-stream UAF -> device-side "index out of bounds" assert).
        gpu_slabs: list[torch.Tensor] = []
        host_slabs: list[torch.Tensor] = []
        try:
            with torch.cuda.stream(self.copy_stream):
                base_ids = (
                    torch.tensor([s for _, _, s in allocs], dtype=torch.long, device=device)
                    + self.base
                )
                for layer, (k_cache, _v_cache) in enumerate(
                    zip(self.gpu_k_caches, self.gpu_v_caches)
                ):
                    # cpu_cache layer: [S, 2, P, H, C] -> gather slots -> split KV ->
                    # transpose to [n, H, P, C] each -> cat -> [n, H, P, 2C]
                    _timing.start("pool.cpu_gather")
                    g = self.cpu_cache[layer].index_select(0, cpu_slots_t)
                    _timing.end("pool.cpu_gather")
                    _timing.start("pool.transform")
                    k = g[:, 0].permute(0, 2, 1, 3)
                    v = g[:, 1].permute(0, 2, 1, 3)
                    combined = torch.cat([k, v], dim=-1).contiguous()
                    _timing.end("pool.transform")
                    _timing.start("pool.h2d")
                    combined_gpu = combined.to(device, non_blocking=True)
                    k_cache.index_copy_(0, base_ids, combined_gpu)
                    _timing.end("pool.h2d")
                    host_slabs.append(combined)
                    gpu_slabs.append(combined_gpu)
                event = torch.cuda.Event()
                event.record(self.copy_stream)
            # One ordering point for the batch: attention reads staged pages only
            # after this. The temporaries above are now safe to drop.
            _timing.start("pool.h2d_sync")
            torch.cuda.current_stream(device=device).wait_stream(self.copy_stream)
            event.synchronize()
            _timing.end("pool.h2d_sync")
        except Exception:
            # Copies failed: hand the reserved slots back so nothing leaks, and
            # leave the occupancy LRU untouched (entries are only inserted on
            # success below). Re-raise: a failed fetch must not be silent.
            for _, _, slot in allocs:
                self._free.append(slot)
            raise
        del host_slabs, gpu_slabs
        if dbg is not None:
            dbg(f"[POOL DEBUG] ensure_resident staged {len(allocs)} blocks "
                f"(slots={[s for _, _, s in allocs]}), skipped {len(skipped)}")
        # Phase 3: publish occupancy only now that the copies succeeded.
        for ref, _cs, slot in allocs:
            self._lru[ref] = slot
            self.residency.mark_staging(ref, slot)
            result[ref] = slot
        return result, skipped

    def _allocate_slot(self, incoming: BlockRef, exclude: set[int] | None = None) -> int:
        """Reserve a staging slot; the caller publishes it into the LRU only
        after the copy succeeds. Score-aware eviction when full: always evicts
        the lowest-scored STAGING-resident entry. `_lru` entries whose
        residency is not STAGING are desync artifacts -- they are never used as
        eviction victims (evicting one would flip a live GPU block to "CPU"
        with a stale cpu_slot) and are reported once. Raises when nothing is
        left."""
        if self._free:
            slot = self._free.pop()
        else:
            lru_candidates = []
            desynced = []
            for ref, sl in self._lru.items():
                if exclude is not None and sl in exclude:
                    continue
                loc = self.residency.get(ref)
                if loc is not None and loc.residency == Residency.STAGING:
                    lru_candidates.append(ref)
                else:
                    desynced.append(ref)
            if desynced and not _DESYNC_WARNED:
                _DESYNC_WARNED["done"] = True
                logger.warning(
                    "KVStagingPool: %d occupancy-LRU entries have non-STAGING "
                    "residency (desync); they are ignored as eviction "
                    "candidates. First offenders: %s",
                    len(desynced), desynced[:3],
                )
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
        return slot

    def release(self, ref: BlockRef) -> None:
        """Recycle `ref`'s staging slot back to the free list. The residency
        map must stay truthful: the block's content still lives in its CPU
        slot (staging is a copy), so flip STAGING -> CPU."""
        slot = self._lru.pop(ref, None)
        if slot is None:
            return
        self._free.append(slot)
        loc = self.residency.get(ref)
        if (
            loc is not None
            and loc.residency == Residency.STAGING
            and loc.staging_slot == slot
            and loc.cpu_slot is not None
        ):
            self.residency.mark_cpu(ref, loc.cpu_slot)


