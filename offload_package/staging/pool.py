# SPDX-License-Identifier: Apache-2.0
"""GPU staging pool for offloaded KV pages (tail pages of the KV tensors)."""

from __future__ import annotations

import os as _os
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
        eviction_scorer=None,
        eviction_policy: str = "scorer",
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
        # Eviction policy is independent of the offload scorer: which staged
        # block to drop under pool pressure decides re-fetch volume, while
        # the offload scorer decides which blocks get offloaded at all.
        # Default: inherit the offload scorer (legacy behavior, and what the
        # unit tests that assert score-aware eviction rely on).
        self._eviction_scorer = eviction_scorer or offload_manager.scorer
        # "lru": victims come from the pool's own occupancy order, which is
        # touched on EVERY layer's use (fetch, fill or already-staged pick).
        # So when layer L allocates, the blocks layer L-1 used are the most
        # recent and evicted last; at layer 0, the previous step's last
        # layer's blocks are. Other policies rank by the eviction scorer.
        self._eviction_policy = str(eviction_policy)
        self._copy_stream: torch.cuda.Stream | None = None
        # Ring of pinned staging slabs [layers, num_slots, H, P, 2C]:
        # per-layer H2Ds are issued from PINNED slices (pageable sources
        # stall each copy, ~2.4ms/16KB measured at 16k), and the ring lets a
        # batch return WITHOUT syncing its own copies -- the next-but-one
        # batch syncs this buffer's event, by which time the transfers have
        # overlapped with a full batch of CPU work. Measured: the per-batch
        # host event sync was 68% of fetch time.
        self._pinned_slabs: list[torch.Tensor] = []
        self._slab_events: list[object | None] = []
        self._slab_pending: list[list[torch.Tensor]] = []
        self._slab_turn = 0
        self._SLAB_RING = 2
        self._lru: OrderedDict[BlockRef, int] = OrderedDict()
        self._free = list(range(num_slots - 1, -1, -1))
        # In-page token count (GPU cache layout [num_blocks, H, P, 2C]).
        self._page_size = int(gpu_k_caches[0].shape[2])
        # data_ptr -> layer index: the wrapper identifies the calling layer
        # by its KV cache tensor (cache_id = k_cache.data_ptr()).
        self._layer_of_ptr = {
            int(c.data_ptr()): i for i, c in enumerate(self.gpu_k_caches)
        }
        # Token-granular fetch scratch: pinned [ring][max_tok, H, 2C] buffers
        # with the same deferred-event discipline as the block slab ring.
        self._tok_slabs: list[torch.Tensor] = []
        self._tok_events: list[object | None] = []
        self._tok_pending: list[list[torch.Tensor]] = []
        self._tok_turn = 0
        self._TOK_RING = 2
        self._tok_capacity = 0

    @property
    def _page_bytes(self) -> int:
        src = self.gpu_k_caches[0]          # [num_blocks, H, P, 2C]
        n = src.numel() // src.shape[0]
        return int(n) * src.element_size()

    @property
    def _block_bytes(self) -> int:
        return self._page_bytes * len(self.gpu_k_caches)

    def layer_of(self, cache_id: int) -> int:
        idx = self._layer_of_ptr.get(int(cache_id))
        if idx is None:
            raise RuntimeError(
                f"KV cache {int(cache_id)} is not one of the staged layers; "
                "the token fetch path needs the calling layer's cache id."
            )
        return idx

    def _acquire_tok_slab(self, n: int) -> torch.Tensor:
        """Pinned scratch for one token fetch batch, ring-buffered like the
        block slabs: the host sync for buffer i's transfers happens two
        batches later, overlapping with CPU work."""
        src = self.gpu_k_caches[0]          # [num_blocks, H, P, 2C]
        H, C2 = int(src.shape[1]), int(src.shape[3])
        if not self._tok_slabs or n > self._tok_capacity:
            cap = max(n, self._tok_capacity, 4096)
            self._tok_slabs = [
                torch.empty((cap, H, C2), dtype=src.dtype,
                            device="cpu", pin_memory=True)
                for _ in range(self._TOK_RING)
            ]
            self._tok_events = [None] * self._TOK_RING
            self._tok_pending = [[] for _ in range(self._TOK_RING)]
            self._tok_capacity = cap
        idx = self._tok_turn
        self._tok_turn = (self._tok_turn + 1) % len(self._tok_slabs)
        self._tok_cur = idx
        ev = self._tok_events[idx]
        if ev is not None:
            _timing.start("pool.h2d_sync")
            ev.synchronize()
            _timing.end("pool.h2d_sync")
            self._tok_events[idx] = None
            self._tok_pending[idx].clear()
        return self._tok_slabs[idx]

    def _reset_tok_ring(self) -> None:
        if self._tok_slabs:
            self.copy_stream.synchronize()
        self._tok_events = [None] * len(self._tok_slabs)
        for pending in self._tok_pending:
            pending.clear()

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        """Lazily initialize the copy stream."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.gpu_k_caches[0].device)
        return self._copy_stream

    def _acquire_slab(self, n: int) -> torch.Tensor:
        """Next pinned slab from the ring, deferring the host sync.

        Allocates the ring lazily from the GPU cache layout (the one layout
        every post-transform consumer uses: [layers, num_slots, H, P, 2C]).
        Before handing out buffer i, syncs the event recorded on it RING
        batches ago -- by then its transfers have overlapped with a full
        batch of CPU work, so the wait is ~free instead of the 22 ms/batch
        measured for syncing THIS batch's copies under load. Returns the
        buffer and stashes the index on ``self._slab_cur`` for the launch
        side, which records a fresh event per batch.
        """
        if not self._pinned_slabs:
            src = self.gpu_k_caches[0]      # [num_blocks, H, P, 2C]
            H, P, C2 = int(src.shape[1]), int(src.shape[2]), int(src.shape[3])
            layers = len(self.gpu_k_caches)
            for _ in range(self._SLAB_RING):
                self._pinned_slabs.append(torch.empty(
                    (layers, self.num_slots, H, P, C2),
                    dtype=src.dtype, device="cpu", pin_memory=True,
                ))
                self._slab_events.append(None)
                self._slab_pending.append([])
        assert n <= self.num_slots
        idx = self._slab_turn
        self._slab_turn = (self._slab_turn + 1) % len(self._pinned_slabs)
        self._slab_cur = idx
        ev = self._slab_events[idx]
        if ev is not None:
            _timing.start("pool.h2d_sync")
            ev.synchronize()
            _timing.end("pool.h2d_sync")
            self._slab_events[idx] = None
            self._slab_pending[idx].clear()
        return self._pinned_slabs[idx]

    def _reset_slab_ring(self) -> None:
        """Drain all in-flight slab transfers and clear ring state (error path)."""
        if self._pinned_slabs:
            self.copy_stream.synchronize()
        self._slab_events = [None] * len(self._pinned_slabs)
        for pending in self._slab_pending:
            pending.clear()
        if self._tok_slabs:
            self._tok_events = [None] * len(self._tok_slabs)
            for pending in self._tok_pending:
                pending.clear()

    def _touch(self, refs) -> None:
        """Mark staged refs as used by the current layer (most recent)."""
        for ref in refs:
            if ref in self._lru:
                self._lru.move_to_end(ref)

    def _victim_order(self, protect_set, device_protect: bool = False) -> list[BlockRef]:
        """Evictable staged refs, first victim first.

        With the residency's device mirror and ``device_protect``, blocks the
        CURRENT layer selected (dev_last_use == tick, written by the
        vectorized discovery) are protected too, and LRU order comes from
        dev_last_use -- every layer touches its whole selection in one
        scatter, so no per-ref host touch is needed.
        """
        candidates: list[BlockRef] = []
        desynced: list[BlockRef] = []
        for r in self._lru.keys():          # oldest touch first
            if r in protect_set:
                continue
            loc = self.residency.get(r)
            if loc is not None and loc.residency == Residency.STAGING:
                candidates.append(r)
            else:
                desynced.append(r)
        if desynced and not _DESYNC_WARNED:
            _DESYNC_WARNED["done"] = True
            logger.warning(
                "KVStagingPool: %d occupancy-LRU entries have non-STAGING "
                "residency (desync); they are ignored as eviction "
                "candidates. First offenders: %s",
                len(desynced), desynced[:3],
            )
        use_dev = device_protect and getattr(self.residency, "has_device_mirror", False)
        if candidates and use_dev:
            lu = self.residency.last_use_of(candidates)
            tick = self.residency.tick
            kept = [(r, u) for r, u in zip(candidates, lu) if u != tick]
            candidates = [r for r, _ in kept]
            if self._eviction_policy == "lru":
                # Stable: ties keep occupancy (insertion/touch) order.
                return [r for r, _ in sorted(kept, key=lambda t: t[1])]
        if not candidates or self._eviction_policy == "lru":
            return candidates
        order = torch.argsort(self._eviction_scorer.scores(candidates))
        return [candidates[i] for i in order.tolist()]

    def ensure_resident(
        self,
        refs: Sequence[BlockRef],
        protect: Sequence[BlockRef] | None = None,
        device_protect: bool = True,
    ) -> tuple[dict[BlockRef, int], list[BlockRef]]:
        """Stage `refs` into GPU staging pages.

        ``protect`` lists refs that are ALREADY staged and needed by the
        caller's current gather (this step's selection). They are never
        eviction victims: the caller's patched indices point at their slots
        and clobbering one would feed the kernel the wrong block's bytes at
        a weight-1 position. When protection leaves no evictable candidate,
        refs are skipped (reported in `skipped`) and the caller falls back
        to the chunked path.

        Returns ``(staged, skipped)``: `staged` maps every ref that is
        GPU- or staging-resident afterwards to its staging slot; `skipped`
        lists refs that could not be allocated. Callers must treat a
        non-empty `skipped` as a partial failure: with true freeing the
        un-staged refs' index entries point at null pages that the kernel
        reads with weight 1.
        """
        protect_set = set(protect) if protect else frozenset()
        self._touch(protect_set)
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
        _timing.start("pool.alloc")
        # Eviction order computed AT MOST ONCE per call: score all evictable
        # candidates once, argsort ascending, pop victims in order as slots
        # are needed. The per-call protect set is fixed, and blocks staged by
        # THIS batch are not in the occupancy LRU until phase 3 publishes
        # them, so the once-computed order is valid for every allocation in
        # the batch. (The old per-slot _allocate_slot rescanned the full LRU
        # and rebuilt the score tensor per allocated slot: 4.3 ms/batch
        # measured at 16k.)
        victims: list[BlockRef] = []
        victim_pos = 0
        victims_built = False
        for ref, cpu_slot in to_fetch:
            slot = None
            if self._free:
                slot = self._free.pop()
            else:
                if not victims_built:
                    victims_built = True
                    # Refs this call already resolved to a staged slot are in
                    # `result` and must not be evicted under them (only LRU
                    # order puts them last by itself).
                    victims = self._victim_order(protect_set | set(result), device_protect)
                    victim_pos = 0
                while victim_pos < len(victims):
                    evicted = victims[victim_pos]
                    victim_pos += 1
                    old = self.residency.get(evicted)
                    if old is None or old.cpu_slot is None:
                        continue  # desynced away between build and use
                    self.residency.mark_cpu(evicted, old.cpu_slot)
                    slot = self._lru.pop(evicted)
                    break
            if slot is None:
                skipped.append(ref)
                continue
            allocs.append((ref, cpu_slot, slot))
        _timing.end("pool.alloc")

        # Phase 2: ONE gather over all layers + ONE uniform layout transform.
        # cpu_cache is a single [L, S, 2, P, H, C] tensor, so index_select
        # on the slot axis fetches every layer's slots in a single dispatch.
        # The transform [2, P, H, C] -> [H, P, 2C] is one permute+reshape for
        # the whole batch (K in [..., :C], V in [..., C:2C] — byte-identical
        # to the old per-layer cat, verified by the staging roundtrip tests),
        # then one host memcpy into the pinned ring buffer. Replaces ~88 tiny
        # per-layer CPU dispatches with ~4.
        if not allocs:
            return result, skipped
        device = self.gpu_k_caches[0].device
        cpu_slots_t = torch.tensor(
            [cs for _, cs, _ in allocs], dtype=torch.long
        )
        n = len(allocs)
        dbg = None
        try:
            from offload_package.debug import dbg as _dbg_fetch
            dbg = _dbg_fetch
        except Exception:
            pass
        gpu_slabs: list[torch.Tensor] = []
        slab = self._acquire_slab(n)
        try:
            _timing.note_bytes("pool.d2h_block", n * self._block_bytes)
            _timing.start("pool.cpu_gather")
            gathered = self.cpu_cache.index_select(1, cpu_slots_t)  # [L, n, 2, P, H, C]
            _timing.end("pool.cpu_gather")
            _timing.start("pool.transform")
            combined = (
                gathered.permute(0, 1, 4, 3, 2, 5)          # [L, n, H, P, 2, C]
                .reshape(*gathered.shape[:2], *slab.shape[2:])  # [L, n, H, P, 2C]
            )
            slab[:, :n].copy_(combined)
            _timing.end("pool.transform")
            # WAR ordering: staging pages written below may still be read by
            # kernels already queued on the compute stream (earlier layers /
            # the previous step). The copy stream must not overtake them.
            self.copy_stream.wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(self.copy_stream):
                base_ids = (
                    torch.tensor([s for _, _, s in allocs], dtype=torch.long, device=device)
                    + self.base
                )
                for layer, (k_cache, _v_cache) in enumerate(
                    zip(self.gpu_k_caches, self.gpu_v_caches)
                ):
                    _timing.start("pool.h2d")
                    combined_gpu = slab[layer, :n].to(device, non_blocking=True)
                    k_cache.index_copy_(0, base_ids, combined_gpu)
                    _timing.end("pool.h2d")
                    gpu_slabs.append(combined_gpu)
                event = torch.cuda.Event()
                event.record(self.copy_stream)
            # GPU ordering point ONLY: the default stream (decode) must observe
            # these copies. wait_stream is a stream dependency, not a host
            # block -- the CPU returns immediately and phase 3 publishes the
            # slots while the transfers are still in flight; the pinned slab
            # is ring-buffered, so its reuse syncs this event two batches
            # later. Eviction of an in-flight-staged block is safe for the
            # same reason: the overwrite lands in the OTHER ring buffer.
            torch.cuda.current_stream(device=device).wait_stream(self.copy_stream)
            self._slab_events[self._slab_cur] = event
            self._slab_pending[self._slab_cur].extend(gpu_slabs)
        except Exception:
            # Copies failed: hand the reserved slots back so nothing leaks, and
            # leave the occupancy LRU untouched (entries are only inserted on
            # success below). Drain the copy stream so no slab buffer is left
            # with a phantom in-flight event. Re-raise: a failed fetch must
            # not be silent.
            self._reset_slab_ring()
            for _, _, slot in allocs:
                self._free.append(slot)
            raise
        if dbg is not None:
            dbg(f"[POOL DEBUG] ensure_resident staged {len(allocs)} blocks, "
                f"skipped {len(skipped)}")
        # Phase 3: publish occupancy only now that the copies succeeded.
        for ref, _cs, slot in allocs:
            self._lru[ref] = slot
            self.residency.mark_staging(ref, slot)
            result[ref] = slot
        return result, skipped

    def _allocate_slot(
        self,
        incoming: BlockRef,
        exclude: set[int] | None = None,
        protect: set[BlockRef] | frozenset = frozenset(),
    ) -> int:
        """Reserve a staging slot; the caller publishes it into the LRU only
        after the copy succeeds. Score-aware eviction when full: always evicts
        the lowest-scored STAGING-resident entry that is NOT in `protect`
        (protected refs are in the caller's current gather; their slots must
        survive this batch). `_lru` entries whose residency is not STAGING are
        desync artifacts -- they are never used as eviction victims (evicting
        one would flip a live GPU block to "CPU" with a stale cpu_slot) and
        are reported once. Raises when nothing is left."""
        if self._free:
            slot = self._free.pop()
        else:
            lru_candidates = []
            desynced = []
            for ref, sl in self._lru.items():
                if exclude is not None and sl in exclude:
                    continue
                if ref in protect:
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
            scores = self._eviction_scorer.scores(lru_candidates)
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


