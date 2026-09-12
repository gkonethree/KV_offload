from __future__ import annotations

from collections.abc import Sequence

import torch

from ..scoring.base import BlockRef, BlockScorer
from ..scoring.norm_kernel import compute_block_kv_norms
from ..staging.residency import Residency, ResidencyTable


class CPUBlockPool:
    def __init__(self, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        self._free = list(range(num_slots - 1, -1, -1))

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("CPU offload pool exhausted")
        return self._free.pop()

    def free(self, slot: int) -> None:
        self._free.append(int(slot))

    @property
    def num_free(self) -> int:
        return len(self._free)


class KVOffloadManager:
    """Score full logical blocks and copy cold blocks from GPU to CPU."""

    def __init__(
        self,
        *,
        scorer: BlockScorer,
        residency: ResidencyTable,
        gpu_k_caches: Sequence[torch.Tensor],
        gpu_v_caches: Sequence[torch.Tensor],
        cpu_cache: torch.Tensor,
        gpu_budget_blocks: int,
        kv_layout: str = "NHD",
    ) -> None:
        if len(gpu_k_caches) != len(gpu_v_caches):
            raise ValueError("gpu K/V cache lists must have the same length")
        if not gpu_k_caches:
            raise ValueError("at least one KV layer is required")
        if cpu_cache.ndim != 6:
            raise ValueError("cpu_cache must have shape [L,S,2,P,H,D]")
        if cpu_cache.shape[0] != len(gpu_k_caches):
            raise ValueError("CPU cache layer count does not match GPU cache")
        if kv_layout not in ("NHD", "HND"):
            raise ValueError("kv_layout must be NHD or HND")
        if gpu_budget_blocks < 0:
            raise ValueError("gpu_budget_blocks must be >= 0")

        self.scorer = scorer
        self.residency = residency
        self.gpu_k_caches = list(gpu_k_caches)
        self.gpu_v_caches = list(gpu_v_caches)
        self.cpu_cache = cpu_cache
        self.gpu_budget_blocks = int(gpu_budget_blocks)
        self.kv_layout = kv_layout
        self._cpu_pool = CPUBlockPool(cpu_cache.shape[1])
        self._copy_stream = torch.cuda.Stream(device=self.gpu_k_caches[0].device)
        self._store_events: dict[int, torch.cuda.Event] = {}

    def score_block(self, ref: BlockRef, gpu_block_id: int) -> None:
        # The representative layer is sufficient for a query-agnostic scorer.
        stats = compute_block_kv_norms(
            self.gpu_k_caches[0],
            self.gpu_v_caches[0],
            [gpu_block_id],
            logical_block_ids=[ref],
            layout=self.kv_layout,
        )
        self.scorer.update(stats)

    def on_block_full(self, ref: BlockRef) -> None:
        loc = self.residency.get(ref)
        if loc is None or loc.gpu_block_id is None:
            return
        try:
            self.score_block(ref, loc.gpu_block_id)
        except Exception:
            # Log and leave block as incomplete so it is re-scored on next write.
            raise
        self.residency.mark_complete(ref)

    def maybe_offload(self, candidates: Sequence[BlockRef]) -> list[BlockRef]:
        gpu_candidates = [
            ref for ref in candidates
            if (loc := self.residency.get(ref)) is not None
            and loc.residency == Residency.GPU
            and loc.complete
        ]
        if len(gpu_candidates) <= self.gpu_budget_blocks:
            return []

        refs_to_evict = self.scorer.select_for_eviction(
            gpu_candidates, self.gpu_budget_blocks
        )
        offloaded: list[BlockRef] = []
        for ref in refs_to_evict:
            loc = self.residency.get(ref)
            if loc is None or loc.gpu_block_id is None or loc.residency != Residency.GPU:
                continue
            cpu_slot = self._copy_gpu_to_cpu(loc.gpu_block_id)
            self.residency.mark_cpu(ref, cpu_slot)
            offloaded.append(ref)
        return offloaded

    def _copy_gpu_to_cpu(self, gpu_block_id: int) -> int:
        slot = self._cpu_pool.allocate()
        with torch.cuda.stream(self._copy_stream):
            # vLLM's K/V caches are separate per attention layer.
            for layer, (k_cache, v_cache) in enumerate(
                zip(self.gpu_k_caches, self.gpu_v_caches)
            ):
                self.cpu_cache[layer, slot, 0].copy_(
                    k_cache[gpu_block_id], non_blocking=True
                )
                self.cpu_cache[layer, slot, 1].copy_(
                    v_cache[gpu_block_id], non_blocking=True
                )
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        self._store_events[slot] = event
        return slot

    def wait_cpu_slot(self, slot: int) -> None:
        event = self._store_events.get(slot)
        if event is not None:
            event.synchronize()
            self._store_events.pop(slot, None)

    def free(self, ref: BlockRef) -> None:
        loc = self.residency.remove(ref)
        if loc is not None and loc.cpu_slot is not None:
            self._cpu_pool.free(loc.cpu_slot)
        self.scorer.forget([ref])
