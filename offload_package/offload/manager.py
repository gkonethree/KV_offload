from __future__ import annotations

from collections.abc import Sequence

import torch

from ..scoring.base import BlockRef, BlockScorer, BlockKVStats
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
        """Score block using all attention layers, aggregate by mean."""
        # Skip warmup requests
        if str(ref.request_id).startswith("_warmup_"):
            return
            
        all_v_norms = []
        all_k_norms = []
        for k_cache, v_cache in zip(self.gpu_k_caches, self.gpu_v_caches):
            stats = compute_block_kv_norms(
                k_cache,
                v_cache,
                [gpu_block_id],
                logical_block_ids=[ref],
                layout=self.kv_layout,
            )
            all_v_norms.append(stats.v_norm)
            all_k_norms.append(stats.k_norm)
        # Mean across layers
        v_norm = torch.stack(all_v_norms).mean(dim=0)
        k_norm = torch.stack(all_k_norms).mean(dim=0)
        
        self.scorer.update(BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=[ref]))

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
        
    def offload_fraction(
        self,
        candidates: Sequence[BlockRef],
        fraction: float,
        scored_blocks: set[BlockRef] | None = None,
    ) -> list[BlockRef]:
        """Offload the lowest-scoring fraction of complete GPU blocks."""

        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction must be in [0, 1)")

        # Only consider blocks that have been scored
        if scored_blocks is None:
            scored_blocks = set()
        
        scored_candidates = [
            ref
            for ref in candidates
            if ref in scored_blocks
            and (loc := self.residency.get(ref)) is not None
            and loc.residency == Residency.GPU
            and loc.complete
        ]

        n_offload = int(len(scored_candidates) * fraction)

        if n_offload <= 0:
            return []

        # select_for_eviction() interprets budget as the number to KEEP.
        keep = len(scored_candidates) - n_offload

        refs_to_evict = self.scorer.select_for_eviction(
            scored_candidates,
            keep,
        )

        offloaded = []
        events = []

        for ref in refs_to_evict:
            loc = self.residency.get(ref)

            if (
                loc is None
                or loc.gpu_block_id is None
                or loc.residency != Residency.GPU
            ):
                continue

            try:
                cpu_slot, event = self._copy_gpu_to_cpu(loc.gpu_block_id)
                events.append(event)

                self.residency.mark_cpu(
                    ref,
                    cpu_slot,
                )

                offloaded.append(ref)
            except Exception as e:
                continue

        # Wait for all async copies to complete
        for event in events:
            event.synchronize()

        return offloaded

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
        events = []
        for ref in refs_to_evict:
            loc = self.residency.get(ref)
            if loc is None or loc.gpu_block_id is None or loc.residency != Residency.GPU:
                continue
            cpu_slot, event = self._copy_gpu_to_cpu(loc.gpu_block_id)
            events.append(event)
            self.residency.mark_cpu(ref, cpu_slot)
            offloaded.append(ref)
        # Wait for all async copies to complete
        for event in events:
            event.synchronize()
        return offloaded

    def _copy_gpu_to_cpu(self, gpu_block_id: int) -> tuple[int, torch.cuda.Event]:
        slot = self._cpu_pool.allocate()
        with torch.cuda.stream(self._copy_stream):
            # vLLM's K/V caches are separate per attention layer.
            for layer, (k_cache, v_cache) in enumerate(
                zip(self.gpu_k_caches, self.gpu_v_caches)
            ):
                # GPU cache has shape [num_kv_heads, page_size, 2*head_dim]
                # CPU cache expects [page_size, num_kv_heads, 2*head_dim]
                k_block = k_cache[gpu_block_id]
                v_block = v_cache[gpu_block_id]
                
                # Transpose from [num_kv_heads, page_size, 2*head_dim] 
                # to [page_size, num_kv_heads, 2*head_dim]
                k_block = k_block.permute(1, 0, 2)
                v_block = v_block.permute(1, 0, 2)
                
                self.cpu_cache[layer, slot, 0].copy_(k_block, non_blocking=True)
                self.cpu_cache[layer, slot, 1].copy_(v_block, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        self._store_events[slot] = event
        return slot, event

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
