from __future__ import annotations

from collections.abc import Sequence

import torch

from ..scoring.base import BlockRef, BlockScorer, BlockKVStats
from ..scoring.norm_kernel import compute_block_kv_norms
from ..staging.residency import Residency, ResidencyTable
from offload_package.debug import dbg as _dbg


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
            if k_cache is v_cache:
                # Combined per-layer page [num_blocks, H, N, 2*head_dim]: K in
                # [..., :head_dim], V in [..., head_dim:]. After the split the
                # views are [B, H, N, C] which is HND ordering.
                head_dim = k_cache.shape[-1] // 2
                k_in = k_cache[..., :head_dim]
                v_in = v_cache[..., head_dim:]
                layout = "HND"
            else:
                k_in, v_in, layout = k_cache, v_cache, self.kv_layout
            stats = compute_block_kv_norms(
                k_in,
                v_in,
                [gpu_block_id],
                logical_block_ids=[ref],
                layout=layout,
            )
            all_v_norms.append(stats.v_norm)
            all_k_norms.append(stats.k_norm)
        # Mean across layers
        v_norm = torch.stack(all_v_norms).mean(dim=0)
        k_norm = torch.stack(all_k_norms).mean(dim=0)

        self.scorer.update(BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=[ref]))

    def score_blocks(self, refs: list[BlockRef], gpu_block_ids: list[int]) -> None:
        """Batch-score many complete blocks: ONE index_select per layer over
        all blocks and ONE host sync total, instead of per-block*per-layer
        syncs. Score values are identical per block to score_block (mean of
        the same norms over the same layers is linear).
        """
        items = [
            (ref, gid)
            for ref, gid in zip(refs, gpu_block_ids)
            if not str(ref.request_id).startswith("_warmup_")
        ]
        if not items:
            return
        refs = [r for r, _ in items]
        gpu_ids = torch.tensor(
            [g for _, g in items], dtype=torch.long,
            device=self.gpu_k_caches[0].device,
        )
        all_v_norms = []
        all_k_norms = []
        for k_cache, v_cache in zip(self.gpu_k_caches, self.gpu_v_caches):
            if k_cache is v_cache:
                head_dim = k_cache.shape[-1] // 2
                k_in = k_cache[..., :head_dim]
                v_in = v_cache[..., head_dim:]
                layout = "HND"
            else:
                k_in, v_in, layout = k_cache, v_cache, self.kv_layout
            stats = compute_block_kv_norms(
                k_in,
                v_in,
                gpu_ids.tolist(),
                logical_block_ids=refs,
                layout=layout,
            )
            all_v_norms.append(stats.v_norm)
            all_k_norms.append(stats.k_norm)
        v_norm = torch.stack(all_v_norms).mean(dim=0)
        k_norm = torch.stack(all_k_norms).mean(dim=0)
        self.scorer.update(BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=refs))

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
        _dbg(f"[OFFLOAD MANAGER DEBUG] offload_fraction called: candidates={len(candidates)}, fraction={fraction}")

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
        _dbg(f"[OFFLOAD MANAGER DEBUG] scored_candidates={len(scored_candidates)}, scored_blocks={len(scored_blocks)}, candidates={len(candidates)}")

        # Use ceiling to ensure at least 1 block is offloaded when fraction > 0
        n_offload = max(1, int(len(scored_candidates) * fraction + 0.5))
        _dbg(f"[OFFLOAD MANAGER DEBUG] n_offload={n_offload}, scored_candidates={len(scored_candidates)}, fraction={fraction}")

        if n_offload <= 0 or len(scored_candidates) == 0:
            _dbg(f"[OFFLOAD MANAGER DEBUG] returning empty: n_offload={n_offload}, scored_candidates={len(scored_candidates)}")
            return []

        # select_for_eviction() interprets budget as the number to KEEP.
        keep = len(scored_candidates) - n_offload

        refs_to_evict = self.scorer.select_for_eviction(
            scored_candidates,
            keep,
        )
        _dbg(f"[OFFLOAD MANAGER DEBUG] refs_to_evict={refs_to_evict}")

        offloaded = []
        events = []

        for ref in refs_to_evict:
            loc = self.residency.get(ref)
            _dbg(f"[OFFLOAD MANAGER DEBUG] Processing ref={ref}, loc={loc}")

            if loc is None:
                _dbg(f"[OFFLOAD MANAGER DEBUG] Skipping: loc is None")
                continue
            if loc.gpu_block_id is None:
                _dbg(f"[OFFLOAD MANAGER DEBUG] Skipping: gpu_block_id is None")
                continue
            if loc.residency != Residency.GPU:
                _dbg(f"[OFFLOAD MANAGER DEBUG] Skipping: residency={loc.residency}")
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
                _dbg(f"[OFFLOAD MANAGER DEBUG] Exception during copy: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Wait for all async copies to complete
        for event in events:
            event.synchronize()

        _dbg(f"[OFFLOAD MANAGER DEBUG] returning offloaded={len(offloaded)}")
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
        _dbg(f"[OFFLOAD MANAGER DEBUG] returning offloaded={len(offloaded)}")
        return offloaded

    def _copy_gpu_to_cpu(self, gpu_block_id: int) -> tuple[int, torch.cuda.Event]:
        slot = self._cpu_pool.allocate()
        with torch.cuda.stream(self._copy_stream):
            # vLLM's K/V caches are separate per attention layer.
            # GPU cache has shape [num_pages, num_kv_heads, page_size, head_dim*2] with K/V combined
            # CPU cache expects separate K/V: [layers, slots, 2, page_size, num_kv_heads, head_dim]
            for layer, (k_cache, v_cache) in enumerate(
                zip(self.gpu_k_caches, self.gpu_v_caches)
            ):
                k_block = k_cache[gpu_block_id]
                v_block = v_cache[gpu_block_id]
                
                # GPU cache: [num_kv_heads, page_size, 2*head_dim] with K and V combined
                # Split K and V from combined dimension
                head_dim = k_block.shape[-1] // 2
                k_block = k_block[..., :head_dim]  # [num_kv_heads, page_size, head_dim]
                v_block = v_block[..., head_dim:]  # [num_kv_heads, page_size, head_dim]
                
                # Transpose from [num_kv_heads, page_size, head_dim] to [page_size, num_kv_heads, head_dim]
                k_block = k_block.permute(1, 0, 2)
                v_block = v_block.permute(1, 0, 2)
                
                # Copy to CPU cache
                # cpu_cache: [layers, slots, 2, page_size, num_kv_heads, head_dim]
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
