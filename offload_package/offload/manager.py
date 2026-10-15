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

        fused = None
        if self._fused_scorer_ok():
            fused = self._fused_block_norms([g for _, g in items])

        if fused is not None:
            # fused -> (k_norm, v_norm) [L, N, H]; mean over layers -> [N, H]
            v_norm = fused[1].mean(dim=0)
            k_norm = fused[0].mean(dim=0)
        else:
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

    def _fused_scorer_ok(self) -> bool:
        """True when the current caches fit the fused scorer layout."""
        import os as _os
        if _os.environ.get("OFFLOAD_PACKAGE_FUSED_SCORER", "1") == "0":
            return False
        if not (self.gpu_k_caches and self.gpu_v_caches):
            return False
        impl = _os.environ.get("OFFLOAD_PACKAGE_SCORER_IMPL", "cuda")
        if impl == "cuda":
            try:
                from offload_package.scoring.cuda_norms import available
                if not available():
                    return False
            except Exception:
                return False
        else:
            try:
                from offload_package.offload.triton_score import _HAS_TRITON
                if not _HAS_TRITON:
                    return False
            except Exception:
                return False
        first = self.gpu_k_caches[0]
        return (
            first.dim() == 4
            and first.dtype in (torch.float16, torch.bfloat16)
            and first.stride(3) == 1
            and self.gpu_k_caches[0] is self.gpu_v_caches[0]
        )

    def _fused_block_norms(self, gpu_block_ids: list[int]):
        import os as _os
        impl = _os.environ.get("OFFLOAD_PACKAGE_SCORER_IMPL", "cuda")
        if impl == "cuda":
            from offload_package.scoring.cuda_norms import score_blocks_fused
        else:
            from offload_package.offload.triton_score import score_blocks_fused
        return score_blocks_fused(self.gpu_k_caches, gpu_block_ids)

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
        """Offload the lowest-scoring ``fraction`` of the scored, complete,
        GPU-resident ``candidates`` (rounded to nearest; 0 when fraction=0)."""
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction must be in [0, 1)")
        eligible = self._eligible(candidates, scored_blocks)
        n_offload = int(len(eligible) * fraction + 0.5)
        return self.offload_lowest(eligible, n_offload)

    def _eligible(
        self,
        candidates: Sequence[BlockRef],
        scored_blocks: set[BlockRef] | None,
    ) -> list[BlockRef]:
        scored_blocks = scored_blocks if scored_blocks is not None else set()
        return [
            ref
            for ref in candidates
            if ref in scored_blocks
            and (loc := self.residency.get(ref)) is not None
            and loc.residency == Residency.GPU
            and loc.complete
            and loc.gpu_block_id is not None
        ]

    def offload_lowest(
        self,
        eligible: Sequence[BlockRef],
        n_offload: int,
    ) -> list[BlockRef]:
        """Copy the ``n_offload`` lowest-scored refs of ``eligible`` (all
        scored, complete, GPU-resident) to CPU and mark them CPU-resident.

        Raises on copy failure (including CPU pool exhaustion): a silent
        partial offload would desync the caller's accounting.
        """
        _dbg(f"[OFFLOAD MANAGER DEBUG] offload_lowest: eligible={len(eligible)}, n_offload={n_offload}")
        if n_offload <= 0 or not eligible:
            return []
        n_offload = min(n_offload, len(eligible))
        # select_for_eviction() interprets budget as the number to KEEP.
        refs_to_evict = self.scorer.select_for_eviction(
            list(eligible), len(eligible) - n_offload,
        )
        gpu_ids = [self.residency.get(ref).gpu_block_id for ref in refs_to_evict]
        if not refs_to_evict:
            return []
        # One batched pass: per-layer index_select + one big D2H burst.
        cpu_slots, event = self._copy_gpu_to_cpu_batch(gpu_ids)
        event.synchronize()
        for ref, slot in zip(refs_to_evict, cpu_slots):
            self.residency.mark_cpu(ref, slot)
        _dbg(f"[OFFLOAD MANAGER DEBUG] offloaded={len(refs_to_evict)}")
        return list(refs_to_evict)

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
        # Keep only blocks that still have a live GPU residency
        refs_with_gpu: list[BlockRef] = []
        gpu_ids: list[int] = []
        for ref in refs_to_evict:
            loc = self.residency.get(ref)
            if loc is None or loc.gpu_block_id is None or loc.residency != Residency.GPU:
                continue
            refs_with_gpu.append(ref)
            gpu_ids.append(loc.gpu_block_id)

        offloaded: list[BlockRef] = []
        if refs_with_gpu:
            cpu_slots, event = self._copy_gpu_to_cpu_batch(gpu_ids)
            event.synchronize()
            for ref, cpu_slot in zip(refs_with_gpu, cpu_slots):
                self.residency.mark_cpu(ref, cpu_slot)
                offloaded.append(ref)
        _dbg(f"[OFFLOAD MANAGER DEBUG] returning offloaded={len(offloaded)}")
        return offloaded

    def _copy_gpu_to_cpu_batch(
        self, gpu_block_ids: list[int]
    ) -> tuple[list[int], torch.cuda.Event]:
        """Batched store: copy many pages GPU→CPU with per-layer giant IO.

        Per layer: one index_select gather over shared pages, one contiguous
        slab D2H (stream-ordered), then one host scatter into the pinned CPU
        cache. Replaces the old per-block loop (~44 python ops + 44 tiny DMA
        launches per block) with ~L kernels + 1 PCIe burst per event.
        """
        n = len(gpu_block_ids)
        if n <= 0:
            raise ValueError("empty batch")
        if n > self._cpu_pool.num_free:
            raise RuntimeError(
                f"CPU offload pool exhausted: need {n}, free {self._cpu_pool.num_free}"
            )

        slots = [self._cpu_pool.allocate() for _ in range(n)]
        dev = self.gpu_k_caches[0].device
        gpu_ids_t = torch.tensor(gpu_block_ids, dtype=torch.long, device=dev)
        slot_ids_t = torch.tensor(slots, dtype=torch.long)

        # Run the whole batch on the CURRENT stream and keep every temporary
        # referenced until the event sync: no side-stream UAF, no allocator
        # recycling of in-flight reads.
        gathered_hold: list[torch.Tensor] = []
        slab_hold_gpu: list[torch.Tensor] = []
        host_hold: list[torch.Tensor] = []
        try:
            for layer, (k_cache, _v_cache) in enumerate(
                zip(self.gpu_k_caches, self.gpu_v_caches)
            ):
                gathered = k_cache.index_select(0, gpu_ids_t)       # [n, H, N, 2C]
                head_dim = gathered.shape[-1] // 2
                k_part = gathered[..., :head_dim].permute(0, 2, 1, 3)  # [n, N, H, C]
                v_part = gathered[..., head_dim:].permute(0, 2, 1, 3)
                slab = torch.stack([k_part, v_part], dim=1).contiguous()  # [n, 2, N, H, C]
                slab_host = slab.to("cpu", non_blocking=True)
                gathered_hold.append(gathered)
                slab_hold_gpu.append(slab)
                host_hold.append(slab_host)
            event_order = torch.cuda.Event()
            event_order.record()
            event_order.synchronize()
            for layer, slab_host in enumerate(host_hold):
                self.cpu_cache[layer].index_copy_(0, slot_ids_t, slab_host)
        except Exception:
            for s in slots:
                self._cpu_pool.free(s)
            raise
        finally:
            del gathered_hold, slab_hold_gpu, host_hold

        for slot in slots:
            self._store_events[slot] = event_order
        return slots, event_order

    def _copy_gpu_to_cpu(self, gpu_block_id: int) -> tuple[int, torch.cuda.Event]:
        """Single-block store path (kept for small-batch callers)."""
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
