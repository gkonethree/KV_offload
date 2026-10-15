# SPDX-License-Identifier: Apache-2.0
"""Per-layer GPU staging pool: stage a block only for the layer that needs it.

KVStagingPool stages a block for ALL layers into one shared slot, so a block
any one layer selects costs every layer's bytes and a slot in every layer.
Layers select different blocks, so the shared pool either has to hold nearly
the whole context or thrashes between layers within a step.

Here each layer owns the same tail pages of its own KV tensor (identical GPU
memory to KVStagingPool with the same ``num_slots``). A fetch copies one
layer's slice of a block (``1/num_layers`` of the bytes). Staged copies are
tracked here only: the shared ResidencyTable keeps offloaded blocks as CPU, so
the scheduler, the offload manager and the build-time index remediation (which
nulls CPU entries every step) behave as before.

Because one paged-index array is shared by all layers of a step and a slot id
only means something within its own layer, every layer re-points the entries
of the offloaded blocks it selected. To keep that off the host, the pool keeps
per-request DEVICE tables, indexed [layer, block_idx]:
  slot_tbl   staging slot of the block in that layer, -1 if not staged
  last_use   tick of the last step this layer read it (LRU order)
plus an is-offloaded mask per request. The wrapper re-points staged blocks with
one torch.where and only syncs to learn whether anything is missing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.logger import init_logger

from offload_package import timing as _timing
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.base import BlockRef
from offload_package.staging.residency import Residency, ResidencyTable

logger = init_logger(__name__)


class PerLayerStagingPool:
    per_layer = True

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
        ring: int = 4,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be > 0")
        if staging_base_page + num_slots > gpu_k_caches[0].shape[0]:
            raise ValueError("staging pages do not fit in GPU cache")
        self.num_slots = int(num_slots)
        self.base = int(staging_base_page)
        self.gpu_k_caches = list(gpu_k_caches)
        self.gpu_v_caches = list(gpu_v_caches)
        self.cpu_cache = cpu_cache                  # [L, S, 2, P, H, C] pinned
        self.residency = residency
        self.offload_manager = offload_manager
        self.num_layers = len(self.gpu_k_caches)
        self.device = self.gpu_k_caches[0].device
        # Host-side ownership: which ref occupies which slot, per layer.
        self._maps: list[dict[BlockRef, int]] = [{} for _ in range(self.num_layers)]
        self._free: list[list[int]] = [
            list(range(self.num_slots - 1, -1, -1)) for _ in range(self.num_layers)]
        # Device tables per request (see module docstring).
        self._slot_tbl: dict[str, torch.Tensor] = {}
        self._last_use: dict[str, torch.Tensor] = {}
        self._cpu_mask: dict[str, tuple[int, torch.Tensor]] = {}
        self.tick = 0
        self._layer_of_ptr = {t.data_ptr(): i for i, t in enumerate(self.gpu_k_caches)}
        self._copy_stream: torch.cuda.Stream | None = None
        self._ring = int(ring)
        self._slabs: list[torch.Tensor] = []
        self._slab_events: list[object | None] = []
        self._slab_turn = 0
        self.stats = dict(fetch_calls=0, blocks_fetched=0, bytes_h2d=0,
                          evictions=0, skipped=0, sel_hits=0, sel_misses=0)

    # ---- layer identity / streams ----------------------------------------------

    def layer_of(self, kv_tensor: torch.Tensor) -> int | None:
        """Layer index of a per-layer KV tensor (or a view sharing its storage start)."""
        return self._layer_of_ptr.get(kv_tensor.data_ptr())

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        return self._copy_stream

    # ---- device tables ---------------------------------------------------------

    # The tables are created and mostly updated inside vLLM's forward, which runs
    # under torch.inference_mode, so they are inference tensors; every write
    # (including release() from the scheduler, outside that mode) is wrapped in
    # inference_mode so it stays legal.

    @torch.inference_mode()
    def _grow(self, table: dict, rid: str, n: int, fill: int, dtype) -> torch.Tensor:
        t = table.get(rid)
        if t is None or t.shape[1] < n:
            cap = max(n, 64, 2 * (t.shape[1] if t is not None else 0))
            new = torch.full((self.num_layers, cap), fill, dtype=dtype, device=self.device)
            if t is not None:
                new[:, : t.shape[1]] = t
            table[rid] = t = new
        return t

    def slot_table(self, rid: str, n_blocks: int) -> torch.Tensor:
        """[num_layers, >= n_blocks] int32 device table, -1 = not staged."""
        return self._grow(self._slot_tbl, rid, n_blocks, -1, torch.int32)

    def last_use_table(self, rid: str, n_blocks: int) -> torch.Tensor:
        return self._grow(self._last_use, rid, n_blocks, 0, torch.int64)

    def cpu_mask(self, rid: str, n_blocks: int) -> torch.Tensor:
        """[n_blocks] bool device mask of the request's offloaded (CPU) blocks,
        rebuilt only when the residency table's CPU set changed."""
        ver = self.residency.cpu_version
        hit = self._cpu_mask.get(rid)
        if hit is not None and hit[0] == ver and hit[1].numel() >= n_blocks:
            return hit[1][:n_blocks]
        mask = torch.zeros(n_blocks, dtype=torch.bool)
        for ref in self.residency.refs_for_request(rid):
            loc = self.residency.get(ref)
            if loc is not None and loc.residency == Residency.CPU and ref.block_idx < n_blocks:
                mask[ref.block_idx] = True
        mask = mask.to(self.device)
        self._cpu_mask[rid] = (ver, mask)
        return mask

    @torch.inference_mode()
    def _set_slots(self, layer: int, refs: list[BlockRef], vals: list[int]) -> None:
        by_req: dict[str, tuple[list[int], list[int]]] = {}
        for r, v in zip(refs, vals):
            ix, vs = by_req.setdefault(r.request_id, ([], []))
            ix.append(r.block_idx)
            vs.append(v)
        for rid, (ix, vs) in by_req.items():
            tbl = self.slot_table(rid, max(ix) + 1)
            tbl[layer, torch.tensor(ix, device=self.device)] = torch.tensor(
                vs, dtype=torch.int32, device=self.device)

    # ---- staging --------------------------------------------------------------

    def _acquire_slab(self) -> torch.Tensor:
        if not self._slabs:
            src = self.gpu_k_caches[0]                  # [num_blocks, H, P, 2C]
            shape = (self.num_slots, *src.shape[1:])
            for _ in range(self._ring):
                self._slabs.append(torch.empty(shape, dtype=src.dtype, device="cpu",
                                               pin_memory=True))
                self._slab_events.append(None)
        i = self._slab_turn
        self._slab_turn = (i + 1) % self._ring
        ev = self._slab_events[i]
        if ev is not None:
            _timing.start("plpool.h2d_sync")
            ev.synchronize()
            _timing.end("plpool.h2d_sync")
            self._slab_events[i] = None
        self._slab_cur = i
        return self._slabs[i]

    def _victims(self, layer: int, keep: set) -> list[BlockRef]:
        """This layer's staged refs not in `keep`, least recently used first."""
        cands = [r for r in self._maps[layer] if r not in keep]
        if not cands:
            return []
        by_req: dict[str, list[int]] = {}
        for i, r in enumerate(cands):
            by_req.setdefault(r.request_id, []).append(i)
        ticks = [0] * len(cands)
        for rid, idxs in by_req.items():
            row = self.last_use_table(rid, 1)[layer].cpu().tolist()
            for i in idxs:
                b = cands[i].block_idx
                ticks[i] = row[b] if b < len(row) else 0
        order = sorted(range(len(cands)), key=ticks.__getitem__)
        return [cands[i] for i in order]

    def ensure_resident_layer(
        self,
        layer: int,
        refs: Sequence[BlockRef],
        protect: Sequence[BlockRef] | None = None,
    ) -> tuple[dict[BlockRef, int], list[BlockRef]]:
        """Stage `refs` (CPU-resident blocks) into `layer`'s region.

        Returns ``(staged, skipped)`` like KVStagingPool.ensure_resident:
        `staged` maps every ref resident in this layer afterwards to its slot.
        `protect` refs (already staged here, used by this layer's current
        gather) are never evicted.
        """
        m, free = self._maps[layer], self._free[layer]
        result: dict[BlockRef, int] = {}
        to_fetch: list[tuple[BlockRef, int]] = []
        for ref in dict.fromkeys(refs):
            slot = m.get(ref)
            if slot is not None:
                result[ref] = slot
                continue
            loc = self.residency.get(ref)
            if loc is None:
                raise KeyError(f"Unknown logical block {ref}")
            if loc.residency == Residency.GPU:
                continue
            if loc.cpu_slot is None:
                raise RuntimeError(f"{ref} is not backed by a CPU slot")
            self.offload_manager.wait_cpu_slot(loc.cpu_slot)
            to_fetch.append((ref, loc.cpu_slot))
        if not to_fetch:
            return result, []

        _timing.start("plpool.alloc")
        allocs: list[tuple[BlockRef, int, int]] = []
        skipped: list[BlockRef] = []
        evicted: list[BlockRef] = []
        victims = None
        for ref, cpu_slot in to_fetch:
            if free:
                slot = free.pop()
            else:
                if victims is None:
                    keep = set(protect or ()) | set(result)
                    victims = iter(self._victims(layer, keep))
                victim = next(victims, None)
                if victim is None:
                    skipped.append(ref)
                    continue
                slot = m.pop(victim)
                evicted.append(victim)
            allocs.append((ref, cpu_slot, slot))
        if evicted:
            self._set_slots(layer, evicted, [-1] * len(evicted))
            self.stats['evictions'] += len(evicted)
        _timing.end("plpool.alloc")
        self.stats['skipped'] += len(skipped)
        if not allocs:
            return result, skipped

        n = len(allocs)
        slab = self._acquire_slab()
        try:
            _timing.start("plpool.cpu_gather")
            cpu_slots = torch.tensor([cs for _, cs, _ in allocs], dtype=torch.long)
            g = self.cpu_cache[layer].index_select(0, cpu_slots)     # [n, 2, P, H, C]
            slab[:n].copy_(g.permute(0, 3, 2, 1, 4).reshape(n, *slab.shape[1:]))
            _timing.end("plpool.cpu_gather")
            _timing.start("plpool.h2d")
            cur = torch.cuda.current_stream(device=self.device)
            with torch.cuda.stream(self.copy_stream):
                # Slots being overwritten may still be read by kernels already
                # queued on the compute stream (an earlier chunk of this layer).
                self.copy_stream.wait_stream(cur)
                ids = torch.tensor([s for _, _, s in allocs], dtype=torch.long,
                                   device=self.device) + self.base
                self.gpu_k_caches[layer].index_copy_(
                    0, ids, slab[:n].to(self.device, non_blocking=True))
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
            cur.wait_stream(self.copy_stream)
            self._slab_events[self._slab_cur] = ev
            _timing.end("plpool.h2d")
        except Exception:
            for _, _, slot in allocs:
                free.append(slot)
            raise
        for ref, _cs, slot in allocs:
            m[ref] = slot
            result[ref] = slot
        self._set_slots(layer, [r for r, _, _ in allocs], [s for _, _, s in allocs])
        self.stats['fetch_calls'] += 1
        self.stats['blocks_fetched'] += n
        self.stats['bytes_h2d'] += n * slab[0].numel() * slab.element_size()
        return result, skipped

    def ensure_resident(self, refs, protect=None):
        raise NotImplementedError(
            "PerLayerStagingPool stages per layer; call ensure_resident_layer(layer, ...). "
            "The 1-step-lag prefetch (oracle-family methods) is not supported in "
            "staging_granularity='layer'.")

    # ---- release ---------------------------------------------------------------

    def release(self, ref) -> None:
        """Drop `ref` from every layer (request finished / block freed)."""
        if not isinstance(ref, BlockRef):
            return
        for layer, (m, free) in enumerate(zip(self._maps, self._free)):
            slot = m.pop(ref, None)
            if slot is not None:
                free.append(slot)
                self._set_slots(layer, [ref], [-1])
        rid = ref.request_id
        if not any(r.request_id == rid for mm in self._maps for r in mm):
            self._slot_tbl.pop(rid, None)
            self._last_use.pop(rid, None)
            self._cpu_mask.pop(rid, None)
