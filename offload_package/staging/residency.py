from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch

from ..scoring.base import BlockRef

# Device-mirror state codes (dev_state). 0 doubles as "untracked": a position
# the table does not know never needs a fetch.
DEV_GPU, DEV_STAGING, DEV_CPU = 0, 1, 2


class Residency(Enum):
    GPU = auto()
    CPU = auto()
    STAGING = auto()


@dataclass
class BlockLocation:
    residency: Residency
    gpu_block_id: Optional[int] = None
    cpu_slot: Optional[int] = None
    staging_slot: Optional[int] = None
    complete: bool = False


class ResidencyTable:
    """Authoritative logical-block -> physical-residency map."""

    def __init__(self) -> None:
        self._blocks: dict[BlockRef, BlockLocation] = {}
        # Number of blocks whose ONLY backing store is the CPU copy (i.e.
        # needing a fetch before the kernel can read them). Used as a cheap
        # short-circuit for per-step fetch probes.
        self._num_cpu = 0
        # Bumped on every CPU-set membership change (blocks entering or
        # leaving CPU-only residency). The lag prefetch compares it against
        # the last-seen value to skip its discovery pass when nothing
        # transitioned: with no transitions, the previously staged set is
        # still valid and the same-step in-run fetch covers any new picks.
        self._cpu_version = 0
        # Per-request block index so refs_for_request does not scan the whole
        # table (it runs per request per decode layer on the hot path).
        self._by_request: dict[str, dict[int, BlockRef]] = {}
        # Optional GPU mirror (enable_device_mirror): lets the decode wrapper
        # classify every selected token against residency with a few tensor
        # ops instead of a per-block Python loop. The dict above stays the
        # source of truth; transitions only mark refs dirty (O(1)) and the
        # mirror is brought up to date in one batched write by flush_device().
        self._mirror = False
        # Called with the mirror slot of every released request, before the
        # slot can be reused (TokenStagingStore drops its token map there).
        self.on_slot_release: list = []

    # ---- device mirror ------------------------------------------------------

    def enable_device_mirror(self, *, num_layers: int, max_blocks: int,
                             num_slots: int, device) -> None:
        """Allocate the GPU mirror:

        dev_state [S, MB] int8      DEV_GPU / DEV_STAGING / DEV_CPU per block
        dev_last_use [S, MB] int64  tick of the last layer that selected it

        S = request slots (stable per request, NOT the batch row), MB = max
        blocks per request (ceil(max_model_len / page)).
        """
        self._mirror = True
        self._nl, self._mb, self._dev = int(num_layers), int(max_blocks), device
        self._slot_of: dict[str, int] = {}
        self._free_slots: list[int] = []
        self._num_slots = 0
        self._grow_mirror(max(1, int(num_slots)))
        self._dirty: set[BlockRef] = set()
        self._rows_to_clear: list[int] = []
        # Bumped once per layer-level discovery; dev_last_use == tick marks the
        # blocks the CURRENT layer selected (never evicted while it runs).
        self.tick = 0
        for rid in self._by_request:
            self._assign_slot(rid)
        self._dirty.update(self._blocks)

    @property
    def has_device_mirror(self) -> bool:
        return self._mirror

    def _grow_mirror(self, n: int) -> None:
        old = (getattr(self, "dev_state", None), getattr(self, "dev_last_use", None))
        st = torch.zeros((n, self._mb), dtype=torch.int8, device=self._dev)
        lu = torch.zeros((n, self._mb), dtype=torch.int64, device=self._dev)
        if old[0] is not None:
            k = old[0].shape[0]
            st[:k].copy_(old[0]); lu[:k].copy_(old[1])
        self.dev_state, self.dev_last_use = st, lu
        self._free_slots.extend(range(n - 1, self._num_slots - 1, -1))
        self._num_slots = n

    def _assign_slot(self, request_id: str) -> int:
        slot = self._slot_of.get(request_id)
        if slot is None:
            if not self._free_slots:
                self._grow_mirror(self._num_slots * 2)
            slot = self._free_slots.pop()
            self._slot_of[request_id] = slot
        return slot

    def _release_slot(self, request_id: str) -> None:
        slot = self._slot_of.pop(request_id, None)
        if slot is not None:
            # Cleared at the next flush, BEFORE dirty writes, so a request
            # that reuses the slot in the meantime still ends up correct.
            self._rows_to_clear.append(slot)
            self._free_slots.append(slot)
            for cb in self.on_slot_release:
                cb(slot)

    def _touch(self, ref: BlockRef) -> None:
        if self._mirror:
            self._dirty.add(ref)

    def slots_for(self, request_ids) -> list[int]:
        """Mirror slot per request id (-1 when untracked)."""
        return [self._slot_of.get(str(r), -1) for r in request_ids]

    def flush_device(self) -> None:
        """Write every pending transition into the GPU mirror (batched)."""
        if not self._mirror or not (self._dirty or self._rows_to_clear):
            return
        dev = self._dev
        if self._rows_to_clear:
            rows = torch.tensor(self._rows_to_clear, dtype=torch.long, device=dev)
            self.dev_state[rows] = 0
            self.dev_last_use[rows] = 0
            self._rows_to_clear = []
        rows, blks, states = [], [], []
        mb = self._mb
        for ref in self._dirty:
            slot = self._slot_of.get(ref.request_id)
            if slot is None:
                continue                      # request gone: row was cleared
            if ref.block_idx >= mb:
                raise RuntimeError(
                    f"block {ref.block_idx} exceeds the device mirror width {mb}")
            loc = self._blocks.get(ref)
            rows.append(slot); blks.append(ref.block_idx)
            if loc is None or loc.residency == Residency.GPU:
                states.append(DEV_GPU)
            elif loc.residency == Residency.CPU:
                states.append(DEV_CPU)
            else:
                states.append(DEV_STAGING)
        self._dirty.clear()
        if not rows:
            return
        t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=dev)
        rows_t, blks_t = t(rows), t(blks)
        self.dev_state[rows_t, blks_t] = t(states, torch.int8)

    def last_use_of(self, refs) -> list[int]:
        """dev_last_use for each ref (one gather + one readback)."""
        rows, blks, keep = [], [], []
        for i, ref in enumerate(refs):
            slot = self._slot_of.get(ref.request_id)
            if slot is not None and ref.block_idx < self._mb:
                rows.append(slot); blks.append(ref.block_idx); keep.append(i)
        out = [0] * len(refs)
        if rows:
            dev = self._dev
            vals = self.dev_last_use[
                torch.tensor(rows, dtype=torch.long, device=dev),
                torch.tensor(blks, dtype=torch.long, device=dev)].tolist()
            for i, v in zip(keep, vals):
                out[i] = int(v)
        return out

    @property
    def num_cpu_blocks(self) -> int:
        return self._num_cpu

    @property
    def cpu_version(self) -> int:
        return self._cpu_version

    def mark_gpu(self, ref: BlockRef, gpu_block_id: int, *, complete: bool = False) -> None:
        old = self._blocks.get(ref)
        if old is not None and old.residency == Residency.CPU:
            self._num_cpu -= 1
            self._cpu_version += 1
        self._blocks[ref] = BlockLocation(
            residency=Residency.GPU,
            gpu_block_id=int(gpu_block_id),
            cpu_slot=old.cpu_slot if old else None,
            complete=bool(complete),
        )
        self._by_request.setdefault(ref.request_id, {})[ref.block_idx] = ref
        if self._mirror:
            self._assign_slot(ref.request_id)
            self._dirty.add(ref)

    def mark_complete(self, ref: BlockRef) -> None:
        loc = self._blocks[ref]
        loc.complete = True

    def mark_cpu(self, ref: BlockRef, cpu_slot: int) -> None:
        old = self._blocks.get(ref)
        if old is None or old.gpu_block_id is None:
            raise KeyError(f"No GPU mapping recorded for {ref}")
        if old.residency != Residency.CPU:
            self._num_cpu += 1
            self._cpu_version += 1
        self._blocks[ref] = BlockLocation(
            residency=Residency.CPU,
            gpu_block_id=old.gpu_block_id,
            cpu_slot=int(cpu_slot),
            complete=old.complete,
        )
        self._touch(ref)

    def mark_staging(self, ref: BlockRef, staging_slot: int) -> None:
        old = self._blocks.get(ref)
        if old is None or old.gpu_block_id is None:
            raise KeyError(f"No GPU mapping recorded for {ref}")
        if old.cpu_slot is None:
            raise KeyError(f"No CPU slot recorded for {ref}")
        if old.residency == Residency.CPU:
            self._num_cpu -= 1
            self._cpu_version += 1
        self._blocks[ref] = BlockLocation(
            residency=Residency.STAGING,
            gpu_block_id=old.gpu_block_id,
            cpu_slot=old.cpu_slot,
            staging_slot=int(staging_slot),
            complete=old.complete,
        )
        self._touch(ref)

    def get(self, ref: BlockRef) -> Optional[BlockLocation]:
        return self._blocks.get(ref)

    def remove(self, ref: BlockRef) -> Optional[BlockLocation]:
        old = self._blocks.pop(ref, None)
        if old is not None:
            req_blocks = self._by_request.get(ref.request_id)
            if req_blocks is not None:
                req_blocks.pop(ref.block_idx, None)
                if not req_blocks:
                    self._by_request.pop(ref.request_id, None)
                    if self._mirror:
                        self._release_slot(ref.request_id)
            self._touch(ref)
            if old.residency == Residency.CPU:
                self._num_cpu -= 1
                self._cpu_version += 1
        return old

    def all_gpu_blocks(self, *, complete_only: bool = True) -> list[BlockRef]:
        return [
            ref for ref, loc in self._blocks.items()
            if loc.residency == Residency.GPU and (loc.complete or not complete_only)
        ]

    def refs_for_request(self, request_id: str) -> list[BlockRef]:
        req_blocks = self._by_request.get(request_id)
        if not req_blocks:
            return []
        return [req_blocks[idx] for idx in sorted(req_blocks)]

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for loc in self._blocks.values():
            counts[loc.residency.name] = counts.get(loc.residency.name, 0) + 1
        return f"ResidencyTable({counts})"
