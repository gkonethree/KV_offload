from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from ..scoring.base import BlockRef


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

    def mark_staging(self, ref: BlockRef, staging_slot: int) -> None:
        old = self._blocks[ref]
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
