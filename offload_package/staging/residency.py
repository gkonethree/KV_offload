from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from ..scoring.base import BlockRef


class Residency(Enum):
    GPU = auto()
    CPU = auto()
    STAGING = auto()
    IN_FLIGHT = auto()


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

    def mark_gpu(self, ref: BlockRef, gpu_block_id: int, *, complete: bool = False) -> None:
        old = self._blocks.get(ref)
        self._blocks[ref] = BlockLocation(
            residency=Residency.GPU,
            gpu_block_id=int(gpu_block_id),
            cpu_slot=old.cpu_slot if old else None,
            complete=bool(complete),
        )

    def mark_complete(self, ref: BlockRef) -> None:
        loc = self._blocks[ref]
        loc.complete = True

    def mark_cpu(self, ref: BlockRef, cpu_slot: int) -> None:
        old = self._blocks.get(ref)
        if old is None or old.gpu_block_id is None:
            raise KeyError(f"No GPU mapping recorded for {ref}")
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
        self._blocks[ref] = BlockLocation(
            residency=Residency.STAGING,
            gpu_block_id=old.gpu_block_id,
            cpu_slot=old.cpu_slot,
            staging_slot=int(staging_slot),
            complete=old.complete,
        )

    def mark_in_flight(self, ref: BlockRef, staging_slot: int) -> None:
        old = self._blocks[ref]
        if old.cpu_slot is None:
            raise KeyError(f"No CPU slot recorded for {ref}")
        self._blocks[ref] = BlockLocation(
            residency=Residency.IN_FLIGHT,
            gpu_block_id=old.gpu_block_id,
            cpu_slot=old.cpu_slot,
            staging_slot=int(staging_slot),
            complete=old.complete,
        )

    def get(self, ref: BlockRef) -> Optional[BlockLocation]:
        return self._blocks.get(ref)

    def remove(self, ref: BlockRef) -> Optional[BlockLocation]:
        return self._blocks.pop(ref, None)

    def all_gpu_blocks(self, *, complete_only: bool = True) -> list[BlockRef]:
        return [
            ref for ref, loc in self._blocks.items()
            if loc.residency == Residency.GPU and (loc.complete or not complete_only)
        ]

    def refs_for_request(self, request_id: str) -> list[BlockRef]:
        return sorted(
            (ref for ref in self._blocks if ref.request_id == request_id),
            key=lambda ref: ref.block_idx,
        )

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for loc in self._blocks.values():
            counts[loc.residency.name] = counts.get(loc.residency.name, 0) + 1
        return f"ResidencyTable({counts})"
