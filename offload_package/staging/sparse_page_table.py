from __future__ import annotations

from ..scoring.base import BlockRef
from .residency import Residency, ResidencyTable


class SparsePageTable:
    def __init__(self, residency: ResidencyTable, staging_base_page_idx: int) -> None:
        self.residency = residency
        self.staging_base = int(staging_base_page_idx)

    def resolve(self, ref: BlockRef) -> int | None:
        loc = self.residency.get(ref)
        if loc is None:
            return None
        if loc.residency == Residency.GPU:
            return loc.gpu_block_id
        if loc.residency == Residency.STAGING and loc.staging_slot is not None:
            return self.staging_base + loc.staging_slot
        return None
