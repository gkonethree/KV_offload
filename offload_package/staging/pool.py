from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch

from ..offload.manager import KVOffloadManager
from ..scoring.base import BlockRef
from .residency import Residency, ResidencyTable
# FIX — lazy stream property (same pattern in both classes):
@property
def _copy_stream(self) -> torch.cuda.Stream:
    if self.__copy_stream is None:
        self.__copy_stream = torch.cuda.Stream(
            device=self.gpu_k_caches[0].device
        )
    return self.__copy_stream

# In __init__: self.__copy_stream: torch.cuda.Stream | None = None

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
        self.copy_stream = torch.cuda.Stream(device=self.gpu_k_caches[0].device)
        self._lru: OrderedDict[BlockRef, int] = OrderedDict()
        self._free = list(range(num_slots - 1, -1, -1))

    def ensure_resident(self, refs: Sequence[BlockRef]) -> dict[BlockRef, int]:
        result: dict[BlockRef, int] = {}
        for ref in dict.fromkeys(refs):
            loc = self.residency.get(ref)
            if loc is None:
                raise KeyError(f"Unknown logical block {ref}")
            if loc.residency == Residency.GPU:
                continue
            if loc.residency in (Residency.STAGING, Residency.IN_FLIGHT) \
                    and loc.staging_slot is not None:
                self._lru.move_to_end(ref)
                result[ref] = loc.staging_slot
                continue
            if loc.cpu_slot is None:
                raise RuntimeError(f"{ref} is not backed by a CPU slot")

            self.offload_manager.wait_cpu_slot(loc.cpu_slot)
            slot = self._allocate_slot(ref)
            with torch.cuda.stream(self.copy_stream):
                for layer, (k_cache, v_cache) in enumerate(
                    zip(self.gpu_k_caches, self.gpu_v_caches)
                ):
                    k_cache[self.base + slot].copy_(
                        self.cpu_cache[layer, loc.cpu_slot, 0], non_blocking=True
                    )
                    v_cache[self.base + slot].copy_(
                        self.cpu_cache[layer, loc.cpu_slot, 1], non_blocking=True
                    )
                event = torch.cuda.Event()
                event.record(self.copy_stream)
            torch.cuda.current_stream(device=k_cache.device).wait_event(event)
            self.residency.mark_staging(ref, slot)
            result[ref] = slot
        return result

    def _allocate_slot(self, incoming: BlockRef) -> int:
        if self._free:
            slot = self._free.pop()
        else:
            evicted, slot = self._lru.popitem(last=False)
            old = self.residency.get(evicted)
            if old is None or old.cpu_slot is None:
                raise RuntimeError(f"Cannot evict staging block {evicted}")
            self.residency.mark_cpu(evicted, old.cpu_slot)
        self._lru[incoming] = slot
        return slot

    def release(self, ref: BlockRef) -> None:
        slot = self._lru.pop(ref, None)
        if slot is not None:
            self._free.append(slot)

    def release_all(self) -> None:
        for ref in list(self._lru):
            self.release(ref)

    def physical_page(self, slot: int) -> int:
        if not 0 <= slot < self.num_slots:
            raise ValueError("invalid staging slot")
        return self.base + slot

    @property
    def num_used_slots(self) -> int:
        return len(self._lru)
