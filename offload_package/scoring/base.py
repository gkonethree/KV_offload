from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BlockRef:
    """Stable logical identity for a KV block.

    Physical vLLM block IDs may be reused.  A BlockRef therefore identifies a
    request-local logical block independently of the physical allocator.
    """

    request_id: str
    block_idx: int


@dataclass
class BlockKVStats:
    """Per-block K/V statistics used by query-agnostic scorers."""

    v_norm: torch.Tensor       # [num_blocks, num_kv_heads]
    k_norm: torch.Tensor       # [num_blocks, num_kv_heads]
    block_ids: list[Hashable]


class BlockScorer(ABC):
    """Storage-agnostic interface for selecting cold logical KV blocks."""

    @abstractmethod
    def update(self, stats: BlockKVStats) -> None:
        raise NotImplementedError

    @abstractmethod
    def scores(self, block_ids: Sequence[Hashable]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def select_for_eviction(
        self,
        block_ids: Sequence[Hashable],
        budget: int,
    ) -> list[Hashable]:
        raise NotImplementedError

    def forget(self, block_ids: Sequence[Hashable]) -> None:
        pass

    def reset(self) -> None:
        pass
