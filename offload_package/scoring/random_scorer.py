from __future__ import annotations

import random
from collections.abc import Sequence, Hashable

import torch

from .base import BlockScorer, BlockRef, BlockKVStats


class RandomScorer(BlockScorer):
    """Randomly scores blocks for eviction testing."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._scores: dict[Hashable, float] = {}

    def update(self, stats: BlockKVStats) -> None:
        for ref in stats.block_ids:
            self._scores[ref] = self._rng.random()

    def scores(self, block_ids: Sequence[Hashable]) -> torch.Tensor:
        return torch.tensor([self._scores.get(b, self._rng.random()) for b in block_ids])

    def select_for_eviction(
        self,
        block_ids: Sequence[Hashable],
        budget: int,
    ) -> list[Hashable]:
        scored = [(b, self._scores.get(b, self._rng.random())) for b in block_ids]
        scored.sort(key=lambda x: x[1])
        return [b for b, _ in scored[:len(block_ids) - budget]]

    def forget(self, block_ids: Sequence[Hashable]) -> None:
        for b in block_ids:
            self._scores.pop(b, None)

    def reset(self) -> None:
        self._scores.clear()
        self._rng.seed(self._seed)