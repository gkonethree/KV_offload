from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from .base import BlockKVStats, BlockScorer


class PagedEvictionScorer(BlockScorer):
    """Query-agnostic block importance based on mean ||V||/||K||."""

    def __init__(
        self,
        decay: float = 1.0,
        head_reduction: str = "mean",
        eps: float = 1e-6,
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if head_reduction not in ("mean", "min"):
            raise ValueError("head_reduction must be 'mean' or 'min'")
        self.decay = float(decay)
        self.head_reduction = head_reduction
        self.eps = float(eps)
        self._scores: dict[Hashable, float] = {}

    @staticmethod
    def compute_block_scores(
        stats: BlockKVStats,
        head_reduction: str = "mean",
        eps: float = 1e-6,
    ) -> torch.Tensor:
        ratio = stats.v_norm.float() / (stats.k_norm.float() + eps)
        if head_reduction == "mean":
            return ratio.mean(dim=-1)
        if head_reduction == "min":
            return ratio.min(dim=-1).values
        raise ValueError("head_reduction must be 'mean' or 'min'")

    def update(self, stats: BlockKVStats) -> None:
        new_scores = self.compute_block_scores(
            stats, self.head_reduction, self.eps
        ).detach().cpu().tolist()
        for block_id, new_score in zip(stats.block_ids, new_scores):
            old = self._scores.get(block_id)
            if old is None:
                self._scores[block_id] = float(new_score)
            else:
                # decay=1 => newest observation only; decay=0 => keep old.
                self._scores[block_id] = (
                    self.decay * float(new_score)
                    + (1.0 - self.decay) * old
                )

    def scores(self, block_ids: Sequence[Hashable]) -> torch.Tensor:
        return torch.tensor(
            [self._scores.get(block_id, float('inf')) for block_id in block_ids],
            dtype=torch.float32,
        )

    def select_for_eviction(
        self,
        block_ids: Sequence[Hashable],
        budget: int,
    ) -> list[Hashable]:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if len(block_ids) <= budget:
            return []
        scores = self.scores(block_ids)
        count = len(block_ids) - budget
        order = torch.argsort(scores, stable=True)
        return [block_ids[i] for i in order[:count].tolist()]

    def forget(self, block_ids: Sequence[Hashable]) -> None:
        for block_id in block_ids:
            self._scores.pop(block_id, None)

    def reset(self) -> None:
        self._scores.clear()
