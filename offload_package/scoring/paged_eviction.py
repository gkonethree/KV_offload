from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from .base import BlockKVStats, BlockScorer
from offload_package.debug import dbg as _dbg


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
        # Unknown refs score -inf: they are evicted first from staging (a ref
        # we have no observation for must never block a scored one) and are
        # never promoted. Never +inf -- that made unscored staging residents
        # permanently un-evictable and promoted them first.
        return torch.tensor(
            [self._scores.get(block_id, float("-inf")) for block_id in block_ids],
            dtype=torch.float32,
        )

    def select_for_eviction(
        self,
        block_ids: Sequence[Hashable],
        budget: int,
    ) -> list[Hashable]:
        _dbg(f"[SCORER DEBUG] select_for_eviction: block_ids={len(block_ids)}, budget={budget}")
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if len(block_ids) <= budget:
            _dbg(f"[SCORER DEBUG] returning empty: len={len(block_ids)} <= budget={budget}")
            return []
        unscored = [b for b in block_ids if b not in self._scores]
        assert not unscored, (
            f"eviction candidates must all be scored; unscored={unscored[:5]}"
        )
        scores = self.scores(block_ids)
        _dbg(f"[SCORER DEBUG] scores={scores}")
        count = len(block_ids) - budget
        order = torch.argsort(scores, stable=True)
        _dbg(f"[SCORER DEBUG] count={count}, order={order}")
        return [block_ids[i] for i in order[:count].tolist()]

    def forget(self, block_ids: Sequence[Hashable]) -> None:
        for block_id in block_ids:
            self._scores.pop(block_id, None)

    def reset(self) -> None:
        self._scores.clear()
