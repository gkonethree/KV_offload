"""LRU scorer: evict the blocks least recently selected by sparse attention.

The recency signal is the decode top-k itself: every step the orchestrator
records which blocks the selection touched (union over heads of layer 0's
pattern -- see OffloadOrchestrator.note_selection_host, fed from the
per-step pattern cache). A block's score is its last-use tick:

    higher  = used more recently  -> kept
    lower   = used longer ago    -> offloaded / evicted from staging first
    0       = never selected     -> coldest possible, offloaded first

This correlates directly with the fetch-miss rate the offload pipeline pays
(adjacent decode steps' top-k sets overlap heavily), unlike the
value-norm proxy of PagedEvictionScorer or RandomScorer's noise.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch

from .base import BlockRef, BlockKVStats, BlockScorer


class LRUScorer(BlockScorer):
    """Recency-based eviction keyed on sparse-attention selection."""

    def __init__(self) -> None:
        self._last_used: dict[Hashable, int] = {}
        self._clock: int = 0

    # Extra API beyond BlockScorer; the orchestrator probes for it and no-ops
    # for scorers without it (paged_eviction, random).
    def note_usage(self, request_id: str, block_idxs: Sequence[int]) -> None:
        """Record that this step's top-k selected these blocks of request_id."""
        self._clock += 1
        rid = str(request_id)
        for idx in block_idxs:
            self._last_used[BlockRef(rid, int(idx))] = self._clock

    def update(self, stats: BlockKVStats) -> None:
        # Recency ignores the KV-norm stats, but scoring a block must
        # register it (the "eviction candidates must all be scored" guard);
        # never-selected blocks keep tick 0, the coldest value.
        for ref in stats.block_ids:
            if ref not in self._last_used:
                self._last_used[ref] = 0

    def scores(self, block_ids: Sequence[Hashable]) -> torch.Tensor:
        # Unknown -> 0 (oldest): consistent with the never-selected floor.
        return torch.tensor(
            [float(self._last_used.get(b, 0)) for b in block_ids],
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
        unscored = [b for b in block_ids if b not in self._last_used]
        assert not unscored, (
            f"eviction candidates must all be scored; unscored={unscored[:5]}"
        )
        scores = self.scores(block_ids)
        count = len(block_ids) - budget
        order = torch.argsort(scores, stable=True)
        return [block_ids[i] for i in order[:count].tolist()]

    def forget(self, block_ids: Sequence[Hashable]) -> None:
        for block_id in block_ids:
            self._last_used.pop(block_id, None)

    def reset(self) -> None:
        self._last_used.clear()
