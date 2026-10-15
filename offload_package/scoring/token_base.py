from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TokenRef:
    """Stable logical identity for a single cached KV token.

    ``token_idx`` is the token's logical position within its request and is
    invariant under physical compaction: physical slots move, logical
    positions never do.
    """

    request_id: str
    token_idx: int


class TokenScorer(ABC):
    """Interface for scoring individual KV tokens for token-level offload.

    A TokenScorer answers one question: given the post-RoPE keys of a set of
    tokens, how important is each token at the current decode round?  Higher
    scores mean "keep resident on GPU"; the round orchestrator turns scores
    into a keep/evict split under the configured budget and protections.

    The scoring interface is deliberately storage-agnostic: the caller
    gathers K (post-RoPE) from wherever the token currently lives -- GPU
    pages, a CPU offload store, or a staging area -- and passes dense
    per-layer tensors.  This keeps the scorer correct across block
    compaction (RoPE is baked into the cached keys, so scores do not depend
    on physical slot order).
    """

    @abstractmethod
    def score(
        self,
        k_by_layer: Mapping[int, torch.Tensor],
        round_start: int,
    ) -> torch.Tensor:
        """Score tokens given their post-RoPE keys.

        Args:
            k_by_layer: layer_idx -> ``[num_tokens, num_kv_heads, head_dim]``
                post-RoPE key tensor.  Layers absent from the scorer's stats
                are ignored.
            round_start: absolute logical position of the decode round
                (the position of the most recently decoded token).

        Returns:
            ``[num_tokens]`` float32 importance scores, higher = keep.
        """
        raise NotImplementedError

    def reset(self) -> None:
        pass
