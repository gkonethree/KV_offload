"""Canonical, explicit runtime profiles used by Skylight benchmarks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BMMProfile:
    """Qualified incremental block-minmax selection policy."""

    target_density: float = 0.10
    sink: int = 64
    local: int = 64
    channel_num: int = -1
    sub_page: int = 16
    block_size: int = 16

    def __post_init__(self) -> None:
        if not 0.0 < self.target_density <= 1.0:
            raise ValueError("target_density must be in (0, 1]")
        if self.sink < 0:
            raise ValueError("sink must be >= 0")
        if self.local < 0:
            raise ValueError("local must be >= 0")
        if self.channel_num == 0 or self.channel_num < -1:
            raise ValueError("channel_num must be -1 or a positive integer")
        if self.sub_page <= 0:
            raise ValueError("sub_page must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

    def to_env(self) -> dict[str, str]:
        """Return the complete environment for the validated incremental path."""
        return {
            "SKYLIGHT_SPARSE_METHOD": "block_minmax",
            "SKYLIGHT_SPARSE_TOPK": str(self.target_density),
            "SKYLIGHT_SPARSE_SINK": str(self.sink),
            "SKYLIGHT_SPARSE_LOCAL": str(self.local),
            "SKYLIGHT_SPARSE_CHANNEL_NUM": str(self.channel_num),
            "SKYLIGHT_SPARSE_SUB_PAGE": str(self.sub_page),
            "SKYLIGHT_BLOCK_SIZE": str(self.block_size),
            "SKYLIGHT_INCR_SLOT": "1",
            "SKYLIGHT_INCR_FULLCG": "1",
            "SKYLIGHT_INCR_PIPELINED": "1",
            "SKYLIGHT_FI_BSR": "0",
        }
