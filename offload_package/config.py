from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadConfig:
    """Environment-driven config so no vLLM config schema changes are required."""

    enabled: bool = True
    offload_fraction: float = 0.20
    num_cpu_slots: int = 4096
    num_staging_slots: int = 64
    interval_tokens: int = 32
    scorer_decay: float = 1.0
    scorer_head_reduction: str = "mean"
    scorer_type: str = "paged_eviction"  # "paged_eviction" or "random"
    # Intent knob: whether to recycle GPU pages after offload (table surgery +
    # free). The runtime gate still refuses true freeing under async
    # scheduling / multiprocess / world_size>1 regardless of this flag.
    true_free: bool = True
    # R2 promotion watch: how many highest-scored CPU-resident blocks to
    # pre-stage per active decode request per step; 0 disables promotion.
    promote_per_step: int = 4

    @classmethod
    def from_env(cls) -> "OffloadConfig":
        def flag(name: str, default: bool = False) -> bool:
            return os.getenv(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}

        return cls(
            enabled=flag("OFFLOAD_PACKAGE_ENABLED"),
            offload_fraction=float(os.getenv("OFFLOAD_PACKAGE_OFFLOAD_FRACTION", "0.20")),
            num_cpu_slots=int(os.getenv("OFFLOAD_PACKAGE_CPU_SLOTS", "4096")),
            num_staging_slots=int(os.getenv("OFFLOAD_PACKAGE_STAGING_SLOTS", "64")),
            interval_tokens=int(os.getenv("OFFLOAD_PACKAGE_INTERVAL", "32")),
            scorer_decay=float(os.getenv("OFFLOAD_PACKAGE_SCORER_DECAY", "1.0")),
            scorer_head_reduction=os.getenv("OFFLOAD_PACKAGE_SCORER_HEAD_REDUCTION", "mean"),
            scorer_type=os.getenv("OFFLOAD_PACKAGE_SCORER_TYPE", "paged_eviction"),
            true_free=flag("OFFLOAD_PACKAGE_TRUE_FREE", True),
            promote_per_step=int(os.getenv("OFFLOAD_PACKAGE_PROMOTE_PER_STEP", "4")),
        )

    def __post_init__(self):
        if not 0.0 <= self.offload_fraction < 1.0:
            raise ValueError("offload_fraction must be in [0, 1)")
        if self.scorer_type not in ("paged_eviction", "random"):
            raise ValueError("scorer_type must be 'paged_eviction' or 'random'")
        if self.promote_per_step < 0:
            raise ValueError("promote_per_step must be >= 0")