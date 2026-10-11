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
    scorer_type: str = "paged_eviction"  # "paged_eviction", "random" or "lru"
    # Intent knob: whether to recycle GPU pages after offload (table surgery +
    # free). The runtime gate still refuses true freeing under async
    # scheduling / multiprocess / world_size>1 regardless of this flag.
    true_free: bool = True
    # Staging pool sizing policy: "fixed" (use num_staging_slots as-is) or
    # "auto" (size at startup to cover the hot sparse-selected set).
    staging_policy: str = "fixed"
    # Staging-pool eviction policy -- independent of scorer_type (which
    # selects WHICH blocks get offloaded). "lru": evict the least-recently
    # SELECTED staged block (recency tracks re-selection probability, the
    # actual re-fetch cost driver); "scorer": inherit the offload scorer;
    # "random": uncorrelated control.
    staging_eviction: str = "lru"
    # Auto mode: cap staging at this fraction of the GPU KV pool.
    staging_max_pool_frac: float = 0.25
    # Auto mode: multiply the computed estimate by this headroom factor.
    staging_safety_margin: float = 1.10

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
            staging_policy=os.getenv("OFFLOAD_PACKAGE_STAGING_POLICY", "fixed"),
            staging_max_pool_frac=float(os.getenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25")),
            staging_safety_margin=float(os.getenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10")),
            staging_eviction=os.getenv("OFFLOAD_PACKAGE_STAGING_EVICTION", "lru"),
        )

    def __post_init__(self):
        if not 0.0 <= self.offload_fraction < 1.0:
            raise ValueError("offload_fraction must be in [0, 1)")
        if self.scorer_type not in ("paged_eviction", "random", "lru"):
            raise ValueError("scorer_type must be 'paged_eviction', 'random' or 'lru'")
        if self.staging_policy not in ("fixed", "auto"):
            raise ValueError("staging_policy must be 'fixed' or 'auto'")
        if not 0.0 < self.staging_max_pool_frac <= 1.0:
            raise ValueError("staging_max_pool_frac must be in (0, 1]")
        if self.staging_eviction not in ("scorer", "lru", "random"):
            raise ValueError("staging_eviction must be 'scorer', 'lru' or 'random'")
        if self.staging_safety_margin < 1.0:
            raise ValueError("staging_safety_margin must be >= 1.0")