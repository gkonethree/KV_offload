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
        )

    def __post_init__(self):
        if not 0.0 <= self.offload_fraction < 1.0:
            raise ValueError("offload_fraction must be in [0, 1)")