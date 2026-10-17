from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadConfig:
    """Environment-driven config so no vLLM config schema changes are required."""

    enabled: bool = True
    offload_fraction: float = 0.20
    # Absolute alternative to offload_fraction: when set, each request keeps
    # at most this many offloadable blocks GPU-resident (sink, local-window
    # and the partially-filled tail are pinned and not counted).
    gpu_block_budget_per_request: int | None = None
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
    # Recall granularity: "block" fetches whole 16-token x all-layer columns
    # of every cold selected block; "per_layer_tokens" fetches only each
    # layer's selected in-page offsets of that layer's own slice (per-layer
    # 16-bit masks gate validity; incremental fill into the same page).
    fetch_granularity: str = "per_layer_tokens"
    # Staging storage, implied by fetch_granularity: "token" (TokenStagingStore,
    # each layer's cold tokens in its own staging tail) for per_layer_tokens,
    # "page" (whole-page KVStagingPool) for block. None = derive it.
    staging_store: str | None = None
    # "block": offload whole blocks (default). "token": score each token
    # per layer (FastKVzip gate), offload the lowest-scored ones per layer,
    # and compact survivors so emptied pages return to vLLM's allocator.
    offload_granularity: str = "block"
    token_scorer: str = "fastkvzip"  # "kvzip" (context reconstruction) or "random"
    kvzip_device: str | None = None   # GPU for KVzip's private HF model (None = worker's)
    kvzip_chunk: int = 2000           # context tokens per reconstruction chunk
    token_sink: int = 64          # first tokens never offloaded
    token_window: int = 128       # most recent tokens never offloaded
    token_budget: int | None = None   # absolute per-layer keep count
    # Per-(layer, head) budgets (HeadTokenOffloadEngine, staging shared by all
    # layers and heads): "global" (default) = one ranking over all (layer,
    # head, token) scores, FastKVzip's own policy; "layer" = equal total per
    # layer; "uniform" = equal count per (layer, head); "cells_shared" = the
    # shared policy on the per-head machinery (parity checks). "shared" =
    # the older TokenOffloadEngine: every KV head of a layer keeps the same
    # tokens, the same count in every layer, per-layer staging.
    token_budget_scope: str = "global"
    num_cpu_tokens: int | None = None  # CPU token-pool capacity
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
            gpu_block_budget_per_request=(
                int(os.environ["OFFLOAD_PACKAGE_GPU_BLOCK_BUDGET"])
                if os.getenv("OFFLOAD_PACKAGE_GPU_BLOCK_BUDGET") else None),
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
            fetch_granularity=os.getenv(
                "OFFLOAD_PACKAGE_FETCH_GRANULARITY", "per_layer_tokens"),
            staging_store=os.getenv("OFFLOAD_PACKAGE_STAGING_STORE") or None,
        )

    def __post_init__(self):
        if not 0.0 <= self.offload_fraction < 1.0:
            raise ValueError("offload_fraction must be in [0, 1)")
        if (self.gpu_block_budget_per_request is not None
                and self.gpu_block_budget_per_request < 0):
            raise ValueError("gpu_block_budget_per_request must be >= 0")
        if self.scorer_type not in ("paged_eviction", "random", "lru"):
            raise ValueError("scorer_type must be 'paged_eviction', 'random' or 'lru'")
        if self.staging_policy not in ("fixed", "auto"):
            raise ValueError("staging_policy must be 'fixed' or 'auto'")
        if not 0.0 < self.staging_max_pool_frac <= 1.0:
            raise ValueError("staging_max_pool_frac must be in (0, 1]")
        if self.staging_eviction not in ("scorer", "lru", "random"):
            raise ValueError("staging_eviction must be 'scorer', 'lru' or 'random'")
        if self.fetch_granularity not in ("block", "per_layer_tokens"):
            raise ValueError(
                "fetch_granularity must be 'block' or 'per_layer_tokens'")
        if self.staging_store is None:
            object.__setattr__(self, "staging_store", (
                "token" if self.fetch_granularity == "per_layer_tokens" else "page"))
        if self.staging_store not in ("token", "page"):
            raise ValueError("staging_store must be 'token' or 'page'")
        if (self.staging_store == "token") != (self.fetch_granularity == "per_layer_tokens"):
            raise ValueError("staging_store='token' goes with fetch_granularity="
                             "'per_layer_tokens', 'page' with 'block'")
        if self.staging_safety_margin < 1.0:
            raise ValueError("staging_safety_margin must be >= 1.0")
        if self.offload_granularity not in ("block", "token"):
            raise ValueError("offload_granularity must be 'block' or 'token'")
        if self.offload_granularity == "token" and self.token_scorer not in (
                "fastkvzip", "kvzip", "random"):
            raise ValueError("token offloading supports token_scorer='fastkvzip', 'kvzip' or "
                             "'random'")
        if self.token_budget_scope not in ("shared", "global", "layer", "uniform", "cells_shared"):
            raise ValueError("token_budget_scope must be shared, global, layer, uniform "
                             "or cells_shared")