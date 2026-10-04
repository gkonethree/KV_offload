# SPDX-License-Identifier: Apache-2.0
"""Shared types to avoid circular imports."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from offload_package.debug import dbg as _dbg

if TYPE_CHECKING:
    from offload_package.integration.manager import OffloadOrchestrator
    from offload_package.staging.pool import KVStagingPool
    from offload_package.vllm_adapter import KVCacheView


@dataclass(frozen=True)
class OrchestratorConfig:
    offload_fraction: float
    num_cpu_slots: int
    num_staging_slots: int
    W: int
    page_size: int
    dtype: torch.dtype
    device: torch.device
    gpu_budget_blocks: int
    kv_layout: str = "NHD"
    scorer_decay: float = 1.0
    scorer_head_reduction: str = "mean"
    scorer_type: str = "paged_eviction"  # "paged_eviction" or "random"


def build_orchestrator(
    cfg: OrchestratorConfig,
    *,
    kv_cache_view: "KVCacheView",
) -> "OffloadOrchestrator":
    """Build the offload runtime using pre-parsed KVCacheView."""
    from offload_package.controller import KVCompressionController
    from offload_package.integration.manager import OffloadOrchestrator
    from offload_package.offload.manager import KVOffloadManager
    from offload_package.scoring.paged_eviction import PagedEvictionScorer
    from offload_package.scoring.random_scorer import RandomScorer
    from offload_package.staging.pool import KVStagingPool
    from offload_package.staging.residency import ResidencyTable

    k_caches = kv_cache_view.k_caches
    v_caches = kv_cache_view.v_caches

    if not k_caches:
        raise ValueError("kv_cache_view.k_caches cannot be empty")

    # Validate staging pages fit
    staging_base = kv_cache_view.staging_base_page
    for idx, cache in enumerate(k_caches):
        if staging_base + cfg.num_staging_slots > cache.shape[0]:
            raise ValueError(f"staging pages exceed KV cache capacity: {staging_base} + {cfg.num_staging_slots} = {staging_base + cfg.num_staging_slots} > {cache.shape[0]}")

    if cfg.gpu_budget_blocks > staging_base:
        raise ValueError("gpu_budget_blocks must fit in the non-staging page range")

    # Debug: print GPU cache shape
    _dbg(f"[BUILD DEBUG] GPU k_cache shape: {k_caches[0].shape}")
    _dbg(f"[BUILD DEBUG] cfg.page_size={cfg.page_size}, num_kv_heads={kv_cache_view.num_kv_heads}, head_dim={kv_cache_view.head_dim}")
    _dbg(f"[BUILD DEBUG] CPU cache shape will be: ({len(k_caches)}, {cfg.num_cpu_slots}, 2, {cfg.page_size}, {kv_cache_view.num_kv_heads}, {kv_cache_view.head_dim})")

    cpu_cache = torch.empty(
        (
            len(k_caches),
            cfg.num_cpu_slots,
            2,
            cfg.page_size,
            kv_cache_view.num_kv_heads,
            kv_cache_view.head_dim,
        ),
        dtype=cfg.dtype,
        device="cpu",
        pin_memory=True,
    )

    # Select scorer based on config
    if cfg.scorer_type == "random":
        scorer = RandomScorer()
    else:
        scorer = PagedEvictionScorer(
            decay=cfg.scorer_decay,
            head_reduction=cfg.scorer_head_reduction,
        )

    residency = ResidencyTable()
    offload_manager = KVOffloadManager(
        scorer=scorer,
        residency=residency,
        gpu_k_caches=k_caches,
        gpu_v_caches=v_caches,
        cpu_cache=cpu_cache,
        gpu_budget_blocks=cfg.gpu_budget_blocks,
        kv_layout=kv_cache_view.kv_layout,
    )
    staging_pool = KVStagingPool(
        num_slots=cfg.num_staging_slots,
        staging_base_page=staging_base,
        gpu_k_caches=k_caches,
        gpu_v_caches=v_caches,
        cpu_cache=cpu_cache,
        residency=residency,
        offload_manager=offload_manager,
    )
    controller = KVCompressionController(cfg.W, cfg.page_size)
    return OffloadOrchestrator(
        offload_manager=offload_manager,
        staging_pool=staging_pool,
        residency=residency,
        controller=controller,
        page_size=cfg.page_size,
        staging_base_page_idx=staging_base,
        offload_fraction=cfg.offload_fraction,
    )


def normalize_runner_kv_caches(
    runner_kv_caches,
    kv_cache_config,
    staging_slots: int = 0,
):
    """
    Normalize vLLM ModelRunner.kv_caches into layers, k_caches, v_caches.

    Supported formats:
    - List of (K, V) tuples
    - Per-layer combined tensors [num_blocks, num_kv_heads, block_size,
      2*head_dim] with K in [..., :head_dim] and V in [..., head_dim:] — this
      fork's layout. For combined tensors the SAME tensor object is used for
      k_caches[i] and v_caches[i]; consumers that need separate K/V split the
      last axis (see KVOffloadManager.score_block/_copy_gpu_to_cpu).

    A flat even-length list of combined tensors is explicitly rejected: this
    fork never produces alternating [K0, V0, K1, V1, ...] lists, and guessing
    that layout silently halves the layer count.
    """
    from offload_package.vllm_adapter import KVCacheView, _get_attention_spec

    spec = _get_attention_spec(kv_cache_config)
    num_kv_heads = spec.num_kv_heads
    head_dim = spec.head_size
    block_size = spec.block_size

    layers = []
    k_caches = []
    v_caches = []

    first_cache = runner_kv_caches[0]

    if isinstance(first_cache, (tuple, list)) and len(first_cache) == 2:
        # Format: [(K0, V0), (K1, V1), ...] - explicit per-layer (K, V) pairs
        for layer_idx, cache in enumerate(runner_kv_caches):
            if not isinstance(cache, (tuple, list)) or len(cache) != 2:
                raise TypeError(f"Expected (K, V) tuple at index {layer_idx}")
            k_cache, v_cache = cache[0], cache[1]
            if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
                raise TypeError(f"K and V must be tensors at index {layer_idx}")
            layers.append((k_cache, v_cache))
            k_caches.append(k_cache)
            v_caches.append(v_cache)
    elif isinstance(first_cache, torch.Tensor):
        # This fork: one combined tensor per layer, [num_blocks, H, N, 2*C].
        expected_last = 2 * head_dim
        for layer_idx, cache in enumerate(runner_kv_caches):
            if not isinstance(cache, torch.Tensor):
                raise TypeError(f"Expected tensor at index {layer_idx}")
            if (
                cache.ndim != 4
                or cache.shape[1] != num_kv_heads
                or cache.shape[2] != block_size
                or cache.shape[-1] != expected_last
            ):
                raise ValueError(
                    "Expected combined per-layer KV cache with shape "
                    f"[num_blocks, num_kv_heads={num_kv_heads}, "
                    f"block_size={block_size}, 2*head_dim={expected_last}], "
                    f"got {tuple(cache.shape)} at index {layer_idx}"
                )
            layers.append(cache)
            k_caches.append(cache)
            v_caches.append(cache)
    else:
        raise TypeError(f"Unexpected cache type at index 0: {type(first_cache)}")

    # Staging pool occupies the tail pages of the per-layer tensors.
    staging_base_page = k_caches[0].shape[0] - staging_slots if k_caches else 0

    return KVCacheView(
        layers=layers,
        k_caches=k_caches,
        v_caches=v_caches,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_layout="HND",
        staging_base_page=staging_base_page,
    )
