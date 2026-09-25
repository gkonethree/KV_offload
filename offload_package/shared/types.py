# SPDX-License-Identifier: Apache-2.0
"""Shared types to avoid circular imports."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

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

    # Verify dimensions match config
    if k_caches[0].shape[-1] != cfg.head_dim if hasattr(cfg, 'head_dim') else True:
        pass  # Config should match

    # Debug: print GPU cache shape
    print(f"[BUILD DEBUG] GPU k_cache shape: {k_caches[0].shape}", flush=True)
    print(f"[BUILD DEBUG] cfg.page_size={cfg.page_size}, num_kv_heads={kv_cache_view.num_kv_heads}, head_dim={kv_cache_view.head_dim}", flush=True)
    print(f"[BUILD DEBUG] CPU cache shape will be: ({len(k_caches)}, {cfg.num_cpu_slots}, 2, {cfg.page_size}, {kv_cache_view.num_kv_heads}, {kv_cache_view.head_dim})", flush=True)

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

    residency = ResidencyTable()
    scorer = PagedEvictionScorer(
        decay=cfg.scorer_decay,
        head_reduction=cfg.scorer_head_reduction,
    )
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
        staging_base_page=kv_cache_view.staging_base_page,
        gpu_k_caches=k_caches,
        gpu_v_caches=v_caches,
        cpu_cache=cpu_cache,
        residency=None,  # Will be set after residency creation
        offload_manager=None,  # Will be set after offload_manager creation
    )
    controller = KVCompressionController(cfg.W, cfg.page_size)
    
    # Create offload_manager with proper dependencies
    offload_manager = KVOffloadManager(
        scorer=scorer,
        residency=None,  # Will be set after residency creation
        gpu_k_caches=k_caches,
        gpu_v_caches=v_caches,
        cpu_cache=cpu_cache,
        gpu_budget_blocks=cfg.gpu_budget_blocks,
        kv_layout=kv_cache_view.kv_layout,
    )
    
    residency = ResidencyTable()
    offload_manager.residency = residency
    
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
    
    Handles multiple formats:
    - List of (K, V) tuples
    - Flat list of alternating K, V tensors
    - List of combined K/V tensors
    """
    from offload_package.vllm_adapter import _get_attention_spec, _split_kv_cache_with_spec
    
    spec = _get_attention_spec(kv_cache_config)
    layers = []
    k_caches = []
    v_caches = []
    
    # Check the format of the first cache to determine the structure
    first_cache = runner_kv_caches[0]
    
    if isinstance(first_cache, (tuple, list)) and len(first_cache) == 2:
        # Format: [(K0, V0), (K1, V1), ...] - list of (K, V) tuples
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
        # Could be flat list [K0, V0, K1, V1, ...] or combined tensors
        # Check if it's a flat alternating list by comparing shapes
        if len(runner_kv_caches) % 2 == 0:
            # Likely flat alternating list [K0, V0, K1, V1, ...]
            for i in range(0, len(runner_kv_caches), 2):
                k_cache = runner_kv_caches[i]
                v_cache = runner_kv_caches[i + 1]
                if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
                    raise TypeError(f"Expected tensor at index {i} or {i+1}")
                layers.append((k_cache, v_cache))
                k_caches.append(k_cache)
                v_caches.append(v_cache)
        else:
            # Try to split as combined tensor
            for layer_idx, cache in enumerate(runner_kv_caches):
                k_cache, v_cache = _split_kv_cache_with_spec(cache, spec)
                layers.append(cache)
                k_caches.append(k_cache)
                v_caches.append(v_cache)
    else:
        raise TypeError(f"Unexpected cache type at index 0: {type(first_cache)}")
    
    from offload_package.vllm_adapter import KVCacheView
    
    # Determine staging_base_page
    if k_caches:
        staging_base_page = k_caches[0].shape[0] - staging_slots
    else:
        staging_base_page = 0
    
    # Get num_kv_heads and head_dim from spec
    from offload_package.vllm_adapter import _get_attention_spec
    spec = _get_attention_spec(kv_cache_config)
    num_kv_heads = spec.num_kv_heads
    head_dim = spec.head_size
    block_size = spec.block_size
    kv_layout = 'NHD'  # Default layout
    
    return KVCacheView(
        layers=layers,
        k_caches=k_caches,
        v_caches=v_caches,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_layout=kv_layout,
        staging_base_page=staging_base_page,
    )
