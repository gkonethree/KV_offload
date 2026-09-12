from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from .controller import KVCompressionController
from .integration.manager import OffloadOrchestrator
from .offload.manager import KVOffloadManager
from .scoring.paged_eviction import PagedEvictionScorer
from .staging.pool import KVStagingPool
from .staging.residency import ResidencyTable


@dataclass(frozen=True)
class OrchestratorConfig:
    gpu_budget_blocks: int
    num_cpu_slots: int
    num_staging_slots: int
    interval_tokens: int
    page_size: int
    dtype: torch.dtype
    device: torch.device
    kv_layout: str = "NHD"
    scorer_decay: float = 1.0
    scorer_head_reduction: str = "mean"


def split_vllm_kv_cache(
    cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a vLLM layer cache into separate K and V page tensors."""
    if isinstance(cache, (tuple, list)):
        if len(cache) != 2 or not all(isinstance(x, torch.Tensor) for x in cache):
            raise TypeError("KV cache sequence must contain exactly (K, V)")
        return cache[0], cache[1]
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Unsupported KV cache type: {type(cache)!r}")
    if cache.ndim < 4:
        raise ValueError("KV cache tensor has too few dimensions")
    if cache.shape[0] == 2:
        return cache[0], cache[1]
    if cache.shape[1] == 2:
        return cache[:, 0], cache[:, 1]
    raise ValueError(
        "Cannot infer K/V axis; expected cache shape [2,P,...] or [P,2,...]"
    )


def build_orchestrator(
    cfg: OrchestratorConfig,
    *,
    runner_kv_caches: Sequence[torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]],
    staging_base_page_idx: int,
) -> OffloadOrchestrator:
    """Build the offload runtime around the existing vLLM KV cache tensors.

    The last ``num_staging_slots`` pages of each layer cache must be reserved
    for staging by the vLLM integration patch; BlockPool must never allocate
    those physical page IDs.
    """
    split = [split_vllm_kv_cache(cache) for cache in runner_kv_caches]
    k_caches = [kv[0] for kv in split]
    v_caches = [kv[1] for kv in split]
    if not k_caches:
        raise ValueError("runner_kv_caches cannot be empty")

    for cache in k_caches:
        if staging_base_page_idx + cfg.num_staging_slots > cache.shape[0]:
            raise ValueError("staging pages exceed KV cache capacity")

    if cfg.gpu_budget_blocks > staging_base_page_idx:
        raise ValueError("gpu_budget_blocks must fit in the non-staging page range")

    num_heads = k_caches[0].shape[2] if cfg.kv_layout == "NHD" else k_caches[0].shape[1]
    head_dim = k_caches[0].shape[-1]
    cpu_cache = torch.empty(
        (
            len(k_caches),
            cfg.num_cpu_slots,
            2,
            cfg.page_size,
            num_heads,
            head_dim,
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
        kv_layout=cfg.kv_layout,
    )
    staging_pool = KVStagingPool(
        num_slots=cfg.num_staging_slots,
        staging_base_page=staging_base_page_idx,
        gpu_k_caches=k_caches,
        gpu_v_caches=v_caches,
        cpu_cache=cpu_cache,
        residency=residency,
        offload_manager=offload_manager,
    )
    controller = KVCompressionController(cfg.interval_tokens, cfg.page_size)
    return OffloadOrchestrator(
        offload_manager=offload_manager,
        staging_pool=staging_pool,
        residency=residency,
        controller=controller,
        page_size=cfg.page_size,
        staging_base_page_idx=staging_base_page_idx,
    )
