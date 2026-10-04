from __future__ import annotations

from collections.abc import Sequence

import sys
import torch

from ..config import OffloadConfig
from ..vllm_adapter import KVCacheView
from ..shared.types import (
    OrchestratorConfig,
    build_orchestrator,
    normalize_runner_kv_caches,
)
from offload_package.debug import dbg as _dbg


def initialize_for_runner(
    runner,
    *,
    normal_gpu_pages: int,
    block_size: int,
    kv_layout: str = "NHD",
    offload_config: "OffloadConfig | None" = None,
) -> object | None:
    """Create an orchestrator after V2 runner KV caches are initialized.
    
    normal_gpu_pages should be the actual number of GPU pages allocated for KV cache,
    after accounting for staging slots reserved by the cache allocator hook.
    """
    _dbg("[DEBUG] initialize_for_runner: START")
    if offload_config is None:
        cfg = OffloadConfig.from_env()
    else:
        cfg = offload_config
    _dbg(f"[DEBUG] Config loaded: enabled={cfg.enabled}")
    if not cfg.enabled:
        return None
    if not getattr(runner, "kv_caches", None):
        raise RuntimeError("KV caches are not initialized")
    if not getattr(runner, "kv_cache_config", None):
        raise RuntimeError("KV cache config not available on runner")

    _dbg("[DEBUG] Normalizing KV caches...")
    # The actual allocated GPU pages is normal_gpu_pages (already accounts for staging slots)
    # We need to pass the staging slots to normalize_runner_kv_caches so it can compute
    # the correct staging_base_page.
    staging_slots = cfg.num_staging_slots

    kv_cache_view = normalize_runner_kv_caches(
        runner.kv_caches, runner.kv_cache_config, staging_slots=cfg.num_staging_slots
    )
    dtype = kv_cache_view.dtype
    device = kv_cache_view.device
    _dbg(f"[DEBUG] KV cache view: dtype={dtype}, device={device}, num_layers={kv_cache_view.num_layers}")

    # normal_gpu_pages is the actual number of GPU pages allocated (after cache allocator
    # reserved space for staging slots). The cache allocator hook already reduced the
    # allocated size by cfg.num_staging_slots.
    staging_slots = cfg.num_staging_slots
    gpu_budget_blocks = int(normal_gpu_pages * (1.0 - cfg.offload_fraction))
    
    # If no room for staging, disable offloading
    if staging_slots == 0 or normal_gpu_pages <= gpu_budget_blocks:
        _dbg("[DEBUG] No room for staging slots, disabling offloading")
        return None

    _dbg(f"[DEBUG] Staging config: staging_slots={staging_slots}, gpu_budget_blocks={gpu_budget_blocks}, normal_gpu_pages={normal_gpu_pages}, offload_fraction={cfg.offload_fraction}")
    
    # If no room for staging, disable offloading
    if staging_slots == 0:
        _dbg("[DEBUG] No room for staging slots, disabling offloading")
        return None

    _dbg("[DEBUG] Building orchestrator...")
    orchestrator = build_orchestrator(
        OrchestratorConfig(
            offload_fraction=cfg.offload_fraction,
            num_cpu_slots=cfg.num_cpu_slots,
            num_staging_slots=staging_slots,
            W=cfg.interval_tokens,
            page_size=block_size,
            dtype=dtype,
            device=device,
            kv_layout=kv_cache_view.kv_layout,
            scorer_decay=cfg.scorer_decay,
            scorer_head_reduction=cfg.scorer_head_reduction,
            gpu_budget_blocks=gpu_budget_blocks,
        ),
        kv_cache_view=kv_cache_view,
    )
    _dbg("[DEBUG] Orchestrator created successfully!")
    return orchestrator


def prepare_sparse_pages(
    orchestrator,
    request_ids: Sequence[str],
    sparse_idx: torch.Tensor,
    sparse_len: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
) -> torch.Tensor:
    staging_map = orchestrator.prepare_staging(
        request_ids, sparse_idx, sparse_len
    )
    return orchestrator.patch_paged_kv_indices(
        paged_kv_indices,
        paged_kv_indptr,
        request_ids,
        sparse_idx,
        sparse_len,
        staging_map,
    )