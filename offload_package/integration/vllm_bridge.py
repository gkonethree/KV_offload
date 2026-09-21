from __future__ import annotations

from collections.abc import Sequence

import sys
import torch

from ..config import OffloadConfig
from ..vllm_adapter import (
    KVCacheView,
    OrchestratorConfig,
    build_orchestrator,
    normalize_runner_kv_caches,
)


def initialize_for_runner(
    runner,
    *,
    normal_gpu_pages: int,
    block_size: int,
    kv_layout: str = "NHD",
) -> object | None:
    """Create an orchestrator after V2 runner KV caches are initialized."""
    print("[DEBUG] initialize_for_runner: START", file=sys.stderr, flush=True)
    cfg = OffloadConfig.from_env()
    print(f"[DEBUG] Config loaded: enabled={cfg.enabled}", file=sys.stderr, flush=True)
    if not cfg.enabled:
        return None
    if not getattr(runner, "kv_caches", None):
        raise RuntimeError("KV caches are not initialized")
    if not getattr(runner, "kv_cache_config", None):
        raise RuntimeError("KV cache config not available on runner")

    print("[DEBUG] Normalizing KV caches...", file=sys.stderr, flush=True)
    kv_cache_view = normalize_runner_kv_caches(runner.kv_caches, runner.kv_cache_config)
    dtype = kv_cache_view.dtype
    device = kv_cache_view.device
    print(f"[DEBUG] KV cache view: dtype={dtype}, device={device}, num_layers={kv_cache_view.num_layers}", file=sys.stderr, flush=True)

    # Calculate how many GPU blocks to reserve for normal KV cache operations
    # We need to leave room for staging_slots pages for sparse attention
    # Reserve offload_fraction of blocks for potential eviction to CPU
    gpu_budget_blocks = int(normal_gpu_pages * (1.0 - cfg.offload_fraction))
    
    # Ensure we have room for staging slots
    max_possible_staging = normal_gpu_pages - gpu_budget_blocks
    staging_slots = min(cfg.num_staging_slots, max_possible_staging)
    
    # Final gpu_budget after accounting for staging
    gpu_budget_blocks = normal_gpu_pages - staging_slots
    
    print(f"[DEBUG] Staging config: staging_slots={staging_slots}, gpu_budget_blocks={gpu_budget_blocks}, normal_gpu_pages={normal_gpu_pages}, offload_fraction={cfg.offload_fraction}", file=sys.stderr, flush=True)
    
    # If no room for staging, disable offloading
    if staging_slots == 0:
        print("[DEBUG] No room for staging slots, disabling offloading", file=sys.stderr, flush=True)
        return None

    print("[DEBUG] Building orchestrator...", file=sys.stderr, flush=True)
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
    print("[DEBUG] Orchestrator created successfully!", file=sys.stderr, flush=True)
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