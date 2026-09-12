from __future__ import annotations

from collections.abc import Sequence

import torch

from ..config import OffloadConfig
from ..vllm_adapter import OrchestratorConfig, build_orchestrator


def initialize_for_runner(
    runner,
    *,
    normal_gpu_pages: int,
    block_size: int,
    kv_layout: str = "NHD",
) -> object | None:
    """Create an orchestrator after V2 runner KV caches are initialized."""
    cfg = OffloadConfig.from_env()
    if not cfg.enabled:
        return None
    if not getattr(runner, "kv_caches", None):
        raise RuntimeError("KV caches are not initialized")

    cache0 = runner.kv_caches[0]
    if isinstance(cache0, (tuple, list)):
        dtype = cache0[0].dtype
        device = cache0[0].device
    else:
        dtype = cache0.dtype
        device = cache0.device

    orchestrator = build_orchestrator(
        OrchestratorConfig(
            gpu_budget_blocks=cfg.gpu_budget_blocks,
            num_cpu_slots=cfg.num_cpu_slots,
            num_staging_slots=cfg.num_staging_slots,
            interval_tokens=cfg.interval_tokens,
            page_size=block_size,
            dtype=dtype,
            device=device,
            kv_layout=kv_layout,
            scorer_decay=cfg.scorer_decay,
            scorer_head_reduction=cfg.scorer_head_reduction,
        ),
        runner_kv_caches=runner.kv_caches,
        staging_base_page_idx=normal_gpu_pages,
    )
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
