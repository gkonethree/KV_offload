from __future__ import annotations

from collections.abc import Sequence

import torch

from ..integration.manager import OffloadOrchestrator


def register_prefill(
    orchestrator: OffloadOrchestrator,
    request_id: str,
    gpu_block_ids: Sequence[int],
    prompt_len: int,
) -> None:
    orchestrator.sync_request_blocks(request_id, gpu_block_ids, prompt_len)
    orchestrator.on_prefill_complete(request_id)


def register_decode_batch(
    orchestrator: OffloadOrchestrator,
    request_ids: Sequence[str],
    gpu_block_ids_by_request: Sequence[Sequence[int]],
    seq_lens: Sequence[int],
) -> list:
    for request_id, block_ids, seq_len in zip(
        request_ids, gpu_block_ids_by_request, seq_lens
    ):
        orchestrator.sync_request_blocks(request_id, block_ids, seq_len)
    return orchestrator.on_decode_step(request_ids, 1)


def prepare_sparse_pages(
    orchestrator: OffloadOrchestrator,
    *,
    request_ids: Sequence[str],
    sparse_idx: torch.Tensor,
    sparse_len: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
) -> torch.Tensor:
    staging = orchestrator.prepare_staging(request_ids, sparse_idx, sparse_len)
    return orchestrator.patch_paged_kv_indices(
        paged_kv_indices,
        paged_kv_indptr,
        request_ids,
        sparse_idx,
        sparse_len,
        staging,
    )
