from __future__ import annotations

from collections.abc import Sequence

import torch

from ..scoring.base import BlockRef
from ..staging.residency import Residency, ResidencyTable


def collect_needed_block_refs(
    request_ids: Sequence[str],
    sparse_idx: torch.Tensor,
    sparse_len: torch.Tensor,
    page_size: int,
) -> list[BlockRef]:
    """Return unique logical blocks touched by the sparse token selections."""
    if sparse_idx.ndim != 3 or sparse_len.shape != sparse_idx.shape[:2] + (1,):
        raise ValueError("invalid sparse_idx/sparse_len shape")
    if len(request_ids) != sparse_idx.shape[0]:
        raise ValueError("request_ids must match sparse batch size")

    refs: list[BlockRef] = []
    seen: set[BlockRef] = set()
    for b, request_id in enumerate(request_ids):
        for q in range(sparse_idx.shape[1]):
            n = int(sparse_len[b, q, 0].item())
            if n <= 0:
                continue
            tokens = sparse_idx[b, q, :n]
            blocks = torch.div(tokens, page_size, rounding_mode="floor").unique()
            for block_idx in blocks.tolist():
                ref = BlockRef(str(request_id), int(block_idx))
                if ref not in seen:
                    seen.add(ref)
                    refs.append(ref)
    return refs


def patch_paged_kv_indices_with_staging(
    original_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    request_ids: Sequence[str],
    sparse_idx: torch.Tensor,
    sparse_len: torch.Tensor,
    page_size: int,
    staging_base_page_idx: int,
    residency: ResidencyTable,
    staging_map: dict[BlockRef, int],
) -> torch.Tensor:
    """Patch the flat vLLM page-index array at the correct request-local offsets.

    This fixes a key bug in the original implementation: searching for an old
    physical block ID in the flattened array is incorrect because physical IDs
    can repeat and the flat array has request-local block ordering.
    """
    req_id_to_batch_idx = {rid: i for i, rid in enumerate(request_ids)}
    if not staging_map:
        return original_indices
    if paged_kv_indptr.ndim != 1 or paged_kv_indptr.numel() != len(request_ids) + 1:
        raise ValueError("paged_kv_indptr must have B+1 entries")

    patched = original_indices.clone()
    indptr = paged_kv_indptr.detach().cpu().tolist()

    needed = collect_needed_block_refs(request_ids, sparse_idx, sparse_len, page_size)
    for ref in needed:
        slot = staging_map.get(ref)
        if slot is None:
            continue
        b = req_id_to_batch_idx[ref.request_id]
        flat_pos = int(indptr[b]) + ref.block_idx
        if flat_pos >= int(indptr[b + 1]):
            raise IndexError(
                f"logical block {ref.block_idx} is outside request {ref.request_id}'s page table"
            )
        patched[flat_pos] = int(staging_base_page_idx + slot)
    return patched
