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

    if sparse_idx.numel() == 0:
        return []

    # All-tensor fast path: per-row valid lengths from ONE host sync; per-row
    # dedup on device; one host transfer per row of block ids. Identical
    # output set to the per-head loop this replaced.
    blocks_all = torch.div(sparse_idx, page_size, rounding_mode="floor")  # [B,Hq,S]
    lens = sparse_len[:, 0, 0].clamp(0, blocks_all.shape[-1])            # [B]
    lens_host = lens.detach().cpu().tolist()                              # the one sync

    refs: list[BlockRef] = []
    seen: set[BlockRef] = set()
    for b, request_id in enumerate(request_ids):
        n = int(lens_host[b])
        if n <= 0:
            continue
        uniq = blocks_all[b, :, :n].unique()  # [u] block indices valid for this row
        rid = str(request_id)
        for block_idx in uniq.tolist():
            ref = BlockRef(rid, int(block_idx))
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def patch_paged_kv_indices_with_staging(
    original_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    request_ids: Sequence[str],
    sparse_idx: torch.Tensor | None,
    sparse_len: torch.Tensor | None,
    page_size: int,
    staging_base_page_idx: int,
    residency: ResidencyTable,
    staging_map: dict[BlockRef, int],
    null_block_id: int | None = None,
) -> torch.Tensor:
    """Patch the flat vLLM page-index array so sparse attention reads the right pages.

    For each request row (decode rows are the leading rows of the batch):
    - Selected, offloaded blocks fetch into staging and their positions are
      rewritten to ``staging_base_page_idx + slot`` (from ``staging_map`` or
      from the residency table for blocks still resident in staging).
    - CPU-resident blocks not currently staged are rewritten to
      ``null_block_id`` when ``null_block_id`` is not None. Their original GPU
      pages may have been freed and reallocated, so the kernel must never read
      them. (When true freeing is off, pass ``null_block_id=None`` to keep the
      still-valid original page ids.)

    ``sparse_idx``/``sparse_len`` may be None: then no staging slots are
    referenced, only the residency-based null fill runs.

    This fixes a key bug in the original implementation: searching for an old
    physical block ID in the flattened array is incorrect because physical IDs
    can repeat and the flat array has request-local block ordering.
    """
    req_id_to_batch_idx = {rid: i for i, rid in enumerate(request_ids)}
    if paged_kv_indptr.ndim != 1 or paged_kv_indptr.numel() < len(request_ids) + 1:
        raise ValueError("paged_kv_indptr must have B+1 entries")

    patched = original_indices.clone()
    indptr = paged_kv_indptr.detach().cpu().tolist()

    # 1) Newly staged blocks (this step's fetch) -> staging slots.
    if sparse_idx is not None:
        needed = collect_needed_block_refs(
            request_ids, sparse_idx, sparse_len, page_size
        )
        for ref in needed:
            slot = staging_map.get(ref)
            if slot is None:
                continue
            b = req_id_to_batch_idx.get(ref.request_id)
            if b is None:
                continue
            flat_pos = int(indptr[b]) + ref.block_idx
            if flat_pos >= int(indptr[b + 1]):
                continue  # pattern computed for a previous batch composition
            patched[flat_pos] = int(staging_base_page_idx + slot)

    # 2) Residency remediation for every block row of every request:
    #    STAGING -> its staging slot; CPU -> null block when freeing is on.
    #    Vectorized: build (positions, values) pairs only for blocks whose
    #    residency actually differs from GPU, then one device scatter.
    patch_pos: list[int] = []
    patch_val: list[int] = []
    for b, req_id in enumerate(request_ids):
        start, end = int(indptr[b]), int(indptr[b + 1])
        n_blocks = end - start
        if n_blocks <= 0:
            continue
        refs = residency.refs_for_request(str(req_id))
        if not refs or len(refs) != n_blocks:
            # Row length does not match tracked state (batch-composition
            # drift); skip this row rather than write a wrong page.
            continue
        accounting = [ref for ref in refs
                      if (loc := residency.get(ref)) is not None
                      and loc.residency != Residency.GPU]
        if not accounting:
            continue
        rid = str(req_id)
        for ref in accounting:
            loc = residency.get(ref)
            if loc.residency == Residency.STAGING and loc.staging_slot is not None:
                patch_pos.append(start + ref.block_idx)
                patch_val.append(int(staging_base_page_idx + loc.staging_slot))
            elif loc.residency == Residency.CPU and null_block_id is not None:
                patch_pos.append(start + ref.block_idx)
                patch_val.append(int(null_block_id))

    if patch_pos:
        dev = patched.device
        patched.index_put_(
            (torch.tensor(patch_pos, dtype=torch.long, device=dev),),
            torch.tensor(patch_val, dtype=patched.dtype, device=dev),
        )

    return patched
