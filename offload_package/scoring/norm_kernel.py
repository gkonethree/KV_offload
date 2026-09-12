from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import BlockKVStats


def compute_block_kv_norms(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    physical_block_ids: Sequence[int],
    *,
    logical_block_ids: Sequence[object] | None = None,
    layout: str = "NHD",
) -> BlockKVStats:
    """Compute mean token-wise L2 norms for selected physical pages."""

    if layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if len(physical_block_ids) != (len(logical_block_ids) if logical_block_ids is not None else len(physical_block_ids)):
        raise ValueError("physical_block_ids and logical_block_ids must have equal length")

    logical_ids = list(
        physical_block_ids if logical_block_ids is None else logical_block_ids
    )
    if not physical_block_ids:
        num_heads = k_cache.shape[2] if layout == "NHD" else k_cache.shape[1]
        empty = torch.empty(
            (0, num_heads), dtype=torch.float32, device=k_cache.device
        )
        return BlockKVStats(empty, empty, logical_ids)

    index = torch.as_tensor(
        list(physical_block_ids), dtype=torch.long, device=k_cache.device
    )
    k_sel = k_cache.index_select(0, index).float()
    v_sel = v_cache.index_select(0, index).float()

    if layout == "NHD":
        k_norm = k_sel.norm(dim=-1).mean(dim=1)
        v_norm = v_sel.norm(dim=-1).mean(dim=1)
    else:
        k_norm = k_sel.norm(dim=-1).mean(dim=2)
        v_norm = v_sel.norm(dim=-1).mean(dim=2)

    return BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=logical_ids)
