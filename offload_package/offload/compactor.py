"""Token-granular gather/scatter/permute over the combined per-layer KV layout.

This fork stores one combined tensor per attention layer:
``[num_blocks, num_kv_heads, page_size, 2 * head_dim]`` with K in
``[..., :head_dim]`` and V in ``[..., head_dim:]``.

Token-level offloading needs three primitives over that layout:

* gather K of a set of token slots (for TriAttention scoring),
* gather full K/V of a set of token slots (for the D2H offload copy),
* an order-preserving permute that slides surviving tokens to the dense
  physical prefix so emptied tail pages can be returned to the allocator.

All primitives express a token slot as ``(block_id, in_page_offset)``; the
skylight kernels resolve token positions through the page table the same
way, so anything these primitives place at a slot is exactly what attention
reads for that position.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


def resolve_token_addresses(
    phys_slots: torch.Tensor,
    page_ids: Sequence[int],
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map physical token slots [0, phys_len) to (block_id, offset).

    Args:
        phys_slots: int64 tensor of physical token slots, any device.
        page_ids: the request's physical pages, in block-table order.
        page_size: tokens per page.

    Returns:
        (block_ids, offsets) tensors on the same device as ``phys_slots``.
    """
    if phys_slots.numel() == 0:
        empty = phys_slots.new_empty(0)
        return empty, empty.clone()
    max_slot = int(phys_slots.max().item())
    if max_slot >= len(page_ids) * page_size:
        raise IndexError(
            f"physical slot {max_slot} exceeds the request's page span "
            f"({len(page_ids)} pages x {page_size})"
        )
    page_ids_t = torch.as_tensor(
        page_ids, dtype=torch.long, device=phys_slots.device
    )
    block_idx = torch.div(phys_slots, page_size, rounding_mode="floor")
    offsets = phys_slots % page_size
    return page_ids_t[block_idx], offsets


class KVCompactor:
    """Gather/permute engine over the combined per-layer KV caches."""

    def __init__(
        self,
        kv_caches: Sequence[torch.Tensor],
        page_size: int,
    ) -> None:
        if not kv_caches:
            raise ValueError("at least one KV layer is required")
        first = kv_caches[0]
        if first.ndim != 4:
            raise ValueError(
                "combined KV cache must be [num_blocks, num_kv_heads, "
                f"page_size, 2*head_dim], got {tuple(first.shape)}"
            )
        if first.shape[2] != page_size:
            raise ValueError(
                f"page_size mismatch: cache holds {first.shape[2]}, "
                f"configured {page_size}"
            )
        if first.shape[-1] % 2:
            raise ValueError("combined KV last dim must be 2*head_dim")
        for cache in kv_caches:
            if cache.shape != first.shape:
                raise ValueError("all layers must share the KV cache shape")
        self.kv_caches = list(kv_caches)
        self.page_size = int(page_size)
        self.device = first.device
        self.num_layers = len(self.kv_caches)
        self.head_dim = first.shape[-1] // 2

    def gather_k(
        self, block_ids: torch.Tensor, offsets: torch.Tensor
    ) -> list[torch.Tensor]:
        """Per-layer K of the addressed tokens: [n, num_kv_heads, head_dim]."""
        head_dim = self.head_dim
        return [
            cache[block_ids, :, offsets, :head_dim] for cache in self.kv_caches
        ]

    def gather_kv(
        self, block_ids: torch.Tensor, offsets: torch.Tensor
    ) -> list[torch.Tensor]:
        """Per-layer K/V of the addressed tokens: [n, num_kv_heads, 2*head_dim]."""
        return [cache[block_ids, :, offsets, :] for cache in self.kv_caches]

    def scatter_kv(
        self,
        block_ids: torch.Tensor,
        offsets: torch.Tensor,
        gathered: Sequence[torch.Tensor],
    ) -> None:
        """Write gathered per-layer token K/V into the addressed slots.

        ``gathered[l]`` must be [n, num_kv_heads, 2*head_dim].
        """
        if len(gathered) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} gathered layers, "
                f"got {len(gathered)}"
            )
        for cache, chunk in zip(self.kv_caches, gathered):
            cache[block_ids, :, offsets, :] = chunk

    def permute(
        self,
        src_block_ids: torch.Tensor,
        src_offsets: torch.Tensor,
        dst_block_ids: torch.Tensor,
        dst_offsets: torch.Tensor,
    ) -> None:
        """Move tokens src -> dst in every layer, overlap-safe.

        The gather materializes a fresh tensor in stream order before any
        scatter write launches, so overlapping source/destination ranges
        cannot corrupt each other.
        """
        gathered = self.gather_kv(src_block_ids, src_offsets)
        self.scatter_kv(dst_block_ids, dst_offsets, gathered)
