"""Reserve the tail GPU KV pages for the offload staging pool.

Approach: instead of extending the physical KV tensors (fragile against the
fork's flat-buffer + strided-view allocator), the last ``num_staging_slots``
blocks are permanently excluded from the scheduler's ``BlockPool`` so the
allocator never hands them to a request. The pages remain physically present
and addressable in the per-layer KV views, so the staging pool can read/write
them at ``staging_base .. num_blocks-1`` without ever racing with request
allocations.

This works identically for the V1 and V2 runners and for both single- and
multi-layer groups, with no changes to vLLM's allocation geometry.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def reserve_staging_tail(block_pool, num_slots: int) -> list[int]:
    """Remove the tail ``num_slots`` pages from the scheduler BlockPool.

    Uses ``BlockPool.touch`` once at startup: a free block is removed from the
    free queue and its ref_cnt pinned at 1, so the block can never be
    allocated, evicted, or re-queued. Reserved pages stay addressable by the
    staging pool (they are inside the per-layer KV views).

    Returns the reserved block ids. Raises if the pool is too small.
    """
    if num_slots <= 0:
        return []
    n = block_pool.num_gpu_blocks
    if num_slots >= n:
        raise ValueError(
            f"num_staging_slots ({num_slots}) must be < num_gpu_blocks ({n})"
        )
    reserved_ids = list(range(n - num_slots, n))
    reserved_blocks = [block_pool.blocks[i] for i in reserved_ids]
    for blk in reserved_blocks:
        if blk.is_null:
            raise ValueError("staging tail overlaps the pool null block")
        if blk.ref_cnt != 0:
            raise RuntimeError(
                f"staging tail block {blk.block_id} is already in use "
                "(reserve_staging_tail must run before any allocation)"
            )
    block_pool.touch(reserved_blocks)
    logger.info(
        "Reserved %d staging pages: block ids %d..%d (pool has %d allocatable left)",
        num_slots, reserved_ids[0], reserved_ids[-1],
        block_pool.get_num_free_blocks(),
    )
    return reserved_ids


def install() -> None:
    """Deprecated no-op.

    Older revisions extended the physical KV tensors via a hook on an
    allocator symbol that no longer exists in this fork. Tail reservation via
    :func:`reserve_staging_tail` (called from
    ``OffloadPackageScheduler.bind_gpu_block_pool``) supersedes it.
    """
    logger.debug("cache_allocator.install() is a no-op (see reserve_staging_tail)")
    return None
