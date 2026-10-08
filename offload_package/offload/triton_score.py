# SPDX-License-Identifier: Apache-2.0
"""Fused per-block K/V norm scorer (Triton).

One kernel launch computes mean-per-token L2 norms for BOTH K and V halves of
every selected page across ALL layers: out[l, b, h] = [mean_t ||K_t||,
mean_t ||V_t||]. This replaces the per-layer index_select + norm + stack loop
in KVOffloadManager.score_blocks with a single launch.

Parity contract: value-equal (within fp32 accumulation tolerance) to
compute_block_kv_norms' per-head mean-over-token norms (mean of per-token
norms, not norm of means).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton is optional
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _kv_norm_kernel(
        LAYER_PTRS,          # int64 [L] base pointers, combined per-layer [P, H, N, 2C]
        BLOCK_IDS,           # int32 [B] page ids to score
        OUT,                 # fp32 [L, B, H, 2]
        nblocks,             # runtime int
        H,                   # num kv heads (runtime)
        N,                   # page size (runtime)
        C,                   # head_dim (runtime)
        C2,                  # 2*head_dim combined width (elements)
        stride_page,         # ELEMENTS per page
        stride_head,         # ELEMENTS per head
        stride_tok,          # ELEMENTS per token
        BLOCK_C: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        rest = pid // H
        b = rest % nblocks
        l = rest // nblocks

        base_i = tl.load(LAYER_PTRS + l)
        if IS_BF16:
            base = base_i.to(tl.pointer_type(tl.bfloat16))
        else:
            base = base_i.to(tl.pointer_type(tl.float16))
        gid = tl.load(BLOCK_IDS + b).to(tl.int64)

        page_base = base + gid * stride_page + h * stride_head

        offs = tl.arange(0, BLOCK_C)
        mask = offs < C

        acc_k = 0.0
        acc_v = 0.0
        for t in range(N):
            kv = tl.load(page_base + t * stride_tok + offs, mask=mask, other=0.0).to(tl.float32)
            vv = tl.load(page_base + t * stride_tok + offs + C, mask=mask, other=0.0).to(tl.float32)
            acc_k += tl.sqrt(tl.sum(kv * kv))
            acc_v += tl.sqrt(tl.sum(vv * vv))

        k_mean = acc_k / N
        v_mean = acc_v / N

        out_off = ((l * nblocks + b) * H + h) * 2
        tl.store(OUT + out_off, k_mean)
        tl.store(OUT + out_off + 1, v_mean)


def score_blocks_fused(
    gpu_caches: list[torch.Tensor],
    gpu_block_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (k_norm, v_norm) each [L, B, H] — per-head mean-over-token norms.

    Only valid when every layer cache shares this fork's combined
    [P, H, N, 2C] fp16 layout with dim-1-contiguous tail. Raises when Triton
    is unavailable or the layout doesn't fit; callers fall back to
    score_blocks.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton is not available")
    if not gpu_caches:
        raise ValueError("no layers")

    if not gpu_block_ids:
        L = len(gpu_caches)
        H = gpu_caches[0].shape[1]
        dev = gpu_caches[0].device
        empty = torch.empty((L, 0, H), dtype=torch.float32, device=dev)
        return empty, empty

    first = gpu_caches[0]
    L = len(gpu_caches)
    B = len(gpu_block_ids)
    P, H, N, C2 = first.shape
    if C2 % 2 != 0:
        raise ValueError(f"combined last dim must be even (K||V), got {C2}")
    C = C2 // 2
    dev = first.device
    if first.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype for triton scorer: {first.dtype}")

    ptrs = torch.tensor([c.data_ptr() for c in gpu_caches], dtype=torch.int64, device=dev)
    ids = torch.tensor(gpu_block_ids, dtype=torch.int32, device=dev)
    out = torch.empty((L, B, H, 2), dtype=torch.float32, device=dev)

    # torch strides are in ELEMENTS; Triton pointer arithmetic on a
    # float16-typed pointer scales by the element size, so pass them as-is.
    stride_page = int(first.stride(0))
    stride_head = int(first.stride(1))
    stride_tok = int(first.stride(2))
    if int(first.stride(3)) != 1:
        raise ValueError("last dim must be contiguous (stride 1 in elements)")

    BLOCK_C = triton.next_power_of_2(C)
    grid = (L * B * H,)
    _kv_norm_kernel[grid](
        ptrs, ids, out, B, H, N, C, C2,
        stride_page, stride_head, stride_tok, BLOCK_C,
        first.dtype == torch.bfloat16,
    )
    return out[..., 0], out[..., 1]
