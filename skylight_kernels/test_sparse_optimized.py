"""
Correctness tests for sparse_optimized.BatchDecodeWithPagedKVCacheWrapper.

Compares the optimized sparse wrapper against the reference pure-PyTorch
sparse implementation under `sparse/`. Sweeps a range of common decode
shapes plus a few sparse-specific edge cases (varying sparse_len per
(batch, q_head), zero-length entries, sparsity patterns, soft cap).
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from sparse import BatchDecodeWithPagedKVCacheWrapper as RefSparseWrapper
from sparse_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OptSparseWrapper,
)


def _make_inputs(
    *,
    batch_size: int,
    context_len: int,
    page_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    max_sparse_context: int,
    dtype: torch.dtype,
    kv_layout: str,
    device: torch.device,
    seed: int = 0,
    allow_zero_sparse: bool = True,
) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    pages_per_req = math.ceil(context_len / page_size)
    total_pages = batch_size * pages_per_req
    last_page_len_value = context_len % page_size or page_size

    kv_indptr = torch.arange(
        0, total_pages + 1, pages_per_req, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), last_page_len_value, dtype=torch.int32, device=device
    )
    if kv_layout == "NHD":
        kv_cache = torch.randn(
            total_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
    else:
        kv_cache = torch.randn(
            total_pages, 2, num_kv_heads, page_size, head_dim,
            dtype=dtype, device=device,
        )
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)

    max_sparse_context = min(max_sparse_context, context_len)
    low_len = 0 if allow_zero_sparse else 1
    sparse_len = torch.randint(
        low=low_len,
        high=max_sparse_context + 1,
        size=(batch_size, num_q_heads, 1),
        dtype=torch.int32,
        device=device,
    )
    sparse_idx = torch.randint(
        low=0,
        high=context_len,
        size=(batch_size, num_q_heads, max_sparse_context),
        dtype=torch.int64,
        device=device,
    )
    sparse_weights = (
        torch.rand(
            batch_size, num_q_heads, max_sparse_context,
            dtype=torch.float32, device=device,
        )
        + 1e-3
    )
    return (kv_indptr, kv_indices, kv_last_page_len, kv_cache, q,
            sparse_len, sparse_idx, sparse_weights)


# Each entry: (B, ctx, page, H_kv, H_q, D, max_sparse, dtype, layout, soft_cap)
CONFIGS = [
    (4,   256, 16,  4,  16, 64,   64,   torch.float16,  "NHD", 0.0),
    (4,   256, 16,  4,  16, 128,  64,   torch.float16,  "NHD", 0.0),
    (8,  1024, 16,  8,  32, 128,  128,  torch.float16,  "NHD", 0.0),
    (8,  2048, 16,  8,  64, 128,  256,  torch.float16,  "NHD", 0.0),
    (4,  1024, 16,  4,  16, 128,  64,   torch.bfloat16, "NHD", 0.0),
    (4,   256, 16,  4,  16, 64,   32,   torch.float16,  "HND", 0.0),
    (8,  2048, 16,  8,  64, 128,  192,  torch.float16,  "HND", 0.0),
    (1,  4096, 16,  8,  8,  128,  256,  torch.float16,  "NHD", 0.0),
    (16,  512, 16,  4,  16, 128,  96,   torch.float16,  "NHD", 0.0),
    # soft cap variants
    (4,  1024, 16,  4,  16, 128,  64,   torch.float16,  "NHD", 30.0),
    # group_size=1 (MHA)
    (2,  1024, 16,  16, 16, 128,  128,  torch.float16,  "NHD", 0.0),
    # group_size=8 (heavy GQA)
    (4,  1024, 16,  4,  32, 128,  128,  torch.float16,  "NHD", 0.0),
    # head_dim=256, group_size=6 — Qwen3.5-27B full-attention layers
    # (24 q_heads / 4 kv_heads = 6, head_dim 256, bf16). This is the
    # ONLY config that exercises the launcher's BDY=4 branch
    # (HD=64→BDY=16, HD=128→BDY=8, HD=256→BDY=4). Without this
    # coverage the wrapper produces garbage on Qwen3.5-27B even at
    # full-coverage sparse budgets (sink+local ≥ seq_len).
    (1,  2048, 16,  4,  24, 256,  512,  torch.bfloat16, "NHD", 0.0),
    (2,  1024, 16,  4,  24, 256,  256,  torch.bfloat16, "HND", 0.0),
]


def _ids(configs):
    return [
        (
            f"B{c[0]}-ctx{c[1]}-Hkv{c[3]}-Hq{c[4]}-D{c[5]}"
            f"-S{c[6]}-{str(c[7]).split('.')[-1]}-{c[8]}"
            + ("-cap" if c[9] > 0 else "")
        )
        for c in configs
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "B,ctx,page,H_kv,H_q,D,maxS,dtype,layout,soft_cap", CONFIGS, ids=_ids(CONFIGS)
)
def test_sparse_optimized_matches_reference(
    B, ctx, page, H_kv, H_q, D, maxS, dtype, layout, soft_cap
):
    device = torch.device("cuda")
    (kv_indptr, kv_indices, kv_last_page_len, kv_cache, q,
     sparse_len, sparse_idx, sparse_weights) = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        max_sparse_context=maxS,
        dtype=dtype, kv_layout=layout, device=device, seed=0,
    )

    ws_ref = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_opt = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    ref = RefSparseWrapper(ws_ref, layout)
    opt = OptSparseWrapper(ws_opt, layout)
    plan_kwargs = dict(
        pos_encoding_mode="NONE",
        q_data_type=dtype, kv_data_type=dtype,
        logits_soft_cap=soft_cap if soft_cap > 0 else None,
    )
    for w in (ref, opt):
        w.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page, **plan_kwargs,
        )

    ref_out, ref_lse = ref.run(
        q, kv_cache, sparse_len, sparse_idx, sparse_weights, return_lse=True
    )
    opt_out, opt_lse = opt.run(
        q, kv_cache, sparse_len, sparse_idx, sparse_weights, return_lse=True
    )

    assert ref_out.shape == opt_out.shape
    assert ref_lse.shape == opt_lse.shape

    rtol_o, atol_o = (5e-2, 5e-2) if dtype == torch.float16 else (1e-1, 1e-1)
    rtol_l, atol_l = (5e-2, 5e-2) if dtype == torch.float16 else (1e-1, 1e-1)

    out_diff = (opt_out.float() - ref_out.float()).abs()
    out_max = out_diff.max().item()
    assert torch.allclose(
        opt_out.float(), ref_out.float(), rtol=rtol_o, atol=atol_o
    ), f"out diff max={out_max:.4g}"

    # LSE for zero-sparse-len rows is -inf in the reference; allow nan-equality
    lse_diff = (opt_lse.float() - ref_lse.float()).abs()
    finite = torch.isfinite(ref_lse) & torch.isfinite(opt_lse)
    if finite.any():
        max_lse_finite = lse_diff[finite].max().item()
        assert max_lse_finite <= max(
            atol_l, rtol_l * ref_lse[finite].abs().max().item()
        ), f"lse finite-diff max={max_lse_finite:.4g}"
    # Both should agree on which rows are -inf (zero-length).
    inf_ref = ~torch.isfinite(ref_lse) & (ref_lse < 0)
    inf_opt = ~torch.isfinite(opt_lse) & (opt_lse < 0)
    assert torch.equal(inf_ref, inf_opt), (
        "ref/opt disagree on which (b,qh) rows have -inf lse"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_zero_sparse_len_returns_zeros():
    """Edge case: every (b, qh) has sparse_len=0 -> output zeros, lse=-inf."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D, maxS = 2, 256, 16, 4, 16, 128, 32

    (kv_indptr, kv_indices, kv_last_page_len, kv_cache, q,
     sparse_len, sparse_idx, sparse_weights) = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        max_sparse_context=maxS,
        dtype=torch.float16, kv_layout="NHD", device=device, seed=1,
    )
    sparse_len.zero_()

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSparseWrapper(ws, "NHD")
    opt.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16, kv_data_type=torch.float16,
    )
    out, lse = opt.run(
        q, kv_cache, sparse_len, sparse_idx, sparse_weights, return_lse=True
    )
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.all(lse == float("-inf"))
