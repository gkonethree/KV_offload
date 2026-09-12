"""
Correctness tests for original_optimized.BatchDecodeWithPagedKVCacheWrapper.

Compares the standalone CUDA-backed wrapper against:
  * the pure-PyTorch reference in `original`
  * the production FlashInfer wrapper (when installed)

Sweeps a few common decode shapes (batch, ctx, kv heads, q heads, head dim,
page size, dtype, layout) so the kernel is exercised on every dispatch path.
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from original import BatchDecodeWithPagedKVCacheWrapper as OriginalWrapper
from original_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OptimizedWrapper,
)


def _get_flashinfer_wrapper_cls():
    flashinfer = pytest.importorskip("flashinfer")
    wrapper_cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if wrapper_cls is not None:
        return wrapper_cls
    return flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper


def _make_inputs(
    *,
    batch_size: int,
    context_len: int,
    page_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    kv_layout: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q


CONFIGS = [
    # (B, ctx, page, H_kv, H_q, D, dtype, layout)
    (4,   256, 16, 4,  16, 64,  torch.float16,  "NHD"),
    (4,   256, 16, 4,  16, 128, torch.float16,  "NHD"),
    (8,   1024, 16, 8, 32, 128, torch.float16,  "NHD"),
    (8,   2048, 16, 8, 64, 128, torch.float16,  "NHD"),
    (4,   1024, 16, 4, 16, 128, torch.bfloat16, "NHD"),
    (4,   256, 16, 4,  16, 64,  torch.float16,  "HND"),
    (8,   2048, 16, 8, 64, 128, torch.float16,  "HND"),
    (1,   4096, 16, 8, 8,  128, torch.float16,  "NHD"),
    (16,  512, 16, 4,  16, 128, torch.float16,  "NHD"),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "B,ctx,page,H_kv,H_q,D,dtype,layout", CONFIGS,
    ids=[f"B{c[0]}-ctx{c[1]}-Hkv{c[3]}-Hq{c[4]}-D{c[5]}-{str(c[6]).split('.')[-1]}-{c[7]}"
         for c in CONFIGS],
)
def test_optimized_matches_original(B, ctx, page, H_kv, H_q, D, dtype, layout):
    torch.manual_seed(0)
    device = torch.device("cuda")

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout=layout, device=device,
    )

    ws_orig = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_opt = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    orig = OriginalWrapper(ws_orig, layout)
    opt = OptimizedWrapper(ws_opt, layout)
    for w in (orig, opt):
        w.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page,
            pos_encoding_mode="NONE",
            q_data_type=dtype, kv_data_type=dtype,
        )

    orig_out, orig_lse = orig.run(q, kv_cache, return_lse=True)
    opt_out, opt_lse = opt.run(q, kv_cache, return_lse=True)

    assert orig_out.shape == opt_out.shape
    assert orig_lse.shape == opt_lse.shape
    rtol, atol = (5e-2, 5e-2) if dtype == torch.float16 else (1e-1, 1e-1)
    assert torch.allclose(opt_out.float(), orig_out.float(), rtol=rtol, atol=atol), (
        f"out diff max={ (opt_out.float()-orig_out.float()).abs().max().item():.4g}"
    )
    assert torch.allclose(opt_lse.float(), orig_lse.float(), rtol=rtol, atol=atol), (
        f"lse diff max={ (opt_lse.float()-orig_lse.float()).abs().max().item():.4g}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optimized_matches_flashinfer_default():
    """Sanity test mirroring the exact shape used in test_original.py."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 64
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device,
    )

    Flash = _get_flashinfer_wrapper_cls()
    ws_f = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_o = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    flash_w = Flash(ws_f, "NHD")
    opt_w = OptimizedWrapper(ws_o, "NHD")
    for w in (flash_w, opt_w):
        w.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page,
            pos_encoding_mode="NONE",
            q_data_type=dtype, kv_data_type=dtype,
        )

    fout, flse = flash_w.run(q, kv_cache, return_lse=True)
    oout, olse = opt_w.run(q, kv_cache, return_lse=True)

    assert torch.allclose(oout, fout, rtol=5e-2, atol=5e-2)
    assert torch.allclose(olse, flse, rtol=5e-2, atol=5e-2)


# (H_kv, H_q) pairs: group_size = H_q // H_kv
GROUP_SIZE_CONFIGS = [
    (8, 8, 1),    # MHA
    (8, 16, 2),
    (8, 32, 4),
    (8, 48, 6),
    (8, 64, 8),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "H_kv,H_q,group_size",
    GROUP_SIZE_CONFIGS,
    ids=[f"gs{g}" for _, _, g in GROUP_SIZE_CONFIGS],
)
def test_optimized_matches_original_across_group_sizes(H_kv, H_q, group_size):
    """Optimized CUDA decode should match the PyTorch reference for every supported GQA group size."""
    assert H_q // H_kv == group_size
    torch.manual_seed(group_size)
    device = torch.device("cuda")
    dtype = torch.float16
    B, ctx, page, D = 4, 512, 16, 128

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B,
        context_len=ctx,
        page_size=page,
        num_kv_heads=H_kv,
        num_q_heads=H_q,
        head_dim=D,
        dtype=dtype,
        kv_layout="NHD",
        device=device,
    )

    ws_orig = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_opt = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    orig = OriginalWrapper(ws_orig, "NHD")
    opt = OptimizedWrapper(ws_opt, "NHD")
    for w in (orig, opt):
        w.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            H_q,
            H_kv,
            D,
            page,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    orig_out, orig_lse = orig.run(q, kv_cache, return_lse=True)
    opt_out, opt_lse = opt.run(q, kv_cache, return_lse=True)

    rtol, atol = 5e-2, 5e-2
    assert torch.allclose(opt_out.float(), orig_out.float(), rtol=rtol, atol=atol), (
        f"group_size={group_size}: out max diff="
        f"{(opt_out.float() - orig_out.float()).abs().max().item():.4g}"
    )
    assert torch.allclose(opt_lse.float(), orig_lse.float(), rtol=rtol, atol=atol), (
        f"group_size={group_size}: lse max diff="
        f"{(opt_lse.float() - orig_lse.float()).abs().max().item():.4g}"
    )
