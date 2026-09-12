"""
Correctness tests for sparse_oracle_topk.BatchDecodeWithPagedKVCacheWrapper.

Two checks:

1. When ``topk == 1.0`` (keep all KV positions) the oracle top-k wrapper must
   equal the dense ``original`` wrapper exactly (modulo fp tolerance), because
   picking every token with all-ones weights reduces to plain softmax attention.

2. When ``topk < 1.0`` the wrapper output must match a tiny vanilla-PyTorch
   oracle top-k reference defined inline in this file.

Plus a couple of edge cases (clamping when ``topk * L > L``, basic shapes
and finiteness).
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from original import BatchDecodeWithPagedKVCacheWrapper as DenseWrapper
from sparse_oracle_topk import (
    BatchDecodeWithPagedKVCacheWrapper as OracleTopKWrapper,
)
from sparse_oracle_topk.batch_decode_with_paged_kv_cache_wrapper import (
    _topk_fraction_to_k_eff,
)


def _make_paged_inputs(
    *,
    batch_size: int,
    context_len: int,
    page_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
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
    # NHD layout: [P, 2, page_size, H_kv, D]
    kv_cache = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=dtype, device=device,
    )
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)
    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q


def _gather_kv_per_request(
    kv_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    batch_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct [L, H_kv, D] K and V tensors for one request (NHD layout)."""
    start = int(kv_indptr[batch_idx].item())
    end = int(kv_indptr[batch_idx + 1].item())
    page_ids = kv_indices[start:end].tolist()
    last = int(kv_last_page_len[batch_idx].item())
    k_pages, v_pages = [], []
    for i, pid in enumerate(page_ids):
        k_page = kv_cache[pid, 0]  # [page_size, H_kv, D]
        v_page = kv_cache[pid, 1]
        if i == len(page_ids) - 1:
            k_page = k_page[:last]
            v_page = v_page[:last]
        k_pages.append(k_page)
        v_pages.append(v_page)
    return torch.cat(k_pages, dim=0), torch.cat(v_pages, dim=0)


def _oracle_topk_reference(
    q: torch.Tensor,                     # [B, H_q, D]
    kv_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    *,
    topk: float,
    num_kv_heads: int,
    sm_scale: float,
    channel_num: int = None,
) -> torch.Tensor:
    """Per-(batch, head) oracle top-k attention; returns [B, H_q, D].

    The selection dot product uses only the first ``channel_num`` channels of
    Q and K (defaulting to the full ``head_dim``), but the attention itself
    is computed over the full head dim once the indices are picked.
    """
    B, H_q, D = q.shape
    if channel_num is None:
        channel_num = D
    group_size = H_q // num_kv_heads
    out = torch.empty_like(q)
    for b in range(B):
        k_seq, v_seq = _gather_kv_per_request(
            kv_cache, kv_indptr, kv_indices, kv_last_page_len, b
        )
        # broadcast KV to query-head dim
        k_seq = k_seq.repeat_interleave(group_size, dim=1)  # [L, H_q, D]
        v_seq = v_seq.repeat_interleave(group_size, dim=1)
        L = k_seq.shape[0]
        k_eff = _topk_fraction_to_k_eff(topk, L)
        for h in range(H_q):
            sel_q = q[b, h, :channel_num].float()
            sel_k = k_seq[:, h, :channel_num].float()
            sel_scores = sel_q @ sel_k.T  # [L]; sm_scale not applied (monotonic)
            top_idx = torch.topk(sel_scores, k=k_eff, dim=-1).indices
            full_scores = (q[b, h].float() @ k_seq[:, h].float().T) * sm_scale
            attn = torch.softmax(full_scores[top_idx], dim=-1)
            v_sel = v_seq[top_idx, h].float()  # [k_eff, D]
            out[b, h] = (attn.unsqueeze(-1) * v_sel).sum(dim=0).to(q.dtype)
    return out


# (B, ctx, page, H_kv, H_q, D, dtype)
SMOKE_CONFIGS = [
    (2,  64,  16, 4,  8,  64,  torch.float16),
    (2, 256,  16, 4, 16, 128,  torch.float16),
    (4, 128,  16, 8, 16, 128,  torch.bfloat16),
]


def _ids(cfgs):
    return [
        f"B{c[0]}-ctx{c[1]}-Hkv{c[3]}-Hq{c[4]}-D{c[5]}-{str(c[6]).split('.')[-1]}"
        for c in cfgs
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "B,ctx,page,H_kv,H_q,D,dtype", SMOKE_CONFIGS, ids=_ids(SMOKE_CONFIGS)
)
def test_topk_equals_full_context_matches_dense(B, ctx, page, H_kv, H_q, D, dtype):
    """topk == 1.0 => oracle top-k attention == dense attention."""
    device = torch.device("cuda")
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=0,
    )

    ws_d = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_s = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    dense = DenseWrapper(ws_d, "NHD")
    oracle = OracleTopKWrapper(ws_s, "NHD")
    plan_kwargs = dict(q_data_type=dtype, kv_data_type=dtype)
    for w in (dense, oracle):
        w.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page, **plan_kwargs,
        )

    dense_out = dense.run(q, kv_cache)
    oracle_out = oracle.run(q, kv_cache, 1.0)

    assert oracle_out.shape == q.shape
    assert torch.isfinite(oracle_out).all()

    rtol, atol = (5e-2, 5e-2) if dtype == torch.float16 else (1e-1, 1e-1)
    assert torch.allclose(
        oracle_out.float(), dense_out.float(), rtol=rtol, atol=atol
    ), f"max abs diff vs dense = {(oracle_out.float()-dense_out.float()).abs().max():.4g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "B,ctx,page,H_kv,H_q,D,dtype", SMOKE_CONFIGS, ids=_ids(SMOKE_CONFIGS)
)
def test_topk_subset_matches_python_reference(B, ctx, page, H_kv, H_q, D, dtype):
    """topk < 1.0 => matches a vanilla-PyTorch oracle top-k reference."""
    device = torch.device("cuda")
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=1,
    )
    topk = 0.25  # same as k = ctx//4 when ctx divisible by 4

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OracleTopKWrapper(ws, "NHD")
    oracle.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )

    sm_scale = 1.0 / math.sqrt(D)
    ref_out = _oracle_topk_reference(
        q, kv_cache, kv_indptr, kv_indices, kv_last_page_len,
        topk=topk, num_kv_heads=H_kv, sm_scale=sm_scale,
    )
    oracle_out = oracle.run(q, kv_cache, topk)

    assert oracle_out.shape == q.shape
    assert torch.isfinite(oracle_out).all()

    rtol, atol = (5e-2, 5e-2) if dtype == torch.float16 else (1e-1, 1e-1)
    diff = (oracle_out.float() - ref_out.float()).abs()
    assert torch.allclose(
        oracle_out.float(), ref_out.float(), rtol=rtol, atol=atol
    ), f"max abs diff vs python ref = {diff.max():.4g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channel_num", [16, 32, 64])
def test_channel_num_matches_python_reference(channel_num):
    """Selection over only the first ``channel_num`` channels matches the
    matching python reference (which slices Q/K the same way for selection
    but uses full head_dim for the actual attention)."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=42,
    )
    topk = 64 / 256  # keep 64 of 256 tokens

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OracleTopKWrapper(ws, "NHD")
    oracle.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )

    sm_scale = 1.0 / math.sqrt(D)
    ref_out = _oracle_topk_reference(
        q, kv_cache, kv_indptr, kv_indices, kv_last_page_len,
        topk=topk, num_kv_heads=H_kv, sm_scale=sm_scale,
        channel_num=channel_num,
    )
    oracle_out = oracle.run(q, kv_cache, topk, channel_num=channel_num)

    rtol, atol = 5e-2, 5e-2
    diff = (oracle_out.float() - ref_out.float()).abs()
    assert torch.allclose(
        oracle_out.float(), ref_out.float(), rtol=rtol, atol=atol
    ), f"channel_num={channel_num}: max abs diff vs python ref = {diff.max():.4g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_channel_num_full_equals_default():
    """Passing channel_num == head_dim must match leaving channel_num
    unspecified (which defaults to head_dim)."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 128, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=7,
    )

    ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OracleTopKWrapper(ws, "NHD")
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)

    topk = 32 / 128
    out_default = oracle.run(q, kv_cache, topk)
    out_full = oracle.run(q, kv_cache, topk, channel_num=D)
    assert torch.equal(out_default, out_full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_channel_num_raises():
    """channel_num <= 0 or > head_dim should raise ValueError."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 1, 32, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=5,
    )

    ws = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OracleTopKWrapper(ws, "NHD")
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)

    topk = 8 / 32
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, topk, channel_num=0)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, topk, channel_num=D + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_larger_than_seqlen_is_clamped():
    """Passing topk * L > L must not crash and must equal the dense answer."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 48, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=2,
    )

    ws_d = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_s = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    dense = DenseWrapper(ws_d, "NHD")
    oracle = OracleTopKWrapper(ws_s, "NHD")
    for w in (dense, oracle):
        w.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
               q_data_type=dtype, kv_data_type=dtype)

    dense_out = dense.run(q, kv_cache)
    # Ask for many more tokens than available; wrapper should clamp internally.
    oracle_out = oracle.run(q, kv_cache, 10.0)

    assert torch.allclose(
        oracle_out.float(), dense_out.float(), rtol=5e-2, atol=5e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_returns_lse_shape_and_finite():
    """Sanity-check the (out, lse) tuple form."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 64, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=3,
    )

    ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OracleTopKWrapper(ws, "NHD")
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)

    out, lse = oracle.run(q, kv_cache, 16 / 64, return_lse=True)
    assert out.shape == q.shape
    assert lse.shape == (B, H_q)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_topk_raises():
    """topk <= 0 should raise before doing any work."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 1, 32, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_paged_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, device=device, seed=4,
    )

    ws = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OracleTopKWrapper(ws, "NHD")
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)

    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, 0.0)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, -1.0)
