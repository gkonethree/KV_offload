"""
Correctness tests for sparse_oracle_topk_optimized.BatchDecodeWithPagedKVCacheWrapper.

Compares the optimized oracle top-k wrapper (custom CUDA score kernel +
torch.topk + sparse_optimized decode kernel) against the pure-PyTorch
reference under `sparse_oracle_topk/`.

Both implementations select the per-(batch, query-head) oracle top-k of the
Q @ K^T scores (``k = max(1, min(L, int(round(topk * L))))``) and run plain
softmax over the selected tokens. As long as the two paths agree on the same
top-k indices (modulo ties at the boundary cut by fp16/bf16 rounding), the
outputs must agree to fp16/bf16 attention tolerance.
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from sparse_oracle_topk import (
    BatchDecodeWithPagedKVCacheWrapper as RefOracleTopKWrapper,
)
from sparse_oracle_topk_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OptOracleTopKWrapper,
)
from original_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as DenseOptWrapper,
)


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


# (B, ctx, page, H_kv, H_q, D, topk, dtype, layout)
# ``topk`` is a float fraction of KV length: k = max(1, min(L, round(topk*L))).
CONFIGS = [
    (2,  64,  16,  4,  8,  64,  16 / 64,  torch.float16,  "NHD"),
    (4, 128,  16,  4, 16, 128,  32 / 128,  torch.float16,  "NHD"),
    (4, 256,  16,  8, 32, 128,  64 / 256,  torch.float16,  "NHD"),
    (8, 512,  16,  8, 32, 128,  128 / 512,  torch.float16,  "NHD"),
    (4, 1024, 16,  4, 16, 128,  64 / 1024,  torch.bfloat16, "NHD"),
    (2, 256,  16,  4,  4, 128,  16 / 256,  torch.float16,  "NHD"),  # group_size=1 (MHA)
    (4, 512,  16,  4, 32, 128,  32 / 512,  torch.float16,  "NHD"),  # group_size=8
    (4, 512,  16,  4, 24, 128,  32 / 512,  torch.float16,  "NHD"),  # group_size=6 (Qwen3.5-27B: 24q / 4kv)
    (2, 96,   16,  4,  8, 128,  1.0,  torch.float16,  "NHD"),  # topk>=1: clamp to L
]


def _ids(cfgs):
    return [
        f"B{c[0]}-ctx{c[1]}-Hkv{c[3]}-Hq{c[4]}-D{c[5]}-topk{c[6]:g}"
        f"-{str(c[7]).split('.')[-1]}-{c[8]}"
        for c in cfgs
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "B,ctx,page,H_kv,H_q,D,topk,dtype,layout", CONFIGS, ids=_ids(CONFIGS)
)
def test_optimized_matches_python_reference(
    B, ctx, page, H_kv, H_q, D, topk, dtype, layout
):
    device = torch.device("cuda")
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout=layout, device=device, seed=0,
    )

    ws_ref = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_opt = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ref = RefOracleTopKWrapper(ws_ref, layout)
    # Exercise the production (sync-free) path: vLLM-style max_model_len.
    opt = OptOracleTopKWrapper(ws_opt, layout, max_seq_len=ctx)
    plan_kwargs = dict(q_data_type=dtype, kv_data_type=dtype)
    for w in (ref, opt):
        w.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page, **plan_kwargs,
        )

    ref_out = ref.run(q, kv_cache, topk)
    opt_out = opt.run(q, kv_cache, topk)

    assert ref_out.shape == opt_out.shape == q.shape
    assert torch.isfinite(opt_out).all(), "optimized output has non-finite values"

    # Both paths now write fp16/bf16 selection scores. Different summation
    # orders (PyTorch GEMM vs warp-shuffle reduction) can flip a few
    # boundary ties in the top-k; this shows up as a small number of
    # positions disagreeing by an attention-output-magnitude amount. Loose
    # atol/rtol tolerates that while still catching genuine kernel bugs.
    rtol, atol = 1e-1, 1e-1
    diff = (opt_out.float() - ref_out.float()).abs()
    assert torch.allclose(
        opt_out.float(), ref_out.float(), rtol=rtol, atol=atol
    ), f"max abs diff opt-vs-ref = {diff.max():.4g} (mean={diff.mean():.4g})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_equals_full_context_matches_dense():
    """topk == 1.0 => oracle top-k attention must equal the dense decoder."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=2,
    )

    ws_d = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_s = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    dense = DenseOptWrapper(ws_d, "NHD")
    oracle = OptOracleTopKWrapper(ws_s, "NHD", max_seq_len=ctx)
    for w in (dense, oracle):
        w.plan(kv_indptr, kv_indices, kv_last_page_len,
               H_q, H_kv, D, page,
               q_data_type=dtype, kv_data_type=dtype)

    dense_out = dense.run(q, kv_cache)
    oracle_out = oracle.run(q, kv_cache, 1.0)

    assert torch.allclose(
        oracle_out.float(), dense_out.float(), rtol=5e-2, atol=5e-2
    ), f"max abs diff = {(oracle_out.float()-dense_out.float()).abs().max():.4g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_top_k_too_large_is_clamped():
    """topk * L > L should not crash; the wrapper clamps to L_max internally."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 64, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=3,
    )
    ws_d = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_o = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    dense = DenseOptWrapper(ws_d, "NHD")
    oracle = OptOracleTopKWrapper(ws_o, "NHD", max_seq_len=ctx)
    for w in (dense, oracle):
        w.plan(kv_indptr, kv_indices, kv_last_page_len,
               H_q, H_kv, D, page,
               q_data_type=dtype, kv_data_type=dtype)
    dense_out = dense.run(q, kv_cache)
    oracle_out = oracle.run(q, kv_cache, 100.0)
    assert torch.allclose(
        oracle_out.float(), dense_out.float(), rtol=5e-2, atol=5e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_returns_lse_shape_and_finite():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=4,
    )
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OptOracleTopKWrapper(ws, "NHD", max_seq_len=ctx)
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len,
                H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)
    out, lse = oracle.run(q, kv_cache, 32 / 256, return_lse=True)
    assert out.shape == q.shape
    assert lse.shape == (B, H_q)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_topk_raises():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 1, 32, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=5,
    )
    ws = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OptOracleTopKWrapper(ws, "NHD")
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len,
                H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, 0.0)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, -1.0)


# ---------------------------------------------------------------------- channel_num

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channel_num", [16, 32, 64, 128])
def test_optimized_channel_num_matches_python_reference(channel_num):
    """The optimized wrapper with a non-default ``channel_num`` must match
    the python reference run with the same ``channel_num``."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 512, 16, 4, 16, 128
    dtype = torch.float16
    topk = 64 / 512
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=11,
    )
    ws_ref = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_opt = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ref = RefOracleTopKWrapper(ws_ref, "NHD")
    opt = OptOracleTopKWrapper(ws_opt, "NHD", max_seq_len=ctx)
    for w in (ref, opt):
        w.plan(kv_indptr, kv_indices, kv_last_page_len,
               H_q, H_kv, D, page,
               q_data_type=dtype, kv_data_type=dtype)

    ref_out = ref.run(q, kv_cache, topk, channel_num=channel_num)
    opt_out = opt.run(q, kv_cache, topk, channel_num=channel_num)

    # See note in test_optimized_matches_python_reference: fp16 score storage
    # plus warp-vs-GEMM reduction order can flip boundary ties.
    diff = (opt_out.float() - ref_out.float()).abs()
    assert torch.allclose(
        opt_out.float(), ref_out.float(), rtol=1e-1, atol=1e-1
    ), f"channel_num={channel_num}: max abs diff = {diff.max():.4g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_channel_num_raises():
    """channel_num must be in [1, head_dim] and a multiple of vec lane."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 1, 64, 16, 4, 8, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=6,
    )
    ws = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OptOracleTopKWrapper(ws, "NHD")
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len,
                H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)
    topk = 8 / 64
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, topk, channel_num=0)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, topk, channel_num=D + 1)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, topk, channel_num=12)  # not a multiple of 8


# ---------------------------------------------------------------------- max_seq_len

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "max_seq_len_factor", [1.0, 2.0, 4.0],
    ids=["exact", "2x_padding", "4x_padding"],
)
def test_max_seq_len_override_matches_auto(max_seq_len_factor):
    """Explicit ``max_seq_len`` (vLLM's max_model_len path) must produce the
    same output as auto-derived ``L_max`` for any value >= the true per-batch
    max sequence length. The score kernel pads the extra
    ``[L_b, max_seq_len)`` range with -inf so top-k correctness is preserved.
    """
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    topk = 64 / 256
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=7,
    )

    ws_auto = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_cfg = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    auto = OptOracleTopKWrapper(ws_auto, "NHD")
    cfg = OptOracleTopKWrapper(
        ws_cfg, "NHD", max_seq_len=int(ctx * max_seq_len_factor)
    )
    for w in (auto, cfg):
        w.plan(kv_indptr, kv_indices, kv_last_page_len,
               H_q, H_kv, D, page,
               q_data_type=dtype, kv_data_type=dtype)

    out_auto = auto.run(q, kv_cache, topk)
    # Padded max_seq_len must pass n_keys=true max so k matches the auto path
    # without any plan()-time GPU read.
    out_cfg = cfg.run(q, kv_cache, topk, n_keys=ctx)

    diff = (out_cfg.float() - out_auto.float()).abs()
    assert torch.allclose(
        out_cfg.float(), out_auto.float(), rtol=1e-3, atol=1e-3
    ), (
        f"max_seq_len_factor={max_seq_len_factor}: "
        f"max abs diff cfg-vs-auto = {diff.max():.4g}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_seq_len_per_call_override():
    """Per-call ``run(max_seq_len=...)`` must work with and without
    ``__init__(max_seq_len=...)`` and must take precedence over the
    constructor value."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 128, 16, 4, 8, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=8,
    )

    ws_auto = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_pc = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_ov = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    auto = OptOracleTopKWrapper(ws_auto, "NHD")
    per_call = OptOracleTopKWrapper(ws_pc, "NHD")  # no ctor max_seq_len
    override = OptOracleTopKWrapper(ws_ov, "NHD", max_seq_len=ctx * 8)
    for w in (auto, per_call, override):
        w.plan(kv_indptr, kv_indices, kv_last_page_len,
               H_q, H_kv, D, page,
               q_data_type=dtype, kv_data_type=dtype)

    topk = 32 / 128
    out_auto = auto.run(q, kv_cache, topk)
    out_pc = per_call.run(q, kv_cache, topk, max_seq_len=ctx)
    # Per-call value (ctx) must override the (deliberately wrong) ctor value
    # (ctx * 8) and still match the auto path.
    out_ov = override.run(q, kv_cache, topk, max_seq_len=ctx)

    for name, out in (("per_call", out_pc), ("override", out_ov)):
        diff = (out.float() - out_auto.float()).abs()
        assert torch.allclose(
            out.float(), out_auto.float(), rtol=1e-3, atol=1e-3
        ), f"{name}: max abs diff vs auto = {diff.max():.4g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_max_seq_len_raises():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 1, 64, 16, 4, 8, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=9,
    )
    ws = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)

    with pytest.raises(ValueError):
        OptOracleTopKWrapper(ws, "NHD", max_seq_len=0)
    with pytest.raises(ValueError):
        OptOracleTopKWrapper(ws, "NHD", max_seq_len=-5)
    with pytest.raises(ValueError):
        OptOracleTopKWrapper(ws, "NHD", n_keys=0)

    oracle = OptOracleTopKWrapper(ws, "NHD")
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len,
                H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, 8 / 64, max_seq_len=0)
    with pytest.raises(ValueError):
        oracle.set_max_seq_len(0)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, 8 / 64, n_keys=0)
    with pytest.raises(ValueError):
        oracle.run(q, kv_cache, 8 / 64, n_keys=999999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_seq_len_path_is_sync_free():
    """When max_seq_len is supplied, run() must NOT touch _lmax_cache (which
    would imply a `.item()` sync was performed)."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 128, 16, 4, 8, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=10,
    )
    ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    oracle = OptOracleTopKWrapper(ws, "NHD", max_seq_len=ctx)
    oracle.plan(kv_indptr, kv_indices, kv_last_page_len,
                H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype)

    assert not hasattr(oracle, "_lmax_cache")
    for _ in range(3):
        _ = oracle.run(q, kv_cache, 32 / 128)
    # Still no cache entry => the device-side .item() path was never taken.
    assert not hasattr(oracle, "_lmax_cache"), (
        "_lmax_cache populated => .item() sync occurred on the max_seq_len path"
    )
