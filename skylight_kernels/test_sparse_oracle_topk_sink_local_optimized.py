"""
Correctness tests for sparse_oracle_topk_sink_local_optimized.

Compares the optimized wrapper (CUDA score kernel + topk + sparse decode) against
the pure-Python reference under ``sparse_oracle_topk_sink_local/``.

Both implementations select tokens from three regions:
  1. Sink    – first ``sink_size`` tokens
  2. Top-k   – ``k = round(topk * mid_len)`` tokens from the middle range
  3. Local   – last ``local_size`` tokens

The outputs must agree to fp16/bf16 attention tolerance.  A small number of
boundary ties may flip between the two paths (different fp16 summation orders),
so we use a loose atol/rtol of 0.1 (same as the existing oracle-topk tests).
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest
import torch

from sparse_oracle_topk_sink_local import (
    BatchDecodeWithPagedKVCacheWrapper as RefSinkLocalWrapper,
)
from sparse_oracle_topk_sink_local_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OptSinkLocalWrapper,
)
from original_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as DenseOptWrapper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _plan_both(ref, opt, kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
               dtype, ctx):
    kwargs = dict(q_data_type=dtype, kv_data_type=dtype)
    ref.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page, **kwargs)
    opt.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page, **kwargs)


# ---------------------------------------------------------------------------
# Main parametrised test: opt vs Python reference
# (B, ctx, page, H_kv, H_q, D, topk, sink, local, dtype, layout)
# ---------------------------------------------------------------------------

CONFIGS = [
    # sink=0, local=0 → must match pure oracle topk
    (2,  64, 16,  4,  8,  64, 16/64,   0,   0, torch.float16, "NHD"),
    (4, 128, 16,  4, 16, 128, 32/128,  0,   0, torch.float16, "NHD"),
    # sink only
    (4, 256, 16,  4, 16, 128, 32/256,  4,   0, torch.float16, "NHD"),
    (4, 256, 16,  4, 16, 128, 32/256, 16,   0, torch.float16, "NHD"),
    # local only
    (4, 256, 16,  4, 16, 128, 32/256,  0,  32, torch.float16, "NHD"),
    (4, 256, 16,  4, 16, 128, 32/256,  0,  64, torch.float16, "NHD"),
    # sink + local
    (4, 512, 16,  4, 16, 128, 64/512,  8,  64, torch.float16, "NHD"),
    (8, 512, 16,  8, 32, 128,128/512, 16, 128, torch.float16, "NHD"),
    (4,1024, 16,  4, 16, 128, 64/1024,16, 256, torch.bfloat16,"NHD"),
    # GQA group_size=1 (MHA)
    (2, 256, 16,  4,  4, 128, 16/256,  4,  32, torch.float16, "NHD"),
    # group_size=8
    (4, 512, 16,  4, 32, 128, 32/512,  8,  64, torch.float16, "NHD"),
    # group_size=6 (Qwen3.5-27B GQA: 24 q-heads / 4 kv-heads)
    (4, 1024, 16,  4, 24, 128, 64/1024, 16, 64, torch.float16, "NHD"),
    # channel_num < head_dim
    (4, 512, 16,  4, 16, 128, 64/512,  8,  64, torch.float16, "NHD"),
]


def _ids(cfgs):
    return [
        f"B{c[0]}-ctx{c[1]}-Hkv{c[3]}-Hq{c[4]}-D{c[5]}"
        f"-topk{c[6]:g}-sink{c[7]}-local{c[8]}"
        f"-{str(c[9]).split('.')[-1]}"
        for c in cfgs
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "B,ctx,page,H_kv,H_q,D,topk,sink,local,dtype,layout",
    CONFIGS, ids=_ids(CONFIGS),
)
def test_optimized_matches_python_reference(
    B, ctx, page, H_kv, H_q, D, topk, sink, local, dtype, layout,
):
    device = torch.device("cuda")
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout=layout, device=device, seed=0,
    )

    ws_ref = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_opt = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ref = RefSinkLocalWrapper(ws_ref, layout)
    opt = OptSinkLocalWrapper(ws_opt, layout, max_seq_len=ctx)
    _plan_both(ref, opt, kv_indptr, kv_indices, kv_last_page_len,
               H_q, H_kv, D, page, dtype, ctx)

    ref_out = ref.run(q, kv_cache, topk, sink_size=sink, local_size=local)
    opt_out = opt.run(q, kv_cache, topk, sink_size=sink, local_size=local)

    assert ref_out.shape == opt_out.shape == q.shape
    assert torch.isfinite(opt_out).all(), "optimized output has non-finite values"

    diff = (opt_out.float() - ref_out.float()).abs()
    assert torch.allclose(
        opt_out.float(), ref_out.float(), rtol=1e-1, atol=1e-1
    ), (
        f"sink={sink} local={local}: max abs diff = {diff.max():.4g} "
        f"(mean={diff.mean():.4g})"
    )


# ---------------------------------------------------------------------------
# channel_num < head_dim
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channel_num", [8, 16, 32, 64])
def test_channel_num_matches_reference(channel_num):
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 512, 16, 4, 16, 128
    dtype = torch.float16
    topk = 64 / 512
    sink, local = 8, 64
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=20,
    )
    ws_ref = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_opt = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ref = RefSinkLocalWrapper(ws_ref, "NHD")
    opt = OptSinkLocalWrapper(ws_opt, "NHD", max_seq_len=ctx)
    _plan_both(ref, opt, kv_indptr, kv_indices, kv_last_page_len,
               H_q, H_kv, D, page, dtype, ctx)

    ref_out = ref.run(q, kv_cache, topk, sink_size=sink, local_size=local,
                      channel_num=channel_num)
    opt_out = opt.run(q, kv_cache, topk, sink_size=sink, local_size=local,
                      channel_num=channel_num)

    diff = (opt_out.float() - ref_out.float()).abs()
    assert torch.allclose(
        opt_out.float(), ref_out.float(), rtol=1e-1, atol=1e-1
    ), f"channel_num={channel_num}: max abs diff = {diff.max():.4g}"


# ---------------------------------------------------------------------------
# topk=1.0 + sink=0 + local=0  →  must match dense decoder
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_topk_no_sink_local_matches_dense():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=1,
    )
    ws_d = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_s = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    dense = DenseOptWrapper(ws_d, "NHD")
    opt = OptSinkLocalWrapper(ws_s, "NHD", max_seq_len=ctx)
    kw = dict(q_data_type=dtype, kv_data_type=dtype)
    dense.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page, **kw)
    opt.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page, **kw)

    dense_out = dense.run(q, kv_cache)
    opt_out = opt.run(q, kv_cache, 1.0, sink_size=0, local_size=0)

    assert torch.allclose(
        opt_out.float(), dense_out.float(), rtol=5e-2, atol=5e-2
    ), f"max abs diff = {(opt_out.float()-dense_out.float()).abs().max():.4g}"


# ---------------------------------------------------------------------------
# return_lse=True
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_return_lse_shape_and_finite():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=2,
    )
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=ctx)
    opt.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
             q_data_type=dtype, kv_data_type=dtype)
    out, lse = opt.run(q, kv_cache, 32/256, sink_size=4, local_size=32, return_lse=True)
    assert out.shape == q.shape
    assert lse.shape == (B, H_q)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()


# ---------------------------------------------------------------------------
# Constructor / per-call default resolution
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_constructor_defaults_override():
    """Values supplied to __init__ must be overridable per run() call."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=3,
    )
    ws1 = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws2 = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    kw = dict(q_data_type=dtype, kv_data_type=dtype)

    # Wrapper with ctor defaults
    opt_ctor = OptSinkLocalWrapper(
        ws1, "NHD", max_seq_len=ctx,
        topk=32/256, sink_size=4, local_size=32,
    )
    opt_ctor.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page, **kw)
    out_ctor = opt_ctor.run(q, kv_cache)   # uses ctor topk/sink/local

    # Wrapper without ctor defaults; pass per call
    opt_call = OptSinkLocalWrapper(ws2, "NHD", max_seq_len=ctx)
    opt_call.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page, **kw)
    out_call = opt_call.run(q, kv_cache, 32/256, sink_size=4, local_size=32)

    diff = (out_ctor.float() - out_call.float()).abs()
    # flashinfer.top_k is non-deterministic at tie boundaries → allow fp16-level noise
    assert torch.allclose(
        out_ctor.float(), out_call.float(), rtol=1e-3, atol=1e-3
    ), f"ctor vs per-call: max diff = {diff.max():.4g}"


# ---------------------------------------------------------------------------
# max_seq_len sync-free path must not touch _lmax_cache
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_max_seq_len_path_is_sync_free():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 128, 16, 4, 8, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=4,
    )
    ws = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=ctx)
    opt.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
             q_data_type=dtype, kv_data_type=dtype)
    assert not hasattr(opt, "_lmax_cache")
    for _ in range(3):
        opt.run(q, kv_cache, 32/128, sink_size=4, local_size=16)
    assert not hasattr(opt, "_lmax_cache"), (
        "_lmax_cache was populated → .item() sync occurred on max_seq_len path"
    )


# ---------------------------------------------------------------------------
# Input-validation guards
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_topk_raises():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 1, 64, 16, 4, 8, 64
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=5,
    )
    ws = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD")
    opt.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
             q_data_type=dtype, kv_data_type=dtype)
    with pytest.raises(ValueError):
        opt.run(q, kv_cache, 0.0)
    with pytest.raises(ValueError):
        opt.run(q, kv_cache, -0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_channel_num_raises():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 1, 64, 16, 4, 8, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=6,
    )
    ws = torch.zeros(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD")
    opt.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
             q_data_type=dtype, kv_data_type=dtype)
    with pytest.raises(ValueError):
        opt.run(q, kv_cache, 0.5, channel_num=0)
    with pytest.raises(ValueError):
        opt.run(q, kv_cache, 0.5, channel_num=D + 1)
    with pytest.raises(ValueError):
        opt.run(q, kv_cache, 0.5, channel_num=12)  # not multiple of 8


# ---------------------------------------------------------------------------
# Setters work and take effect on the next run() call
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_setters_take_effect():
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=7,
    )
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=ctx)
    opt.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
             q_data_type=dtype, kv_data_type=dtype)

    opt.set_topk(32/256)
    opt.set_sink_size(8)
    opt.set_local_size(32)

    # Should not raise
    out = opt.run(q, kv_cache)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Edge case the kernel didn't previously handle: context shorter than the
# combined sink+local window leaves no middle range for top-K scoring.
#
# vLLM's memory-profile pass and CUDA-graph warmup run dummy forwards with
# very short seq_lens (often 1 per request). When ``sink_size + local_size
# > L``, the middle range ``[sink_size, L - local_size)`` is empty / negative.
# The wrapper used to call ``compute_scores_compact`` with
# ``token_end <= token_start`` and the C op asserted
# "token_end must be > token_start", crashing the engine before it ever
# served a request.
#
# The wrapper now gates the scoring step on ``k_eff_mid > 0``: when there's
# no middle to score, it skips the kernel entirely and falls through to
# sink + local indices only — which together cover the entire context in
# this case, so the output is equivalent to dense attention.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_short_context_skips_topk_scoring_and_matches_dense():
    """Short context (sink+local > L) must not crash; output equals dense.

    When sink_size + local_size >= context_len, the union of [0, sink) and
    [L-local, L) covers every token, so the attention is effectively dense.
    """
    device = torch.device("cuda")
    # ctx (16) far smaller than sink+local (192). Mimics vllm's memory-profile
    # dummy-run shape multiplied up to something the kernel can plan on.
    B, ctx, page = 2, 16, 16
    H_kv, H_q, D = 4, 4, 64
    sink, local = 64, 128
    topk_frac = 0.05
    dtype = torch.float16
    layout = "NHD"

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout=layout, device=device, seed=0,
    )

    ws_opt = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws_opt, layout, max_seq_len=ctx)
    opt.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )

    # Must not raise — this was the crash point before the fix.
    out = opt.run(q, kv_cache, topk_frac, sink_size=sink, local_size=local)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()

    # When sink + local >= ctx, the selected positions union to [0, L),
    # i.e. dense attention. Compare against the dense wrapper.
    ws_dense = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    dense = DenseOptWrapper(ws_dense, layout)
    dense.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )
    dense_out = dense.run(q, kv_cache)

    assert torch.allclose(
        out.float(), dense_out.float(), rtol=1e-1, atol=1e-1,
    ), (
        f"output should match dense when sink+local cover full context "
        f"(B={B} ctx={ctx} sink={sink} local={local}); "
        f"max abs diff = {(out.float() - dense_out.float()).abs().max():.4g}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_seq_len_1_does_not_crash():
    """Extreme case: context_len=1 (the actual shape vLLM's profile pass uses)."""
    device = torch.device("cuda")
    B, ctx, page = 1, 1, 16  # paged kv allocates one page, last_page_len=1
    H_kv, H_q, D = 4, 4, 64
    sink, local = 128, 256  # both larger than ctx=1
    dtype = torch.float16
    layout = "NHD"

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout=layout, device=device, seed=0,
    )

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    # max_seq_len must be >= ctx; pass a generous value like vllm would.
    opt = OptSinkLocalWrapper(ws, layout, max_seq_len=4096)
    opt.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )
    out = opt.run(q, kv_cache, 0.05, sink_size=sink, local_size=local)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()

    # With ctx=1, attention degenerates to the single token's V → matches dense.
    # Strengthens the test from "doesn't crash" to "computes the right answer".
    ws_dense = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    dense = DenseOptWrapper(ws_dense, layout)
    dense.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )
    dense_out = dense.run(q, kv_cache)
    assert torch.allclose(
        out.float(), dense_out.float(), rtol=1e-1, atol=1e-1,
    ), (
        f"ctx=1 should equal dense; "
        f"max abs diff = {(out.float() - dense_out.float()).abs().max():.4g}"
    )


# ---------------------------------------------------------------------------
# Regression: per-batch L_b cap on sparse_len
# ---------------------------------------------------------------------------
# Before the per-request cap, sparse_len was a constant total_k = sink + local + k_topk
# broadcast across all batches. When a request's actual context length L_b was
# smaller than total_k (short request mixed with long ones in the same batch),
# the kernel read kv_indices[indptr[b] + page_idx] for page_idx >= pages(b),
# falling off the end of request b's page slice into request b+1's pages — i.e.
# attending to a neighboring request's KV. The fix caps sparse_len per-request
# at L_b.
#
# This test runs a batch with mixed context lengths through the optimized
# kernel, then runs each request individually as B=1 (no neighbors, so the
# bug cannot manifest there), and asserts the per-request outputs match.
# Without the fix, the short-context batches' outputs differ from their B=1
# baselines because the kernel reads from neighboring batches' KV.

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_batch_short_contexts_dont_read_neighbors():
    device = torch.device("cuda")
    page = 16
    H_kv, H_q, D = 4, 16, 128
    dtype = torch.float16
    sink, local = 64, 64
    topk = 0.10
    layout = "NHD"

    # Mix: batch 0 has L_b=20 (<< sink+local=128), batches 1-2 have plenty.
    ctx_per_req = [20, 200, 800]
    pages_per_req = [math.ceil(c / page) for c in ctx_per_req]
    total_pages = sum(pages_per_req)

    torch.manual_seed(7)
    kv_cache = torch.randn(
        total_pages, 2, page, H_kv, D, dtype=dtype, device=device,
    )

    # indptr[b] = cumulative pages preceding batch b.
    indptr_vals = [0]
    for p in pages_per_req:
        indptr_vals.append(indptr_vals[-1] + p)
    kv_indptr = torch.tensor(indptr_vals, dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [c % page or page for c in ctx_per_req],
        dtype=torch.int32, device=device,
    )
    q = torch.randn(len(ctx_per_req), H_q, D, dtype=dtype, device=device)

    # ---- mixed-batch path (the one that used to read neighbors) ----
    ws_mixed = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    mixed = OptSinkLocalWrapper(ws_mixed, layout, max_seq_len=max(ctx_per_req))
    mixed.plan(
        kv_indptr, kv_indices, last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )
    out_mixed = mixed.run(q, kv_cache, topk, sink_size=sink, local_size=local)

    assert torch.isfinite(out_mixed).all(), \
        "mixed-batch output contains NaN/Inf — kernel likely read past page slice"

    # ---- per-request reference: B=1, no neighbors, can't manifest the bug ----
    for b, (ctx_b, pages_b) in enumerate(zip(ctx_per_req, pages_per_req)):
        page_start = sum(pages_per_req[:b])
        page_end = page_start + pages_b
        single_kv = kv_cache[page_start:page_end].contiguous()
        single_indptr = torch.tensor([0, pages_b], dtype=torch.int32, device=device)
        single_indices = torch.arange(pages_b, dtype=torch.int32, device=device)
        single_last = last_page_len[b:b + 1].contiguous()
        single_q = q[b:b + 1]

        ws_ref = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        ref = OptSinkLocalWrapper(ws_ref, layout, max_seq_len=max(ctx_per_req))
        ref.plan(
            single_indptr, single_indices, single_last,
            H_q, H_kv, D, page,
            q_data_type=dtype, kv_data_type=dtype,
        )
        out_single = ref.run(single_q, single_kv, topk, sink_size=sink, local_size=local)

        diff = (out_mixed[b:b + 1].float() - out_single.float()).abs()
        assert torch.allclose(
            out_mixed[b:b + 1].float(), out_single.float(),
            rtol=1e-1, atol=1e-1,
        ), (
            f"batch {b} (L_b={ctx_b}): mixed-batch output differs from "
            f"single-batch reference (max abs diff = {diff.max():.4g}). "
            f"Most likely cause: kernel read past request {b}'s page slice "
            f"into request {b + 1}'s KV (the constant-total_k sparse_len OOB bug)."
        )


# ---------------------------------------------------------------------------
# Sparsity stats instrumentation
# ---------------------------------------------------------------------------
#
# The wrapper records per-call sparsity stats (Python-side, sync-free) for
# downstream prometheus instrumentation. These tests pin the contract:
#
#   1. Initial stats are zero.
#   2. After a fully-saturating (topk=1.0, sink=0, local=0) run, last_fraction
#      == 1.0 — exercises the "looks like dense" boundary.
#   3. After a configured sparse run, last_selected matches the formula
#      sink + k_eff_mid + local (capped at n_keys by the per-request L_b cap).
#   4. cumulative_* and call_count accumulate across multiple run() calls.
#   5. get_sparsity_stats() returns a snapshot — mutating the returned object
#      does NOT affect the wrapper's internal state (catches "accidentally
#      returning the live reference" bugs that could let downstream code
#      poison the wrapper's counters).


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparsity_stats_initial_state():
    """A freshly-constructed wrapper has all-zero sparsity stats."""
    device = torch.device("cuda")
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=256)

    stats = opt.get_sparsity_stats()
    assert stats.last_selected == 0
    assert stats.last_total == 0
    assert stats.last_fraction == 0.0
    assert stats.cumulative_selected == 0
    assert stats.cumulative_total == 0
    assert stats.call_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparsity_stats_dense_topk_one():
    """topk=1.0, sink=0, local=0 → last_fraction == 1.0 (boundary check).

    This is the silent-dense-fallback canary: if our backend ever silently
    dispatched the dense decode path, the wrapper would still observe
    selected == total → fraction == 1.0. Conversely, when the operator
    truly configures dense-ish behaviour (topk=1.0), we expect 1.0.
    """
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    dtype = torch.float16

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=0,
    )

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=ctx)
    opt.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )

    opt.run(q, kv_cache, 1.0, sink_size=0, local_size=0)

    stats = opt.get_sparsity_stats()
    assert stats.last_total == ctx
    assert stats.last_selected == ctx
    assert stats.last_fraction == 1.0
    assert stats.call_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparsity_stats_sparse_run_matches_formula():
    """last_selected matches sink + k_eff_mid + local for a known config."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 4, 256, 16, 4, 16, 128
    sink, local = 8, 16
    topk = 0.25
    dtype = torch.float16

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=1,
    )

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=ctx)
    opt.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )

    opt.run(q, kv_cache, topk, sink_size=sink, local_size=local)

    # Mirror _topk_fraction_to_k_eff: round, floor at 1, cap at mid_len.
    mid_len = ctx - sink - local
    k_eff_mid = (
        max(1, min(int(round(topk * mid_len)), mid_len)) if mid_len > 0 else 0
    )
    expected_selected = sink + k_eff_mid + local  # < ctx → no LMAX cap

    stats = opt.get_sparsity_stats()
    assert stats.last_total == ctx
    assert stats.last_selected == expected_selected
    assert stats.last_fraction == pytest.approx(expected_selected / ctx)
    assert stats.call_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparsity_stats_cumulative_increments_over_runs():
    """cumulative_* and call_count accumulate across multiple run() calls."""
    device = torch.device("cuda")
    B, ctx, page, H_kv, H_q, D = 2, 128, 16, 4, 8, 64
    sink, local = 4, 4
    topk = 0.5
    dtype = torch.float16

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
        batch_size=B, context_len=ctx, page_size=page,
        num_kv_heads=H_kv, num_q_heads=H_q, head_dim=D,
        dtype=dtype, kv_layout="NHD", device=device, seed=2,
    )

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=ctx)
    opt.plan(
        kv_indptr, kv_indices, kv_last_page_len,
        H_q, H_kv, D, page,
        q_data_type=dtype, kv_data_type=dtype,
    )

    n_calls = 3
    for _ in range(n_calls):
        opt.run(q, kv_cache, topk, sink_size=sink, local_size=local)

    mid_len = ctx - sink - local
    k_eff_mid = max(1, min(int(round(topk * mid_len)), mid_len))
    expected_selected_per_call = sink + k_eff_mid + local

    stats = opt.get_sparsity_stats()
    assert stats.call_count == n_calls
    assert stats.cumulative_total == n_calls * ctx
    assert stats.cumulative_selected == n_calls * expected_selected_per_call
    # last_* still reflects the most recent call (same as expected_per_call).
    assert stats.last_total == ctx
    assert stats.last_selected == expected_selected_per_call


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sparsity_stats_snapshot_is_isolated_copy():
    """get_sparsity_stats() returns a copy; mutation doesn't poison the wrapper."""
    device = torch.device("cuda")
    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    opt = OptSinkLocalWrapper(ws, "NHD", max_seq_len=256)

    snap_a = opt.get_sparsity_stats()
    snap_a.call_count = 9999
    snap_a.last_fraction = 0.42

    snap_b = opt.get_sparsity_stats()
    assert snap_b.call_count == 0  # not poisoned by snap_a mutation
    assert snap_b.last_fraction == 0.0
    assert snap_a is not snap_b
