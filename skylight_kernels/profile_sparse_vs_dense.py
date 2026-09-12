#!/usr/bin/env python3
"""
Compare latency: `sparse_optimized` (sparse decode, varying sparsity) vs
`original_optimized` (full dense decode) on the same shapes.

Sparsity is parametrised by a sparsity *ratio* r in (0, 1] -- the average
sparse_len = r * context_len. We run both impls on the same kv_cache/q so
the comparison is apples-to-apples.

Outputs a table: (shape, sparsity, dense_ms, sparse_ms, speedup).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseWrapper
from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as SparseWrapper


SHAPES = [
    # (B, ctx, H_kv, H_q, D, page)
    (1,  4096,  8,  32, 128, 16),
    (4,  4096,  8,  32, 128, 16),
    (8,  4096,  8,  32, 128, 16),
    (1,  16384, 8,  32, 128, 16),
    (4,  16384, 8,  32, 128, 16),
    (8,  16384, 8,  32, 128, 16),
    (1,  4096,  8,  8,  128, 16),  # MHA (group_size=1)
    (4,  4096,  8,  8,  128, 16),
]


SPARSITIES = [0.5, 0.25, 0.10, 0.05, 0.02, 0.01]


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _build_inputs(B, ctx, page, H_kv, H_q, D, max_S, dtype, device, seed=0):
    torch.manual_seed(seed)
    pages_per_req = math.ceil(ctx / page)
    total_pages = B * pages_per_req
    last_page_len_value = ctx % page or page
    kv_indptr = torch.arange(
        0, total_pages + 1, pages_per_req, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (B,), last_page_len_value, dtype=torch.int32, device=device
    )
    kv_cache = torch.randn(
        total_pages, 2, page, H_kv, D, dtype=dtype, device=device
    )
    q = torch.randn(B, H_q, D, dtype=dtype, device=device)
    sparse_idx = torch.randint(
        0, ctx, (B, H_q, max_S), dtype=torch.int64, device=device
    )
    sparse_weights = torch.rand(
        B, H_q, max_S, dtype=torch.float32, device=device
    ) + 1e-3
    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q, sparse_idx, sparse_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)

    print(f"device: {torch.cuda.get_device_name(device)}, dtype: {args.dtype}")
    print()
    header = (
        f"{'shape':>34s}  {'sparsity':>9s}  {'avg_S':>6s}  "
        f"{'dense_ms':>9s}  {'sparse_ms':>10s}  {'speedup':>8s}"
    )
    print(header)
    print("-" * len(header))

    for B, ctx, H_kv, H_q, D, page in SHAPES:
        # Build dense inputs once (max sparsity context = ctx so we can vary).
        max_S = ctx
        (kv_indptr, kv_indices, kv_last_page_len, kv_cache, q,
         sparse_idx_full, sparse_weights_full) = _build_inputs(
            B, ctx, page, H_kv, H_q, D, max_S, dtype, device, seed=0
        )

        # Plan dense and time once
        ws_d = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        dense = DenseWrapper(ws_d, "NHD")
        dense.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
                   q_data_type=dtype, kv_data_type=dtype)
        dense_ms = _bench(lambda: dense.run(q, kv_cache), args.warmup, args.iters)

        # Plan sparse once (independent of sparsity ratio)
        ws_s = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        sparse = SparseWrapper(ws_s, "NHD")
        sparse.plan(kv_indptr, kv_indices, kv_last_page_len, H_q, H_kv, D, page,
                    q_data_type=dtype, kv_data_type=dtype)

        for r in SPARSITIES:
            avg_S = max(1, int(round(r * ctx)))
            sparse_len = torch.full(
                (B, H_q, 1), avg_S, dtype=torch.int32, device=device
            )
            # We can re-use the same sparse_idx and sparse_weights tensors;
            # only the first avg_S entries per row are read.
            sparse_ms = _bench(
                lambda: sparse.run(
                    q, kv_cache, sparse_len, sparse_idx_full, sparse_weights_full
                ),
                args.warmup, args.iters,
            )
            speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")
            print(
                f"B{B:<2d}-ctx{ctx:>5d}-H{H_kv}/{H_q}-D{D:<4d}  "
                f"{r:>9.2%}  {avg_S:>6d}  "
                f"{dense_ms:>9.4f}  {sparse_ms:>10.4f}  {speedup:>7.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
