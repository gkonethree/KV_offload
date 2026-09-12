#!/usr/bin/env python3
"""Focused sparse-vs-dense comparison at long context lengths."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseWrapper
from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as SparseWrapper


CTXS = [32768, 65536, 131072]
BATCHES = [16]
H_KV_LIST = [8, 16, 32]
H_Q = 32
HEAD_DIM = 128
PAGE = 16
DTYPE = torch.float16
SPARSITIES = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


def _bench(fn, warmup=10, iters=100):
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
    return (kv_indptr, kv_indices, kv_last_page_len, kv_cache, q,
            sparse_idx, sparse_weights)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}, dtype: float16, "
          f"H_q={H_Q}, D={HEAD_DIM}, page={PAGE}, batches={BATCHES}")
    print()

    for ctx in CTXS:
        for H_kv in H_KV_LIST:
            print(f"=== context_len = {ctx}  |  H_q/H_kv = {H_Q}/{H_kv}  "
                  f"(group_size={H_Q // H_kv}) ===")
            header = (
                f"{'B':>3s}  {'sparsity':>9s}  {'avg_S':>7s}  "
                f"{'dense_ms':>9s}  {'sparse_ms':>10s}  {'speedup':>8s}"
            )
            print(header)
            print("-" * len(header))
            for B in BATCHES:
                max_S = ctx
                (kv_indptr, kv_indices, kv_last_page_len, kv_cache, q,
                 sparse_idx, sparse_weights) = _build_inputs(
                    B, ctx, PAGE, H_kv, H_Q, HEAD_DIM, max_S, DTYPE, device
                )
                ws_d = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
                dense = DenseWrapper(ws_d, "NHD")
                dense.plan(kv_indptr, kv_indices, kv_last_page_len, H_Q, H_kv, HEAD_DIM, PAGE,
                           q_data_type=DTYPE, kv_data_type=DTYPE)
                dense_ms = _bench(lambda: dense.run(q, kv_cache))

                ws_s = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
                sparse = SparseWrapper(ws_s, "NHD")
                sparse.plan(kv_indptr, kv_indices, kv_last_page_len, H_Q, H_kv, HEAD_DIM, PAGE,
                            q_data_type=DTYPE, kv_data_type=DTYPE)

                for r in SPARSITIES:
                    avg_S = max(1, int(round(r * ctx)))
                    if avg_S > max_S:
                        continue
                    sparse_len = torch.full(
                        (B, H_Q, 1), avg_S, dtype=torch.int32, device=device
                    )
                    sparse_ms = _bench(
                        lambda: sparse.run(
                            q, kv_cache, sparse_len, sparse_idx, sparse_weights
                        )
                    )
                    speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")
                    print(
                        f"{B:>3d}  {r:>9.2%}  {avg_S:>7d}  "
                        f"{dense_ms:>9.4f}  {sparse_ms:>10.4f}  {speedup:>7.2f}x"
                    )
                del kv_cache, q, sparse_idx, sparse_weights
                torch.cuda.empty_cache()
            print()


if __name__ == "__main__":
    main()
