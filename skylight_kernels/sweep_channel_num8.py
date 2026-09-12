#!/usr/bin/env python3
"""
Comparative sweep: sparse_oracle_topk_optimized vs FlashInfer dense decode at
channel_num = 8.

Fixed:
  - context_len     = 128K
  - head_dim        = 128
  - num_q_heads     = 32
  - channel_num     = 8        (selection over only the first 8 channels of Q/K)
  - dtype           = float16
  - page_size       = 16

Swept:
  - num_kv_heads ∈ {8, 32}     (group_size 4 = GQA; group_size 1 = MHA)
  - batch_size   ∈ {1, 4, 8, 16}
  - sparsity     ∈ {50%, 20%, 10%, 5%, 2%, 1%}

Reports per-call latency and speedup for every (B, H_kv, sparsity) cell.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse_oracle_topk_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OracleTopKOptWrapper,
)


def _flashinfer_cls():
    import flashinfer  # type: ignore
    cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if cls is None:
        cls = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
    return cls


def _bench(fn, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _make_inputs(*, B, ctx, page, H_kv, H_q, D, dtype, device, seed=0):
    torch.manual_seed(seed)
    pages_per_req = math.ceil(ctx / page)
    total_pages = B * pages_per_req
    last_pl = ctx % page or page
    indptr = torch.arange(0, total_pages + 1, pages_per_req,
                          dtype=torch.int32, device=device)
    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    last = torch.full((B,), last_pl, dtype=torch.int32, device=device)
    kv = torch.randn(total_pages, 2, page, H_kv, D, dtype=dtype, device=device)
    q = torch.randn(B, H_q, D, dtype=dtype, device=device)
    return indptr, indices, last, kv, q


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    dtype = torch.float16

    CTX = 131072
    PAGE = 16
    H_q = 32
    D = 128
    CHANNEL_NUM = 8
    SPARSITIES = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]
    BATCH_SIZES = [1, 4, 8, 16]
    NUM_KV_HEADS = [8, 32]

    fi_cls = _flashinfer_cls()

    print(f"device       : {torch.cuda.get_device_name(device)}")
    print(f"dtype        : float16")
    print(f"context_len  : {CTX}   page_size: {PAGE}   head_dim: {D}")
    print(f"num_q_heads  : {H_q}")
    print(f"channel_num  : {CHANNEL_NUM} (of {D})")
    print()

    rows = []  # (B, H_kv, sparsity, k_eff, dense_ms, sparse_ms, speedup)

    for H_kv in NUM_KV_HEADS:
        if H_q % H_kv != 0:
            continue
        group_size = H_q // H_kv
        for B in BATCH_SIZES:
            print(f"-- B={B:>2d}  H_kv={H_kv:>2d} (group_size={group_size}) --",
                  flush=True)
            indptr, indices, last, kv_cache, q = _make_inputs(
                B=B, ctx=CTX, page=PAGE, H_kv=H_kv, H_q=H_q, D=D,
                dtype=dtype, device=device, seed=0,
            )

            ws_d = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            dense = fi_cls(ws_d, "NHD")
            dense.plan(
                indptr, indices, last, H_q, H_kv, D, PAGE,
                pos_encoding_mode="NONE",
                q_data_type=dtype, kv_data_type=dtype,
            )
            dense_ms = _bench(lambda: dense.run(q, kv_cache))

            # Construct with the engine-wide max_seq_len so run() goes through
            # the sync-free fast path (skips the per-call .item() on L_max).
            ws_s = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            oracle = OracleTopKOptWrapper(ws_s, "NHD", max_seq_len=CTX)
            oracle.plan(
                indptr, indices, last, H_q, H_kv, D, PAGE,
                q_data_type=dtype, kv_data_type=dtype,
            )

            for r in SPARSITIES:
                k_eff = max(1, min(CTX, int(round(float(r) * CTX))))
                sparse_ms = _bench(
                    lambda top=r: oracle.run(
                        q, kv_cache, top, channel_num=CHANNEL_NUM
                    )
                )
                speedup = dense_ms / sparse_ms
                rows.append((B, H_kv, r, k_eff, dense_ms, sparse_ms, speedup))

            del kv_cache, q, dense, oracle, ws_d, ws_s
            torch.cuda.empty_cache()

    # ---- Pretty print --------------------------------------------------------
    print()
    print("=" * 96)
    print(f"Results: channel_num={CHANNEL_NUM}  ctx={CTX}  H_q={H_q}  D={D}  "
          f"dtype=float16")
    print("=" * 96)
    header = (
        f"{'B':>3s}  {'H_kv':>4s}  {'group':>5s}  {'sparsity':>9s}  "
        f"{'k_eff':>7s}  {'dense_ms':>10s}  {'sparse_ms':>10s}  {'speedup':>8s}"
    )
    print(header)
    print("-" * len(header))
    for (B, H_kv, r, k_eff, dense_ms, sparse_ms, speedup) in rows:
        print(
            f"{B:>3d}  {H_kv:>4d}  {H_q // H_kv:>5d}  {r:>9.2%}  "
            f"{k_eff:>7d}  {dense_ms:>10.4f}  {sparse_ms:>10.4f}  "
            f"{speedup:>7.2f}x"
        )

    # ---- Speedup-only matrix per H_kv ---------------------------------------
    for H_kv in NUM_KV_HEADS:
        print()
        print(f"--- speedup matrix (H_kv={H_kv}, group_size={H_q // H_kv}) ---")
        col = "  ".join(f"{r:>6.0%}" for r in SPARSITIES)
        print(f"{'B \\ sparsity':>12s}  {col}")
        for B in BATCH_SIZES:
            cells = []
            for r in SPARSITIES:
                row = next(
                    (x for x in rows
                     if x[0] == B and x[1] == H_kv and x[2] == r),
                    None,
                )
                cells.append(f"{row[6]:>5.2f}x" if row else f"{'--':>6s}")
            print(f"{B:>12d}  {'  '.join(cells)}")


if __name__ == "__main__":
    main()
