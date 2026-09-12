#!/usr/bin/env python3
"""
Dense baseline sweep mirroring the channel_num=8 sparsity sweep:
  - context_len  = 128K
  - num_q_heads  = 32
  - num_kv_heads ∈ {8, 32}
  - batch_size   ∈ {1, 4, 8, 16}
  - head_dim     = 128, fp16, page_size 16, NHD layout

Reuses ``bench_one`` from ``profile_optimized_decode.py`` (which itself
uses the helpers from ``profile_flashinfer_decode.py`` style construction)
to time both:
  - flashinfer.BatchDecodeWithPagedKVCacheWrapper
  - original_optimized.BatchDecodeWithPagedKVCacheWrapper

so the numbers are directly comparable to the sparse sweep.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_optimized_decode import bench_one


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")

    CTX = 131072
    PAGE = 16
    H_q = 32
    D = 128
    DTYPE = torch.float16
    LAYOUT = "NHD"
    BATCH_SIZES = [1, 4, 8, 16]
    NUM_KV_HEADS = [8, 32]
    WARMUP = 50
    ITERS = 200

    print(f"device       : {torch.cuda.get_device_name(device)}")
    print(f"dtype        : float16")
    print(f"context_len  : {CTX}   page_size: {PAGE}   head_dim: {D}")
    print(f"num_q_heads  : {H_q}")
    print(f"warmup/iters : {WARMUP}/{ITERS}\n")

    rows = []
    for H_kv in NUM_KV_HEADS:
        for B in BATCH_SIZES:
            t0 = time.time()
            f_ms, o_ms = bench_one(
                B, CTX, PAGE, H_kv, H_q, D, DTYPE, LAYOUT,
                warmup=WARMUP, iters=ITERS,
                device=device, ws_size_mb=256,
            )
            elapsed = time.time() - t0
            print(f"-- B={B:>2d}  H_kv={H_kv:>2d} (group_size={H_q // H_kv})  "
                  f"flash={f_ms:.4f}ms  opt={o_ms:.4f}ms  "
                  f"opt/flash={o_ms/f_ms:.3f}  ({elapsed:.1f}s)",
                  flush=True)
            rows.append((B, H_kv, f_ms, o_ms))
            torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print(f"Dense decode @ ctx=128K, H_q=32, D=128, fp16")
    print("=" * 78)
    header = (
        f"{'B':>3s}  {'H_kv':>4s}  {'group':>5s}  "
        f"{'flashinfer_ms':>14s}  {'orig_opt_ms':>12s}  {'opt/flash':>10s}"
    )
    print(header)
    print("-" * len(header))
    for (B, H_kv, f_ms, o_ms) in rows:
        print(f"{B:>3d}  {H_kv:>4d}  {H_q // H_kv:>5d}  "
              f"{f_ms:>14.4f}  {o_ms:>12.4f}  {o_ms / f_ms:>9.3f}x")

    # Per-H_kv side-by-side matrix, easy to scan.
    for H_kv in NUM_KV_HEADS:
        print()
        print(f"--- decode latency (ms), H_kv={H_kv}, group_size={H_q // H_kv} ---")
        print(f"{'B':>3s}  {'flashinfer_ms':>14s}  {'orig_opt_ms':>12s}")
        for B in BATCH_SIZES:
            row = next(x for x in rows if x[0] == B and x[1] == H_kv)
            print(f"{B:>3d}  {row[2]:>14.4f}  {row[3]:>12.4f}")


if __name__ == "__main__":
    main()
