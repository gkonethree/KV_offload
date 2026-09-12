#!/usr/bin/env python3
"""
Compare latency of original_optimized.BatchDecodeWithPagedKVCacheWrapper vs
the FlashInfer reference, on a sweep of common decode shapes.

Examples:
  python profile_optimized_decode.py
  python profile_optimized_decode.py --iters 500 --warmup 50
"""

from __future__ import annotations

import argparse
import math
import time
from typing import List, Tuple

import torch

from original_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OptimizedWrapper,
)


def _flashinfer_cls():
    import flashinfer  # type: ignore
    cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if cls is not None:
        return cls
    return flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper


def make_inputs(B, ctx, page, H_kv, H_q, D, dtype, layout, device):
    pages_per_req = math.ceil(ctx / page)
    total_pages = B * pages_per_req
    last_page_len_value = ctx % page or page

    kv_indptr = torch.arange(0, total_pages + 1, pages_per_req,
                             dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((B,), last_page_len_value,
                                  dtype=torch.int32, device=device)
    if layout == "NHD":
        kv_cache = torch.randn(total_pages, 2, page, H_kv, D,
                               dtype=dtype, device=device)
    else:
        kv_cache = torch.randn(total_pages, 2, H_kv, page, D,
                               dtype=dtype, device=device)
    q = torch.randn(B, H_q, D, dtype=dtype, device=device)
    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q


def time_run(wrapper, q, kv_cache, warmup, iters):
    for _ in range(warmup):
        _ = wrapper.run(q, kv_cache)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = None
    for _ in range(iters):
        out = wrapper.run(q, kv_cache)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, out


def bench_one(B, ctx, page, H_kv, H_q, D, dtype, layout, *,
              warmup, iters, device, ws_size_mb):
    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = make_inputs(
        B, ctx, page, H_kv, H_q, D, dtype, layout, device,
    )

    Flash = _flashinfer_cls()
    ws_f = torch.zeros(ws_size_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_o = torch.zeros(ws_size_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    flash = Flash(ws_f, layout)
    opt = OptimizedWrapper(ws_o, layout)
    for w in (flash, opt):
        w.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page,
            pos_encoding_mode="NONE",
            q_data_type=dtype, kv_data_type=dtype,
        )

    fout, _ = time_run(flash, q, kv_cache, warmup, iters)
    oout, _ = time_run(opt, q, kv_cache, warmup, iters)

    return fout, oout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--workspace-mb", type=int, default=128)
    p.add_argument("--config", default="all",
                   help="Either 'all' or comma-separated index list (0-based)")
    return p.parse_args()


SHAPES: List[Tuple] = [
    # (B, ctx, page, H_kv, H_q, D, dtype, layout)
    (1,   1024, 16, 8, 32, 128, torch.float16, "NHD"),
    (4,   1024, 16, 8, 32, 128, torch.float16, "NHD"),
    (8,   1024, 16, 8, 32, 128, torch.float16, "NHD"),
    (16,  1024, 16, 8, 32, 128, torch.float16, "NHD"),
    (32,  1024, 16, 8, 32, 128, torch.float16, "NHD"),
    (8,   2048, 16, 8, 32, 128, torch.float16, "NHD"),
    (8,   4096, 16, 8, 32, 128, torch.float16, "NHD"),
    (8,   8192, 16, 8, 32, 128, torch.float16, "NHD"),
    # large contexts (32K and 128K)
    (1,  32768, 16, 8, 32, 128, torch.float16, "NHD"),
    (4,  32768, 16, 8, 32, 128, torch.float16, "NHD"),
    (1, 131072, 16, 8, 32, 128, torch.float16, "NHD"),
    # group_size=1 (MHA), group_size=4, group_size=8
    (4,   2048, 16, 8, 64, 128, torch.float16, "NHD"),
    (4,   2048, 16, 8, 8,  128, torch.float16, "NHD"),
    (1,   8192, 16, 8, 32, 128, torch.float16, "NHD"),
    (1,  16384, 16, 8, 32, 128, torch.float16, "NHD"),
    # head dim variations
    (4,   2048, 16, 8, 32, 64,  torch.float16, "NHD"),
    (4,   2048, 16, 8, 32, 128, torch.bfloat16, "NHD"),
    (4,   2048, 16, 8, 32, 128, torch.float16, "HND"),
    # MHA H_q=H_kv (group_size=1)
    (1,   1024, 16, 32, 32, 128, torch.float16, "NHD"),
    (1,  32768, 16, 32, 32, 128, torch.float16, "NHD"),
]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    if args.config == "all":
        idxs = list(range(len(SHAPES)))
    else:
        idxs = [int(x) for x in args.config.split(",")]

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Iters/measurement: warmup={args.warmup}, iters={args.iters}\n")
    header = f"{'B':>3}  {'ctx':>5}  {'pg':>3}  {'Hkv':>3}  {'Hq':>3}  {'D':>3}  {'dtype':>8}  {'layout':>6}  {'flash_ms':>10}  {'opt_ms':>10}  {'opt/flash':>9}"
    print(header)
    print("-" * len(header))
    for i in idxs:
        B, ctx, page, H_kv, H_q, D, dtype, layout = SHAPES[i]
        f_ms, o_ms = bench_one(
            B, ctx, page, H_kv, H_q, D, dtype, layout,
            warmup=args.warmup, iters=args.iters,
            device=device, ws_size_mb=args.workspace_mb,
        )
        ratio = o_ms / f_ms
        dtype_s = str(dtype).split(".")[-1]
        print(f"{B:>3}  {ctx:>5}  {page:>3}  {H_kv:>3}  {H_q:>3}  {D:>3}  {dtype_s:>8}  {layout:>6}  "
              f"{f_ms:>10.4f}  {o_ms:>10.4f}  {ratio:>9.3f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nscript_elapsed_s: {time.time() - t0:.1f}")
