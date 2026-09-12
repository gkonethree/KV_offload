#!/usr/bin/env python3
"""
Speedup profile: sparse_oracle_topk_sink_local_optimized vs FlashInfer dense decode.

Fixed: B=16, context_len=128K, channel_num=8.
Sweep: sparsities {50%, 20%, 10%, 5%, 2%, 1%} × sink/local configs.

Reports per-call latency (ms) and speedup vs FlashInfer dense decode.

Usage:
  python profile_sparse_oracle_topk_sink_local_optimized.py
  python profile_sparse_oracle_topk_sink_local_optimized.py \\
      --sparsities 0.1 0.05 0.01 \\
      --sink-size 64 --local-size 1024 \\
      --channel-num 8
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse_oracle_topk_sink_local_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as SinkLocalOptWrapper,
)
from sparse_oracle_topk_optimized.cuda_ops import get_oracle_topk_ops
from sparse_optimized.cuda_ops import get_sparse_decode_ops


DEFAULT_SPARSITIES = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]

# Sink+local configs to sweep: (sink_size, local_size, label)
DEFAULT_SINK_LOCAL_CONFIGS = [
    (0,    0,    "no-sink-local"),
    (64,   1024, "sink64+local1024"),
    (64,   4096, "sink64+local4096"),
]


def _flashinfer_wrapper_cls():
    import flashinfer  # type: ignore
    cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if cls is None:
        cls = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
    return cls


def _bench(fn, warmup: int = 10, iters: int = 50) -> float:
    """Average milliseconds per call via CUDA events."""
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


def _make_inputs(
    *,
    B: int, ctx: int, page: int, H_kv: int, H_q: int, D: int,
    dtype: torch.dtype, device: torch.device, seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        total_pages, 2, page, H_kv, D, dtype=dtype, device=device,
    )
    q = torch.randn(B, H_q, D, dtype=dtype, device=device)
    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q


def _kv_strides(H_kv, page, D, layout="NHD"):
    if layout == "NHD":
        kv_stride_page = 2 * page * H_kv * D
        kv_stride_n = H_kv * D
        kv_stride_h = D
        kv_v_offset = page * H_kv * D
    else:
        kv_stride_page = 2 * H_kv * page * D
        kv_stride_n = D
        kv_stride_h = page * D
        kv_v_offset = H_kv * page * D
    return kv_stride_page, kv_stride_n, kv_stride_h, kv_v_offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=131072)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument(
        "--num-kv-heads-list", type=int, nargs="+", default=[8],
        help="KV heads; produces group_size = H_q / H_kv.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--sparsities", type=float, nargs="+", default=DEFAULT_SPARSITIES,
        help="topk fractions (k = round(topk * mid_len)).",
    )
    parser.add_argument("--channel-num", type=int, default=8,
                        help="Selection channel count (must be in {8,16,32,64,128,256}).")
    parser.add_argument("--sink-size", type=int, default=None,
                        help="Override sink_size; if omitted sweeps DEFAULT_SINK_LOCAL_CONFIGS.")
    parser.add_argument("--local-size", type=int, default=None,
                        help="Override local_size; if omitted sweeps DEFAULT_SINK_LOCAL_CONFIGS.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Override sink/local sweep if both are specified
    if args.sink_size is not None and args.local_size is not None:
        sink_local_configs = [
            (args.sink_size, args.local_size,
             f"sink{args.sink_size}+local{args.local_size}")
        ]
    else:
        sink_local_configs = DEFAULT_SINK_LOCAL_CONFIGS

    fi_cls = _flashinfer_wrapper_cls()

    print(f"device       : {torch.cuda.get_device_name(device)}")
    print(f"dtype        : {args.dtype}")
    print(f"batch_size   : {args.batch_size}")
    print(f"context_len  : {args.context_len}")
    print(f"page_size    : {args.page_size}")
    print(f"H_q          : {args.num_q_heads}")
    print(f"H_kv list    : {args.num_kv_heads_list}")
    print(f"head_dim     : {args.head_dim}")
    print(f"channel_num  : {args.channel_num}")
    print(f"warmup/iters : {args.warmup}/{args.iters}")
    print()

    B = args.batch_size
    ctx = args.context_len
    page = args.page_size
    D = args.head_dim
    H_q = args.num_q_heads
    channel_num = args.channel_num

    for H_kv in args.num_kv_heads_list:
        if H_q % H_kv != 0:
            print(f"skip H_kv={H_kv} (H_q={H_q} not divisible)")
            continue
        group_size = H_q // H_kv
        print(
            f"=== H_q/H_kv = {H_q}/{H_kv}  (group_size={group_size}) "
            f"context_len={ctx}  B={B}  channel_num={channel_num} ==="
        )

        kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = _make_inputs(
            B=B, ctx=ctx, page=page, H_kv=H_kv, H_q=H_q, D=D,
            dtype=dtype, device=device, seed=args.seed,
        )

        # ---- Dense FlashInfer reference ------------------------------------
        ws_d = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        dense = fi_cls(ws_d, "NHD")
        dense.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page,
            pos_encoding_mode="NONE",
            q_data_type=dtype, kv_data_type=dtype,
        )
        dense_ms = _bench(lambda: dense.run(q, kv_cache),
                          warmup=args.warmup, iters=args.iters)
        print(f"  FlashInfer dense: {dense_ms:.4f} ms")
        print()

        oracle_ops = get_oracle_topk_ops()
        sparse_ops = get_sparse_decode_ops()
        kv_stride_page, kv_stride_n, kv_stride_h, kv_v_offset = _kv_strides(
            H_kv, page, D, "NHD"
        )

        # Score-kernel cost (shared across all sink/local configs since it
        # covers the full sequence before masking).
        def _bench_score(cn=channel_num):
            return oracle_ops.oracle_topk_compute_scores(
                q, kv_cache, kv_stride_page, kv_stride_n, kv_stride_h,
                kv_indptr, kv_indices, kv_last_page_len, H_kv, page, ctx, cn,
            )
        score_ms = _bench(_bench_score, warmup=args.warmup, iters=args.iters)

        for sink_sz, local_sz, label in sink_local_configs:
            print(f"  --- {label}  (sink={sink_sz}, local={local_sz}) ---")

            ws_s = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
            sparse_wrapper = SinkLocalOptWrapper(
                ws_s, "NHD",
                max_seq_len=ctx,
                sink_size=sink_sz,
                local_size=local_sz,
            )
            sparse_wrapper.plan(
                kv_indptr, kv_indices, kv_last_page_len,
                H_q, H_kv, D, page,
                q_data_type=dtype, kv_data_type=dtype,
            )

            header = (
                f"  {'topk':>9s}  {'k_eff_mid':>9s}  {'total_k':>7s}  "
                f"{'dense_ms':>9s}  {'sparse_ms':>10s}  {'speedup':>8s}  | "
                f"{'score_ms':>9s}  {'topk_ms':>8s}  {'decode_ms':>10s}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))

            for r in args.sparsities:
                n_keys_mid = max(0, ctx - sink_sz - local_sz)
                k_eff_mid = max(1, min(n_keys_mid, int(round(float(r) * n_keys_mid))))
                total_k = sink_sz + k_eff_mid + local_sz

                try:
                    sparse_ms = _bench(
                        lambda top=r, cn=channel_num: sparse_wrapper.run(
                            q, kv_cache, top, channel_num=cn
                        ),
                        warmup=args.warmup, iters=args.iters,
                    )

                    # Stage breakdown: score → topk → decode
                    scores_snap = _bench_score()

                    # Mask sink/local positions (matches what the wrapper does)
                    if sink_sz > 0:
                        scores_snap[:, :, :sink_sz] = float("-inf")
                    if local_sz > 0:
                        pages_count = kv_indptr[1:] - kv_indptr[:-1]
                        L_per = (
                            (pages_count - 1).clamp(min=0).to(torch.int64) * page
                            + kv_last_page_len.to(torch.int64)
                        )
                        pos = torch.arange(ctx, dtype=torch.int64, device=device).view(1, 1, ctx)
                        L_per_view = L_per.view(B, 1, 1)
                        local_start = (L_per_view - local_sz).clamp(min=sink_sz)
                        local_mask = (pos >= local_start) & (pos < L_per_view)
                        scores_snap.masked_fill_(local_mask, float("-inf"))

                    topk_ms = _bench(
                        lambda s=scores_snap, k=k_eff_mid: torch.topk(
                            s, k=k, dim=-1, sorted=False
                        ),
                        warmup=args.warmup, iters=args.iters,
                    )
                    topk_idx_snap = torch.topk(
                        scores_snap, k=k_eff_mid, dim=-1, sorted=False
                    ).indices

                    # Build combined index for decode stage timing
                    parts = []
                    if sink_sz > 0:
                        sink_idx = (
                            torch.arange(sink_sz, dtype=torch.int64, device=device)
                            .view(1, 1, -1).expand(B, H_q, -1)
                        )
                        parts.append(sink_idx)
                    parts.append(topk_idx_snap)
                    if local_sz > 0:
                        L_per_long = L_per.long()
                        local_start_idx = (L_per_long - local_sz).clamp(min=sink_sz)
                        local_offsets = torch.arange(local_sz, dtype=torch.int64, device=device)
                        local_idx = (
                            local_start_idx.view(B, 1, 1) + local_offsets.view(1, 1, -1)
                        ).expand(B, H_q, -1)
                        parts.append(local_idx)
                    combined_idx = torch.cat(parts, dim=-1).contiguous()

                    sparse_len_t = torch.full(
                        (B, H_q, 1), total_k, dtype=torch.int32, device=device
                    )
                    sparse_w = torch.ones(
                        (B, H_q, total_k), dtype=torch.float32, device=device
                    )
                    k_split = int(sparse_ops.pick_k_split(B, H_q, total_k, 4))

                    def _bench_decode():
                        return sparse_ops.sparse_decode_run(
                            q, kv_cache, kv_cache,
                            kv_stride_page, kv_stride_n, kv_stride_h, kv_v_offset,
                            kv_indptr, kv_indices, sparse_len_t, combined_idx, sparse_w,
                            H_kv, page,
                            1.0 / math.sqrt(D),
                            0.0, int(k_split), False,
                        )
                    decode_ms = _bench(_bench_decode, warmup=args.warmup, iters=args.iters)

                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  {r:>9.4f}  {k_eff_mid:>9d}  {total_k:>7d}  "
                        f"  ERROR: {exc}"
                    )
                    continue

                speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")
                print(
                    f"  {r:>9.4f}  {k_eff_mid:>9d}  {total_k:>7d}  "
                    f"{dense_ms:>9.4f}  {sparse_ms:>10.4f}  {speedup:>7.2f}x  | "
                    f"{score_ms:>9.4f}  {topk_ms:>8.4f}  {decode_ms:>10.4f}"
                )
            print()
            del sparse_wrapper, ws_s

        del kv_cache, q, dense, ws_d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
