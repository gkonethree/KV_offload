#!/usr/bin/env python3
"""
Speedup profile: sparse_oracle_topk_optimized vs FlashInfer dense decode.

For B = 16 and context_len = 128K, sweep oracle top-k sparsity levels
{50%, 20%, 10%, 5%, 2%, 1%} and compare:
  - dense: flashinfer.BatchDecodeWithPagedKVCacheWrapper.run(q, kv)  (no sparsity)
  - sparse: sparse_oracle_topk_optimized.BatchDecodeWithPagedKVCacheWrapper.run(
              q, kv, topk=sparsity, channel_num=...)  # k = round(topk * ctx)

Optionally sweep over ``--channel-num-list`` to see how restricting the
selection dot product to the first ``channel_num`` channels of Q/K shrinks
the score-kernel time. ``channel_num`` must be one of {8, 16, 32, 64, 128, 256}
and ``<= head_dim``.

Reports per-call latency and speedup. Per request the user, additionally
sweeps a few common (H_q, H_kv) ratios.

Usage:
  python profile_sparse_oracle_topk_optimized.py
  python profile_sparse_oracle_topk_optimized.py --channel-num-list 16 32 64 128
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse_oracle_topk_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OracleTopKOptWrapper,
)
from sparse_oracle_topk_optimized.cuda_ops import get_oracle_topk_ops
from sparse_optimized.cuda_ops import get_sparse_decode_ops


# Defaults requested by the spec: B=16, ctx=128K, sparsity 50%..1%.
DEFAULT_SPARSITIES = [0.50, 0.20, 0.10, 0.05, 0.02, 0.01]


def _flashinfer_wrapper_cls():
    import flashinfer  # type: ignore
    cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if cls is None:
        cls = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
    return cls


def _bench(fn, warmup: int = 10, iters: int = 50) -> float:
    """Average milliseconds per call."""
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
        total_pages, 2, page, H_kv, D, dtype=dtype, device=device
    )
    q = torch.randn(B, H_q, D, dtype=dtype, device=device)
    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=131072)  # 128K
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument(
        "--num-kv-heads-list", type=int, nargs="+", default=[8],
        help="KV heads to sweep; produces group_size = H_q / H_kv.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--sparsities", type=float, nargs="+", default=DEFAULT_SPARSITIES,
        help="Sparsity levels as float ``topk`` (k = round(topk * context_len)).",
    )
    parser.add_argument(
        "--channel-num-list", type=int, nargs="+", default=[128],
        help="Selection-channel sizes to sweep (must be in {8,16,32,64,128,256} "
             "and <= head_dim). 128 = use full head_dim (default).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    fi_cls = _flashinfer_wrapper_cls()

    print(f"device       : {torch.cuda.get_device_name(device)}")
    print(f"dtype        : {args.dtype}")
    print(f"batch_size   : {args.batch_size}")
    print(f"context_len  : {args.context_len}")
    print(f"page_size    : {args.page_size}")
    print(f"H_q          : {args.num_q_heads}")
    print(f"H_kv list    : {args.num_kv_heads_list}")
    print(f"head_dim     : {args.head_dim}")
    print(f"warmup/iters : {args.warmup}/{args.iters}")
    print()

    B = args.batch_size
    ctx = args.context_len
    page = args.page_size
    D = args.head_dim
    H_q = args.num_q_heads

    for H_kv in args.num_kv_heads_list:
        if H_q % H_kv != 0:
            print(f"skip H_kv={H_kv} (H_q={H_q} not divisible)")
            continue
        group_size = H_q // H_kv
        print(f"=== H_q/H_kv = {H_q}/{H_kv}  (group_size={group_size}) "
              f"context_len={ctx}  B={B} ===")

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

        # ---- Optimized oracle top-k ----------------------------------------
        # Pass ``max_seq_len=ctx`` (vLLM's max_model_len analogue) so run() goes
        # through the sync-free fast path and the timings reflect what a real
        # serving engine would see.
        ws_s = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        oracle = OracleTopKOptWrapper(ws_s, "NHD", max_seq_len=ctx)
        oracle.plan(
            kv_indptr, kv_indices, kv_last_page_len,
            H_q, H_kv, D, page,
            q_data_type=dtype, kv_data_type=dtype,
        )

        # Stage-isolated benchmarks share the same kv_cache / planned wrappers
        # so we can attribute the overall sparse cost to its three stages.
        oracle_ops = get_oracle_topk_ops()
        sparse_ops = get_sparse_decode_ops()
        if H_q % H_kv != 0:
            continue
        if D not in (64, 128, 256):
            raise NotImplementedError(f"head_dim {D} not supported by score kernel")

        if oracle._kv_layout == "NHD":
            kv_stride_page = 2 * page * H_kv * D
            kv_stride_n = H_kv * D
            kv_stride_h = D
            kv_v_offset = page * H_kv * D
        else:
            kv_stride_page = 2 * H_kv * page * D
            kv_stride_n = D
            kv_stride_h = page * D
            kv_v_offset = H_kv * page * D

        for channel_num in args.channel_num_list:
            if channel_num > D:
                print(f"  skip channel_num={channel_num} (> head_dim={D})")
                continue
            print(f"  -- channel_num = {channel_num} "
                  f"(of head_dim {D})  dense_ms={dense_ms:.4f}")

            # The score kernel cost is constant w.r.t. k_eff; cache it per
            # channel_num.
            def _bench_score(cn=channel_num):
                return oracle_ops.oracle_topk_compute_scores(
                    q, kv_cache, kv_stride_page, kv_stride_n, kv_stride_h,
                    kv_indptr, kv_indices, kv_last_page_len, H_kv, page, ctx,
                    cn,
                )
            score_ms = _bench(_bench_score, warmup=args.warmup, iters=args.iters)

            header = (
                f"  {'topk':>9s}  {'k_eff':>7s}  {'flashinfer_ms':>14s}  "
                f"{'sparse_ms':>10s}  {'speedup':>8s}  | "
                f"{'score_ms':>9s}  {'topk_ms':>8s}  {'decode_ms':>10s}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for r in args.sparsities:
                k_eff = max(1, min(ctx, int(round(float(r) * ctx))))
                try:
                    sparse_ms = _bench(
                        lambda top=r, cn=channel_num: oracle.run(
                            q, kv_cache, top, channel_num=cn
                        ),
                        warmup=args.warmup, iters=args.iters,
                    )
                    # Stage timings (top-k & sparse decode separately).
                    scores_one = _bench_score()
                    topk_ms = _bench(
                        lambda: torch.topk(scores_one, k=k_eff, dim=-1, sorted=False),
                        warmup=args.warmup, iters=args.iters,
                    )
                    topk_idx = torch.topk(
                        scores_one, k=k_eff, dim=-1, sorted=False
                    ).indices.contiguous()
                    sparse_len = torch.full(
                        (B, H_q, 1), k_eff, dtype=torch.int32, device=device
                    )
                    sparse_w = torch.ones(
                        (B, H_q, k_eff), dtype=torch.float32, device=device
                    )
                    k_split = sparse_ops.pick_k_split(B, H_q, k_eff, 4)
                    def _bench_decode():
                        return sparse_ops.sparse_decode_run(
                            q, kv_cache, kv_cache,
                            kv_stride_page, kv_stride_n, kv_stride_h, kv_v_offset,
                            kv_indptr, kv_indices, sparse_len, topk_idx, sparse_w,
                            H_kv, page, 1.0 / math.sqrt(D), 0.0, int(k_split), False,
                        )
                    decode_ms = _bench(_bench_decode, warmup=args.warmup, iters=args.iters)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {r:>9.4f}  {k_eff:>7d}  {dense_ms:>14.4f}  "
                          f"  ERROR: {exc}")
                    continue
                speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")
                print(
                    f"  {r:>9.4f}  {k_eff:>7d}  {dense_ms:>14.4f}  "
                    f"{sparse_ms:>10.4f}  {speedup:>7.2f}x  | "
                    f"{score_ms:>9.4f}  {topk_ms:>8.4f}  {decode_ms:>10.4f}"
                )
            print()
        del kv_cache, q, dense, oracle, ws_d, ws_s
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
