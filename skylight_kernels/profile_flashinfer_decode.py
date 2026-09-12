#!/usr/bin/env python3
"""
Profile flashinfer decode attention for synthetic paged KV-cache inputs.

Example:
  python profile_flashinfer_decode.py \
    --batch-size 8 \
    --context-len 2048 \
    --num-kv-heads 8 \
    --num-q-heads 64 \
    --head-dim 128
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run "
            "with synthetic paged KV-cache and query tensors."
        )
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--num-q-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--kv-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--use-tensor-cores", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_decode_wrapper_cls():
    import flashinfer  # type: ignore

    wrapper_cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if wrapper_cls is not None:
        return wrapper_cls
    return flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper


def make_synthetic_decode_inputs(
    batch_size: int,
    context_len: int,
    page_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    kv_layout: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pages_per_req = math.ceil(context_len / page_size)
    total_pages = batch_size * pages_per_req
    last_page_len_value = context_len % page_size
    if last_page_len_value == 0:
        last_page_len_value = page_size

    kv_indptr = torch.arange(
        0,
        total_pages + 1,
        pages_per_req,
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,),
        last_page_len_value,
        dtype=torch.int32,
        device=device,
    )

    if kv_layout == "NHD":
        kv_cache = torch.randn(
            total_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
    else:
        kv_cache = torch.randn(
            total_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=dtype,
            device=device,
        )

    q = torch.randn(
        batch_size,
        num_q_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flashinfer decode profiling.")

    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads.")
    if args.context_len < 1:
        raise ValueError("context_len must be >= 1.")
    if args.page_size < 1:
        raise ValueError("page_size must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    wrapper_cls = get_decode_wrapper_cls()

    workspace_buffer = torch.zeros(
        args.workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device
    )
    decode_wrapper = wrapper_cls(
        workspace_buffer,
        args.kv_layout,
        use_tensor_cores=args.use_tensor_cores,
    )

    kv_indptr, kv_indices, kv_last_page_len, kv_cache, q = make_synthetic_decode_inputs(
        batch_size=args.batch_size,
        context_len=args.context_len,
        page_size=args.page_size,
        num_kv_heads=args.num_kv_heads,
        num_q_heads=args.num_q_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        kv_layout=args.kv_layout,
        device=device,
    )

    decode_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    for _ in range(args.warmup_iters):
        _ = decode_wrapper.run(q, kv_cache)
    torch.cuda.synchronize()

    # Whole-loop timing captures launch + kernel + sync behavior.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = None
    for _ in range(args.iters):
        out = decode_wrapper.run(q, kv_cache)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)

    assert out is not None
    avg_ms = total_ms / args.iters
    tok_per_s = args.batch_size * (1000.0 / avg_ms)

    pages_per_req = math.ceil(args.context_len / args.page_size)
    total_pages = args.batch_size * pages_per_req
    kv_cache_bytes = kv_cache.numel() * kv_cache.element_size()

    print("=== FlashInfer Decode Profiling ===")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"dtype: {args.dtype}")
    print(f"batch_size: {args.batch_size}")
    print(f"context_len: {args.context_len}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"num_q_heads: {args.num_q_heads}")
    print(f"head_dim: {args.head_dim}")
    print(f"page_size: {args.page_size}")
    print(f"kv_layout: {args.kv_layout}")
    print(f"use_tensor_cores: {args.use_tensor_cores}")
    print(f"pages_per_request: {pages_per_req}")
    print(f"total_pages: {total_pages}")
    print(f"kv_cache_size_mb: {kv_cache_bytes / 1024**2:.2f}")
    print(f"output_shape: {tuple(out.shape)}")
    print(f"warmup_iters: {args.warmup_iters}")
    print(f"profile_iters: {args.iters}")
    print(f"avg_run_latency_ms: {avg_ms:.4f}")
    print(f"throughput_tokens_per_s: {tok_per_s:.2f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f"script_elapsed_s: {t1 - t0:.3f}")
