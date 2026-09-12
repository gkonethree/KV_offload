#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

import torch


def _load_wrapper_cls(folder: str):
    folder_path = Path(folder).resolve()
    module_path = folder_path / "batch_decode_with_paged_kv_cache_wrapper.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing wrapper module: {module_path}")

    spec = importlib.util.spec_from_file_location(f"wrapper_{folder_path.name}", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatchDecodeWithPagedKVCacheWrapper


def _build_inputs(
    *,
    batch_size: int,
    context_len: int,
    page_size: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
    max_sparse_context: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    torch.manual_seed(seed)
    pages_per_req = math.ceil(context_len / page_size)
    total_pages = batch_size * pages_per_req
    last_page_len_value = context_len % page_size or page_size

    kv_indptr = torch.arange(0, total_pages + 1, pages_per_req, dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((batch_size,), last_page_len_value, dtype=torch.int32, device=device)

    kv_cache = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    q = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)

    max_sparse_context = min(max_sparse_context, context_len)
    sparse_len = torch.randint(
        1, max_sparse_context + 1, (batch_size, num_q_heads, 1), dtype=torch.int32, device=device
    )
    sparse_idx = torch.randint(
        0, context_len, (batch_size, num_q_heads, max_sparse_context), dtype=torch.int64, device=device
    )
    sparse_weights = torch.rand(
        batch_size, num_q_heads, max_sparse_context, dtype=torch.float32, device=device
    )
    sparse_weights = sparse_weights + 1e-3

    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q, sparse_len, sparse_idx, sparse_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile sparse BatchDecodeWithPagedKVCacheWrapper.")
    parser.add_argument("--impl-folder", required=True, help="Folder with sparse implementation.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-q-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-sparse-context", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")
    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads.")

    WrapperCls = _load_wrapper_cls(args.impl_folder)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    (
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        kv_cache,
        q,
        sparse_len,
        sparse_idx,
        sparse_weights,
    ) = _build_inputs(
        batch_size=args.batch_size,
        context_len=args.context_len,
        page_size=args.page_size,
        num_kv_heads=args.num_kv_heads,
        num_q_heads=args.num_q_heads,
        head_dim=args.head_dim,
        max_sparse_context=args.max_sparse_context,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    workspace = torch.zeros(args.workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = WrapperCls(workspace, "NHD")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_dim,
        args.page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        pos_encoding_mode="NONE",
    )

    for _ in range(args.warmup_iters):
        _ = wrapper.run(q, kv_cache, sparse_len, sparse_idx, sparse_weights)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        _ = wrapper.run(q, kv_cache, sparse_len, sparse_idx, sparse_weights)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / args.iters
    tok_per_s = args.batch_size * (1000.0 / avg_ms)
    avg_sparse_len = sparse_len.to(torch.float32).mean().item()

    print("=== Sparse Decode Profiling ===")
    print(f"implementation_folder: {Path(args.impl_folder).resolve()}")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"dtype: {args.dtype}")
    print(f"batch_size: {args.batch_size}")
    print(f"context_len: {args.context_len}")
    print(f"num_kv_heads: {args.num_kv_heads}")
    print(f"num_q_heads: {args.num_q_heads}")
    print(f"head_dim: {args.head_dim}")
    print(f"page_size: {args.page_size}")
    print(f"max_sparse_context: {args.max_sparse_context}")
    print(f"avg_sparse_len: {avg_sparse_len:.2f}")
    print(f"warmup_iters: {args.warmup_iters}")
    print(f"profile_iters: {args.iters}")
    print(f"avg_run_latency_ms: {avg_ms:.4f}")
    print(f"throughput_tokens_per_s: {tok_per_s:.2f}")


if __name__ == "__main__":
    main()
