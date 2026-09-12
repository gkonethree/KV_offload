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


def _make_synthetic_inputs(
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
):
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
        low=1,
        high=max_sparse_context + 1,
        size=(batch_size, num_q_heads, 1),
        dtype=torch.int32,
        device=device,
    )
    sparse_idx = torch.randint(
        low=0,
        high=context_len,
        size=(batch_size, num_q_heads, max_sparse_context),
        dtype=torch.int64,
        device=device,
    )
    sparse_weights = torch.rand(
        batch_size, num_q_heads, max_sparse_context, dtype=torch.float32, device=device
    )
    sparse_weights = sparse_weights + 1e-3

    return kv_indptr, kv_indices, kv_last_page_len, kv_cache, q, sparse_len, sparse_idx, sparse_weights


def run_correctness_test(
    impl_folder: str,
    ref_folder: str,
    *,
    device: str,
    dtype: str,
    seed: int,
    trials: int,
) -> None:
    impl_cls = _load_wrapper_cls(impl_folder)
    ref_cls = _load_wrapper_cls(ref_folder)

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    torch.manual_seed(seed)
    dev = torch.device(device)

    configs = [
        (4, 256, 16, 4, 16, 64, 64),
        (8, 512, 16, 8, 32, 128, 96),
        (2, 1024, 16, 8, 64, 128, 128),
    ]

    for trial in range(trials):
        cfg = configs[trial % len(configs)]
        batch_size, context_len, page_size, num_kv_heads, num_q_heads, head_dim, max_sparse_context = cfg

        (
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            kv_cache,
            q,
            sparse_len,
            sparse_idx,
            sparse_weights,
        ) = _make_synthetic_inputs(
            batch_size=batch_size,
            context_len=context_len,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            max_sparse_context=max_sparse_context,
            dtype=torch_dtype,
            device=dev,
        )

        ws_ref = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        ws_impl = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
        ref = ref_cls(ws_ref, "NHD")
        impl = impl_cls(ws_impl, "NHD")

        ref.plan(
            kv_indptr, kv_indices, kv_last_page_len, num_q_heads, num_kv_heads, head_dim, page_size
        )
        impl.plan(
            kv_indptr, kv_indices, kv_last_page_len, num_q_heads, num_kv_heads, head_dim, page_size
        )

        ref_out, ref_lse = ref.run(q, kv_cache, sparse_len, sparse_idx, sparse_weights, return_lse=True)
        impl_out, impl_lse = impl.run(q, kv_cache, sparse_len, sparse_idx, sparse_weights, return_lse=True)

        out_ok = torch.allclose(impl_out, ref_out, rtol=3e-2, atol=3e-2)
        lse_ok = torch.allclose(impl_lse, ref_lse, rtol=3e-2, atol=3e-2, equal_nan=True)
        if not (out_ok and lse_ok):
            out_diff = (impl_out - ref_out).abs().max().item()
            lse_diff = (impl_lse - ref_lse).abs().nan_to_num().max().item()
            raise AssertionError(
                f"Mismatch on trial {trial}: out_ok={out_ok}, lse_ok={lse_ok}, "
                f"max_out_diff={out_diff:.6f}, max_lse_diff={lse_diff:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparse wrapper correctness test.")
    parser.add_argument("--impl-folder", required=True, help="Folder containing optimized sparse implementation.")
    parser.add_argument("--ref-folder", default="sparse", help="Folder containing reference sparse implementation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    run_correctness_test(
        args.impl_folder,
        args.ref_folder,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        trials=args.trials,
    )
    print("Sparse correctness test PASSED.")


if __name__ == "__main__":
    main()
