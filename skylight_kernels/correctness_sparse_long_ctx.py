#!/usr/bin/env python3
"""Correctness sweep for sparse_optimized at long contexts.

Mirrors the matrix used by `profile_sparse_long_ctx.py`:
  - context_len in {32K, 64K, 128K}
  - group_size in {4, 2, 1}  (i.e. H_q / H_kv with H_q=32, H_kv in {8,16,32})
  - sparsity in {20%, 10%, 5%, 2%, 1%, 0.5%, 0.2%, 0.1%}
but with B=1 and a smaller H_q (and proportionally smaller H_kv) so the pure
PyTorch reference (which Python-loops over batch * heads) finishes quickly
while still exercising every group_size we benchmarked.

Reports per (ctx, group_size, sparsity):
  - max |opt_out - ref_out|
  - max |opt_lse - ref_lse| over finite entries
  - PASS / FAIL versus float16 tolerances.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse import BatchDecodeWithPagedKVCacheWrapper as RefSparseWrapper
from sparse_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OptSparseWrapper,
)


CTXS = [32768, 65536, 131072]
# (group_size, H_q, H_kv).  H_q/H_kv match profiling matrix (4, 2, 1).
HEAD_CONFIGS = [
    (4, 4, 1),
    (2, 4, 2),
    (1, 4, 4),
]
B = 1
HEAD_DIM = 128
PAGE = 16
DTYPE = torch.float16
SPARSITIES = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

ATOL = 5e-2
RTOL = 5e-2


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
    print(
        f"device: {torch.cuda.get_device_name(device)}, dtype: float16, "
        f"D={HEAD_DIM}, page={PAGE}, B={B}"
    )
    print(f"tolerances: atol={ATOL}, rtol={RTOL} (float16)")
    print()

    overall_pass = True
    for ctx in CTXS:
        for group_size, H_q, H_kv in HEAD_CONFIGS:
            print(
                f"=== context_len = {ctx}  |  H_q/H_kv = {H_q}/{H_kv}  "
                f"(group_size={group_size}) ==="
            )
            header = (
                f"{'sparsity':>9s}  {'avg_S':>7s}  {'max|dout|':>10s}  "
                f"{'max|dlse|':>10s}  {'status':>7s}"
            )
            print(header)
            print("-" * len(header))

            max_S = ctx
            (kv_indptr, kv_indices, kv_last_page_len, kv_cache, q,
             sparse_idx, sparse_weights) = _build_inputs(
                B, ctx, PAGE, H_kv, H_q, HEAD_DIM, max_S, DTYPE, device
            )

            ws_ref = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            ws_opt = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            ref = RefSparseWrapper(ws_ref, "NHD")
            opt = OptSparseWrapper(ws_opt, "NHD")
            for w in (ref, opt):
                w.plan(
                    kv_indptr, kv_indices, kv_last_page_len,
                    H_q, H_kv, HEAD_DIM, PAGE,
                    pos_encoding_mode="NONE",
                    q_data_type=DTYPE, kv_data_type=DTYPE,
                )

            for r in SPARSITIES:
                avg_S = max(1, int(round(r * ctx)))
                if avg_S > max_S:
                    continue
                sparse_len = torch.full(
                    (B, H_q, 1), avg_S, dtype=torch.int32, device=device
                )

                ref_out, ref_lse = ref.run(
                    q, kv_cache, sparse_len, sparse_idx, sparse_weights,
                    return_lse=True,
                )
                opt_out, opt_lse = opt.run(
                    q, kv_cache, sparse_len, sparse_idx, sparse_weights,
                    return_lse=True,
                )

                d_out = (opt_out.float() - ref_out.float()).abs()
                max_d_out = d_out.max().item()
                ok_out = torch.allclose(
                    opt_out.float(), ref_out.float(),
                    rtol=RTOL, atol=ATOL,
                )

                finite = torch.isfinite(ref_lse) & torch.isfinite(opt_lse)
                if finite.any():
                    d_lse = (opt_lse[finite].float()
                             - ref_lse[finite].float()).abs()
                    max_d_lse = d_lse.max().item()
                    lse_ref_max = ref_lse[finite].abs().max().item()
                    ok_lse = max_d_lse <= max(ATOL, RTOL * lse_ref_max)
                else:
                    max_d_lse = 0.0
                    ok_lse = True
                inf_ref = ~torch.isfinite(ref_lse) & (ref_lse < 0)
                inf_opt = ~torch.isfinite(opt_lse) & (opt_lse < 0)
                ok_lse = ok_lse and torch.equal(inf_ref, inf_opt)

                ok = ok_out and ok_lse
                overall_pass = overall_pass and ok
                status = "PASS" if ok else "FAIL"
                print(
                    f"{r:>9.2%}  {avg_S:>7d}  "
                    f"{max_d_out:>10.4g}  {max_d_lse:>10.4g}  "
                    f"{status:>7s}"
                )

            del kv_cache, q, sparse_idx, sparse_weights
            torch.cuda.empty_cache()
            print()

    print("=" * 32)
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
