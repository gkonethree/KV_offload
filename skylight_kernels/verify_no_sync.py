#!/usr/bin/env python3
"""
Verify that ``sparse_oracle_topk_optimized.BatchDecodeWithPagedKVCacheWrapper``
performs **zero host-device synchronizations** per ``run()`` call when
constructed with ``max_seq_len=...`` (the vLLM ``max_model_len`` path).

Three independent checks are run, each of which fails LOUDLY if any sync
slips in:

  Check 1 - cache-witness static check:
      The wrapper uses ``_lmax_cache`` as a one-shot host-side cache for
      the ``L_per.max().item()`` result. If the sync-free branch is taken,
      this attribute is never created. We assert it stays absent across
      many calls.

  Check 2 - PyTorch sync debug mode:
      ``torch.cuda.set_sync_debug_mode("error")`` makes any implicit
      GPU->CPU sync raise a RuntimeError. We arm it after warmup and run
      ``oracle.run(...)`` repeatedly; if anything in the score kernel,
      flashinfer.top_k call, sparse_decode kernel, or the wrapper Python
      layer hits a sync point, this raises immediately.

  Check 3 - CUDA Graph capture:
      A CUDA Graph captures only async stream work; capture FAILS if any
      sync (or any tensor->host op) executes during capture. Successfully
      capturing and replaying ``run(...)`` inside a graph is the strongest
      possible empirical proof that the call is end-to-end async.

Compares the sync-free path against the auto-derived path: with no
``max_seq_len`` supplied, both Check 1 and Check 2 must FAIL on the first
call (proving the syncs are real). The script reports both outcomes.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparse_oracle_topk_optimized import (
    BatchDecodeWithPagedKVCacheWrapper as OracleTopKOptWrapper,
)


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


def _make_wrapper(*, max_seq_len, indptr, indices, last, H_q, H_kv, D, page,
                  dtype, device):
    ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = OracleTopKOptWrapper(ws, "NHD", max_seq_len=max_seq_len)
    w.plan(indptr, indices, last, H_q, H_kv, D, page,
           q_data_type=dtype, kv_data_type=dtype)
    return w, ws


def check_1_lmax_cache(oracle, q, kv, topk, *, expect_sync_free):
    """``_lmax_cache`` is set iff ``_max_seq_len()`` (the .item() path) ran."""
    assert not hasattr(oracle, "_lmax_cache")
    for _ in range(5):
        _ = oracle.run(q, kv, topk, channel_num=8)
    cached = hasattr(oracle, "_lmax_cache")
    if expect_sync_free:
        assert not cached, (
            "_lmax_cache populated => .item() sync occurred on the "
            "max_seq_len path"
        )
        print("  [check 1] OK: _lmax_cache not populated => no .item() sync.")
    else:
        assert cached, (
            "_lmax_cache NOT populated => unexpected: the auto path should "
            "have called _max_seq_len()"
        )
        print("  [check 1] (auto) _lmax_cache populated as expected (= sync).")


def check_2_sync_debug(oracle, q, kv, topk, *, expect_sync_free):
    """``set_sync_debug_mode('error')`` raises on any implicit D2H sync."""
    # Warm up first (JIT, allocator priming, kernel autotune, FlashInfer
    # state) so we measure only the steady-state run() cost.
    for _ in range(3):
        _ = oracle.run(q, kv, topk, channel_num=8)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    raised = None
    try:
        for _ in range(5):
            _ = oracle.run(q, kv, topk, channel_num=8)
        torch.cuda.synchronize()  # the only legal sync, after the loop
    except RuntimeError as exc:  # noqa: BLE001
        raised = exc
    finally:
        torch.cuda.set_sync_debug_mode("default")

    if expect_sync_free:
        assert raised is None, (
            f"sync detected on the max_seq_len path:\n  {raised}"
        )
        print("  [check 2] OK: 5 run() calls under sync_debug_mode='error' "
              "with NO sync raised.")
    else:
        assert raised is not None, (
            "auto path was expected to sync but didn't (unexpected)"
        )
        msg = str(raised).splitlines()[0]
        print(f"  [check 2] (auto) sync_debug raised as expected: {msg!r}")


def check_3_cuda_graph(oracle, q, kv, topk, *, expect_sync_free):
    """CUDA Graph capture fails if any sync runs during capture."""
    # Warm up.
    for _ in range(3):
        _ = oracle.run(q, kv, topk, channel_num=8)
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            _ = oracle.run(q, kv, topk, channel_num=8)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    captured_out = None
    captured = False
    err = None
    try:
        with torch.cuda.graph(g, stream=s):
            captured_out = oracle.run(q, kv, topk, channel_num=8)
        captured = True
    except Exception as exc:  # noqa: BLE001
        err = exc

    if expect_sync_free:
        if not captured:
            raise AssertionError(
                f"CUDA Graph capture FAILED on the max_seq_len path:\n"
                f"  {err!r}\n"
                f"  ({type(err).__name__})"
            )
        # Replay -- the captured graph should produce a finite output.
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(captured_out).all(), \
            "captured graph output had non-finite values"
        print("  [check 3] OK: run() captured into a CUDA Graph and "
              "replayed cleanly.")
    else:
        # The auto path may or may not fail capture depending on torch
        # version; .item() inside capture is illegal in some versions but
        # silently sync-on-replay in others.
        if captured:
            print("  [check 3] (auto) capture surprisingly SUCCEEDED "
                  "(possibly torch lets .item() sync through capture).")
        else:
            print(f"  [check 3] (auto) capture failed as expected: "
                  f"{type(err).__name__}: "
                  f"{str(err).splitlines()[0]}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    dtype = torch.float16
    B, ctx, page, H_kv, H_q, D = 4, 4096, 16, 8, 32, 128
    topk = 256 / 4096  # same k_eff as legacy top_k=256 at ctx=4096
    indptr, indices, last, kv, q = _make_inputs(
        B=B, ctx=ctx, page=page, H_kv=H_kv, H_q=H_q, D=D,
        dtype=dtype, device=device, seed=0,
    )

    print(f"device   : {torch.cuda.get_device_name(device)}")
    print(f"dtype    : {dtype}")
    print(f"shape    : B={B} ctx={ctx} H_kv={H_kv} H_q={H_q} D={D} topk={topk}")
    print()

    print("== Sync-free path: max_seq_len=ctx supplied ==")
    oracle, _ws = _make_wrapper(
        max_seq_len=ctx, indptr=indptr, indices=indices, last=last,
        H_q=H_q, H_kv=H_kv, D=D, page=page, dtype=dtype, device=device,
    )
    check_1_lmax_cache(oracle, q, kv, topk, expect_sync_free=True)
    check_2_sync_debug(oracle, q, kv, topk, expect_sync_free=True)
    check_3_cuda_graph(oracle, q, kv, topk, expect_sync_free=True)
    del oracle, _ws
    torch.cuda.empty_cache()

    print()
    print("== Control: max_seq_len NOT supplied (auto path with .item() sync) ==")
    oracle2, _ws2 = _make_wrapper(
        max_seq_len=None, indptr=indptr, indices=indices, last=last,
        H_q=H_q, H_kv=H_kv, D=D, page=page, dtype=dtype, device=device,
    )
    check_1_lmax_cache(oracle2, q, kv, topk, expect_sync_free=False)
    # NOTE: check_2 on the auto path is interesting only on the FIRST run().
    # After that, the .item() result is cached and the path is effectively
    # sync-free. So we re-construct to test the very first call.
    oracle3, _ws3 = _make_wrapper(
        max_seq_len=None, indptr=indptr, indices=indices, last=last,
        H_q=H_q, H_kv=H_kv, D=D, page=page, dtype=dtype, device=device,
    )
    # Run a few warmups to JIT etc., but do NOT call oracle3.run yet (else
    # _lmax_cache will be populated and sync-free).
    # Instead, prime via a sibling instance that already JIT-compiled.
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    raised = None
    try:
        # First run on a fresh wrapper => MUST go through .item().
        _ = oracle3.run(q, kv, topk, channel_num=8)
    except RuntimeError as exc:  # noqa: BLE001
        raised = exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert raised is not None, (
        "auto path on a FRESH wrapper was expected to sync but didn't"
    )
    print(f"  [check 2] (auto, fresh wrapper) sync_debug raised as "
          f"expected: {str(raised).splitlines()[0]!r}")

    print()
    print("================================================================")
    print("ALL CHECKS PASSED. run(..., max_seq_len=ctx) is sync-free.")
    print("================================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
