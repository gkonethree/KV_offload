#!/usr/bin/env python3
"""K1/K2 evidence: batched store copy + fused scorer vs old per-block paths.

Shapes mirror the 64k TinyLlama setup: L=22 layers, page=[H=4, N=16, 2C=128]
fp16 combined, and store batches of 512 and 1869 blocks (the latter is the
measured prefill-complete offload size at 64k).
"""
import time
import torch

import sys
sys.path.insert(0, ".")

from offload_package.offload.manager import KVOffloadManager, CPUBlockPool
from offload_package.offload.triton_score import score_blocks_fused, _HAS_TRITON
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.scoring.norm_kernel import compute_block_kv_norms
from offload_package.staging.residency import ResidencyTable


def bench(name, fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def main():
    torch.manual_seed(0)
    dev = "cuda"
    L, P, H, N, C = 22, 4096, 4, 16, 64

    k_caches = [torch.randn(P, H, N, 2 * C, dtype=torch.float16, device=dev) for _ in range(L)]
    v_caches = k_caches
    cpu_slots = 4096
    cpu_cache = torch.empty((L, cpu_slots, 2, N, H, C), dtype=torch.float16, device="cpu", pin_memory=True)

    res = ResidencyTable()
    scorer = PagedEvictionScorer()
    mgr = KVOffloadManager(
        scorer=scorer, residency=res, gpu_k_caches=k_caches, gpu_v_caches=v_caches,
        cpu_cache=cpu_cache, gpu_budget_blocks=P - 128, kv_layout="HND",
    )

    # ---- K1: store path, batched vs per-block reference
    for n_blocks in [512, 1869]:
        ids = list(range(100, 100 + n_blocks))

        def store_old():
            mgr._cpu_pool._free = list(range(cpu_slots - 1, -1, -1))
            for gid in ids:
                mgr._copy_gpu_to_cpu(gid)
            torch.cuda.synchronize()

        def store_new():
            mgr._cpu_pool._free = list(range(cpu_slots - 1, -1, -1))
            mgr._copy_gpu_to_cpu_batch(ids)
            torch.cuda.synchronize()

        # reset pool state between runs
        mgr._cpu_pool._free = list(range(cpu_slots - 1, -1, -1))
        t_old = bench(f"store_old{n_blocks}", store_old)
        mgr._cpu_pool._free = list(range(cpu_slots - 1, -1, -1))
        t_new = bench(f"store_new{n_blocks}", store_new)
        print(f"[store {n_blocks} blocks]  old={t_old*1000:7.1f}ms  new={t_new*1000:7.1f}ms  ({t_old/t_new:6.1f}x)")

    # ---- K2: fused scorer vs per-layer torch path
    ids = list(range(50, 1000))
    k_, v_ = None, None

    def score_torch_path():
        all_v, all_k = [], []
        for kc, vc in zip(k_caches, v_caches):
            hk, hv = kc[..., :C], vc[..., C:]
            st = compute_block_kv_norms(hk, hv, ids, layout="HND")
            all_k.append(st.k_norm); all_v.append(st.v_norm)
        return torch.stack(all_k).mean(0), torch.stack(all_v).mean(0)

    def score_fused():
        if not _HAS_TRITON: return None
        return score_blocks_fused(k_caches, ids)

    t_old = bench("score_old", score_torch_path)
    if _HAS_TRITON:
        t_new = bench("score_fused", score_fused)
        print(f"[score {len(ids)} blocks x {L} layers]  old(torch)={t_old*1000:7.1f}ms  fused={t_new*1000:7.1f}ms  ({t_old/t_new:6.1f}x)")
    else:
        print("[score] triton unavailable; skipped")


if __name__ == "__main__":
    main()
