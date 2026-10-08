#!/usr/bin/env python3
"""Stress repro for the batched store path (K1) under allocator churn.

Runs the same batched GPU→CPU copy repeatedly while the caching allocator is
actively churned, to surface an async-use-after-free if one exists. Prints
deterministic aggregates (no asserts expected unless the race is real).
"""
import time
import torch
import sys

sys.path.insert(0, ".")

from offload_package.offload.manager import KVOffloadManager, CPUBlockPool
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.residency import ResidencyTable

def main():
    torch.manual_seed(0)
    dev = "cuda"
    L, P, H, N, C = 22, 4096, 4, 16, 64  # TinyLlama-style combined layout
    BATCH = 64
    N_ITERS = 60

    k_caches = [torch.randn(P, H, N, 2 * C, dtype=torch.float16, device=dev) for _ in range(L)]
    cpu_cache = torch.empty((L, 2048, 2, N, H, C), dtype=torch.float16, device="cpu", pin_memory=True)

    res = ResidencyTable()
    scorer = PagedEvictionScorer()
    mgr = KVOffloadManager(
        scorer=scorer, residency=res, gpu_k_caches=k_caches, gpu_v_caches=k_caches,
        cpu_cache=cpu_cache, gpu_budget_blocks=P - 128, kv_layout="HND",
    )

    # churn allocator in the background: allocate/free big tensors each iter
    churn = [torch.empty(1 << 20, device=dev, dtype=torch.float16) for _ in range(4)]

    bad = 0
    for it in range(N_ITERS):
        # Free slots + fresh ids each iteration to force allocator recycling
        ids = [(it * BATCH + i) % (P - 1) for i in range(BATCH)]
        slots, ev = mgr._copy_gpu_to_cpu_batch(ids)
        ev.synchronize()
        # free those CPU slots again so pool isn't exhausted
        for s in slots:
            mgr._cpu_pool.free(s)
        del churn[3]
        churn.append(torch.empty(1 << 20, device=dev, dtype=torch.float16))
        # Assert content: pick the first slot of this batch, compare against source
        slot = slots[0]
        gid = ids[0]
        for l in range(L):
            ref = k_caches[l][gid]              # [H, N, 2C]
            k_ref = ref[..., :C].permute(1, 0, 2)  # K half -> [N, H, C]
            got = cpu_cache[l, slot, 0]          # [N, H, C]
            if not torch.equal(got, k_ref.cpu()):
                print(f"iter {it} layer {l}: CONTENT MISMATCH")
                bad += 1
                break
        if bad:
            break

    print(f"done: iters={N_ITERS}, mismatches={bad}")


if __name__ == "__main__":
    main()
