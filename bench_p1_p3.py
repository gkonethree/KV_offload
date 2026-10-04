#!/usr/bin/env python3
"""P1+P3 evidence: time the OLD vs NEW implementations of the Python hot paths
at 64k-shaped tensors. Old implementations are inlined copies from before the
P1/P3 edits; new ones imported from the package."""

import time
import torch

import sys
sys.path.insert(0, ".")

from offload_package.selection.adapter import collect_needed_block_refs, patch_paged_kv_indices_with_staging
from offload_package.scoring.base import BlockRef, BlockKVStats
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.scoring.norm_kernel import compute_block_kv_norms
from offload_package.staging.residency import ResidencyTable, Residency

PAGE = 16


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
    B, Hq, S, PAGES = 4, 32, 6715, 3800  # 64k-scale shape
    request_ids = [f"req{i}" for i in range(B)]

    # --- OLD collect: per-(row,head) loop with .item() syncs + per-head unique
    sparse_idx_gpu = torch.randint(0, 65536, (B, Hq, S), device=dev)
    sparse_len_gpu = torch.full((B, Hq, 1), S, dtype=torch.int32, device=dev)

    def collect_old():
        refs, seen = [], set()
        for b, rid in enumerate(request_ids):
            for q in range(sparse_idx_gpu.shape[1]):
                n = int(sparse_len_gpu[b, q, 0].item())
                tokens = sparse_idx_gpu[b, q, :n]
                blocks = torch.div(tokens, PAGE, rounding_mode="floor").unique()
                for bi in blocks.tolist():
                    ref = BlockRef(rid, int(bi))
                    if ref not in seen:
                        seen.add(ref)
                        refs.append(ref)
        return refs

    def collect_new():
        return collect_needed_block_refs(request_ids, sparse_idx_gpu, sparse_len_gpu, PAGE)

    a, b = collect_old(), collect_new()
    assert set(a) == set(b), "collect output mismatch"
    t_old = bench("collect_old", collect_old)
    t_new = bench("collect_new", collect_new)
    print(f"[collect] old={t_old*1000:.1f}ms new={t_new*1000:.1f}ms  ({t_old / t_new:.1f}x faster)")

    # --- OLD patch: per-position loop; residency with ~30% offloaded
    residency = ResidencyTable()
    n_pages = PAGES
    staging_base = 90000
    for b, rid in enumerate(request_ids):
        for blk in range(n_pages):
            residency.mark_gpu(BlockRef(rid, blk), gpu_block_id=1000 + blk, complete=True)
    # 30% CPU, a few STAGING
    for b, rid in enumerate(request_ids):
        for blk in range(0, n_pages, 3):
            residency.mark_cpu(BlockRef(rid, blk), cpu_slot=blk % 2048)
        for i, blk in enumerate(range(1, n_pages, 997)):
            residency.mark_cpu(BlockRef(rid, blk), cpu_slot=blk % 2048)
            residency.mark_staging(BlockRef(rid, blk), staging_slot=i % 64)

    indptr_cpu = torch.tensor([i * n_pages for i in range(B + 1)], dtype=torch.int32)
    indices_gpu = torch.randint(0, 80000, (B * n_pages,), dtype=torch.int32, device=dev)

    def patch_old():
        req2b = {rid: i for i, rid in enumerate(request_ids)}
        idx = indptr_cpu.tolist()
        patched = indices_gpu.clone()
        for b, rid in enumerate(request_ids):
            start, end = int(idx[b]), int(idx[b + 1])
            for flat_pos in range(start, end):
                ref = BlockRef(str(rid), flat_pos - start)
                loc = residency.get(ref)
                if loc is None or loc.residency == Residency.GPU:
                    continue
                if loc.residency == Residency.STAGING and loc.staging_slot is not None:
                    patched[flat_pos] = int(staging_base + loc.staging_slot)
                elif loc.residency == Residency.CPU:
                    patched[flat_pos] = 0
        return patched

    def patch_new():
        return patch_paged_kv_indices_with_staging(
            indices_gpu, indptr_cpu, request_ids, None, None, PAGE,
            staging_base, residency, {}, null_block_id=0,
        )

    a, b = patch_old(), patch_new()
    assert torch.equal(a, b), "patch output mismatch"
    t_old = bench("patch_old", patch_old)
    t_new = bench("patch_new", patch_new)
    print(f"[patch]   old={t_old*1000:.1f}ms new={t_new*1000:.1f}ms  ({t_old / t_new:.1f}x faster)")

    # --- OLD ensure_resident per-block x per-layer; NEW batched
    L, H, N, C = 22, 4, 16, 64
    N_GPU = 92048
    n_fetch = 48
    k_combined_layers = [
        torch.randn(N_GPU, H, N, 2 * C, dtype=torch.float16, device=dev) for _ in range(L)
    ]
    gpu_k = k_combined_layers
    gpu_v = k_combined_layers
    cpu_cache = torch.empty((L, 2048, 2, N, H, C), dtype=torch.float16, device="cpu", pin_memory=True)

    from offload_package.offload.manager import KVOffloadManager
    from offload_package.staging.pool import KVStagingPool

    def make_pool(base):
        res = ResidencyTable()
        mgr = KVOffloadManager(
            scorer=PagedEvictionScorer(), residency=res,
            gpu_k_caches=gpu_k, gpu_v_caches=gpu_v,
            cpu_cache=cpu_cache, gpu_budget_blocks=base, kv_layout="HND",
        )
        return KVStagingPool(
            num_slots=2048, staging_base_page=base, gpu_k_caches=gpu_k,
            gpu_v_caches=gpu_v, cpu_cache=cpu_cache, residency=res, offload_manager=mgr,
        ), res, mgr

    pool, res_, mgr_ = make_pool(staging_base)
    refs = [BlockRef("req0", i) for i in range(n_fetch)]
    for i, ref in enumerate(refs):
        res_.mark_gpu(ref, gpu_block_id=10 + i, complete=True)
        res_.mark_cpu(ref, cpu_slot=i)

    # OLD-STYLE copy loop (per block, per layer), same content
    def fetch_old():
        for ref in refs:
            loc = res_.get(ref)
            slot = base_ids = None
            slot = (ref.block_idx)  # pretend slot == idx for the copy cost run
            for layer, (kc, vc) in enumerate(zip(gpu_k, gpu_v)):
                k_src = cpu_cache[layer, loc.cpu_slot, 0].permute(1, 0, 2).contiguous()
                v_src = cpu_cache[layer, loc.cpu_slot, 1].permute(1, 0, 2).contiguous()
                combined = torch.cat([k_src, v_src], dim=-1)
                kc[staging_base + slot].copy_(combined, non_blocking=True)
        torch.cuda.synchronize()

    def fetch_new():
        pool.ensure_resident(refs)
        torch.cuda.synchronize()

    t_old = bench("fetch_old", fetch_old)
    t_new = bench("fetch_new", fetch_new)
    print(f"[fetch {n_fetch} blocks x {L} layers] old={t_old*1000:.1f}ms new={t_new*1000:.1f}ms  ({t_old / t_new:.1f}x faster)")

    # --- P3 scorer batching: old per-block x per-layer vs batched score_blocks
    N_BLOCKS = 2000
    refs2 = [BlockRef("req0", i) for i in range(N_BLOCKS)]
    gids = [100 + i for i in range(N_BLOCKS)]

    k_cache0 = k_combined_layers[0]
    v_cache0 = k_combined_layers[0]

    def score_old():
        # old score_block: per block, per layer — we only use one layer here
        for ref, gid in zip(refs2, gids):
            stats = compute_block_kv_norms(k_cache0, v_cache0, [gid], logical_block_ids=[ref], layout="HND")
            s = BlockKVStats(v_norm=stats.v_norm, k_norm=stats.k_norm, block_ids=[ref])
            # drop: only measuring the kernel/sync cost
    def score_new():
        # batched via compute_block_kv_norms once over all ids (what score_blocks does per layer)
        compute_block_kv_norms(k_cache0, v_cache0, gids, logical_block_ids=refs2, layout="HND")

    t_old = bench("score_old", score_old, iters=1, warmup=0)
    t_new = bench("score_new", score_new)
    print(f"[score {N_BLOCKS} blocks, 1 layer] old={t_old*1000:.1f}ms new={t_new*1000:.1f}ms  ({t_old / t_new:.1f}x faster)")


if __name__ == "__main__":
    main()
