"""KVStagingPool semantics: (staged, skipped) contract, eviction invariants,
and the leak-free copy-failure path.

Covers the F2/F3 hardening:
- ensure_resident returns (staged_map, skipped); skipped is non-empty only
  when staging is exhausted with no evictable candidate.
- Eviction candidates must be STAGING-resident; occupancy-LRU entries whose
  residency is not STAGING (desync artifacts) are never evicted -- evicting
  one would flip a live GPU block to "CPU" with a stale cpu_slot.
- A failed copy batch returns reserved slots to the free list and leaves the
  occupancy LRU untouched (entries are published only after copies succeed).
"""
from __future__ import annotations

import pytest
import torch

from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.base import BlockKVStats, BlockRef
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable


def _make_pool(num_pages=16, num_slots=4, num_cpu_slots=32, layers=1):
    N, H, C = 8, 2, 32
    combined = []
    for _ in range(layers):
        kc = torch.randn(num_pages, N, H, C, dtype=torch.float16, device="cuda")
        vc = torch.randn(num_pages, N, H, C, dtype=torch.float16, device="cuda")
        from offload_package.tests.conftest import combined_caches
        combined_caches([kc], [vc])  # warm import
        combined.append(
            torch.cat([kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3)], dim=-1)
            .contiguous()
        )
    cpu_cache = torch.zeros(
        (layers, num_cpu_slots, 2, N, H, C),
        dtype=torch.float16, device="cpu", pin_memory=True,
    )
    residency = ResidencyTable()
    mgr = KVOffloadManager(
        scorer=PagedEvictionScorer(), residency=residency,
        gpu_k_caches=combined, gpu_v_caches=combined,
        cpu_cache=cpu_cache, gpu_budget_blocks=0, kv_layout="HND",
    )
    pool = KVStagingPool(
        num_slots=num_slots,
        staging_base_page=num_pages - num_slots,
        gpu_k_caches=combined, gpu_v_caches=combined,
        cpu_cache=cpu_cache, residency=residency, offload_manager=mgr,
    )
    return pool, mgr, residency


def _offload_blocks(mgr, residency, refs):
    """Register refs on GPU, score them, and copy them to CPU slots."""
    for i, ref in enumerate(refs):
        residency.mark_gpu(ref, i, complete=True)
    v_norm = torch.ones(len(refs), 1)
    k_norm = torch.ones(len(refs), 1)
    mgr.scorer.update(BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=refs))
    return mgr.offload_fraction(refs, 0.99, set(refs))


class TestEnsureResidentContract:
    def test_success_returns_empty_skipped(self):
        pool, mgr, res = _make_pool()
        refs = [BlockRef("r", i) for i in range(2)]
        assert _offload_blocks(mgr, res, refs) == refs
        staged, skipped = pool.ensure_resident(refs)
        assert not skipped
        assert set(staged) == set(refs)
        for ref in refs:
            assert res.get(ref).residency == Residency.STAGING
            assert pool._lru[ref] == staged[ref]

    def test_capacity_overflow_evicts_lowest_scored(self):
        pool, mgr, res = _make_pool(num_slots=2)
        old_refs = [BlockRef("old", 0), BlockRef("old", 1)]
        assert len(_offload_blocks(mgr, res, old_refs)) == 2
        staged, skipped = pool.ensure_resident(old_refs)
        assert not skipped and len(staged) == 2

        # Two NEW cold blocks: both previously-staged (scored 4 and 3) are
        # evicted to make room; the newcomers fit.
        new_refs = [BlockRef("new", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, new_refs)) == 2
        mgr.scorer._scores.clear()
        all_refs = old_refs + new_refs
        v_norm = torch.tensor([[4.0], [3.0], [2.0], [1.5]])
        mgr.scorer.update(BlockKVStats(
            v_norm=v_norm, k_norm=torch.ones(len(all_refs), 1),
            block_ids=all_refs))

        staged, skipped = pool.ensure_resident(new_refs)
        assert not skipped
        assert set(staged) == set(new_refs)
        # The lowest-scored previously-staged blocks were evicted to CPU.
        assert res.get(old_refs[0]).residency == Residency.CPU
        assert res.get(old_refs[1]).residency == Residency.CPU
        for ref in new_refs:
            assert res.get(ref).residency == Residency.STAGING

    def test_same_batch_never_evicts_its_own_allocations(self):
        """A single ensure_resident call can stage at most the pool's current
        capacity: refs allocated earlier in the batch are not yet published
        as STAGING, so they are never evicted to make room for later ones.
        This is the contract the wrapper's chunked path relies on when it
        splits the cold set into capacity-sized groups."""
        pool, mgr, res = _make_pool(num_slots=2)
        refs = [BlockRef("r", i) for i in range(4)]
        assert len(_offload_blocks(mgr, res, refs)) == 4
        staged, skipped = pool.ensure_resident(refs)
        assert len(staged) == 2
        assert set(skipped) == set(refs[2:])
        for ref in refs[:2]:
            assert res.get(ref).residency == Residency.STAGING
        for ref in refs[2:]:
            assert res.get(ref).residency == Residency.CPU


class TestDesyncEvictionGuard:
    def test_desynced_lru_entries_are_not_evicted(self):
        """Occupancy-LRU entries whose residency is not STAGING must never be
        eviction victims; a full pool of desync entries reports skipped
        instead of flipping live GPU blocks to CPU."""
        pool, mgr, res = _make_pool(num_slots=2)
        holders = [BlockRef("r", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, holders)) == 2
        staged, skipped = pool.ensure_resident(holders)
        assert not skipped and len(staged) == 2

        # Simulate the desync: occupancy-LRU still holds both, but residency
        # was re-marked GPU (the state mark_gpu leaves behind).
        for ref in holders:
            res.mark_gpu(ref, 100 + ref.block_idx, complete=True)

        newcomers = [BlockRef("q", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, newcomers)) == 2
        staged, skipped = pool.ensure_resident(newcomers)
        # No STAGING candidates to evict -> newcomers are skipped, and the
        # GPU-resident holders were NOT touched.
        assert skipped == newcomers
        assert staged == {}
        for ref in holders:
            assert res.get(ref).residency == Residency.GPU
        assert len(pool._lru) == 2  # desync entries left in place (reported)


class TestCopyFailureIsLeakFree:
    def test_failed_copy_returns_slots_and_leaves_lru(self, monkeypatch):
        pool, mgr, res = _make_pool(num_slots=4)

        class _Boom:
            def __getitem__(self, layer):
                return self

            def index_select(self, *a, **k):
                raise RuntimeError("simulated copy failure")

        monkeypatch.setattr(pool, "cpu_cache", _Boom())

        refs = [BlockRef("r", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, refs)) == 2
        free_before = list(pool._free)
        with pytest.raises(RuntimeError, match="simulated copy failure"):
            pool.ensure_resident(refs)
        # The two reserved slots were popped from _free and handed back on
        # failure: nothing leaked, contents identical (order may differ).
        assert sorted(pool._free) == sorted(free_before)
        # Occupancy LRU untouched: entries publish only after copy success.
        for ref in refs:
            assert ref not in pool._lru
            assert res.get(ref).residency == Residency.CPU


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
