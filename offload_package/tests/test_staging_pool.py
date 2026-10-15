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
            # Production cpu_cache layer shape [S, 2, P, H, C] so the pinned
            # slab allocation succeeds and the failure fires at the gather.
            shape = (32, 2, 8, 2, 32)

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


class TestEvictionProtection:
    """protect= refs (this step's already-staged selections) must never be
    eviction victims: their index entries are patched at their slots, and a
    clobbered slot feeds the kernel the wrong block's bytes at weight 1."""

    def test_protected_ref_survives_full_pool_fetch(self):
        pool, mgr, res = _make_pool(num_slots=2)
        holders = [BlockRef("r", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, holders)) == 2
        staged, skipped = pool.ensure_resident(holders)
        assert not skipped and len(staged) == 2
        protected_ref = holders[0]

        # Two NEW cold blocks arrive; only the UNPROTECTED holder is
        # evictable, so exactly one newcomer fits and the other is skipped
        # (the caller's cue to fall back to the chunked path).
        newcomers = [BlockRef("q", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, newcomers)) == 2
        staged, skipped = pool.ensure_resident(newcomers, protect=[protected_ref])
        assert len(staged) == 1 and len(skipped) == 1
        assert res.get(protected_ref).residency == Residency.STAGING
        assert res.get(holders[1]).residency == Residency.CPU

    def test_full_protection_skips_instead_of_evicting(self):
        """When every resident is protected and the pool is full, the fetch
        must skip (caller falls back to chunked) rather than clobber a
        protected slot."""
        pool, mgr, res = _make_pool(num_slots=2)
        holders = [BlockRef("r", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, holders)) == 2
        staged, skipped = pool.ensure_resident(holders)
        assert not skipped and len(staged) == 2

        newcomer = BlockRef("q", 0)
        assert _offload_blocks(mgr, res, [newcomer]) == [newcomer]
        staged, skipped = pool.ensure_resident(
            [newcomer], protect=list(holders)
        )
        assert skipped == [newcomer]
        for ref in holders:  # both protected slots intact
            assert res.get(ref).residency == Residency.STAGING

    def test_no_protect_still_evicts(self):
        """Backward compatibility: without protect, eviction behaves as
        before (lowest-scored resident goes)."""
        pool, mgr, res = _make_pool(num_slots=2)
        holders = [BlockRef("r", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, holders)) == 2
        staged, _ = pool.ensure_resident(holders)

        newcomer = BlockRef("q", 0)
        assert _offload_blocks(mgr, res, [newcomer]) == [newcomer]
        staged, skipped = pool.ensure_resident([newcomer])
        assert not skipped
        # one of the holders was evicted back to CPU
        assert sorted(
            str(res.get(h).residency) for h in holders
        ) == ["Residency.CPU", "Residency.STAGING"]


class TestEvictionScorerSplit:
    """staging_eviction is independent of the offload scorer: eviction
    quality must follow recency (the pool's own LRUScorer), while the
    offload scorer keeps its own semantics for which blocks get moved."""

    def test_pool_eviction_scorer_independent_of_offload(self):
        from offload_package.scoring.lru_scorer import LRUScorer

        pool, mgr, res = _make_pool(num_slots=2)
        eviction = LRUScorer()
        pool._eviction_scorer = eviction  # simulate the split wiring

        holders = [BlockRef("r", i) for i in range(2)]
        assert len(_offload_blocks(mgr, res, holders)) == 2
        staged, skipped = pool.ensure_resident(holders)
        assert not skipped and len(staged) == 2

        # Recency: holders[0] used recently, holders[1] long ago.
        eviction.note_usage("r", [0])

        newcomer = BlockRef("q", 0)
        assert _offload_blocks(mgr, res, [newcomer]) == [newcomer]
        staged, skipped = pool.ensure_resident([newcomer], protect=[])
        assert not skipped
        # Eviction followed the POOL's recency: holders[1] (stale) evicted,
        # holders[0] (recent) kept -- even though the offload scorer's
        # paged scores would have ranked them however their norms rank.
        assert res.get(holders[0]).residency == Residency.STAGING
        assert res.get(holders[1]).residency == Residency.CPU

    def test_default_eviction_inherits_offload_scorer(self):
        pool, mgr, res = _make_pool(num_slots=2)
        assert pool._eviction_scorer is mgr.scorer

    def test_split_wiring_via_build_orchestrator(self):
        """build_orchestrator(staging_eviction='lru') gives the pool its own
        LRUScorer while scorer_type='paged_eviction' keeps the offload
        scorer as PagedEvictionScorer."""
        from types import SimpleNamespace
        import torch
        from offload_package.shared.types import (
            OrchestratorConfig, build_orchestrator,
        )
        from offload_package.scoring.paged_eviction import PagedEvictionScorer
        from offload_package.scoring.lru_scorer import LRUScorer

        L, P, H, C = 2, 8, 2, 32
        combined = torch.randn(16, H, P, 2 * C, dtype=torch.float16, device="cuda")
        kv_cache_view = SimpleNamespace(
            k_caches=[combined], v_caches=[combined],
            block_size=P, num_kv_heads=H, head_dim=C, kv_layout="HND",
            staging_base_page=8, num_layers=1, dtype=torch.float16,
            device=combined.device,
        )
        cfg = OrchestratorConfig(
            offload_fraction=0.5, num_cpu_slots=32, num_staging_slots=4,
            W=16, page_size=P, dtype=torch.float16,
            device=combined.device, gpu_budget_blocks=4,
            scorer_type="paged_eviction", staging_eviction="lru",
        )
        orch = build_orchestrator(cfg, kv_cache_view=kv_cache_view)
        assert isinstance(orch.offload_manager.scorer, PagedEvictionScorer)
        assert isinstance(orch.staging_pool._eviction_scorer, LRUScorer)
        assert orch.staging_pool._eviction_scorer is not orch.offload_manager.scorer


class TestLayerOrderedLRU:
    """staging_eviction='lru': every layer's use (fetch, gap fill or an
    already-staged pick passed as `protect`) refreshes recency, so layer L
    evicts before touching what layer L-1 used, and layer 0 of the next step
    keeps what the previous step's last layer used."""

    def _setup(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=3)
        pool._eviction_policy = "lru"
        refs = {n: BlockRef("r", i) for i, n in enumerate("ABCDEF")}
        assert _offload_blocks(mgr, res, list(refs.values())) == list(refs.values())
        return pool, res, refs

    @staticmethod
    def _staged(res, ref):
        return res.get(ref).residency == Residency.STAGING

    def test_previous_layer_blocks_survive(self):
        pool, res, R = self._setup()
        pool.ensure_resident_tokens(0, [(R["A"], [0]), (R["B"], [1])])
        pool.ensure_resident_tokens(1, [(R["C"], [0]), (R["D"], [2])])
        _, skipped = pool.ensure_resident_tokens(2, [(R["E"], [3])])
        assert not skipped
        assert self._staged(res, R["E"])
        assert self._staged(res, R["C"]) and self._staged(res, R["D"])
        assert not self._staged(res, R["A"])          # oldest (layer 0) goes

    def test_next_step_layer0_keeps_last_layers_blocks(self):
        pool, res, R = self._setup()
        pool.ensure_resident_tokens(0, [(R["A"], [0]), (R["B"], [0])])
        pool.ensure_resident_tokens(1, [(R["C"], [0]), (R["D"], [0])])
        # Last layer of the step uses A (already staged -> protect + fill)
        # and B only via protect (fully covered for that layer).
        pool.ensure_resident_tokens(
            2, [(R["A"], [5])], protect=[R["A"], R["B"]])
        pool.ensure_resident_tokens(0, [(R["E"], [0]), (R["F"], [0])])
        assert self._staged(res, R["A"]) and self._staged(res, R["B"])
        assert not self._staged(res, R["C"]) and not self._staged(res, R["D"])

    def test_current_layer_selection_never_evicted(self):
        pool, res, R = self._setup()
        pool.ensure_resident_tokens(0, [(R["A"], [0]), (R["B"], [0]),
                                        (R["C"], [0]), (R["D"], [0])])
        # A is the oldest, but this layer selects it.
        pool.ensure_resident_tokens(1, [(R["E"], [0])], protect=[R["A"]])
        assert self._staged(res, R["A"])
        assert not self._staged(res, R["B"])

    def test_block_mode_uses_same_order(self):
        pool, res, R = self._setup()
        pool.ensure_resident([R["A"], R["B"]])
        pool.ensure_resident([R["C"], R["D"]])
        pool.ensure_resident([R["E"]], protect=[R["A"], R["C"]])
        assert self._staged(res, R["A"]) and self._staged(res, R["C"])
        assert not self._staged(res, R["B"])
