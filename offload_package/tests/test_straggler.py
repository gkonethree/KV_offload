"""Unit tests for the per-step straggler fetch (staging pool + coordinator).

The chunked-exactness test drives the wrapper's ``_lse2_merge`` algebra
directly; the fetch/remap tests run the real engine round (compaction +
offload to the CPU token pool) and verify that a logical-space selection
lands in the staging pages byte-exact and remaps to kernel positions that
resolve through the extended page-table rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_token_offload import (
    NUM_LAYERS,
    PAGE_SIZE,
    StubScorer,
    make_combined_cache,
    token_value,
    write_tokens,
)

from offload_package.integration.straggler import (
    SelectionPlan,
    StragglerFetchCoordinator,
    TokenStagingPool,
)
from offload_package.integration.token_rounds import TokenRoundEngine
from offload_package.offload.compactor import KVCompactor
from offload_package.offload.token_pool import CPUTokenPool

NUM_PAGES = 16
STAGING_BASE = 12
STAGING_PAGES = 4
STAGING_SLOTS = STAGING_PAGES * PAGE_SIZE


def build_engine(
    torch_device,
    *,
    budget=10,
    window=4,
    sink=2,
    num_slots=512,
    staging_pages=STAGING_PAGES,
):
    caches = make_combined_cache(torch_device, num_pages=NUM_PAGES)
    block_ids = list(range(6))
    write_tokens(caches, block_ids, 24)
    compactor = KVCompactor(caches, PAGE_SIZE)
    pool = CPUTokenPool(
        num_layers=NUM_LAYERS,
        num_slots=num_slots,
        num_kv_heads=2,
        combined_head_dim=8,
        dtype=torch.float32,
        device=torch_device,
    )
    engine = TokenRoundEngine(
        scorer=StubScorer(),
        compactor=compactor,
        token_pool=pool,
        page_size=PAGE_SIZE,
        token_budget=budget,
        window_size=window,
        sink_tokens=sink,
        protect_prefill=False,
    )
    staging = TokenStagingPool(
        staging_base_page=STAGING_BASE,
        num_pages=staging_pages,
        page_size=PAGE_SIZE,
    )
    coord = StragglerFetchCoordinator(
        engine=engine,
        token_pool=pool,
        compactor=compactor,
        staging_pool=staging,
        page_size=PAGE_SIZE,
    )
    engine.straggler = coord
    engine.add_request("r1", block_ids, 24)
    engine.on_prefill_complete("r1", 0)
    result = engine.on_round("r1")
    assert result is not None and result.kept_count == budget
    return engine, coord, caches


def orig_rows(torch_device, pages=(0, 1, 2)):
    indptr = torch.tensor([0, len(pages)], dtype=torch.int32, device=torch_device)
    indices = torch.tensor(list(pages), dtype=torch.int32, device=torch_device)
    return indptr, indices


# ---------------------------------------------------------------------------
# TokenStagingPool
# ---------------------------------------------------------------------------


class TestTokenStagingPool:

    def test_place_reuse_and_release(self):
        pool = TokenStagingPool(
            staging_base_page=STAGING_BASE,
            num_pages=2,
            page_size=4,
        )
        placed, unplaced = pool.place_many([("r", 1), ("r", 2)], protect=set())
        assert not unplaced and len(placed) == 2
        # Re-placing touches and returns the same slots.
        again, _ = pool.place_many([("r", 1)], protect=set())
        assert again[("r", 1)] == placed[("r", 1)]
        assert pool.num_staged == 2
        pool.release_tokens("r", [1])
        assert pool.num_staged == 1
        assert pool.slot_of("r", 1) is None

    def test_lru_eviction_respects_protect(self):
        pool = TokenStagingPool(
            staging_base_page=STAGING_BASE, num_pages=1, page_size=2
        )
        pool.place_many([("r", 1), ("r", 2)], protect=set())
        # Touch r/1 so r/2 is the LRU victim.
        pool.slot_of("r", 1)
        placed, unplaced = pool.place_many(
            [("r", 3)], protect={("r", 2)}
        )
        assert not unplaced
        assert pool.slot_of("r", 2) is not None  # protected
        assert pool.slot_of("r", 1) is None      # LRU-evicted

    def test_unplaced_when_all_protected(self):
        pool = TokenStagingPool(
            staging_base_page=STAGING_BASE, num_pages=1, page_size=2
        )
        pool.place_many([("r", 1), ("r", 2)], protect=set())
        placed, unplaced = pool.place_many(
            [("r", 3), ("r", 4)], protect={("r", 1), ("r", 2)}
        )
        assert len(unplaced) == 2 and not placed

    def test_release_request(self):
        pool = TokenStagingPool(
            staging_base_page=STAGING_BASE, num_pages=1, page_size=4
        )
        pool.place_many([("r1", 1), ("r2", 2)], protect=set())
        pool.release_request("r1")
        assert pool.slot_of("r1", 1) is None
        assert pool.slot_of("r2", 2) is not None
        assert pool.pages_of("r1") == []


# ---------------------------------------------------------------------------
# Coordinator: remap + fetch
# ---------------------------------------------------------------------------


class TestCoordinatorApply:

    def test_fast_path_fetch_remap_and_data(self, torch_device):
        engine, coord, caches = build_engine(torch_device)
        state = engine.state("r1")
        kept = sorted(state.logical_of_physical[:10].tolist())
        cpu_logical = [t for t in range(24) if t not in kept]

        sel = torch.tensor(
            [[kept[0], kept[5], cpu_logical[0], cpu_logical[1], cpu_logical[2]]],
            dtype=torch.long,
            device=torch_device,
        )
        indptr, indices = orig_rows(torch_device, pages=state.block_ids)
        plan = coord.apply_selection(sel, ["r1"], indptr, indices)

        assert isinstance(plan, SelectionPlan)
        assert plan.fetch_missing == []          # everything fit
        assert plan.sparse_idx.shape == sel.shape
        # Row extended: physical pages then the request's staging pages.
        row = plan.indices.tolist()
        assert row[: len(state.block_ids)] == list(state.block_ids)
        n_p = len(state.block_ids)
        assert len(row) > n_p                    # staging pages appended

        kernel = plan.sparse_idx[0].tolist()
        logicals = sel[0].tolist()
        for out_pos, logical in zip(kernel, logicals):
            if logical in kept:
                expect = state.logical_of_physical[
                    : state.physical_len
                ].tolist().index(logical)
                assert out_pos == expect
            else:
                page_local = out_pos // PAGE_SIZE
                assert page_local >= n_p         # staging region
                sp = row[page_local]
                offset = out_pos % PAGE_SIZE
                for layer in range(NUM_LAYERS):
                    got = caches[layer][sp, :, offset, :4]
                    assert torch.all(got == token_value(layer, logical)).item()

        # Second application the same step: staged tokens reused, no fetch.
        plan2 = coord.apply_selection(sel, ["r1"], indptr, indices)
        assert plan2.fetch_missing == []
        assert torch.equal(plan2.sparse_idx, plan.sparse_idx)

    def test_staged_reuse_across_steps(self, torch_device):
        engine, coord, caches = build_engine(torch_device)
        state = engine.state("r1")
        cpu_logical = [
            t for t in range(24)
            if t not in set(state.logical_of_physical[:10].tolist())
        ]
        sel = torch.tensor([cpu_logical[:3]], dtype=torch.long, device=torch_device)
        indptr, indices = orig_rows(torch_device, pages=state.block_ids)
        plan = coord.apply_selection(sel, ["r1"], indptr, indices)
        assert plan.fetch_missing == []
        staged_before = coord.staging.num_staged

        # New step: virtuals re-scattered (n_p may have changed), tokens
        # must still be staged -> no refetch.
        coord.begin_step(2, ["r1"])
        plan2 = coord.apply_selection(sel, ["r1"], indptr, indices)
        assert plan2.fetch_missing == []
        assert coord.staging.num_staged == staged_before

    def test_rescatter_survives_page_growth(self, torch_device):
        """Virtual positions must track a growing physical page count."""
        engine, coord, caches = build_engine(torch_device)
        state = engine.state("r1")
        cpu_logical = [
            t for t in range(24)
            if t not in set(state.logical_of_physical[:10].tolist())
        ]
        sel = torch.tensor([cpu_logical[:2]], dtype=torch.long, device=torch_device)
        indptr, indices = orig_rows(torch_device, pages=state.block_ids)
        plan1 = coord.apply_selection(sel, ["r1"], indptr, indices)

        # Simulate a decode step appending a page + token.
        engine.sync_blocks("r1", state.block_ids + [6])
        caches[0][6, :, 0, :4] = token_value(0, 24)
        caches[1][6, :, 0, :4] = token_value(1, 24)
        engine.on_tokens_written("r1", 1)

        coord.begin_step(2, ["r1"])
        indptr2, indices2 = orig_rows(torch_device, pages=state.block_ids)
        plan2 = coord.apply_selection(sel, ["r1"], indptr2, indices2)
        assert plan2.fetch_missing == []
        # Virtual positions shifted by exactly one page.
        shift = PAGE_SIZE * 1
        assert torch.equal(
            plan2.sparse_idx, plan1.sparse_idx + shift
        )
        # And the staged data is still addressable through the new rows.
        row = plan2.indices.tolist()
        n_p = len(state.block_ids)
        for out_pos, logical in zip(
            plan2.sparse_idx[0].tolist(), sel[0].tolist()
        ):
            sp = row[out_pos // PAGE_SIZE]
            off = out_pos % PAGE_SIZE
            for layer in range(NUM_LAYERS):
                got = caches[layer][sp, :, off, :4]
                assert torch.all(got == token_value(layer, logical)).item()

    def test_inflight_position_at_map_capacity(self, torch_device):
        """Regression: the newest token's position can equal the virtual
        map's capacity.  Clamping the vol READ must not clamp the
        in-flight ARITHMETIC -- at decode step 2 of the TinyLlama bench the
        position 1648 with vol capacity 1648 was clamped to 1647, silently
        attending the wrong (previous) token and corrupting greedy output.
        """
        engine, coord, caches = build_engine(torch_device)
        state = engine.state("r1")
        # Reproduce the exact field configuration: capacity 1648 with
        # logical_len == physical_len == 1648 (post-prompt decode step).
        cap = 1648
        vol = torch.full((cap,), -1, dtype=torch.int32, device=torch_device)
        vol[:1648] = torch.arange(1648, device=torch_device,
                                  dtype=torch.int32)
        state.virtual_of_logical = vol
        state.physical_len = state.logical_len = 1648
        # Selection picks logical position 1648 == the map capacity.
        sel = torch.tensor([[1648, 5]], dtype=torch.long, device=torch_device)
        indptr, indices = orig_rows(torch_device, pages=state.block_ids)
        plan = coord.apply_selection(sel, ["r1"], indptr, indices)
        kernel = plan.sparse_idx[0].tolist()
        # The in-flight token maps to the physical tail 1648 (NOT 1647),
        # and position 5 to its resident slot.
        assert kernel[0] == 1648
        assert kernel[1] == 5

    def test_promotion_releases_staging(self, torch_device):
        engine, coord, caches = build_engine(torch_device)
        state = engine.state("r1")
        cpu_logical = [
            t for t in range(24)
            if t not in set(state.logical_of_physical[:10].tolist())
        ]
        sel = torch.tensor([cpu_logical[:2]], dtype=torch.long, device=torch_device)
        indptr, indices = orig_rows(torch_device, pages=state.block_ids)
        assert coord.apply_selection(sel, ["r1"], indptr, indices).fetch_missing == []
        assert coord.staging.num_staged == 2

        # Flip the scorer: staged tokens get promoted at the next round and
        # their staging copies must be released.
        engine.scorer.transform = lambda k: (24.0 - k)
        engine.on_round("r1")
        assert coord.staging.num_staged == 0

    def test_overflow_goes_chunked(self, torch_device):
        # budget 4 -> 20 CPU tokens; staging 2 pages = 8 slots; select 12
        # -> overflow: the whole missing set goes chunked (base pass masks
        # all of them; chunks re-place from the same capacity).
        engine, coord, caches = build_engine(
            torch_device, budget=4, window=0, sink=2, staging_pages=2
        )
        state = engine.state("r1")
        capacity = 2 * PAGE_SIZE
        cpu_logical = [
            t for t in range(24)
            if t not in set(state.logical_of_physical[:4].tolist())
        ]
        n_sel = capacity + 4
        sel = torch.tensor([cpu_logical[:n_sel]], dtype=torch.long, device=torch_device)
        indptr, indices = orig_rows(torch_device, pages=state.block_ids)
        plan = coord.apply_selection(sel, ["r1"], indptr, indices)
        assert len(plan.fetch_missing) == n_sel
        assert coord.staging.num_staged == 0   # partial placements recycled

        # Chunked flow: groups sized by chunk_group_size() place, remap,
        # and release -- slots recycle between groups.
        gs = coord.chunk_group_size()
        assert gs == capacity
        all_groups = [
            plan.fetch_missing[i : i + gs]
            for i in range(0, len(plan.fetch_missing), gs)
        ]
        assert [len(g) for g in all_groups] == [capacity, 4]
        sel_list = sel[0].tolist()
        pos_of = {l: i for i, l in enumerate(sel_list)}
        seen_data = []
        for group in all_groups:
            unplaced = coord.fetch_chunk(group)
            assert unplaced == []
            kernel = coord.remap_kernel_positions(sel, ["r1"])
            indptr_c, indices_c = coord.ext_rows_for(
                ["r1"], indptr, indices, torch_device
            )
            row = indices_c.tolist()
            for req, logical, slot in group:
                out_pos = kernel[0][pos_of[logical]].item()
                assert out_pos >= 0
                sp = row[out_pos // PAGE_SIZE]
                off = out_pos % PAGE_SIZE
                for layer in range(NUM_LAYERS):
                    got = caches[layer][sp, :, off, :4]
                    assert torch.all(
                        got == token_value(layer, logical)
                    ).item()
                seen_data.append(logical)
            coord.release_chunk(group)
            for req, logical, _ in group:
                assert coord.staging.slot_of(req, logical) is None
                assert int(state.virtual_of_logical[logical].item()) == -1
        assert sorted(seen_data) == sorted(cpu_logical[:n_sel])

    def test_invariant_violation_raises(self, torch_device):
        engine, coord, caches = build_engine(torch_device)
        state = engine.state("r1")
        # Corrupt residency: drop a CPU token's pool entry, then select it.
        victim = next(
            t for t in range(24)
            if t not in set(state.logical_of_physical[:10].tolist())
        )
        state.cpu_slot_of_logical.pop(victim)
        state.virtual_of_logical[victim] = -1
        sel = torch.tensor([[victim]], dtype=torch.long, device=torch_device)
        indptr, indices = orig_rows(torch_device, pages=state.block_ids)
        with pytest.raises(RuntimeError, match="neither resident nor"):
            coord.apply_selection(sel, ["r1"], indptr, indices)


# ---------------------------------------------------------------------------
# Chunked merge exactness (wrapper's _lse2_merge algebra)
# ---------------------------------------------------------------------------


class TestChunkedMergeExactness:

    def test_merge_equals_full_softmax(self, torch_device):
        from pq_optimized.batch_decode_with_paged_kv_cache_wrapper import (
            BatchDecodeWithPagedKVCacheWrapper as W,
        )

        g = torch.Generator().manual_seed(7)
        B, Hq, D = 2, 3, 4
        groups = [
            torch.randn(B, Hq, 5, D, generator=g),
            torch.randn(B, Hq, 7, D, generator=g),
            torch.randn(B, Hq, 9, D, generator=g),
        ]
        scores = [
            torch.randn(B, Hq, v.shape[2], generator=g) * 2.0
            for v in groups
        ]

        def partial(v, s):
            # Kernel convention: weights 2^s, lse = log2(sum(2^s)), out is
            # the normalized weighted sum.
            m = s.amax(dim=-1, keepdim=True)          # [B, Hq, 1]
            p = torch.exp2(s - m)                     # [B, Hq, N]
            denom = p.sum(dim=-1, keepdim=True)       # [B, Hq, 1]
            out = (v * p.unsqueeze(-1)).sum(dim=2) / denom
            lse = (m + torch.log2(denom)).squeeze(-1)
            return out, lse

        outs = [partial(v, s) for v, s in zip(groups, scores)]
        out_t, lse_t = outs[0]
        for out_c, lse_c in outs[1:]:
            out_t, lse_t = W._lse2_merge(out_t, lse_t, out_c, lse_c)

        full_v = torch.cat(groups, dim=2)
        full_s = torch.cat(scores, dim=2)
        ref_out, ref_lse = partial(full_v, full_s)
        torch.testing.assert_close(out_t, ref_out, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(lse_t, ref_lse, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# slot_codes.append logical/physical decoupling
# ---------------------------------------------------------------------------


class TestAppendDecoupling:

    def test_code_row_logical_slot_physical(self, torch_device):
        sys.path.insert(0, "skylight/skylight_kernels")
        try:
            from pq_optimized.slot_codes import (
                SlotCodeMaintainer,
                encode_vectorized,
            )
        finally:
            sys.path.pop(0)

        H, PAGE, CH = 2, 4, 2
        # Codebook convention: centroids [H_kv, M, C, ds] with M*ds == CH.
        M_SUB, C_BOOK = 1, 4
        maint = SlotCodeMaintainer(
            h_kv=H, m=M_SUB, page=PAGE, n_slots=4, l_max=64
        )
        kv = torch.zeros(8, H, PAGE, 2 * CH, device=torch_device)
        # Distinctive K per flat slot: cache[block, head, off, :CH].
        for blk in range(8):
            for off in range(PAGE):
                flat = blk * PAGE + off
                kv[blk, :, off, :CH] = float(flat + 1)

        def gather(flat):
            flat = flat.long()
            return kv[flat // PAGE, :, flat % PAGE, :CH]

        cache_id = kv.data_ptr()
        buf = maint.get_buf(kv, 64)
        g = torch.Generator(device="cpu").manual_seed(3)
        centroids = torch.randn(H, M_SUB, C_BOOK, CH, generator=g).to(torch_device)

        indptr = torch.tensor([0, 3], dtype=torch.int32, device=torch_device)
        indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=torch_device)
        slots = torch.tensor([0], dtype=torch.long, device=torch_device)
        # Logical length 21 (code row 20), physical length 10 (slot 9 ->
        # page row 2, offset 1).
        logical = torch.tensor([21], dtype=torch.long, device=torch_device)
        phys = torch.tensor([10], dtype=torch.long, device=torch_device)

        before = buf[0, :, 20, :].clone()
        maint.append(
            cache_id, gather, indptr, indices, slots, logical, centroids,
            phys_lengths=phys,
        )
        written = buf[0, :, 20, :].clone()

        # Row written at the LOGICAL position with the K of the PHYSICAL
        # slot (flat 9), not the logical slot (flat 20).  (No negative
        # assertion here: two different keys can collide onto the same
        # centroid, so "differs from the wrong-slot encode" is not a
        # contract -- "equals the physical-slot encode" is.)
        expect = encode_vectorized(gather(torch.tensor([9], device=torch_device)), centroids)
        torch.testing.assert_close(written, expect[0])
        assert not torch.equal(before, written)
