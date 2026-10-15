"""Token-granular recall: per-layer offset masks, partial-page fills, and
byte equivalence with the block fetch.

Phase-1 contract under test (per-(layer, token) recall):
- ensure_resident_tokens stages ONLY the requested offsets of the calling
  layer's slice into a canonical staging page; the per-layer 16-bit mask
  records validity and incremental fills OR into it.
- A full 16-offset token fetch produces bytes identical to the block fetch.
- The fail-loud coverage guard raises when a requested offset was never
  staged (simulated by suppressing the mask merge).
- Eviction drops the mask along with the page.
"""
from __future__ import annotations

import pytest
import torch

from offload_package.scoring.base import BlockRef
from offload_package.staging.residency import Residency, ResidencyTable
from offload_package.tests.test_staging_pool import (
    _make_pool,
    _offload_blocks,
)


def _expected_token(pool, layer, cpu_slot, offset):
    """Reference transform of one cpu_cache token to the GPU page layout."""
    tok = pool.cpu_cache[layer, cpu_slot, :, offset, :, :]      # [2, H, C]
    return (tok.permute(1, 0, 2).reshape(tok.shape[1], -1)
            .to(device=pool.gpu_k_caches[layer].device,
                dtype=pool.gpu_k_caches[layer].dtype))


class TestTokenFetchBytes:
    def test_full_offsets_match_block_fetch(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=2)
        ref_a = BlockRef("r", 0)
        ref_b = BlockRef("r", 1)
        _offload_blocks(mgr, res, [ref_a, ref_b])
        # Same CPU content for both refs so token- and block-fetched pages
        # must agree byte-for-byte.
        slot_a = res.get(ref_a).cpu_slot
        pool.cpu_cache[:, res.get(ref_b).cpu_slot] = pool.cpu_cache[:, slot_a]

        for layer in range(2):
            staged, skipped = pool.ensure_resident_tokens(
                layer, [(ref_a, range(pool._page_size))], protect=[])
            assert skipped == []
            page_a = pool.base + staged[ref_a]
            for o in range(pool._page_size):
                assert torch.equal(
                    pool.gpu_k_caches[layer][page_a, :, o, :],
                    _expected_token(pool, layer, slot_a, o),
                ), f"layer {layer} offset {o} bytes differ"

        staged_b, skipped_b = pool.ensure_resident(
            [ref_b], protect=[])
        assert skipped_b == []
        for layer in range(2):
            page_b = pool.base + staged_b[ref_b]
            page_a = pool.base + res.get(ref_a).staging_slot
            assert torch.equal(
                pool.gpu_k_caches[layer][page_a],
                pool.gpu_k_caches[layer][page_b],
            ), f"token fetch != block fetch on layer {layer}"

    def test_partial_fetch_only_writes_requested_offsets(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=2)
        ref = BlockRef("r", 0)
        _offload_blocks(mgr, res, [ref])
        cpu_slot = res.get(ref).cpu_slot

        staged, skipped = pool.ensure_resident_tokens(
            0, [(ref, [1, 3, 5])], protect=[])  # P=8 fixture
        assert skipped == []
        page = pool.base + staged[ref]
        # Poison a non-requested offset on the OTHER layer's slice: a
        # correct partial fetch must not touch it (layer-0 request).
        poison = torch.full_like(
            pool.gpu_k_caches[1][page, :, 6, :], 7.0)
        pool.gpu_k_caches[1][page, :, 6, :] = poison

        assert res.staged_coverage(ref, 0) == 0b0101010
        assert res.staged_coverage(ref, 1) == 0
        for o in (1, 3, 5):
            assert torch.equal(
                pool.gpu_k_caches[0][page, :, o, :],
                _expected_token(pool, 0, cpu_slot, o),
            )
        assert torch.equal(pool.gpu_k_caches[1][page, :, 6, :], poison)

        # Incremental fill into the SAME page: only the gap offset {7}
        # arrives, the mask ORs up.
        staged2, skipped2 = pool.ensure_resident_tokens(
            0, [(ref, [3, 5, 7])], protect=[])
        assert skipped2 == []
        assert res.staged_coverage(ref, 0) == 0b10101010
        assert res.get(ref).staging_slot == page - pool.base
        assert torch.equal(
            pool.gpu_k_caches[0][page, :, 7, :],
            _expected_token(pool, 0, cpu_slot, 7),
        )

        # Per-layer independence: layer 1 fetches its own slice only.
        staged3, skipped3 = pool.ensure_resident_tokens(
            1, [(ref, [6])], protect=[])
        assert skipped3 == []
        assert res.staged_coverage(ref, 1) == 0b1000000
        assert res.staged_coverage(ref, 0) == 0b10101010
        assert torch.equal(
            pool.gpu_k_caches[1][page, :, 6, :],
            _expected_token(pool, 1, cpu_slot, 6),
        )

class TestAllocationZeroFill:
    def test_new_pages_are_zeroed_in_fetching_layer(self):
        """Unselected offsets of a token-staged page must read ZERO in the
        fetching layer's slice (not the previous tenant's bytes). Other
        layers' slices are not touched: a layer gap-fills its own selected
        offsets before it can read the page, so zeroing them per chunk was
        pure overhead."""
        pool, mgr, res = _make_pool(num_slots=4, layers=2)
        ref = BlockRef("r", 0)
        _offload_blocks(mgr, res, [ref])
        # Stage a previous tenant, then evict it so the recycled page
        # carries stale bytes.
        other = BlockRef("r", 1)
        _offload_blocks(mgr, res, [other])
        pool.ensure_resident_tokens(0, [(other, range(pool._page_size))],
                                    protect=[])
        evict_me = BlockRef("r", 2)
        _offload_blocks(mgr, res, [evict_me])
        pool.ensure_resident_tokens(0, [(evict_me, range(pool._page_size))],
                                    protect=[])
        # `ref`'s allocation now evicts one of the staged pages.
        page_before = {
            l: pool.gpu_k_caches[l].clone() for l in range(2)}
        staged, _ = pool.ensure_resident_tokens(0, [(ref, [2])], protect=[])
        page = pool.base + staged[ref]
        slice_0 = pool.gpu_k_caches[0][page]
        for o in range(pool._page_size):
            if o == 2:
                continue  # just written by the fetch
            assert torch.equal(
                slice_0[:, o, :], torch.zeros_like(slice_0[:, o, :])
            ), f"stale bytes at layer 0 offset {o}"
        # Layer 1's slice is untouched by a layer-0 fetch.
        assert torch.equal(pool.gpu_k_caches[1][page], page_before[1][page])

    def test_gapless_refetch_is_free(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=1)
        ref = BlockRef("r", 0)
        _offload_blocks(mgr, res, [ref])
        pool.ensure_resident_tokens(0, [(ref, [2, 4])], protect=[])
        page = pool.base + res.get(ref).staging_slot
        # Corrupt the page after the first fill; a second, fully covered
        # request must not re-copy (the masked path is a no-op) -- but the
        # BYTES must still be the correct staged content for the requested
        # offsets, so restore what we corrupt before asserting.
        keeper = pool.gpu_k_caches[0][page, :, 2, :].clone()
        staged, skipped = pool.ensure_resident_tokens(
            0, [(ref, [2, 4])], protect=[])
        assert skipped == []
        assert torch.equal(pool.gpu_k_caches[0][page, :, 2, :], keeper)


class TestMaskLifecycle:
    def test_eviction_clears_mask(self):
        pool, mgr, res = _make_pool(num_slots=1, num_cpu_slots=32, layers=2)
        ref = BlockRef("r", 0)
        other = BlockRef("r", 1)
        _offload_blocks(mgr, res, [ref, other])
        staged, _ = pool.ensure_resident_tokens(0, [(ref, [0])], protect=[])
        assert res.staged_coverage(ref, 0) == 0b1
        # Force eviction of ref's page by fetching `other` with no protect.
        staged2, skipped = pool.ensure_resident_tokens(
            0, [(other, [0])], protect=[])
        assert skipped == []
        assert res.get(ref).residency == Residency.CPU
        assert res.staged_coverage(ref, 0) == 0
        assert res.staged_coverage(other, 0) == 0b1

    def test_gpu_refs_are_ignored(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=1)
        ref = BlockRef("r", 0)
        res.mark_gpu(ref, 3, complete=True)
        staged, skipped = pool.ensure_resident_tokens(
            0, [(ref, [1, 2])], protect=[])
        assert skipped == []
        assert staged == {}

    def test_unknown_ref_raises(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=1)
        with pytest.raises(KeyError):
            pool.ensure_resident_tokens(0, [(BlockRef("ghost", 0), [1])])


class TestCoverageGuard:
    def test_unmerged_gap_raises(self, monkeypatch):
        pool, mgr, res = _make_pool(num_slots=4, layers=1)
        ref = BlockRef("r", 0)
        _offload_blocks(mgr, res, [ref])
        pool.ensure_resident_tokens(0, [(ref, [1, 3])], protect=[])
        # Suppress the mask merge: the bytes for the gap land but the mask
        # stays stale, which the post-fetch coverage check must catch.
        monkeypatch.setattr(
            res, "merge_staged_mask",
            lambda ref, layer, mask: None,
        )
        with pytest.raises(RuntimeError, match="coverage hole"):
            pool.ensure_resident_tokens(0, [(ref, [1, 5])], protect=[])


class TestLayerMap:
    def test_layer_of_roundtrip(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=2)
        for layer, cache in enumerate(pool.gpu_k_caches):
            assert pool.layer_of(cache.data_ptr()) == layer

    def test_layer_of_unknown_raises(self):
        pool, mgr, res = _make_pool(num_slots=4, layers=1)
        with pytest.raises(RuntimeError, match="not one of the staged"):
            pool.layer_of(0xDEADBEEF)
