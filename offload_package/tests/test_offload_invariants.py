"""
Invariant tests for KV offload mechanism.

These tests assert externally observable properties that must hold regardless
of internal implementation. Each test is red on the current codebase and
will turn green only after the corresponding fix is applied.
"""

from __future__ import annotations

import pytest
import torch

from offload_package.integration.scheduler_connector import OffloadPackageScheduler
from offload_package.integration.worker_connector import OffloadPackageWorker
from offload_package.offload.manager import KVOffloadManager, CPUBlockPool
from offload_package.scoring.base import BlockRef, BlockKVStats
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable, BlockLocation
from offload_package.controller import KVCompressionController
from offload_package.shared.types import build_orchestrator, normalize_runner_kv_caches, OrchestratorConfig
from offload_package.vllm_adapter import KVCacheView
from offload_package.config import OffloadConfig


class TestTrueFreeAccounting:
    """BlockPool accounting for true GPU block freeing (Bug 1)."""

    def test_offloaded_block_returns_to_free_pool(self, monkeypatch):
        """Offloaded blocks must return to BlockPool free list (ref_cnt == 0)."""
        pytest.skip("Requires vLLM BlockPool integration - run after fix B2")

    def test_no_double_free_at_request_finish(self):
        """After offload+free, request finish must not double-free blocks."""
        pytest.skip("Requires scheduler + BlockPool integration - run after fix B2")

    def test_shared_prefix_block_never_offloaded(self):
        """Prefix-shared blocks (ref_cnt > 1) must be excluded from offload candidates."""
        pytest.skip("Requires orchestrator+scorer+BlockPool integration - run after fix B2")


class TestLayoutIntegrity:
    """Per-layer tensor layout correctness (Bug 4)."""

    def test_normalize_pairs_one_tensor_per_layer(self):
        """normalize_runner_kv_caches must map 1 input tensor -> 1 K/V pair per layer."""
        num_layers = 22
        head_dim = 64
        num_kv_heads = 4
        page_size = 16
        num_blocks = 100

        # Build synthetic combined tensors with unique sentinel per layer
        tensors = []
        for layer in range(num_layers):
            # Combined [num_blocks, H, N, 2*head_dim] with K in [:,:,:,:head_dim], V in [:,:,:,head_dim:]
            t = torch.zeros(num_blocks, num_kv_heads, page_size, 2 * head_dim, dtype=torch.float16)
            # Stamp layer ID in K channel 0, token 0, head 0
            t[:, 0, 0, 0] = float(layer)
            # Stamp layer ID + 100 in V channel 0, token 0, head 0  
            t[:, 0, 0, head_dim] = float(layer + 100)
            tensors.append(t)

        from offload_package.shared.types import normalize_runner_kv_caches

        # Minimal fake KVCacheConfig
        class FakeSpec:
            block_size = 16
            num_kv_heads = 4
            head_size = 64

        class FakeGroup:
            layer_names = [f"layer_{i}" for i in range(num_layers)]
            kv_cache_spec = FakeSpec()

        class FakeConfig:
            kv_cache_groups = [FakeGroup()]
            kv_cache_tensors = []

        view = normalize_runner_kv_caches(tensors, FakeConfig(), staging_slots=0)

        # Must produce exactly 22 layers, 22 K, 22 V
        assert len(view.layers) == num_layers, f"Expected {num_layers} layers, got {len(view.layers)}"
        assert len(view.k_caches) == num_layers, f"Expected {num_layers} k_caches, got {len(view.k_caches)}"
        assert len(view.v_caches) == num_layers, f"Expected {num_layers} v_caches, got {len(view.v_caches)}"

        # Combined layout: k_caches[i] and v_caches[i] are the same tensor, and
        # it must be layer i's tensor (K sentinel at [.., :head_dim], V at [.., head_dim:]).
        for i in range(num_layers):
            assert view.k_caches[i] is view.v_caches[i], f"layer {i}: expected combined tensor shared by k/v"
            k_layer_id = int(view.k_caches[i][0, 0, 0, 0].item())
            v_layer_id = int(view.v_caches[i][0, 0, 0, head_dim].item()) - 100
            assert k_layer_id == i, f"k_caches[{i}] has layer {k_layer_id}, expected {i}"
            assert v_layer_id == i, f"v_caches[{i}] has layer {v_layer_id}, expected {i}"

        # Layout must be HND (fork's [B, H, N, C] page ordering)
        assert view.kv_layout == "HND", f"kv_layout={view.kv_layout}"

    def test_offload_fetch_roundtrip_parity(self):
        """Offload a block to CPU, fetch to staging; staging must equal original page."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.integration.manager import OffloadOrchestrator
        from offload_package.offload.manager import KVOffloadManager
        from offload_package.scoring.paged_eviction import PagedEvictionScorer
        from offload_package.staging.pool import KVStagingPool
        from offload_package.staging.residency import Residency, ResidencyTable
        from offload_package.vllm_adapter import KVCacheView

        num_layers, num_pages = 3, 16
        H, N, C = 2, 16, 64           # heads, page tokens, head_dim
        staging_slots = 4
        staging_base = num_pages - staging_slots
        head_dim = C

        # Combined per-layer tensors [pages, H, N, 2C] with distinct content per layer/page
        combined_layers = []
        for l in range(num_layers):
            t = torch.zeros(num_pages, H, N, 2 * C, dtype=torch.float16, device="cuda")
            for p in range(num_pages):
                t[p].fill_(float(l * 100 + p) / 100.0)
            combined_layers.append(t)
        k_caches = list(combined_layers)
        v_caches = list(combined_layers)
        view = KVCacheView(
            layers=combined_layers,
            k_caches=k_caches,
            v_caches=v_caches,
            block_size=N,
            num_kv_heads=H,
            head_dim=head_dim,
            kv_layout="HND",
            staging_base_page=staging_base,
        )

        cpu_cache = torch.empty(
            (num_layers, 8, 2, N, H, C), dtype=torch.float16,
            device="cpu", pin_memory=True,
        )
        residency = ResidencyTable()
        scorer = PagedEvictionScorer()
        offload_manager = KVOffloadManager(
            scorer=scorer, residency=residency,
            gpu_k_caches=k_caches, gpu_v_caches=v_caches,
            cpu_cache=cpu_cache, gpu_budget_blocks=staging_base, kv_layout="HND",
        )
        staging_pool = KVStagingPool(
            num_slots=staging_slots, staging_base_page=staging_base,
            gpu_k_caches=k_caches, gpu_v_caches=v_caches, cpu_cache=cpu_cache,
            residency=residency, offload_manager=offload_manager,
        )

        ref = BlockRef("req-x", 0)
        original_pages = [combined_layers[l][3].clone() for l in range(num_layers)]
        residency.mark_gpu(ref, 3, complete=True)

        offloaded = offload_manager.offload_fraction([ref], 0.99, {ref})
        assert offloaded == [ref], "block should have been offloaded"
        loc = residency.get(ref)
        assert loc.residency == Residency.CPU and loc.cpu_slot is not None

        staging_map = staging_pool.ensure_resident([ref])
        assert ref in staging_map
        loc = residency.get(ref)
        assert loc.residency == Residency.STAGING

        slot = staging_map[ref]
        for l in range(num_layers):
            staged = combined_layers[l][staging_base + slot]
            assert torch.equal(staged, original_pages[l]), (
                f"layer {l}: staged page content mismatch after offload+fetch"
            )

    def test_scorer_uses_true_kv_norms(self):
        """compute_block_kv_norms must produce per-head mean-over-token norms.

        HND layout [B, H, N, D]: norm over head_dim (D), then mean over tokens
        (N), keeping one value per head per block.
        """
        from offload_package.scoring.norm_kernel import compute_block_kv_norms
        from offload_package.scoring.base import BlockRef

        num_blocks = 1
        num_kv_heads = 4
        page_size = 16
        head_dim = 64

        k_cache = torch.zeros(num_blocks, num_kv_heads, page_size, head_dim, dtype=torch.float32)
        v_cache = torch.zeros(num_blocks, num_kv_heads, page_size, head_dim, dtype=torch.float32)
        k_cache.fill_(1.0)
        v_cache.fill_(2.0)

        stats = compute_block_kv_norms(
            k_cache, v_cache, [0],
            logical_block_ids=[BlockRef("test", 0)],
            layout="HND",
        )

        # Per (block, head): ||K_token|| = sqrt(head_dim) = 8; mean over 16 tokens = 8.
        assert stats.k_norm.shape == (1, num_kv_heads), f"k_norm shape {stats.k_norm.shape}"
        assert stats.v_norm.shape == (1, num_kv_heads), f"v_norm shape {stats.v_norm.shape}"

        expected_k = head_dim ** 0.5          # 8.0
        expected_v = 2.0 * head_dim ** 0.5    # 16.0
        assert torch.allclose(stats.k_norm, torch.full_like(stats.k_norm, expected_k), atol=1e-3)
        assert torch.allclose(stats.v_norm, torch.full_like(stats.v_norm, expected_v), atol=1e-3)

        ratio = stats.v_norm / stats.k_norm
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-3)


class TestStagingReachesAttention:
    """Staged data must be visible to attention kernel (Bug 2)."""

    def test_patch_applied_in_decode_metadata(self):
        """After staging fetch, paged_kv_indices must point to staging_base + slot.

        CPU-only adapter-level test over patch_paged_kv_indices_with_staging:
        staged refs -> staging slots; CPU-resident unstaged refs -> null (when
        true freeing is on); GPU-resident refs -> untouched.
        """
        from offload_package.selection.adapter import patch_paged_kv_indices_with_staging
        from offload_package.staging.residency import ResidencyTable

        residency = ResidencyTable()
        staging_base = 1000
        # One request, 6 blocks (pages) originally at GPU ids 10..15
        gpu_ids = [10, 11, 12, 13, 14, 15]
        for idx, gid in enumerate(gpu_ids):
            residency.mark_gpu(BlockRef("r0", idx), gid, complete=True)

        # Block 2 offloaded (CPU, slot 7); block 4 offloaded then staged (slot 5).
        residency.mark_cpu(BlockRef("r0", 2), cpu_slot=7)
        residency.mark_cpu(BlockRef("r0", 4), cpu_slot=3)
        residency.mark_staging(BlockRef("r0", 4), staging_slot=5)

        indices = torch.tensor(gpu_ids, dtype=torch.int32)
        indptr = torch.tensor([0, 6], dtype=torch.int32)

        staged = patch_paged_kv_indices_with_staging(
            indices, indptr, ["r0"], sparse_idx=None, sparse_len=None,
            page_size=16, staging_base_page_idx=staging_base,
            residency=residency, staging_map={},
            null_block_id=0,
        )

        assert int(staged[0]) == 10 and int(staged[1]) == 11          # GPU stays
        assert int(staged[2]) == 0                                    # CPU -> null
        assert int(staged[3]) == 13                                   # GPU stays
        assert int(staged[4]) == staging_base + 5                     # staged slot
        assert int(staged[5]) == 15                                   # GPU stays

        # Without null fill (shadow mode), CPU-resident ids stay as-is.
        nofill = patch_paged_kv_indices_with_staging(
            indices, indptr, ["r0"], sparse_idx=None, sparse_len=None,
            page_size=16, staging_base_page_idx=staging_base,
            residency=residency, staging_map={},
            null_block_id=None,
        )
        assert int(nofill[2]) == 12                                   # CPU -> unchanged
        assert int(nofill[4]) == staging_base + 5                     # staged slot still applied

    def test_sparse_output_matches_dense_on_offloaded_blocks(self):
        """Oracle wrapper output on staged blocks must match dense reference."""
        pytest.skip("Requires kernel-level integration - run after fix B3")

    def test_staging_fetch_visible_before_attention(self):
        """Fetch copy event must complete before attention kernel launch."""
        pytest.skip("Requires CUDA event integration - run after fix B3")


class TestAllocatorIsolation:
    """Staging tail must not overlap allocator space (Bug 5)."""

    def test_staging_tail_outside_allocator(self):
        """Reserved tail pages must never be handed to a request by BlockPool."""
        from vllm.v1.core.block_pool import BlockPool
        from offload_package.cache_allocator import reserve_staging_tail

        pool = BlockPool(num_gpu_blocks=100, enable_caching=False, hash_block_size=16)
        # 100 blocks; null_block (id 0) is popped at init -> 99 allocatable.
        reserved = reserve_staging_tail(pool, 8)
        assert reserved == list(range(92, 100))
        assert pool.get_num_free_blocks() == 91

        # Drain the pool: the reserved tail must never appear in an allocation.
        got = pool.get_new_blocks(91)
        got_ids = {b.block_id for b in got}
        assert got_ids.isdisjoint(reserved)
        assert max(got_ids) < 92

        # One more allocation must fail (reserved blocks are truly unavailable).
        with pytest.raises(ValueError):
            pool.get_new_blocks(1)


class TestPromotionWatch:
    """R2 promotion: highest-scored CPU-resident blocks get pre-staged."""

    def _orch_with_cpu_blocks(self, n_cpu=6):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.integration.manager import OffloadOrchestrator
        from offload_package.offload.manager import KVOffloadManager
        from offload_package.scoring.paged_eviction import PagedEvictionScorer
        from offload_package.scoring.base import BlockKVStats
        from offload_package.staging.pool import KVStagingPool

        num_layers, num_pages = 1, 64
        H, N, C = 2, 16, 64
        staging_slots = 4
        staging_base = num_pages - staging_slots

        combined = torch.randn(num_pages, H, N, 2 * C, dtype=torch.float16, device="cuda")
        cpu_cache = torch.empty(
            (num_layers, 32, 2, N, H, C), dtype=torch.float16, device="cpu", pin_memory=True
        )
        residency = ResidencyTable()
        scorer = PagedEvictionScorer()
        offload_manager = KVOffloadManager(
            scorer=scorer, residency=residency,
            gpu_k_caches=[combined], gpu_v_caches=[combined],
            cpu_cache=cpu_cache, gpu_budget_blocks=staging_base, kv_layout="HND",
        )
        staging_pool = KVStagingPool(
            num_slots=staging_slots, staging_base_page=staging_base,
            gpu_k_caches=[combined], gpu_v_caches=[combined],
            cpu_cache=cpu_cache, residency=residency, offload_manager=offload_manager,
        )
        from offload_package.controller import KVCompressionController
        orch = OffloadOrchestrator(
            offload_manager=offload_manager, staging_pool=staging_pool,
            residency=residency, controller=KVCompressionController(16, 16),
            page_size=N, staging_base_page_idx=staging_base, offload_fraction=0.5,
        )

        refs = [BlockRef("req1", i) for i in range(n_cpu)]
        # scores ascending in block index: block i has score i+1
        stats = BlockKVStats(
            v_norm=torch.arange(1, n_cpu + 1, dtype=torch.float32).unsqueeze(1),
            k_norm=torch.ones(n_cpu, 1, dtype=torch.float32),
            block_ids=refs,
        )
        scorer.update(stats)
        for i, ref in enumerate(refs):
            residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        return orch, residency, refs, staging_base

    @staticmethod
    def _cpu_mark(residency, refs):
        for i, ref in enumerate(refs):
            residency.mark_cpu(ref, cpu_slot=i)

    def test_promote_picks_highest_scored(self):
        orch, residency, refs, staging_base = self._orch_with_cpu_blocks(6)
        self._cpu_mark(residency, refs)
        n0 = residency.num_cpu_blocks
        staged = orch.promote_hot_cpu_blocks("req1", max_count=2)
        assert len(staged) == 2
        # The two highest-scored blocks (idx 5, 4) must be staged
        assert BlockRef("req1", 5) in staged
        assert BlockRef("req1", 4) in staged
        assert BlockRef("req1", 0) not in staged
        assert residency.num_cpu_blocks == n0 - 2
        for ref, slot in staged.items():
            loc = residency.get(ref)
            assert loc.residency == Residency.STAGING
            assert loc.staging_slot == slot
            assert 0 <= slot < 4  # within pool

    def test_promote_noop_when_nothing_cold(self):
        orch, residency, refs, _ = self._orch_with_cpu_blocks(3)
        self._cpu_mark(residency, refs)
        for ref in refs:  # flip all to staging
            residency.mark_staging(ref, staging_slot=ref.block_idx)
        assert residency.num_cpu_blocks == 0
        assert orch.promote_hot_cpu_blocks("req1", 2) == {}

    def test_promote_bounded_by_max_count(self):
        orch, residency, refs, _ = self._orch_with_cpu_blocks(6)
        self._cpu_mark(residency, refs)
        staged = orch.promote_hot_cpu_blocks("req1", max_count=1)
        assert len(staged) == 1
        assert BlockRef("req1", 5) in staged

    def test_promote_zero_disables(self):
        orch, residency, refs, _ = self._orch_with_cpu_blocks(3)
        self._cpu_mark(residency, refs)
        assert orch.promote_hot_cpu_blocks("req1", max_count=0) == {}
        assert residency.num_cpu_blocks == 3

    def test_promoted_content_matches_original(self):
        """A promoted block's staging page must equal the original GPU page."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        orch, _, refs, staging_base = self._orch_with_cpu_blocks(8)
        # Real offload (GPU->CPU copy) via offload_fraction on half the blocks.
        offloaded = orch.offload_manager.offload_fraction(refs, 0.5, set(refs))
        assert offloaded  # some went to CPU for real
        originals = {}
        combined = orch.staging_pool.gpu_k_caches[0]
        for ref in offloaded:
            originals[ref] = combined[ref.block_idx].clone()

        staged = orch.promote_hot_cpu_blocks("req1", max_count=4)
        assert staged
        for ref, slot in staged.items():
            staged_page = combined[staging_base + slot]
            assert torch.equal(staged_page, originals[ref]), (
                f"{ref}: staged page content mismatch after offload+promote"
            )


class TestLeakFreeLifecycle:
    """Preemption and request finish must not leak CPU slots (Bug 6)."""

    def test_preemption_releases_cpu_slots(self):
        """Preempted request's CPU slots + residency + scorer entries must be freed."""
        pytest.skip("Requires scheduler preemption hook - run after fix B6")


class TestEndToEndGuard:
    """Full integration smoke test."""

    def test_offload_integrated_output_equality(self):
        """Greedy output with offload must equal dense baseline (token-for-token)."""
        pytest.skip("End-to-end validation - run after all fixes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])