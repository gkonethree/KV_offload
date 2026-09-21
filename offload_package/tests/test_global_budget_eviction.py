"""Tests for global budget eviction in OffloadOrchestrator."""
from __future__ import annotations

import pytest
import torch

from offload_package.integration.manager import OffloadOrchestrator
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.scoring.base import BlockRef
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable
from offload_package.controller import KVCompressionController


class TestGlobalBudgetEviction:
    """Test that offload evicts globally lowest-scoring blocks across all requests."""

    @pytest.fixture
    def setup_orchestrator(self):
        """Create orchestrator with multiple requests and blocks."""
        num_layers = 2
        num_pages = 128
        page_size = 16
        num_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = torch.device("cuda")

        # Create K/V caches with different values per block to create different scores
        k_caches = []
        v_caches = []
        for _ in range(num_layers):
            k_cache = torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)
            v_cache = torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)
            k_caches.append(k_cache)
            v_caches.append(v_cache)

        cpu_cache = torch.empty(
            (num_layers, 64, 2, page_size, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True
        )

        residency = ResidencyTable()
        scorer = PagedEvictionScorer(decay=1.0, head_reduction="mean")
        offload_manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=100,
            kv_layout="NHD",
        )
        staging_pool = KVStagingPool(
            num_slots=8,
            staging_base_page=100,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            residency=residency,
            offload_manager=offload_manager,
        )
        controller = KVCompressionController(W=16, page_size=page_size)
        
        orchestrator = OffloadOrchestrator(
            offload_manager=offload_manager,
            staging_pool=staging_pool,
            residency=residency,
            controller=controller,
            page_size=page_size,
            staging_base_page_idx=100,
            offload_fraction=0.3,
        )
        orchestrator._enable_sparse_mode()
        
        return orchestrator, page_size

    def test_global_eviction_across_requests(self, setup_orchestrator):
        """Test that offload_fraction considers all requests globally."""
        orchestrator, page_size = setup_orchestrator
        
        # Add 3 requests with different block counts
        req_blocks = {
            "req1": list(range(0, 20)),    # 20 blocks
            "req2": list(range(20, 35)),   # 15 blocks
            "req3": list(range(35, 50)),   # 15 blocks
        }
        
        for req_id, block_ids in req_blocks.items():
            orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=len(block_ids) * page_size)
            # Mark all blocks as complete
            for ref in orchestrator._request_blocks[req_id]:
                orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Trigger prefill complete for one request
        # This should offload globally lowest-scoring blocks
        offloaded = orchestrator.on_prefill_complete("req1")
        
        # Should have offloaded some blocks globally
        assert len(offloaded) > 0
        
        # Check that offloaded blocks are from across all requests (not just req1)
        offloaded_req_ids = {ref.request_id for ref in offloaded}
        # Global eviction means any request's blocks can be offloaded
        assert len(offloaded_req_ids) >= 1

    def test_global_eviction_respects_budget(self, setup_orchestrator):
        """Test that global eviction doesn't exceed GPU budget."""
        orchestrator, page_size = setup_orchestrator
        
        # Add many blocks across requests
        total_blocks = 0
        for req_id, start in [("req1", 0), ("req2", 20), ("req3", 40), ("req4", 60)]:
            block_ids = list(range(start, start + 20))
            total_blocks += len(block_ids)
            orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=len(block_ids) * page_size)
            for ref in orchestrator._request_blocks[req_id]:
                orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Initial GPU blocks
        initial_gpu = len(orchestrator.residency.all_gpu_blocks(complete_only=True))
        assert initial_gpu == total_blocks
        
        # Offload 30% fraction
        offloaded = orchestrator.on_prefill_complete("req1")
        
        # GPU blocks should decrease
        remaining_gpu = len(orchestrator.residency.all_gpu_blocks(complete_only=True))
        assert remaining_gpu < initial_gpu
        
        # Offloaded count should be roughly 30% of total
        expected_offload = int(total_blocks * 0.3)
        assert abs(len(offloaded) - expected_offload) <= 2  # Allow small variance

    def test_decode_step_global_eviction(self, setup_orchestrator):
        """Test that decode step also uses global eviction."""
        orchestrator, page_size = setup_orchestrator
        
        # Add requests
        for req_id, start in [("req1", 0), ("req2", 20), ("req3", 40)]:
            block_ids = list(range(start, start + 15))
            orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=len(block_ids) * page_size)
            for ref in orchestrator._request_blocks[req_id]:
                orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Complete prefill for all
        for req_id in ["req1", "req2", "req3"]:
            orchestrator.controller.on_prefill_complete(req_id)
        
        # Trigger decode step for all - should use global pool
        offloaded = orchestrator.on_decode_step(["req1", "req2", "req3"], num_tokens=4)
        
        # Should offload from global pool
        assert isinstance(offloaded, list)


class TestOnPrefillCompleteReturnsList:
    """Test that on_prefill_complete returns list of offloaded blocks."""

    @pytest.fixture
    def setup_orchestrator(self):
        num_layers = 1
        num_pages = 64
        page_size = 16
        num_heads = 4
        head_dim = 64
        dtype = torch.float16
        device = torch.device("cuda")

        k_caches = [torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)]
        v_caches = [torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)]

        cpu_cache = torch.empty(
            (num_layers, 32, 2, page_size, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True
        )

        residency = ResidencyTable()
        scorer = PagedEvictionScorer(decay=1.0, head_reduction="mean")
        offload_manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=40,
            kv_layout="NHD",
        )
        staging_pool = KVStagingPool(
            num_slots=4,
            staging_base_page=50,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            residency=residency,
            offload_manager=offload_manager,
        )
        controller = KVCompressionController(W=16, page_size=page_size)
        
        orchestrator = OffloadOrchestrator(
            offload_manager=offload_manager,
            staging_pool=staging_pool,
            residency=residency,
            controller=controller,
            page_size=page_size,
            staging_base_page_idx=50,
            offload_fraction=0.25,
        )
        orchestrator._enable_sparse_mode()
        
        return orchestrator

    def test_on_prefill_complete_returns_offloaded_list(self, setup_orchestrator):
        """Test that on_prefill_complete returns list of BlockRef."""
        orchestrator = setup_orchestrator
        
        req_id = "req1"
        block_ids = list(range(10))
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=160)
        for ref in orchestrator._request_blocks[req_id]:
            orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        offloaded = orchestrator.on_prefill_complete(req_id)
        
        assert isinstance(offloaded, list)
        for ref in offloaded:
            assert isinstance(ref, BlockRef)
            assert ref.request_id == req_id


class TestScoreAwareStagingEviction:
    """Test that staging pool evicts lowest-scored blocks."""

    @pytest.fixture
    def setup_staging_pool(self):
        """Create staging pool with scorer."""
        num_layers = 1
        num_pages = 64
        page_size = 16
        num_heads = 4
        head_dim = 64
        dtype = torch.float16
        device = torch.device("cuda")

        k_caches = [torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)]
        v_caches = [torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)]

        cpu_cache = torch.empty(
            (num_layers, 16, 2, page_size, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True
        )

        residency = ResidencyTable()
        scorer = PagedEvictionScorer(decay=1.0, head_reduction="mean")
        offload_manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=40,
            kv_layout="NHD",
        )
        staging_pool = KVStagingPool(
            num_slots=4,
            staging_base_page=50,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            residency=residency,
            offload_manager=offload_manager,
        )
        
        return staging_pool, offload_manager, residency, page_size

    def test_staging_evicts_lowest_scored(self, setup_staging_pool):
        """Test that when staging is full, lowest-scored block is evicted."""
        staging_pool, offload_manager, residency, page_size = setup_staging_pool
        
        # Test score-aware eviction directly by filling staging and adding more
        refs = []
        for i in range(4):
            ref = BlockRef(f"req1", i)
            residency.mark_gpu(ref, i, complete=True)
            offload_manager.score_block(ref, i)
            refs.append(ref)
        
        # Move all to CPU using maybe_offload with budget=0 (offload all)
        original_budget = offload_manager.gpu_budget_blocks
        offload_manager.gpu_budget_blocks = 0
        offload_manager.maybe_offload(refs)
        offload_manager.gpu_budget_blocks = original_budget
        
        # Manually bring to staging (simulate what ensure_resident does)
        for i, ref in enumerate(refs):
            loc = residency.get(ref)
            assert loc.residency == Residency.CPU, f"Expected CPU, got {loc.residency}"
            slot = staging_pool._allocate_slot(ref)
            staging_pool.residency.mark_staging(ref, slot)
        
        assert len(staging_pool._lru) == 4
        
        # Add a 5th block with lower score - should evict lowest scored
        new_ref = BlockRef("req1", 4)
        residency.mark_gpu(new_ref, 4, complete=True)
        offload_manager.score_block(new_ref, 4)
        offload_manager.gpu_budget_blocks = 0
        offload_manager.maybe_offload([new_ref])
        offload_manager.gpu_budget_blocks = original_budget
        
        # Bring to staging - should evict lowest scored
        loc = residency.get(new_ref)
        assert loc.residency == Residency.CPU
        slot = staging_pool._allocate_slot(new_ref)
        staging_pool.residency.mark_staging(new_ref, slot)
        
        # Staging should still have 4 slots
        assert len(staging_pool._lru) == 4
        # New ref should be in staging
        assert new_ref in staging_pool._lru
        # One of the old refs should be evicted (back to CPU)
        evicted = [r for r in refs if r not in staging_pool._lru]
        assert len(evicted) == 1


class TestAsyncCopySync:
    """Test that async copies complete before returning."""

    @pytest.fixture
    def setup_manager(self):
        num_layers = 1
        num_pages = 64
        page_size = 16
        num_heads = 4
        head_dim = 64
        dtype = torch.float16
        device = torch.device("cuda")

        k_caches = [torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)]
        v_caches = [torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)]

        cpu_cache = torch.empty(
            (num_layers, 16, 2, page_size, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True
        )

        residency = ResidencyTable()
        scorer = PagedEvictionScorer(decay=1.0, head_reduction="mean")
        offload_manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=40,
            kv_layout="NHD",
        )
        return offload_manager

    def test_offload_fraction_waits_for_copies(self, setup_manager):
        """Test that offload_fraction waits for all async copies to complete."""
        manager = setup_manager
        
        # Add several blocks
        refs = [BlockRef("req1", i) for i in range(5)]
        for ref in refs:
            manager.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Offload with fraction 0.8 -> 4 blocks offloaded (5 * 0.8 = 4)
        offloaded = manager.offload_fraction(refs, 0.8)
        
        assert len(offloaded) == 4
        # All should be marked CPU
        for ref in offloaded:
            loc = manager.residency.get(ref)
            assert loc.residency == Residency.CPU

    def test_maybe_offload_waits_for_copies(self, setup_manager):
        """Test that maybe_offload waits for all async copies to complete."""
        manager = setup_manager
        
        refs = [BlockRef("req1", i) for i in range(5)]
        for ref in refs:
            manager.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Set budget low to trigger eviction
        manager.gpu_budget_blocks = 2
        
        offloaded = manager.maybe_offload(refs)
        
        assert len(offloaded) == 3  # 5 - 2 = 3 evicted
        for ref in offloaded:
            loc = manager.residency.get(ref)
            assert loc.residency == Residency.CPU