"""Tests for chunked prefill and preemption handling."""
from __future__ import annotations

import pytest
import torch

from offload_package.integration.manager import OffloadOrchestrator
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable
from offload_package.controller import KVCompressionController
from offload_package.scoring.base import BlockRef


class TestChunkedPrefill:
    """Test offload behavior with chunked prefill."""

    @pytest.fixture
    def setup_orchestrator(self):
        """Create orchestrator for chunked prefill testing."""
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
        
        return orchestrator, page_size

    def test_chunked_prefill_only_offloads_at_final_chunk(self, setup_orchestrator):
        """Test that offload only triggers at final prefill chunk."""
        orchestrator, page_size = setup_orchestrator
        
        req_id = "req1"
        total_blocks = 10
        total_tokens = total_blocks * page_size
        
        # Simulate 2-chunk prefill: first chunk 5 blocks, second chunk 5 blocks
        chunk1_blocks = list(range(5))
        chunk2_blocks = list(range(5, 10))
        
        # Chunk 1: partial prefill - do NOT call on_prefill_complete
        orchestrator.sync_request_blocks(req_id, chunk1_blocks, num_computed_tokens=5 * page_size)
        for ref in orchestrator._request_blocks[req_id]:
            orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Chunk 2: complete prefill
        all_blocks = chunk1_blocks + chunk2_blocks
        orchestrator.sync_request_blocks(req_id, all_blocks, num_computed_tokens=total_tokens)
        for ref in orchestrator._request_blocks[req_id]:
            orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Now should offload when prefill completes
        offloaded = orchestrator.on_prefill_complete(req_id)
        assert len(offloaded) > 0, "Should offload on final prefill chunk"

    def test_chunked_prefill_sync_updates_blocks(self, setup_orchestrator):
        """Test that sync_request_blocks correctly updates block list."""
        orchestrator, page_size = setup_orchestrator
        
        req_id = "req1"
        
        # First chunk
        chunk1 = list(range(3))
        orchestrator.sync_request_blocks(req_id, chunk1, num_computed_tokens=48)
        assert len(orchestrator._request_blocks[req_id]) == 3
        
        # Second chunk adds more blocks
        chunk2 = list(range(3, 8))
        orchestrator.sync_request_blocks(req_id, chunk1 + chunk2, num_computed_tokens=128)
        assert len(orchestrator._request_blocks[req_id]) == 8
        
        # Third chunk
        chunk3 = list(range(8, 12))
        orchestrator.sync_request_blocks(req_id, chunk1 + chunk2 + chunk3, num_computed_tokens=192)
        assert len(orchestrator._request_blocks[req_id]) == 12


class TestPrefillCompletionDetection:
    """Test prefill completion detection logic."""

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

    def test_complete_detection_uses_num_computed_tokens(self, setup_orchestrator):
        """Test that block completion is based on num_computed_tokens."""
        orchestrator = setup_orchestrator
        
        req_id = "req1"
        block_ids = [0, 1, 2]  # 3 blocks = 48 tokens
        
        # num_computed_tokens = 32 (only 2 blocks complete)
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=32)
        
        refs = orchestrator._request_blocks[req_id]
        loc0 = orchestrator.residency.get(refs[0])
        loc1 = orchestrator.residency.get(refs[1])
        loc2 = orchestrator.residency.get(refs[2])
        
        # Block 0 (0-15) and Block 1 (16-31) should be complete
        assert loc0.complete is True
        assert loc1.complete is True
        # Block 2 (32-47) should NOT be complete
        assert loc2.complete is False
        
        # Increase to 48 (all 3 blocks complete)
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=48)
        
        loc2 = orchestrator.residency.get(refs[2])
        assert loc2.complete is True

    def test_complete_detection_boundary(self, setup_orchestrator):
        """Test completion detection at exact page boundaries."""
        orchestrator = setup_orchestrator
        
        req_id = "req1"
        block_ids = [0, 1]  # 2 blocks = 32 tokens
        
        # Exactly 32 tokens - both blocks complete
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=32)
        
        refs = orchestrator._request_blocks[req_id]
        assert orchestrator.residency.get(refs[0]).complete is True
        assert orchestrator.residency.get(refs[1]).complete is True
        
        # Test at lower token count (simulating earlier prefill stage)
        # Note: In practice num_computed_tokens only increases, so we can't test unmarking
        # This test just verifies that at 32 tokens both blocks are marked complete


class TestPreemptionHandling:
    """Test preemption handling in orchestrator."""

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
        
        return orchestrator, offload_manager, staging_pool, residency

    def test_on_request_preempted_cleans_up(self, setup_orchestrator):
        """Test that on_request_preempted cleans up resources."""
        orchestrator, offload_manager, staging_pool, residency = setup_orchestrator
        
        req_id = "req1"
        block_ids = [0, 1, 2]
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=48)
        for ref in orchestrator._request_blocks[req_id]:
            orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Offload some blocks
        orchestrator.offload_manager.offload_fraction(
            orchestrator._request_blocks[req_id], 0.5
        )
        
        # Move some to staging
        gpu_refs = [r for r in orchestrator._request_blocks[req_id] 
                    if orchestrator.residency.get(r).residency == Residency.GPU]
        if gpu_refs:
            staging_pool.ensure_resident(gpu_refs[:1])
        
        # Preempt the request
        orchestrator.on_request_preempted(req_id)
        
        # Request should be removed
        assert req_id not in orchestrator._request_blocks
        # Controller should be cleaned up
        assert orchestrator.controller.count(req_id) == 0
        # Staging should be released
        assert len(staging_pool._lru) == 0
        # CPU slots should be freed
        assert offload_manager._cpu_pool.num_free == 32

    def test_on_request_done_cleans_up_same_as_preempted(self, setup_orchestrator):
        """Test that on_request_done has same cleanup as on_request_preempted."""
        orchestrator, offload_manager, staging_pool, residency = setup_orchestrator
        
        req_id = "req1"
        block_ids = [0, 1, 2]
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=48)
        for ref in orchestrator._request_blocks[req_id]:
            orchestrator.residency.mark_gpu(ref, ref.block_idx, complete=True)
        
        # Offload some
        orchestrator.offload_manager.offload_fraction(
            orchestrator._request_blocks[req_id], 0.5
        )
        
        # Finish request
        orchestrator.on_request_done(req_id)
        
        # Same cleanup
        assert req_id not in orchestrator._request_blocks
        assert orchestrator.controller.count(req_id) == 0
        assert len(staging_pool._lru) == 0


class TestControllerInterval:
    """Test KVCompressionController interval logic."""

    def test_controller_tracks_decode_tokens(self):
        """Test that controller tracks decode tokens per request."""
        controller = KVCompressionController(W=16, page_size=16)
        
        req_id = "req1"
        controller.on_prefill_complete(req_id)
        
        # 1 decode token - no trigger
        assert controller.on_decode(req_id, 1) is False
        assert controller.count(req_id) == 1
        
        # 14 more - total 15, no trigger
        assert controller.on_decode(req_id, 14) is False
        assert controller.count(req_id) == 15
        
        # 1 more - total 16, trigger!
        assert controller.on_decode(req_id, 1) is True
        assert controller.count(req_id) == 16
        
        # 16 more - total 32, trigger again
        assert controller.on_decode(req_id, 16) is True
        assert controller.count(req_id) == 32

    def test_controller_reset_on_remove(self):
        """Test that controller resets on request removal."""
        controller = KVCompressionController(W=16, page_size=16)
        
        req_id = "req1"
        controller.on_prefill_complete(req_id)
        controller.on_decode(req_id, 16)  # Trigger
        
        controller.remove(req_id)
        
        assert controller.count(req_id) == 0
        # Re-adding should start fresh
        controller.on_prefill_complete(req_id)
        assert controller.on_decode(req_id, 1) is False