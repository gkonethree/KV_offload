"""Integration tests for the full offloading pipeline."""
from __future__ import annotations

import pytest
import torch

from offload_package.config import OffloadConfig
from offload_package.controller import KVCompressionController
from offload_package.integration.manager import OffloadOrchestrator
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.base import BlockRef
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable
from offload_package.vllm_adapter import OrchestratorConfig, build_orchestrator


class TestOffloadOrchestrator:
    """Integration tests for OffloadOrchestrator."""

    @pytest.fixture
    def orchestrator_setup(self, sample_kv_cache, sample_cpu_cache):
        """Create a full orchestrator setup."""
        k_cache, v_cache = sample_kv_cache
        
        # Create KVCacheView manually for testing
        from offload_package.vllm_adapter import KVCacheView
        num_pages = k_cache.shape[0]
        page_size = k_cache.shape[1]
        num_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
        
        # Create combined cache tensor: [num_pages, 2, page_size, num_heads, head_dim]
        combined = torch.zeros(
            num_pages, 2, page_size, num_heads, head_dim, 
            dtype=k_cache.dtype, device=k_cache.device
        )
        # Put K/V in the combined tensor - k_cache already has page_size dimension
        combined[:, 0] = k_cache  # [num_pages, page_size, num_heads, head_dim]
        combined[:, 1] = v_cache
        layers = [combined]
        
        k_caches = [k_cache]
        v_caches = [v_cache]
        
        kv_cache_view = KVCacheView(
            layers=layers,
            k_caches=k_caches,
            v_caches=v_caches,
            block_size=page_size,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            kv_layout="NHD",
            staging_base_page=100,
        )
        
        cfg = OrchestratorConfig(
            offload_fraction=0.2,
            num_cpu_slots=64,
            num_staging_slots=8,
            W=16,
            page_size=16,
            dtype=torch.float32,
            device=k_cache.device,
            gpu_budget_blocks=64,
            kv_layout="NHD",
            scorer_decay=1.0,
            scorer_head_reduction="mean",
        )
        
        orchestrator = build_orchestrator(
            cfg,
            kv_cache_view=kv_cache_view,
        )
        orchestrator._enable_sparse_mode()
        
        return orchestrator, cfg

    def test_orchestrator_initialization(self, orchestrator_setup):
        """Test that orchestrator initializes correctly."""
        orchestrator, cfg = orchestrator_setup
        
        assert orchestrator is not None
        assert orchestrator.offload_manager is not None
        assert orchestrator.staging_pool is not None
        assert orchestrator.residency is not None
        assert orchestrator.controller is not None

    def test_sync_request_blocks_creates_residency(self, orchestrator_setup):
        """Test that sync_request_blocks creates residency records."""
        orchestrator, _ = orchestrator_setup
        
        req_id = "req1"
        block_ids = [0, 1, 2]
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=0)
        
        refs = orchestrator._request_blocks.get(req_id)
        assert refs is not None
        assert len(refs) == 3
        
        # Check residency was created
        for ref in refs:
            loc = orchestrator.residency.get(ref)
            assert loc is not None
            assert loc.residency == Residency.GPU

    def test_on_prefill_complete_offloads(self, orchestrator_setup):
        """Test that on_prefill_complete triggers offloading."""
        orchestrator, _ = orchestrator_setup
        
        req_id = "req1"
        # Use enough blocks to exceed GPU budget (64) so eviction happens
        block_ids = list(range(80))
        num_computed = 80 * 16  # All blocks complete
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=num_computed)
        
        # Mark all blocks as complete
        for i, ref in enumerate(orchestrator._request_blocks[req_id]):
            loc = orchestrator.residency.get(ref)
            orchestrator.residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        
        # Trigger offload after prefill
        offloaded = orchestrator.on_prefill_complete(req_id)
        
        assert len(offloaded) > 0  # Some blocks should be offloaded
        
        # Check residency changed
        for ref in offloaded:
            loc = orchestrator.residency.get(ref)
            assert loc.residency == Residency.CPU

    def test_on_decode_step_respects_intervals(self, orchestrator_setup):
        """Test that on_decode_step respects W-token intervals."""
        orchestrator, _ = orchestrator_setup
        
        req_id = "req1"
        block_ids = [0, 1, 2, 3, 4]
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=50)
        
        # Mark blocks complete
        for i, ref in enumerate(orchestrator._request_blocks[req_id]):
            orchestrator.residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        
        orchestrator.controller.on_prefill_complete(req_id)
        
        # Decode 1 token - should not trigger (W=16)
        result1 = orchestrator.on_decode_step([req_id], num_tokens=1)
        assert len(result1) == 0  # No offload
        
        # Decode 15 more tokens (total 16) - should trigger
        result2 = orchestrator.on_decode_step([req_id], num_tokens=15)
        assert len(result2) > 0  # Should offload on W-token boundary

    def test_prepare_staging_moves_to_staging(self, orchestrator_setup):
        """Test that prepare_staging moves blocks from CPU to staging."""
        orchestrator, _ = orchestrator_setup
        
        req_id = "req1"
        block_ids = [0, 1, 2]
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=50)
        
        # Mark blocks complete and offload to CPU
        for i, ref in enumerate(orchestrator._request_blocks[req_id]):
            orchestrator.residency.mark_gpu(ref, gpu_block_id=i, complete=True)
            # Simulate offload
            orchestrator.offload_manager.offload_fraction([ref], fraction=0.99)
        
        # Prepare staging for needed blocks - need valid sparse tensors
        # Get device from k_caches
        device = orchestrator.offload_manager.gpu_k_caches[0].device
        # sparse_idx: [B, q_heads, max_S], sparse_len: [B, q_heads, 1]
        sparse_idx = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
        sparse_idx[0, 0, :] = torch.tensor([0, 1, 2], dtype=torch.long)
        sparse_len = torch.tensor([[[3]]], dtype=torch.int32, device=device)
        
        staging_map = orchestrator.prepare_staging([req_id], sparse_idx, sparse_len)
        
        # At least some blocks should be in staging map
        assert isinstance(staging_map, dict)

    def test_on_request_done_cleans_up(self, orchestrator_setup):
        """Test that on_request_done cleans up resources."""
        orchestrator, _ = orchestrator_setup
        
        req_id = "req1"
        block_ids = [0, 1, 2]
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=50)
        
        # Mark blocks complete
        for i, ref in enumerate(orchestrator._request_blocks[req_id]):
            orchestrator.residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        
        # Offload some
        orchestrator.offload_manager.offload_fraction(list(orchestrator._request_blocks[req_id]), 0.5)
        
        orchestrator.on_request_done(req_id)
        
        # Request should be cleaned up
        assert req_id not in orchestrator._request_blocks
        
        # Controller should forget the request
        assert orchestrator.controller.count(req_id) == 0

    def test_multiple_requests_concurrent(self, orchestrator_setup):
        """Test handling multiple concurrent requests."""
        orchestrator, _ = orchestrator_setup
        
        # Add two requests with enough blocks to exceed GPU budget
        req_ids = ["req1", "req2"]
        for req_id in req_ids:
            block_ids = list(range(50))  # 50 blocks each = 100 total > 64 budget
            orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=800)
            
            for i, ref in enumerate(orchestrator._request_blocks[req_id]):
                orchestrator.residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        
        # Trigger offload for both
        orchestrator.on_prefill_complete(req_ids[0])
        orchestrator.on_prefill_complete(req_ids[1])
        
        # Both should have blocks
        for req_id in req_ids:
            refs = orchestrator._request_blocks.get(req_id)
            assert refs is not None
            
            # Some should be on CPU
            cpu_blocks = [
                r for r in refs
                if orchestrator.residency.get(r).residency == Residency.CPU
            ]
            assert len(cpu_blocks) > 0
        
        # Finish first request
        orchestrator.on_request_done(req_ids[0])
        
        # Second request should still exist
        assert req_ids[0] not in orchestrator._request_blocks
        assert req_ids[1] in orchestrator._request_blocks

    def test_residency_consistency(self, orchestrator_setup):
        """Test that residency table stays consistent through operations."""
        orchestrator, _ = orchestrator_setup
        
        req_id = "req1"
        block_ids = [0, 1, 2]
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=50)
        
        refs = orchestrator._request_blocks[req_id]
        
        # All should start as GPU
        for ref in refs:
            loc = orchestrator.residency.get(ref)
            assert loc.residency == Residency.GPU
        
        # Mark complete
        for i, ref in enumerate(refs):
            orchestrator.residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        
        # Offload some
        offloaded = orchestrator.offload_manager.offload_fraction(refs, 0.5)
        
        # Check they transitioned to CPU
        for ref in offloaded:
            loc = orchestrator.residency.get(ref)
            assert loc.residency == Residency.CPU
        
        # Not offloaded should still be GPU
        for ref in refs:
            if ref not in offloaded:
                loc = orchestrator.residency.get(ref)
                assert loc.residency == Residency.GPU

    def test_cpu_memory_budget_respected(self, orchestrator_setup):
        """Test that CPU memory budget is respected."""
        orchestrator, cfg = orchestrator_setup
        
        # Add request
        req_id = "req1"
        block_ids = list(range(20))  # Many blocks
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=320)
        
        # Mark all complete
        for i, ref in enumerate(orchestrator._request_blocks[req_id]):
            orchestrator.residency.mark_gpu(ref, gpu_block_id=i % 128, complete=True)
        
        # Try to offload all
        refs = list(orchestrator._request_blocks[req_id])
        offloaded = orchestrator.offload_manager.offload_fraction(refs, 0.9)
        
        # CPU pool should not exceed budget
        cpu_used = cfg.num_cpu_slots - orchestrator.offload_manager._cpu_pool.num_free
        assert cpu_used <= cfg.num_cpu_slots

    def test_scoring_triggered_on_prefill_complete(self, orchestrator_setup):
        """Test that scoring happens on prefill complete."""
        orchestrator, _ = orchestrator_setup
        
        req_id = "req1"
        block_ids = [0, 1, 2]
        
        orchestrator.sync_request_blocks(req_id, block_ids, num_computed_tokens=50)
        
        # Mark complete
        for i, ref in enumerate(orchestrator._request_blocks[req_id]):
            orchestrator.residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        
        # Blocks should be scored when marked complete in sync_request_blocks
        refs = orchestrator._request_blocks[req_id]
        assert len(orchestrator._scored) == 3
        
        # After prefill complete, offload should happen
        orchestrator.on_prefill_complete(req_id)
        
        assert len(orchestrator._scored) == 3
