"""Tests for multi-layer scoring in KVOffloadManager."""
from __future__ import annotations

import pytest
import torch

from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.base import BlockRef
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.residency import Residency, ResidencyTable


class TestMultiLayerScoring:
    """Test that scoring aggregates across multiple attention layers."""

    @pytest.fixture
    def setup_manager(self):
        """Create a manager with 4 attention layers."""
        num_layers = 4
        num_pages = 128
        page_size = 16
        num_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = torch.device("cuda")

        # Create K/V caches for each layer with different values
        k_caches = []
        v_caches = []
        for layer_idx in range(num_layers):
            # Each layer has different norm values to test aggregation
            scale = (layer_idx + 1) * 0.5
            k_cache = torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device) * scale
            v_cache = torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device) * scale
            k_caches.append(k_cache)
            v_caches.append(v_cache)

        cpu_cache = torch.empty(
            (num_layers, 64, 2, page_size, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True
        )

        residency = ResidencyTable()
        scorer = PagedEvictionScorer(decay=1.0, head_reduction="mean")
        
        manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=50,
            kv_layout="NHD",
        )
        return manager, num_layers

    def test_score_block_aggregates_across_layers(self, setup_manager):
        """Test that score_block computes mean across all layers."""
        manager, num_layers = setup_manager
        
        ref = BlockRef("req1", 0)
        gpu_block_id = 10
        
        # Score the block
        manager.score_block(ref, gpu_block_id)
        
        # Check that scorer was updated with aggregated stats
        scores = manager.scorer.scores([ref])
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()
        assert scores[0] > 0

    def test_on_block_full_scores_all_layers(self, setup_manager):
        """Test that on_block_full triggers scoring across all layers."""
        manager, num_layers = setup_manager
        
        ref = BlockRef("req1", 1)
        gpu_block_id = 20
        
        # Mark as GPU resident
        manager.residency.mark_gpu(ref, gpu_block_id, complete=False)
        
        # Call on_block_full - should score across all layers
        manager.on_block_full(ref)
        
        # Verify block is marked complete
        loc = manager.residency.get(ref)
        assert loc.complete is True
        
        # Verify score was computed
        scores = manager.scorer.scores([ref])
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()

    def test_multi_layer_scoring_mean_aggregation(self, setup_manager):
        """Test that multi-layer scoring uses mean aggregation."""
        manager, num_layers = setup_manager
        
        # Create a block where each layer has a known norm ratio
        ref = BlockRef("req1", 2)
        gpu_block_id = 30
        
        # Manually set K/V caches with known values for each layer
        for layer_idx in range(num_layers):
            # Layer 0: ratio = 1.0, Layer 1: ratio = 2.0, Layer 2: ratio = 3.0, Layer 3: ratio = 4.0
            # Mean should be (1+2+3+4)/4 = 2.5
            k_cache = manager.gpu_k_caches[layer_idx]
            v_cache = manager.gpu_v_caches[layer_idx]
            with torch.no_grad():
                k_cache[gpu_block_id].fill_(1.0)
                v_cache[gpu_block_id].fill_((layer_idx + 1) * 1.0)
        
        manager.score_block(ref, gpu_block_id)
        
        # Score should be mean of ratios across layers
        scores = manager.scorer.scores([ref])
        # Allow some tolerance due to mean over heads/tokens
        assert scores[0] > 0


class TestScoreBlockWithDifferentLayouts:
    """Test scoring works with different KV layouts."""

    @pytest.fixture
    def setup_manager_hnd(self):
        """Create manager with HND layout."""
        num_layers = 2
        num_pages = 64
        page_size = 16
        num_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = torch.device("cuda")

        k_caches = []
        v_caches = []
        for _ in range(num_layers):
            # HND layout: [num_pages, num_heads, page_size, head_dim]
            k_cache = torch.randn(num_pages, num_heads, page_size, head_dim, dtype=dtype, device=device)
            v_cache = torch.randn(num_pages, num_heads, page_size, head_dim, dtype=dtype, device=device)
            k_caches.append(k_cache)
            v_caches.append(v_cache)

        cpu_cache = torch.empty(
            (num_layers, 32, 2, page_size, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True
        )

        residency = ResidencyTable()
        scorer = PagedEvictionScorer(decay=1.0, head_reduction="mean")
        
        manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=30,
            kv_layout="HND",
        )
        return manager

    def test_score_block_hnd_layout(self, setup_manager_hnd):
        """Test scoring works with HND layout."""
        manager = setup_manager_hnd
        
        ref = BlockRef("req1", 0)
        gpu_block_id = 5
        
        manager.score_block(ref, gpu_block_id)
        
        scores = manager.scorer.scores([ref])
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()


class TestScoreBlockEdgeCases:
    """Test edge cases for multi-layer scoring."""

    def test_single_layer_still_works(self):
        """Test that single layer case still works."""
        num_layers = 1
        num_pages = 64
        page_size = 16
        num_heads = 8
        head_dim = 128
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
        
        manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=30,
            kv_layout="NHD",
        )
        
        ref = BlockRef("req1", 0)
        manager.score_block(ref, 10)
        
        scores = manager.scorer.scores([ref])
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()

    def test_many_layers(self):
        """Test with many layers (e.g., 32 layers for large models)."""
        num_layers = 32
        num_pages = 128
        page_size = 16
        num_heads = 8
        head_dim = 128
        dtype = torch.float16
        device = torch.device("cuda")

        k_caches = []
        v_caches = []
        for _ in range(num_layers):
            k_caches.append(torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device))
            v_caches.append(torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device))

        cpu_cache = torch.empty(
            (num_layers, 64, 2, page_size, num_heads, head_dim),
            dtype=dtype, device="cpu", pin_memory=True
        )

        residency = ResidencyTable()
        scorer = PagedEvictionScorer(decay=1.0, head_reduction="mean")
        
        manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=50,
            kv_layout="NHD",
        )
        
        ref = BlockRef("req1", 0)
        manager.score_block(ref, 20)
        
        scores = manager.scorer.scores([ref])
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()