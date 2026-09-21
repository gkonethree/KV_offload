"""Unit tests for scoring module."""
from __future__ import annotations

import pytest
import torch

from offload_package.scoring.base import BlockKVStats, BlockScorer, BlockRef
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.scoring.norm_kernel import compute_block_kv_norms


class TestPagedEvictionScorer:
    """Tests for PagedEvictionScorer class."""

    def test_init_valid_params(self):
        """Test scorer initialization with valid parameters."""
        scorer = PagedEvictionScorer(decay=0.9, head_reduction="mean", eps=1e-6)
        assert scorer.decay == 0.9
        assert scorer.head_reduction == "mean"
        assert scorer.eps == 1e-6

    def test_init_invalid_decay(self):
        """Test scorer initialization rejects invalid decay."""
        with pytest.raises(ValueError, match="decay must be in"):
            PagedEvictionScorer(decay=-0.1)
        
        with pytest.raises(ValueError, match="decay must be in"):
            PagedEvictionScorer(decay=1.5)

    def test_init_invalid_head_reduction(self):
        """Test scorer initialization rejects invalid head_reduction."""
        with pytest.raises(ValueError, match="head_reduction must be"):
            PagedEvictionScorer(head_reduction="max")

    def test_update_stores_scores(self):
        """Test that update() stores and returns scores."""
        scorer = PagedEvictionScorer()
        
        # Create sample stats
        v_norm = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=torch.float32)
        k_norm = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        block_ids = [BlockRef("req1", 0), BlockRef("req1", 1)]
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=block_ids)
        
        scorer.update(stats)
        
        # Check scores are stored
        scores = scorer.scores(block_ids)
        assert scores.shape == (2,)
        assert not torch.any(torch.isnan(scores))

    def test_scores_with_decay(self):
        """Test that decay parameter correctly averages scores over time."""
        scorer = PagedEvictionScorer(decay=0.5)
        block_ids = [BlockRef("req1", 0)]
        
        # First update: score = 2.0
        v_norm = torch.tensor([[2.0]], dtype=torch.float32)
        k_norm = torch.tensor([[1.0]], dtype=torch.float32)
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=block_ids)
        scorer.update(stats)
        score1 = scorer.scores(block_ids)[0].item()
        assert score1 == pytest.approx(2.0)
        
        # Second update: score = 4.0, with decay=0.5 -> (0.5 * 4.0) + (0.5 * 2.0) = 3.0
        v_norm = torch.tensor([[4.0]], dtype=torch.float32)
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=block_ids)
        scorer.update(stats)
        score2 = scorer.scores(block_ids)[0].item()
        assert score2 == pytest.approx(3.0)

    def test_select_for_eviction(self):
        """Test that select_for_eviction returns lowest-scoring blocks."""
        scorer = PagedEvictionScorer()
        block_ids = [
            BlockRef("req1", 0),
            BlockRef("req1", 1),
            BlockRef("req1", 2),
            BlockRef("req1", 3),
        ]
        
        # Score blocks with different values
        v_norm = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
        k_norm = torch.ones(4, 1, dtype=torch.float32)
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=block_ids)
        scorer.update(stats)
        
        # Keep top 2, evict bottom 2
        to_evict = scorer.select_for_eviction(block_ids, budget=2)
        assert len(to_evict) == 2
        assert block_ids[0] in to_evict  # score 1.0
        assert block_ids[1] in to_evict  # score 2.0

    def test_select_for_eviction_with_budget_exceeded(self):
        """Test select_for_eviction when GPU budget is exceeded."""
        scorer = PagedEvictionScorer()
        block_ids = [
            BlockRef("req1", 0),
            BlockRef("req1", 1),
            BlockRef("req1", 2),
        ]
        
        v_norm = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        k_norm = torch.ones(3, 1, dtype=torch.float32)
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=block_ids)
        scorer.update(stats)
        
        # Budget 1 means keep 1, evict 2
        to_evict = scorer.select_for_eviction(block_ids, budget=1)
        assert len(to_evict) == 2

    def test_select_for_eviction_empty_candidates(self):
        """Test select_for_eviction with no candidates."""
        scorer = PagedEvictionScorer()
        to_evict = scorer.select_for_eviction([], budget=10)
        assert len(to_evict) == 0

    def test_forget_removes_blocks(self):
        """Test that forget() removes score records."""
        scorer = PagedEvictionScorer()
        block_ids = [BlockRef("req1", 0), BlockRef("req1", 1)]
        
        v_norm = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        k_norm = torch.ones(2, 1, dtype=torch.float32)
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=block_ids)
        scorer.update(stats)
        
        # Forget first block
        scorer.forget([block_ids[0]])
        
        # Score should be 0 (not found) after forget
        scores = scorer.scores(block_ids)
        assert scores[0].item() == 0.0
        assert scores[1].item() > 0.0

    def test_reset_clears_all_scores(self):
        """Test that reset() clears all stored scores."""
        scorer = PagedEvictionScorer()
        block_ids = [BlockRef("req1", 0), BlockRef("req1", 1)]
        
        v_norm = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        k_norm = torch.ones(2, 1, dtype=torch.float32)
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=block_ids)
        scorer.update(stats)
        
        scorer.reset()
        
        # All scores should be 0
        scores = scorer.scores(block_ids)
        assert torch.all(scores == 0.0)


class TestComputeBlockKVNorms:
    """Tests for compute_block_kv_norms function."""

    def test_compute_norms_basic(self, sample_kv_cache):
        """Test basic norm computation."""
        k_cache, v_cache = sample_kv_cache
        
        physical_block_ids = [0, 1, 2]
        stats = compute_block_kv_norms(k_cache, v_cache, physical_block_ids, layout="NHD")
        
        # NHD layout: [num_pages, page_size, num_heads, head_dim] -> num_heads at index 2
        assert stats.v_norm.shape == (3, k_cache.shape[2])  # [num_blocks, num_heads]
        assert stats.k_norm.shape == (3, k_cache.shape[2])
        assert len(stats.block_ids) == 3
        assert not torch.any(torch.isnan(stats.v_norm))
        assert not torch.any(torch.isnan(stats.k_norm))

    def test_compute_norms_with_logical_ids(self, sample_kv_cache):
        """Test norm computation with custom logical block IDs."""
        k_cache, v_cache = sample_kv_cache
        
        physical_block_ids = [0, 1]
        logical_block_ids = [BlockRef("req1", 0), BlockRef("req1", 1)]
        stats = compute_block_kv_norms(
            k_cache, v_cache, physical_block_ids, logical_block_ids=logical_block_ids
        )
        
        assert stats.block_ids == logical_block_ids

    def test_compute_norms_empty_input(self, sample_kv_cache):
        """Test norm computation with empty input."""
        k_cache, v_cache = sample_kv_cache
        
        stats = compute_block_kv_norms(k_cache, v_cache, [], layout="NHD")
        
        assert stats.v_norm.shape[0] == 0
        assert stats.k_norm.shape[0] == 0
        assert len(stats.block_ids) == 0

    def test_compute_norms_different_layouts(self, sample_kv_cache):
        """Test norm computation with different KV layouts."""
        k_cache, v_cache = sample_kv_cache
        physical_block_ids = [0, 1]
        
        # NHD layout (tested above)
        stats_nhd = compute_block_kv_norms(
            k_cache, v_cache, physical_block_ids, layout="NHD"
        )
        assert stats_nhd.v_norm.shape == (2, k_cache.shape[2])
        
        # Test invalid layout
        with pytest.raises(ValueError, match="layout must be"):
            compute_block_kv_norms(k_cache, v_cache, physical_block_ids, layout="invalid")

    def test_compute_norms_mismatched_lengths(self, sample_kv_cache):
        """Test that mismatched physical and logical IDs raise error."""
        k_cache, v_cache = sample_kv_cache
        
        with pytest.raises(ValueError, match="equal length"):
            compute_block_kv_norms(
                k_cache, v_cache,
                [0, 1],
                logical_block_ids=[BlockRef("req1", 0)]  # Only 1, not 2
            )
