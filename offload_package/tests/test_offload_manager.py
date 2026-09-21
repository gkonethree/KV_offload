"""Unit tests for KV offload manager."""
from __future__ import annotations

import pytest
import torch

from offload_package.offload.manager import CPUBlockPool, KVOffloadManager
from offload_package.scoring.base import BlockRef
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.residency import Residency, ResidencyTable


class TestCPUBlockPool:
    """Tests for CPUBlockPool class."""

    def test_init_valid_slots(self):
        """Test pool initialization with valid slot count."""
        pool = CPUBlockPool(num_slots=10)
        assert pool.num_free == 10

    def test_init_invalid_slots(self):
        """Test pool initialization rejects invalid slot count."""
        with pytest.raises(ValueError, match="num_slots must be > 0"):
            CPUBlockPool(num_slots=0)
        
        with pytest.raises(ValueError, match="num_slots must be > 0"):
            CPUBlockPool(num_slots=-1)

    def test_allocate_exhausts_pool(self):
        """Test that allocate succeeds until pool is empty."""
        pool = CPUBlockPool(num_slots=3)
        
        slots = [pool.allocate() for _ in range(3)]
        assert len(set(slots)) == 3  # All unique
        
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate()

    def test_free_makes_slot_available(self):
        """Test that free() makes a slot available for re-allocation."""
        pool = CPUBlockPool(num_slots=2)
        
        slot1 = pool.allocate()
        slot2 = pool.allocate()
        assert pool.num_free == 0
        
        pool.free(slot1)
        assert pool.num_free == 1
        
        # Can allocate again
        slot3 = pool.allocate()
        assert slot3 == slot1


class TestKVOffloadManager:
    """Tests for KVOffloadManager class."""

    @pytest.fixture
    def offload_setup(self, sample_kv_cache, sample_cpu_cache):
        """Create a pre-configured offload manager."""
        k_cache, v_cache = sample_kv_cache
        k_caches = [k_cache]
        v_caches = [v_cache]
        
        residency = ResidencyTable()
        scorer = PagedEvictionScorer()
        cpu_cache = sample_cpu_cache
        
        manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=k_caches,
            gpu_v_caches=v_caches,
            cpu_cache=cpu_cache,
            gpu_budget_blocks=64,
            kv_layout="NHD",
        )
        return manager, residency, k_caches, v_caches

    def test_init_valid_params(self, sample_kv_cache, sample_cpu_cache):
        """Test manager initialization with valid parameters."""
        k_cache, v_cache = sample_kv_cache
        residency = ResidencyTable()
        scorer = PagedEvictionScorer()
        
        manager = KVOffloadManager(
            scorer=scorer,
            residency=residency,
            gpu_k_caches=[k_cache],
            gpu_v_caches=[v_cache],
            cpu_cache=sample_cpu_cache,
            gpu_budget_blocks=64,
        )
        assert manager.gpu_budget_blocks == 64

    def test_init_mismatched_k_v_caches(self, sample_kv_cache, sample_cpu_cache):
        """Test that mismatched K/V cache lists raise error."""
        k_cache, v_cache = sample_kv_cache
        residency = ResidencyTable()
        scorer = PagedEvictionScorer()
        
        with pytest.raises(ValueError, match="same length"):
            KVOffloadManager(
                scorer=scorer,
                residency=residency,
                gpu_k_caches=[k_cache, k_cache],
                gpu_v_caches=[v_cache],  # Only 1, not 2
                cpu_cache=sample_cpu_cache,
                gpu_budget_blocks=64,
            )

    def test_score_block(self, offload_setup):
        """Test scoring a single block."""
        manager, residency, k_caches, v_caches = offload_setup
        
        # Mark block as GPU resident
        ref = BlockRef("req1", 0)
        residency.mark_gpu(ref, gpu_block_id=0)
        
        # Score it
        manager.score_block(ref, gpu_block_id=0)
        
        # Verify score was computed (not marked complete - that's done by on_block_full)
        scores = manager.scorer.scores([ref])
        assert scores.shape == (1,)
        assert not torch.isnan(scores).any()
        assert scores[0] > 0

    def test_on_block_full(self, offload_setup):
        """Test on_block_full triggers scoring."""
        manager, residency, k_caches, v_caches = offload_setup
        
        ref = BlockRef("req1", 0)
        residency.mark_gpu(ref, gpu_block_id=0)
        
        # This should score and mark complete
        manager.on_block_full(ref)
        
        loc = residency.get(ref)
        assert loc.complete is True

    def test_maybe_offload_under_budget(self, offload_setup):
        """Test maybe_offload doesn't offload when under budget."""
        manager, residency, k_caches, v_caches = offload_setup
        
        # Add 2 blocks (budget is 64)
        for i in range(2):
            ref = BlockRef("req1", i)
            residency.mark_gpu(ref, gpu_block_id=i, complete=True)
        
        offloaded = manager.maybe_offload([BlockRef("req1", 0), BlockRef("req1", 1)])
        
        assert len(offloaded) == 0

    def test_maybe_offload_over_budget(self, offload_setup):
        """Test maybe_offload triggers when over budget."""
        manager, residency, k_caches, v_caches = offload_setup
        
        # Lower budget to trigger offload
        manager.gpu_budget_blocks = 1
        
        # Add 3 blocks
        refs = []
        for i in range(3):
            ref = BlockRef("req1", i)
            residency.mark_gpu(ref, gpu_block_id=i, complete=True)
            refs.append(ref)
        
        # Score them with different priorities
        import torch
        v_norm = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        k_norm = torch.ones(3, 1, dtype=torch.float32)
        from offload_package.scoring.base import BlockKVStats
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=refs)
        manager.scorer.update(stats)
        
        offloaded = manager.maybe_offload(refs)
        
        # Should offload the lowest-scoring block (1-block budget)
        assert len(offloaded) == 2  # Keep 1, offload 2
        assert refs[0] in offloaded  # Lowest score

    def test_offload_fraction(self, offload_setup):
        """Test offload_fraction offloads correct fraction."""
        manager, residency, k_caches, v_caches = offload_setup
        
        # Add 10 blocks
        refs = []
        for i in range(10):
            ref = BlockRef("req1", i)
            residency.mark_gpu(ref, gpu_block_id=i, complete=True)
            refs.append(ref)
        
        # Score them
        import torch
        v_norm = torch.arange(10, 0, -1, dtype=torch.float32).unsqueeze(1)
        k_norm = torch.ones(10, 1, dtype=torch.float32)
        from offload_package.scoring.base import BlockKVStats
        stats = BlockKVStats(v_norm=v_norm, k_norm=k_norm, block_ids=refs)
        manager.scorer.update(stats)
        
        # Offload 50%
        offloaded = manager.offload_fraction(refs, fraction=0.5)
        
        assert len(offloaded) == 5
        # Check they're marked as CPU resident
        for ref in offloaded:
            loc = residency.get(ref)
            assert loc.residency == Residency.CPU

    def test_free_removes_from_pool(self, offload_setup):
        """Test free() releases CPU slots and scorer state."""
        manager, residency, k_caches, v_caches = offload_setup
        
        ref = BlockRef("req1", 0)
        residency.mark_gpu(ref, gpu_block_id=0)
        residency.mark_cpu(ref, cpu_slot=5)
        
        initial_free = manager._cpu_pool.num_free
        
        manager.free(ref)
        
        # Slot should be freed
        assert manager._cpu_pool.num_free == initial_free + 1
        # Block should be removed
        assert residency.get(ref) is None

    def test_wait_cpu_slot_synchronizes(self, offload_setup):
        """Test wait_cpu_slot waits for copy event."""
        manager, residency, k_caches, v_caches = offload_setup
        
        # Record an event
        event = torch.cuda.Event()
        event.record()
        manager._store_events[0] = event
        
        # Wait should not raise
        manager.wait_cpu_slot(0)
        
        # Event should be cleared
        assert 0 not in manager._store_events
