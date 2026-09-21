"""Unit tests for residency tracking module."""
from __future__ import annotations

import pytest

from offload_package.scoring.base import BlockRef
from offload_package.staging.residency import BlockLocation, Residency, ResidencyTable


class TestBlockLocation:
    """Tests for BlockLocation dataclass."""

    def test_create_gpu_location(self):
        """Test creating a GPU block location."""
        loc = BlockLocation(residency=Residency.GPU, gpu_block_id=5, complete=True)
        assert loc.residency == Residency.GPU
        assert loc.gpu_block_id == 5
        assert loc.complete is True

    def test_create_cpu_location(self):
        """Test creating a CPU block location."""
        loc = BlockLocation(residency=Residency.CPU, gpu_block_id=5, cpu_slot=10)
        assert loc.residency == Residency.CPU
        assert loc.cpu_slot == 10

    def test_create_staging_location(self):
        """Test creating a staging block location."""
        loc = BlockLocation(
            residency=Residency.STAGING, gpu_block_id=5, cpu_slot=10, staging_slot=2
        )
        assert loc.residency == Residency.STAGING
        assert loc.staging_slot == 2


class TestResidencyTable:
    """Tests for ResidencyTable class."""

    def test_mark_gpu(self):
        """Test marking a block as GPU resident."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        table.mark_gpu(ref, gpu_block_id=5, complete=False)
        
        loc = table.get(ref)
        assert loc is not None
        assert loc.residency == Residency.GPU
        assert loc.gpu_block_id == 5
        assert loc.complete is False

    def test_mark_complete(self):
        """Test marking a block as complete."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        table.mark_gpu(ref, gpu_block_id=5, complete=False)
        table.mark_complete(ref)
        
        loc = table.get(ref)
        assert loc.complete is True

    def test_mark_cpu(self):
        """Test marking a block as CPU resident."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        # Must have GPU mapping first
        table.mark_gpu(ref, gpu_block_id=5)
        table.mark_cpu(ref, cpu_slot=10)
        
        loc = table.get(ref)
        assert loc.residency == Residency.CPU
        assert loc.cpu_slot == 10
        assert loc.gpu_block_id == 5  # Preserves GPU mapping

    def test_mark_cpu_without_gpu_mapping_fails(self):
        """Test that marking CPU without GPU mapping raises error."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        with pytest.raises(KeyError):
            table.mark_cpu(ref, cpu_slot=10)

    def test_mark_staging(self):
        """Test marking a block as staging resident."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        table.mark_gpu(ref, gpu_block_id=5)
        table.mark_cpu(ref, cpu_slot=10)
        table.mark_staging(ref, staging_slot=2)
        
        loc = table.get(ref)
        assert loc.residency == Residency.STAGING
        assert loc.staging_slot == 2
        assert loc.cpu_slot == 10  # Preserves CPU slot

    def test_mark_staging_requires_cpu_slot(self):
        """Test that marking staging requires a CPU slot."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        table.mark_gpu(ref, gpu_block_id=5)
        
        with pytest.raises(KeyError):
            table.mark_staging(ref, staging_slot=2)

    def test_get_nonexistent_block(self):
        """Test getting a block that doesn't exist."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        loc = table.get(ref)
        assert loc is None

    def test_remove_block(self):
        """Test removing a block from the table."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        table.mark_gpu(ref, gpu_block_id=5)
        removed = table.remove(ref)
        
        assert removed is not None
        assert table.get(ref) is None

    def test_all_gpu_blocks(self):
        """Test querying all GPU blocks."""
        table = ResidencyTable()
        
        # Add GPU blocks
        gpu_refs = [BlockRef("req1", 0), BlockRef("req1", 1)]
        for ref in gpu_refs:
            table.mark_gpu(ref, gpu_block_id=int(ref.block_idx), complete=True)
        
        # Add a CPU block
        cpu_ref = BlockRef("req1", 2)
        table.mark_gpu(cpu_ref, gpu_block_id=2)
        table.mark_cpu(cpu_ref, cpu_slot=10)
        
        gpu_blocks = table.all_gpu_blocks(complete_only=True)
        assert len(gpu_blocks) == 2
        assert cpu_ref not in gpu_blocks

    def test_refs_for_request(self):
        """Test querying blocks for a specific request."""
        table = ResidencyTable()
        
        # Add blocks for req1
        req1_refs = [BlockRef("req1", 0), BlockRef("req1", 1), BlockRef("req1", 2)]
        for ref in req1_refs:
            table.mark_gpu(ref, gpu_block_id=int(ref.block_idx))
        
        # Add blocks for req2
        req2_refs = [BlockRef("req2", 0), BlockRef("req2", 1)]
        for ref in req2_refs:
            table.mark_gpu(ref, gpu_block_id=int(ref.block_idx))
        
        # Query req1
        req1_blocks = table.refs_for_request("req1")
        assert len(req1_blocks) == 3
        assert all(ref.request_id == "req1" for ref in req1_blocks)
        # Check they're sorted by block index
        assert [ref.block_idx for ref in req1_blocks] == [0, 1, 2]

    def test_len(self):
        """Test getting table size."""
        table = ResidencyTable()
        
        assert len(table) == 0
        
        table.mark_gpu(BlockRef("req1", 0), gpu_block_id=0)
        assert len(table) == 1
        
        table.mark_gpu(BlockRef("req1", 1), gpu_block_id=1)
        assert len(table) == 2

    def test_repr(self):
        """Test string representation."""
        table = ResidencyTable()
        
        table.mark_gpu(BlockRef("req1", 0), gpu_block_id=0)
        table.mark_gpu(BlockRef("req1", 1), gpu_block_id=1)
        cpu_ref = BlockRef("req1", 2)
        table.mark_gpu(cpu_ref, gpu_block_id=2)
        table.mark_cpu(cpu_ref, cpu_slot=10)
        
        repr_str = repr(table)
        assert "GPU" in repr_str
        assert "CPU" in repr_str

    def test_state_machine_transitions(self):
        """Test valid state machine transitions."""
        table = ResidencyTable()
        ref = BlockRef("req1", 0)
        
        # GPU -> CPU
        table.mark_gpu(ref, gpu_block_id=5)
        table.mark_cpu(ref, cpu_slot=10)
        assert table.get(ref).residency == Residency.CPU
        
        # CPU -> STAGING
        table.mark_staging(ref, staging_slot=2)
        assert table.get(ref).residency == Residency.STAGING
        
        # STAGING -> CPU (back to CPU if not used)
        table.mark_cpu(ref, cpu_slot=10)
        assert table.get(ref).residency == Residency.CPU
