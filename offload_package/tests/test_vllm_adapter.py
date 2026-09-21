"""Tests for vLLM adapter and KVCacheView."""
from __future__ import annotations

import pytest
import torch

from offload_package.vllm_adapter import (
    KVCacheView,
    OrchestratorConfig,
    normalize_runner_kv_caches,
    build_orchestrator,
)


class TestKVCacheView:
    """Test KVCacheView dataclass."""

    def test_view_properties(self):
        """Test KVCacheView properties."""
        layers = [
            torch.randn(64, 2, 16, 8, 64, dtype=torch.float16, device="cuda"),
            torch.randn(64, 2, 16, 8, 64, dtype=torch.float16, device="cuda"),
        ]
        k_caches = [torch.randn(64, 16, 8, 64, dtype=torch.float16, device="cuda") for _ in range(2)]
        v_caches = [torch.randn(64, 16, 8, 64, dtype=torch.float16, device="cuda") for _ in range(2)]
        
        view = KVCacheView(
            layers=layers,
            k_caches=k_caches,
            v_caches=v_caches,
            block_size=16,
            num_kv_heads=8,
            head_dim=64,
            kv_layout="NHD",
            staging_base_page=64,
        )
        
        assert view.num_layers == 2
        assert view.device.type == "cuda"
        assert view.dtype == torch.float16


class TestNormalizeRunnerKVCaches:
    """Test normalize_runner_kv_caches function."""

    @pytest.fixture
    def mock_kv_cache_config(self):
        """Create a mock KVCacheConfig with AttentionSpec."""
        from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup
        
        # This would need actual vLLM imports - skip for unit test
        # In real integration tests, we'd use actual vLLM config
        pass


class TestBuildOrchestrator:
    """Test build_orchestrator function."""

    @pytest.fixture
    def kv_cache_view(self):
        """Create a mock KVCacheView."""
        num_layers = 2
        num_pages = 128
        page_size = 16
        num_heads = 8
        head_dim = 64
        dtype = torch.float16
        device = torch.device("cuda")

        layers = [
            torch.randn(num_pages, 2, page_size, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        k_caches = [
            torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        v_caches = [
            torch.randn(num_pages, page_size, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        
        return KVCacheView(
            layers=layers,
            k_caches=k_caches,
            v_caches=v_caches,
            block_size=page_size,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            kv_layout="NHD",
            staging_base_page=100,
        )

    def test_build_orchestrator_creates_components(self, kv_cache_view):
        """Test that build_orchestrator creates all components."""
        cfg = OrchestratorConfig(
            offload_fraction=0.2,
            num_cpu_slots=64,
            num_staging_slots=8,
            W=16,
            page_size=16,
            dtype=torch.float16,
            device=torch.device("cuda"),
            gpu_budget_blocks=80,
            kv_layout="NHD",
            scorer_decay=1.0,
            scorer_head_reduction="mean",
        )
        
        orchestrator = build_orchestrator(cfg, kv_cache_view=kv_cache_view)
        
        assert orchestrator is not None
        assert orchestrator.offload_manager is not None
        assert orchestrator.staging_pool is not None
        assert orchestrator.residency is not None
        assert orchestrator.controller is not None
        assert orchestrator.offload_fraction == 0.2

    def test_build_orchestrator_validates_staging(self, kv_cache_view):
        """Test that build_orchestrator validates staging pages fit."""
        cfg = OrchestratorConfig(
            offload_fraction=0.2,
            num_cpu_slots=64,
            num_staging_slots=50,  # Too many for 128 pages with base 100
            W=16,
            page_size=16,
            dtype=torch.float16,
            device=torch.device("cuda"),
            gpu_budget_blocks=80,
            kv_layout="NHD",
        )
        
        with pytest.raises(ValueError, match="staging pages exceed KV cache capacity"):
            build_orchestrator(cfg, kv_cache_view=kv_cache_view)

    def test_build_orchestrator_validates_budget(self, kv_cache_view):
        """Test that build_orchestrator validates GPU budget."""
        cfg = OrchestratorConfig(
            offload_fraction=0.2,
            num_cpu_slots=64,
            num_staging_slots=8,
            W=16,
            page_size=16,
            dtype=torch.float16,
            device=torch.device("cuda"),
            gpu_budget_blocks=110,  # Exceeds staging_base_page (100)
            kv_layout="NHD",
        )
        
        with pytest.raises(ValueError, match="gpu_budget_blocks must fit"):
            build_orchestrator(cfg, kv_cache_view=kv_cache_view)


class TestIntegrationWithVLLMSpec:
    """Integration tests that would run with actual vLLM."""

    # These tests require actual vLLM and are marked as integration
    @pytest.mark.integration
    def test_normalize_with_real_vllm_config(self):
        """Test with real vLLM KVCacheConfig."""
        # This would be run in integration test environment
        pass
    
    @pytest.mark.integration
    def test_build_with_real_vllm_caches(self):
        """Test with real vLLM runner KV caches."""
        pass