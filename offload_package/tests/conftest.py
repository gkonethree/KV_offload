"""Shared pytest fixtures for offload package tests."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def torch_device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_kv_cache():
    """Create sample K/V cache tensors for testing.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (K_cache, V_cache) with shape [num_pages, page_size, num_heads, head_dim]
    """
    num_pages = 128
    page_size = 16
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    k_cache = torch.randn(num_pages, page_size, num_heads, head_dim, dtype=torch.float32, device=device)
    v_cache = torch.randn(num_pages, page_size, num_heads, head_dim, dtype=torch.float32, device=device)
    
    return k_cache, v_cache


@pytest.fixture
def sample_cpu_cache():
    """Create sample CPU cache tensor for testing.
    
    Returns:
        torch.Tensor: CPU cache with shape [num_layers, num_slots, 2, page_size, num_heads, head_dim]
    """
    num_layers = 1
    num_slots = 64
    page_size = 16
    num_heads = 8
    head_dim = 64
    
    cpu_cache = torch.zeros(
        num_layers, num_slots, 2, page_size, num_heads, head_dim,
        dtype=torch.float32
    )
    return cpu_cache
