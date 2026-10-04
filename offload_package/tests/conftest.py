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


def combined_caches(k_caches, v_caches):
    """Convert separate NHD [P, tokens, H, D] K/V lists into the fork's
    combined per-layer layout [P, H, tokens, 2D] (K in [..., :D]).

    The KVOffloadManager/KVStagingPool copy paths assume a single combined
    tensor per layer (k_caches[i] is v_caches[i]); tests that model the
    production layout must use this conversion.
    """
    out = []
    for kc, vc in zip(k_caches, v_caches):
        P, N, H, D = kc.shape
        t = torch.empty(P, H, N, 2 * D, dtype=kc.dtype, device=kc.device)
        t[..., :D] = kc.permute(0, 2, 1, 3)
        t[..., D:] = vc.permute(0, 2, 1, 3)
        out.append(t)
    return out
