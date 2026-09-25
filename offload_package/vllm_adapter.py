# SPDX-License-Identifier: Apache-2.0
"""vLLM adapter - extracts KV cache info from ModelRunner."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from offload_package.integration.manager import OffloadOrchestrator
    from offload_package.staging.pool import KVStagingPool


@dataclass(frozen=True)
class KVCacheView:
    """
    Canonical representation of vLLM's KV cache for the offload package.

    `layers[i]` is a tuple of (k_cache, v_cache) for layer i.
    The tensors are kept in vLLM's native layout; the offload package
    must not reinterpret the layout here.
    """

    layers: list[tuple[torch.Tensor, torch.Tensor]]
    k_caches: list[torch.Tensor]
    v_caches: list[torch.Tensor]
    block_size: int
    num_kv_heads: int
    head_dim: int
    kv_layout: str
    staging_base_page: int

    @property
    def num_layers(self) -> int:
        return len(self.k_caches)

    @property
    def device(self) -> torch.device:
        return self.k_caches[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.k_caches[0].dtype


def _get_attention_spec(kv_cache_config):
    """Extract AttentionSpec from KVCacheConfig."""
    from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            for layer_name in group.layer_names:
                layer_spec = spec.kv_cache_specs[layer_name]
                if isinstance(layer_spec, AttentionSpec):
                    return layer_spec
        elif hasattr(spec, 'block_size') and hasattr(spec, 'num_kv_heads'):
            return spec
    raise ValueError("No AttentionSpec found in KVCacheConfig")


def _split_kv_cache_with_spec(cache: torch.Tensor | tuple | list, spec) -> tuple[torch.Tensor, torch.Tensor]:
    """Split KV cache tensor into K and V using AttentionSpec.
    
    Handles both combined tensors and already-split (K, V) tuples/lists.
    Supports formats:
    - 5D: [num_pages, block_size, 2, num_kv_heads, head_dim] (NHD)
    - 5D: [num_pages, 2, num_kv_heads, block_size, head_dim] (HND)
    - 4D: [num_pages, num_kv_heads, page_size, 2*head_dim] (K/V combined in last dim)
    """
    if isinstance(cache, (tuple, list)):
        if len(cache) != 2 or not all(isinstance(x, torch.Tensor) for x in cache):
            raise TypeError("KV cache sequence must contain exactly (K, V)")
        return cache[0], cache[1]

    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Unsupported KV cache type: {type(cache).__name__}")

    # Use spec to determine layout and split correctly
    block_size = spec.block_size
    num_kv_heads = spec.num_kv_heads
    head_dim = spec.head_size

    if cache.ndim == 5:
        # NHD: [num_pages, block_size, 2, num_kv_heads, head_dim]
        # or HND: [num_pages, 2, num_kv_heads, block_size, head_dim]
        if cache.shape[2] == 2:
            return cache[:, :, 0], cache[:, :, 1]
        elif cache.shape[1] == 2:
            return cache[:, 0], cache[:, 1]
    elif cache.ndim == 4:
        # [num_pages, num_kv_heads, page_size, 2*head_dim] - K/V combined in last dim
        # Split the last dimension into K and V
        head_dim = cache.shape[-1] // 2
        k = cache[..., :head_dim]
        v = cache[..., head_dim:]
        return k, v
    elif cache.ndim == 4:
        # [num_pages, block_size, num_kv_heads, head_dim] - single tensor (K or V only)
        # This is unexpected for combined cache; assume it's already split elsewhere
        raise ValueError("Expected combined 5D K/V cache, got 4D tensor")

    raise ValueError(f"Cannot split KV cache with unexpected shape {cache.shape}")


def _get_attention_spec(kv_cache_config):
    """Extract AttentionSpec from KVCacheConfig."""
    from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            for layer_name in group.layer_names:
                layer_spec = spec.kv_cache_specs[layer_name]
                if isinstance(layer_spec, AttentionSpec):
                    return layer_spec
        elif hasattr(spec, 'block_size') and hasattr(spec, 'num_kv_heads'):
            return spec
    raise ValueError("No AttentionSpec found in KVCacheConfig")


def _split_kv_cache_with_spec(cache: torch.Tensor | tuple | list, spec) -> tuple[torch.Tensor, torch.Tensor]:
    """Split KV cache tensor into K and V using AttentionSpec."""
    if isinstance(cache, (tuple, list)):
        if len(cache) != 2 or not all(isinstance(x, torch.Tensor) for x in cache):
            raise TypeError("KV cache sequence must contain exactly (K, V)")
        return cache[0], cache[1]

    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Unsupported KV cache type: {type(cache).__name__}")

    # Use spec to determine layout and split correctly
    block_size = spec.block_size
    num_kv_heads = spec.num_kv_heads
    head_dim = spec.head_size

    if cache.ndim == 5:
        # NHD: [num_pages, block_size, 2, num_kv_heads, head_dim]
        # or HND: [num_pages, 2, num_kv_heads, block_size, head_dim]
        if cache.shape[2] == 2:
            return cache[:, :, 0], cache[:, :, 1]
        elif cache.shape[1] == 2:
            return cache[:, 0], cache[:, 1]
    elif cache.ndim == 4:
        # [num_pages, block_size, num_kv_heads, head_dim] - single tensor (K or V only)
        # This is unexpected for combined cache; assume it's already split elsewhere
        raise ValueError("Expected combined 5D K/V cache, got 4D tensor")

    raise ValueError(f"Cannot split KV cache with unexpected shape {cache.shape}")


def normalize_runner_kv_caches(
    runner_kv_caches,
    kv_cache_config,
    staging_slots: int = 0,
):
    """
    Normalize vLLM ModelRunner.kv_caches into layers, k_caches, v_caches.
    
    Handles multiple formats:
    - List of (K, V) tuples
    - Flat list of alternating K, V tensors
    - List of combined K/V tensors
    """
    from offload_package.shared.types import _get_attention_spec, _split_kv_cache_with_spec
    
    spec = _get_attention_spec(kv_cache_config)
    layers = []
    k_caches = []
    v_caches = []
    
    # Check the format of the first cache to determine the structure
    first_cache = runner_kv_caches[0]
    
    if isinstance(first_cache, (tuple, list)) and len(first_cache) == 2:
        # Format: [(K0, V0), (K1, V1), ...] - list of (K, V) tuples
        for layer_idx, cache in enumerate(runner_kv_caches):
            if not isinstance(cache, (tuple, list)) or len(cache) != 2:
                raise TypeError(f"Expected (K, V) tuple at index {layer_idx}")
            k_cache, v_cache = cache[0], cache[1]
            if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
                raise TypeError(f"K and V must be tensors at index {layer_idx}")
            layers.append((k_cache, v_cache))
            k_caches.append(k_cache)
            v_caches.append(v_cache)
    elif isinstance(first_cache, torch.Tensor):
        # Could be flat list [K0, V0, K1, V1, ...] or combined tensors
        # Check if it's a flat alternating list by comparing shapes
        if len(runner_kv_caches) % 2 == 0:
            # Likely flat alternating list [K0, V0, K1, V1, ...]
            for i in range(0, len(runner_kv_caches), 2):
                k_cache = runner_kv_caches[i]
                v_cache = runner_kv_caches[i + 1]
                if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
                    raise TypeError(f"Expected tensor at index {i} or {i+1}")
                layers.append((k_cache, v_cache))
                k_caches.append(k_cache)
                v_caches.append(v_cache)
        else:
            # Try to split as combined tensor
            for layer_idx, cache in enumerate(runner_kv_caches):
                k_cache, v_cache = _split_kv_cache_with_spec(cache, _get_attention_spec({}))
                layers.append((k_cache, v_cache))
                k_caches.append(k_cache)
                v_caches.append(v_cache)
    else:
        raise TypeError(f"Unexpected cache type at index 0: {type(first_cache)}")
    
    return layers, k_caches, v_caches
