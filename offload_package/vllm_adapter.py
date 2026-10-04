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

    `layers[i]` is a tuple of (k_cache, v_cache) for layer i, or the combined
    per-layer tensor for this fork's [num_blocks, H, N, 2*head_dim] layout (in
    which case k_caches[i] is v_caches[i] and consumers split the last axis).
    The tensors are kept in vLLM's native layout; the offload package
    must not reinterpret the layout here.
    """

    layers: list
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
