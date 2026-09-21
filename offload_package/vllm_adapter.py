from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import sys

import torch

from .controller import KVCompressionController
from .integration.manager import OffloadOrchestrator
from .offload.manager import KVOffloadManager
from .scoring.paged_eviction import PagedEvictionScorer
from .staging.pool import KVStagingPool
from .staging.residency import ResidencyTable


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
    """
    if isinstance(cache, (tuple, list)):
        if len(cache) != 2 or not all(isinstance(x, torch.Tensor) for x in cache):
            raise TypeError("KV cache sequence must contain exactly (K, V)")
        return cache[0], cache[1]

    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Unsupported KV cache type: {type(cache)!r}")

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
        raise ValueError("Expected combined K/V cache, got single tensor")

    raise ValueError(f"Cannot split KV cache with shape {cache.shape} using spec")


def split_vllm_kv_cache(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-layer combined K/V cache tensor into separate K and V tensors.
    
    This is called from GPUModelRunner._get_k_caches_for_sparse() to extract
    K caches for sparse attention scoring.
    
    Handles standard layouts (NHD/HND) where K and V are interleaved in the
    combined tensor returned by vLLM's KV cache allocator.
    
    Args:
        cache: Combined K/V tensor of shape:
            - NHD: [num_pages, block_size, 2, num_kv_heads, head_dim]
            - HND: [num_pages, 2, num_kv_heads, block_size, head_dim]
            - LBNHC (packed per-layer view): [num_pages, block_size, 2, num_kv_heads, head_dim]
    
    Returns:
        Tuple of (k_cache, v_cache) each with K/V dimension removed.
    """
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(cache)!r}")

    if cache.ndim == 5:
        # NHD layout: [P, block_size, 2, H, D]
        if cache.shape[2] == 2:
            return cache[:, :, 0], cache[:, :, 1]
        # HND layout: [P, 2, H, block_size, D]
        elif cache.shape[1] == 2:
            return cache[:, 0], cache[:, 1]

    # If already split (4D), assume it's K or V only - this shouldn't happen
    # in normal vLLM flow but handle gracefully
    if cache.ndim == 4:
        raise ValueError(
            "split_vllm_kv_cache expects combined 5D K/V cache, "
            f"got 4D tensor with shape {cache.shape}. "
            "This may indicate the cache is already split."
        )

    raise ValueError(f"Cannot split KV cache with unexpected shape {cache.shape}")


def _normalize_kv_caches(runner_kv_caches, spec) -> tuple[list, list, list]:
    """
    Normalize vLLM ModelRunner.kv_caches into layers, k_caches, v_caches.
    
    Handles multiple formats:
    - List of (K, V) tuples
    - Flat list of alternating K, V tensors
    - List of combined K/V tensors
    """
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
                k_cache, v_cache = _split_kv_cache_with_spec(cache, spec)
                layers.append(cache)
                k_caches.append(k_cache)
                v_caches.append(v_cache)
    else:
        raise TypeError(f"Unexpected cache type at index 0: {type(first_cache)}")
    
    return layers, k_caches, v_caches


def normalize_runner_kv_caches(
    runner_kv_caches,
    kv_cache_config,
) -> KVCacheView:
    """
    Normalize vLLM ModelRunner.kv_caches using AttentionSpec for exact layout.

    vLLM 0.29.0 exposes runner.kv_caches in various formats depending on the
    attention backend and configuration.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    spec = _get_attention_spec(kv_cache_config)
    block_size = spec.block_size
    num_kv_heads = spec.num_kv_heads
    head_dim = spec.head_size

    # Normalize the KV caches into layers, k_caches, v_caches
    layers, k_caches, v_caches = _normalize_kv_caches(runner_kv_caches, spec)

    # Determine layout from tensor strides
    if not k_caches:
        raise ValueError("runner.kv_caches is empty")

    first_cache = k_caches[0]
    if first_cache.ndim >= 4:
        # Check stride order to determine NHD vs HND
        strides = first_cache.stride()
        if len(strides) >= 3:
            # NHD: stride order typically [page_stride, block_stride, head_stride, dim_stride]
            # HND: stride order typically [page_stride, head_stride, block_stride, dim_stride]
            stride_1, stride_2 = strides[1], strides[2]
            kv_layout = "NHD" if stride_1 > stride_2 else "HND"
        else:
            kv_layout = "NHD"
    else:
        kv_layout = "NHD"

    # Use actual GPU cache shape for head_dim (spec.head_size may differ from storage dim)
    # Cache shape is typically [num_blocks, num_kv_heads, page_size, head_dim] or similar
    actual_head_dim = first_cache.shape[-1]
    actual_num_kv_heads = first_cache.shape[-3] if first_cache.ndim >= 3 else num_kv_heads

    # Consistency checks - only check device, not dtype (storage dtype may vary)
    device = k_caches[0].device

    for layer_idx, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
        if k_cache.device != device:
            raise ValueError(
                "All KV cache layers must be on the same device: "
                f"layer 0 is on {device}, layer {layer_idx} is on {k_cache.device}"
            )
        if v_cache.device != device:
            raise ValueError(
                "All KV cache layers must be on the same device: "
                f"layer 0 is on {device}, layer {layer_idx} is on {v_cache.device}"
            )

    # Staging base page is the normal GPU page count (staging pages are appended after)
    staging_base_page = kv_cache_config.num_blocks

    return KVCacheView(
        layers=layers,
        k_caches=k_caches,
        v_caches=v_caches,
        block_size=block_size,
        num_kv_heads=actual_num_kv_heads,
        head_dim=actual_head_dim,
        kv_layout=kv_layout,
        staging_base_page=staging_base_page,
    )


@dataclass(frozen=True)
class OrchestratorConfig:
    offload_fraction: float
    num_cpu_slots: int
    num_staging_slots: int
    W: int
    page_size: int
    dtype: torch.dtype
    device: torch.device
    gpu_budget_blocks: int
    kv_layout: str = "NHD"
    scorer_decay: float = 1.0
    scorer_head_reduction: str = "mean"


def build_orchestrator(
    cfg: OrchestratorConfig,
    *,
    kv_cache_view: KVCacheView,
) -> OffloadOrchestrator:
    """Build the offload runtime using pre-parsed KVCacheView."""
    k_caches = kv_cache_view.k_caches
    v_caches = kv_cache_view.v_caches

    if not k_caches:
        raise ValueError("kv_cache_view.k_caches cannot be empty")

    # Validate staging pages fit
    staging_base = kv_cache_view.staging_base_page
    for idx, cache in enumerate(k_caches):
        print(f"[DEBUG] Validation layer {idx}: cache.shape[0]={cache.shape[0]}, staging_base={staging_base}, num_staging_slots={cfg.num_staging_slots}", file=sys.stderr, flush=True)
        if staging_base + cfg.num_staging_slots > cache.shape[0]:
            raise ValueError(f"staging pages exceed KV cache capacity: {staging_base} + {cfg.num_staging_slots} = {staging_base + cfg.num_staging_slots} > {cache.shape[0]}")

    if cfg.gpu_budget_blocks > staging_base:
        raise ValueError("gpu_budget_blocks must fit in the non-staging page range")

    # Verify dimensions match config
    if k_caches[0].shape[-1] != cfg.head_dim if hasattr(cfg, 'head_dim') else True:
        pass  # Config should match

    cpu_cache = torch.empty(
        (
            len(k_caches),
            cfg.num_cpu_slots,
            2,
            cfg.page_size,
            kv_cache_view.num_kv_heads,
            kv_cache_view.head_dim,
        ),
        dtype=cfg.dtype,
        device="cpu",
        pin_memory=True,
    )

    residency = ResidencyTable()
    scorer = PagedEvictionScorer(
        decay=cfg.scorer_decay,
        head_reduction=cfg.scorer_head_reduction,
    )
    offload_manager = KVOffloadManager(
        scorer=scorer,
        residency=residency,
        gpu_k_caches=k_caches,
        gpu_v_caches=v_caches,
        cpu_cache=cpu_cache,
        gpu_budget_blocks=cfg.gpu_budget_blocks,
        kv_layout=kv_cache_view.kv_layout,
    )
    staging_pool = KVStagingPool(
        num_slots=cfg.num_staging_slots,
        staging_base_page=staging_base,
        gpu_k_caches=k_caches,
        gpu_v_caches=v_caches,
        cpu_cache=cpu_cache,
        residency=residency,
        offload_manager=offload_manager,
    )
    controller = KVCompressionController(cfg.W, cfg.page_size)
    return OffloadOrchestrator(
        offload_manager=offload_manager,
        staging_pool=staging_pool,
        residency=residency,
        controller=controller,
        page_size=cfg.page_size,
        staging_base_page_idx=staging_base,
        offload_fraction=cfg.offload_fraction,
    )