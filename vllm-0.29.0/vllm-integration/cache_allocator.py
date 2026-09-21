from __future__ import annotations

"""Conservative tail-page allocator hook for the vLLM V2 KV cache.

Supports both:
- Standard layout: one KV cache tensor per layer (NHD/HND)
- Packed layout: all layers in one tensor (LBNHC/FlashAttention)

The allocator expands the backing storage by `num_staging_slots` pages per layer.
For packed layouts, expands the shared backing storage once.
"""

import torch

from offload_package.config import OffloadConfig

_INSTALLED = False
_ORIGINAL = None


def install() -> None:
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return

    cfg = OffloadConfig.from_env()
    if not cfg.enabled:
        return

    import vllm.v1.worker.utils as worker_utils
    from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs

    original = worker_utils.allocate_kv_cache
    _ORIGINAL = original

    def allocate_with_tail(kv_cache_config, device, layout, kernel_block_sizes=None):
        # Call original allocator which creates per-layer views (even for packed layouts)
        raw = original(kv_cache_config, device, layout, kernel_block_sizes)
        
        # Group layers by their base storage pointer to detect packed layouts
        layers_by_base_ptr: dict[int, list[tuple[str, torch.Tensor]]] = {}
        for layer_name, tensor in raw.items():
            base_ptr = tensor.untyped_storage().data_ptr()
            layers_by_base_ptr.setdefault(base_ptr, []).append((layer_name, tensor))
        
        # For each unique base storage (handles both standard and packed layouts)
        for base_ptr, layer_tensors in layers_by_base_ptr.items():
            # Get the first layer's tensor to determine spec
            first_layer_name, first_tensor = layer_tensors[0]
            
            matching = [
                group
                for group in kv_cache_config.kv_cache_groups
                if first_layer_name in group.layer_names
            ]
            if len(matching) != 1:
                continue
            spec = matching[0].kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[first_layer_name]
            if not isinstance(spec, AttentionSpec):
                continue

            # Calculate extra bytes needed:
            # - Standard layout (one layer per base ptr): staging_slots per layer
            # - Packed layout (multiple layers per base ptr): staging_slots total for the shared buffer
            is_packed = len(layer_tensors) > 1
            if is_packed:
                # Packed layout: all layers share one tensor, staging slots added once
                extra = cfg.num_staging_slots * spec.page_size_bytes
            else:
                # Standard layout: each layer has its own tensor, staging slots per layer
                extra = cfg.num_staging_slots * spec.page_size_bytes
            if extra <= 0:
                continue

            # Get the original backing storage
            original_storage = first_tensor.untyped_storage()
            original_size = original_storage.nbytes()
            
            # NO COPYING NEEDED! The KV cache starts empty (all zeros).
            # We just need to allocate larger storage with staging slots appended.
            new_storage_tensor = torch.zeros(
                original_size + extra,
                dtype=torch.uint8,
                device=device,
            )
            
            # Get the untyped storage from the new tensor
            new_storage = new_storage_tensor.untyped_storage()
            
            # Recreate views for all layers pointing to the new storage
            # Preserve original view shapes/strides/offsets, but EXPAND first dimension
            import sys
            for layer_name, old_view in layer_tensors:
                # Expand the first dimension (number of pages) to include staging slots
                new_size = list(old_view.size())
                new_size[0] = new_size[0] + cfg.num_staging_slots  # Add staging pages
                
                new_view = torch.as_strided(
                    new_storage_tensor.view(old_view.dtype),
                    size=new_size,
                    stride=old_view.stride(),
                    storage_offset=old_view.storage_offset(),
                )
                print(f"[DEBUG cache_allocator] Layer {layer_name}: old_view.shape={old_view.shape}, new_view.shape={new_view.shape}, new_storage bytes={new_storage.nbytes()}", file=sys.stderr, flush=True)
                raw[layer_name] = new_view
            
        return raw

    worker_utils.allocate_kv_cache = allocate_with_tail
    _INSTALLED = True