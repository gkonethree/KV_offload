from __future__ import annotations

"""Conservative tail-page allocator hook for the vLLM V2 KV cache."""

import torch

from .config import OffloadConfig

_INSTALLED = False
_ORIGINAL = None


def install() -> None:
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return

    cfg = OffloadConfig.from_env()
    if not cfg.enabled:
        return

    import vllm.v1.worker.gpu.attn_utils as attn_utils
    from vllm.v1.kv_cache_interface import AttentionSpec, UniformTypeKVCacheSpecs

    original = attn_utils._allocate_kv_cache
    _ORIGINAL = original

    def allocate_with_tail(kv_cache_config, shared_layers, device):
        raw = original(kv_cache_config, shared_layers, device)
        for kv_tensor in kv_cache_config.kv_cache_tensors:
            if len(kv_tensor.shared_by) != 1:
                raise RuntimeError(
                    "offload-package currently requires one layer per KV cache tensor; "
                    f"got shared_by={kv_tensor.shared_by}"
                )
            layer_name = kv_tensor.shared_by[0]
            matching = [
                group
                for group in kv_cache_config.kv_cache_groups
                if layer_name in group.layer_names
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"Could not determine a unique KV cache group for {layer_name}"
                )
            spec = matching[0].kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[layer_name]
            if not isinstance(spec, AttentionSpec):
                raise RuntimeError(
                    "offload-package staging is supported only for ordinary AttentionSpec caches"
                )

            old = raw[layer_name]
            # The raw tensor is byte-addressed int8 storage.  For packed layouts,
            # block_stride is the correct unit; otherwise page_size_bytes is.
            bytes_per_physical_page = (
                kv_tensor.block_stride if kv_tensor.block_stride > 0 else spec.page_size_bytes
            )
            extra = cfg.num_staging_slots * bytes_per_physical_page
            if extra <= 0:
                continue
            expanded = torch.empty(
                old.numel() + extra,
                dtype=old.dtype,
                device=old.device,
            )
            expanded[: old.numel()].copy_(old)
            raw[layer_name] = expanded
        return raw

    attn_utils._allocate_kv_cache = allocate_with_tail
    _INSTALLED = True