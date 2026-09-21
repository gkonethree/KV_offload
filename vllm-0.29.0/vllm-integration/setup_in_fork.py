"""Helpers to be called from the vLLM fork's GPU worker/model runner."""

from __future__ import annotations


def install_all() -> None:
    # Must run before V2 initialize_kv_cache allocates the raw tensors.
    from .cache_allocator import install as install_cache_allocator

    install_cache_allocator()
