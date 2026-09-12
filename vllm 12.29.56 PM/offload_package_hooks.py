"""Copy this file into vllm/v1/worker/ when applying the integration."""
from __future__ import annotations

from .cache_allocator import install as install_cache_allocator


def install_all() -> None:
    install_cache_allocator()
