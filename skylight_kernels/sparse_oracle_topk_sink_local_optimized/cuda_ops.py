"""Loader for the AOT-compiled compact sink+local score CUDA extension.

Built at install time by skylight_kernels' setup.py via
torch.utils.cpp_extension.CUDAExtension. Importing ``_cuda_ops`` loads
the compiled ``.so``; no JIT runs at runtime.
"""
from __future__ import annotations

from . import _cuda_ops as _OPS


def get_compact_score_ops():
    return _OPS
