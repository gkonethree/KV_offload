"""Loader for the AOT-compiled compact sink+local score CUDA extension.

Built at install time by skylight_kernels' setup.py via
torch.utils.cpp_extension.CUDAExtension. Importing ``_cuda_ops`` loads
the compiled ``.so``; no JIT runs at runtime.
"""
from __future__ import annotations

from . import _cuda_ops as _OPS


def get_compact_score_ops():
    return _OPS


def get_scorer_ops():
    """Returns module with fused scorer operations:
    - oracle_topk_compute_topk_indices: fused score + topk -> indices [B, H_q, k]
    - oracle_topk_compute_scores_compact: compact scores [B, H_q, L_mid]
    - oracle_topk_select_topk: top-k from scores -> indices [B, H_q, k]
    """
    return _OPS


# Convenience function for fused scorer
def oracle_topk_compute_topk_indices(
    q, k_cache, kv_stride_page, kv_stride_n, kv_stride_h,
    indptr, indices, last_page_len,
    H_kv, page_size, token_start, token_end, channel_num, k):
    """Fused: compute compact scores + top-k -> indices [B, H_q, k] with global positions."""
    return _OPS.oracle_topk_compute_topk_indices(
        q, k_cache, kv_stride_page, kv_stride_n, kv_stride_h,
        indptr, indices, last_page_len,
        H_kv, page_size, token_start, token_end, channel_num, k)
