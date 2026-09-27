"""Adapter: Oracle sparse scorer + sparse_optimized kernel.

This adapter mimics the oracle kernel's API but uses:
1. Oracle's score kernels for sparse pattern computation
2. sparse_optimized kernel for the actual sparse decode
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import torch

from offload_package.scoring.sparse_scorer import SparseAttentionScorer, SparsePattern
from original_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    _unpack_paged_kv_cache,
)
from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseOptimizedWrapper


class SparseOptimizedAdapterWrapper:
    """Adapter wrapper that combines Oracle scorer with sparse_optimized kernel.
    
    Constructor API matches oracle kernel:
        topk, sink_size, local_size, channel_num, max_seq_len
    
    run() computes sparse pattern using Oracle scorer, then calls sparse_optimized.
    """
    
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[list[Any]] = None,
        topk: Optional[float] = None,
        channel_num: Optional[int] = None,
        sink_size: Optional[int] = None,
        local_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        # Internal sparse_optimized wrapper (dense decode path from original_optimized)
        self._inner = DenseOptimizedWrapper(
            float_workspace_buffer=float_workspace_buffer,
            kv_layout=kv_layout,
            use_cuda_graph=use_cuda_graph,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
            backend=backend,
            jit_args=jit_args,
        )
        
        # Scorer config from environment or constructor
        self._topk = topk if topk is not None else float(os.environ.get("SKYLIGHT_SPARSE_TOPK", "0.1"))
        self._sink_size = sink_size if sink_size is not None else int(os.environ.get("SKYLIGHT_SPARSE_SINK", "64"))
        self._local_size = local_size if local_size is not None else int(os.environ.get("SKYLIGHT_SPARSE_LOCAL", "64"))
        self._channel_num = channel_num if channel_num is not None else int(os.environ.get("SKYLIGHT_SPARSE_CHANNEL_NUM", "8"))
        self._max_seq_len = max_seq_len if max_seq_len is not None else int(os.environ.get("SKYLIGHT_SPARSE_MAX_SEQ_LEN", "8192"))
        
        # Scorer instance
        self._scorer = SparseAttentionScorer(
            topk=self._topk,
            sink_size=self._sink_size,
            local_size=self._local_size,
            channel_num=self._channel_num,
            max_seq_len=self._max_seq_len,
        )
        
        # Cached plan data
        self._indptr: Optional[torch.Tensor] = None
        self._indices: Optional[torch.Tensor] = None
        self._last_page_len: Optional[torch.Tensor] = None
        self._kv_cache_spec: Optional[tuple] = None
        self._layout = kv_layout
        
        # For offload system: cache last sparse pattern
        self._last_sparse_idx: Optional[torch.Tensor] = None
        self._last_sparse_len: Optional[torch.Tensor] = None
    
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> None:
        """Forward plan to inner wrapper and cache metadata for scorer."""
        self._inner.plan(
            indptr, indices, last_page_len,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=block_tables,
            seq_lens=seq_lens,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        
        # Cache for scorer
        self._indptr = indptr
        self._indices = indices
        self._last_page_len = last_page_len
        self._kv_cache_spec = (num_kv_heads, head_dim, page_size)
    
    def _unpack_paged_kv_cache(
        self,
        paged_kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract K and V caches from paged KV cache."""
        k_cache, v_cache, *_ = _unpack_paged_kv_cache(paged_kv_cache, self._layout)
        return k_cache, v_cache
    
    def _get_kv_strides(
        self,
        paged_kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[int, int, int, int]:
        """Get KV cache strides for scorer."""
        _, _, stride_page, stride_n, stride_h, v_offset, _, _ = _unpack_paged_kv_cache(paged_kv_cache, self._layout)
        return (int(stride_page), int(stride_n), int(stride_h), int(v_offset))
    
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        *args: Any,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        q_len_per_req: Optional[int] = 1,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        kv_cache_sf: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute sparse pattern, then call sparse_optimized kernel."""
        # Ensure plan was called
        if self._indptr is None:
            raise RuntimeError("plan() must be called before run().")
        
        if q_len_per_req is None:
            q_len_per_req = 1
        if q_len_per_req != 1:
            raise NotImplementedError("Only q_len_per_req == 1 supported.")
        
        # 1. Compute sparse pattern using Oracle scorer
        k_cache, v_cache = self._unpack_paged_kv_cache(paged_kv_cache)
        
        pattern = self._scorer.compute_pattern(
            q=q,
            k_cache=k_cache,
            indptr=self._indptr,
            indices=self._indices,
            last_page_len=self._last_page_len,
            kv_strides=self._get_kv_strides(paged_kv_cache),
            page_size=self._kv_cache_spec[2],
            num_kv_heads=self._kv_cache_spec[0],
            head_dim=self._kv_cache_spec[1],
            layout=self._layout,
        )
        
        # 2. Cache pattern for offload system (skylight backend reads this)
        self._last_sparse_idx = pattern.sparse_idx.detach().cpu()
        self._last_sparse_len = pattern.sparse_len.detach().cpu()
        
        # Also write to module-level cache for worker connector
        try:
            import skylight.backend as skylight_backend
            skylight_backend._sparse_pattern_cache['sparse_idx'] = pattern.sparse_idx.detach().cpu()
            skylight_backend._sparse_pattern_cache['sparse_len'] = pattern.sparse_len.detach().cpu()
        except Exception:
            pass  # Silently ignore if skylight backend not available
        
        # 3. Call sparse_optimized kernel with pattern
        return self._inner.run(
            q, paged_kv_cache,
            pattern.sparse_len,
            pattern.sparse_idx,
            pattern.sparse_weights,
            *args,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out=out,
            lse=lse,
            return_lse=return_lse,
            enable_pdl=enable_pdl,
            window_left=window_left,
            sinks=sinks,
            q_len_per_req=q_len_per_req,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            kv_cache_sf=kv_cache_sf,
        )
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
    
    # For offload system: expose cached pattern
    def get_cached_sparse_pattern(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_sparse_idx, self._last_sparse_len