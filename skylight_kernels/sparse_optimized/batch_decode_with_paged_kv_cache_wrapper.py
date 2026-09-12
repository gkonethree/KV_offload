"""
Standalone sparse BatchDecodeWithPagedKVCacheWrapper backed by a custom CUDA
kernel.

Implements the same plan/init API as `original_optimized` but augments
`run(...)` to take per-(batch, q_head) sparse selection arguments:

    sparse_len     : int32 [B, q_heads, 1]
    sparse_idx     : int64 [B, q_heads, max_context_length]
    sparse_weights : float32 [B, q_heads, max_context_length]

The compute path is `csrc/sparse_decode_kernel.cu`. The kernel reuses the
FlashInfer headers extracted under `original_optimized/csrc/flashinfer/` for
vectorized loads and stable log2-domain softmax math.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple, Union, overload

import torch

from original_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as _DenseOptimizedWrapper,
    _unpack_paged_kv_cache,
)

from .cuda_ops import get_sparse_decode_ops


class BatchDecodeWithPagedKVCacheWrapper(_DenseOptimizedWrapper):
    """Sparse paged-KV decode wrapper with a custom CUDA kernel backend."""

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
    ) -> None:
        super().__init__(
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
        self._sparse_ops = get_sparse_decode_ops()
        # K_SPLIT is computed lazily on first run() given the actual max_S of
        # sparse_idx; we cache the choice keyed by max_S to avoid repeat work.
        self._k_split_cache: dict[int, int] = {}
        # Heuristic knob: aim for this many CTAs per SM after split.
        # 4 was empirically the sweet spot on H100 across batch sizes 1..16
        # (tbps=8 helps low-batch a touch more but adds merge overhead at
        # high-batch).
        self._target_blocks_per_sm = 4

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        sparse_len: torch.Tensor,
        sparse_idx: torch.Tensor,
        sparse_weights: torch.Tensor,
        *args: Any,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        q_len_per_req: Optional[int] = 1,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        kv_cache_sf: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        sparse_len: torch.Tensor,
        sparse_idx: torch.Tensor,
        sparse_weights: torch.Tensor,
        *args: Any,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        q_len_per_req: Optional[int] = 1,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        kv_cache_sf: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        sparse_len: torch.Tensor,
        sparse_idx: torch.Tensor,
        sparse_weights: torch.Tensor,
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
        kv_cache_sf: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        del args, enable_pdl, sinks, skip_softmax_threshold_scale_factor, kv_cache_sf
        if not self._planned:
            raise RuntimeError("plan() must be called before run().")
        if q_len_per_req is None:
            q_len_per_req = 1
        if q_len_per_req != 1:
            raise NotImplementedError(
                "Sparse standalone wrapper only supports q_len_per_req == 1."
            )

        (
            k_cache,
            v_cache,
            kv_stride_page,
            kv_stride_axis1,
            kv_stride_axis2,
            kv_v_offset_elem,
            page_size_actual,
            num_kv_heads_actual,
        ) = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        # The kernel expects "stride_n" (position-axis stride) and "stride_h"
        # (head-axis stride) regardless of layout. _unpack_paged_kv_cache
        # returns axis-1 / axis-2 strides of the underlying K tensor.
        if self._kv_layout == "NHD":
            # K shape [P, page_size, H, D]: axis1=position (n), axis2=head (h)
            kv_stride_n = kv_stride_axis1
            kv_stride_h = kv_stride_axis2
        else:
            # HND: K shape [P, H, page_size, D]: axis1=head (h), axis2=position (n)
            kv_stride_n = kv_stride_axis2
            kv_stride_h = kv_stride_axis1

        batch_size = self._paged_kv_last_page_len_buf.numel()
        if q.shape[0] != batch_size:
            raise ValueError(
                f"q.shape[0] ({q.shape[0]}) must equal batch_size ({batch_size})."
            )
        if q.shape[1] != self._num_qo_heads or q.shape[2] != self._head_dim:
            raise ValueError(
                "q must have shape [batch_size, num_qo_heads, head_dim] matching plan()."
            )
        if sparse_len.shape != (batch_size, self._num_qo_heads, 1):
            raise ValueError(
                f"sparse_len must have shape "
                f"{(batch_size, self._num_qo_heads, 1)}, got {tuple(sparse_len.shape)}."
            )
        if sparse_idx.dim() != 3 or sparse_weights.dim() != 3:
            raise ValueError(
                "sparse_idx and sparse_weights must be rank-3 [B, q_heads, max_S]."
            )
        if sparse_idx.shape != sparse_weights.shape:
            raise ValueError(
                "sparse_idx and sparse_weights must have identical shapes."
            )
        if sparse_idx.shape[0] != batch_size or sparse_idx.shape[1] != self._num_qo_heads:
            raise ValueError(
                "sparse_idx/sparse_weights first two dims must be [B, q_heads]."
            )

        local_window_left = self._window_left if window_left is None else window_left
        if local_window_left is not None and local_window_left >= 0:
            raise NotImplementedError(
                "Sparse wrapper does not yet implement sliding window."
            )

        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale

        out_dtype = self._cached_o_data_type or q.dtype
        if out is not None and out.shape != q.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} does not match q shape {tuple(q.shape)}."
            )

        # Make inputs contiguous if necessary (most callers pass contiguous).
        q_in = q if q.is_contiguous() else q.contiguous()
        sparse_len_in = sparse_len.to(torch.int32) if sparse_len.dtype != torch.int32 else sparse_len
        sparse_len_in = sparse_len_in if sparse_len_in.is_contiguous() else sparse_len_in.contiguous()
        sparse_idx_in = sparse_idx.to(torch.int64) if sparse_idx.dtype != torch.int64 else sparse_idx
        sparse_idx_in = sparse_idx_in if sparse_idx_in.is_contiguous() else sparse_idx_in.contiguous()
        sparse_w_in = sparse_weights.to(torch.float32) if sparse_weights.dtype != torch.float32 else sparse_weights
        sparse_w_in = sparse_w_in if sparse_w_in.is_contiguous() else sparse_w_in.contiguous()

        max_S = int(sparse_idx_in.shape[2])
        k_split = self._k_split_cache.get(max_S)
        if k_split is None:
            k_split = int(self._sparse_ops.pick_k_split(
                int(batch_size), int(self._num_qo_heads), int(max_S),
                int(self._target_blocks_per_sm),
            ))
            self._k_split_cache[max_S] = k_split

        out_t, lse_t = self._sparse_ops.sparse_decode_run(
            q_in,
            k_cache,
            v_cache,
            int(kv_stride_page),
            int(kv_stride_n),
            int(kv_stride_h),
            int(kv_v_offset_elem),
            self._kv_indptr_int32,
            self._kv_indices_int32,
            sparse_len_in,
            sparse_idx_in,
            sparse_w_in,
            int(num_kv_heads_actual),
            int(page_size_actual),
            float(sm_scale),
            float(self._logits_soft_cap),
            int(k_split),
            bool(return_lse),
        )

        if out is not None:
            if out_t.dtype != out.dtype:
                out.copy_(out_t.to(out.dtype))
            else:
                out.copy_(out_t)
            out_buf = out
        else:
            out_buf = out_t if out_t.dtype == out_dtype else out_t.to(out_dtype)

        is_float_one = isinstance(v_scale, float) and v_scale == 1.0
        if v_scale is not None and not is_float_one:
            out_buf.mul_(v_scale)

        if return_lse:
            if lse is not None:
                if lse.shape != (q.shape[0], q.shape[1]):
                    raise ValueError(
                        "lse must have shape [batch_size, num_qo_heads]."
                    )
                lse.copy_(lse_t)
                return out_buf, lse
            return out_buf, lse_t
        return out_buf

    run_return_lse = lambda self, *args, **kwargs: self.run(  # noqa: E731
        *args, **kwargs, return_lse=True
    )
