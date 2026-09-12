"""
Standalone BatchDecodeWithPagedKVCacheWrapper implementation.

Adapted from the public API and behavior of FlashInfer:
https://github.com/flashinfer-ai/flashinfer

This file intentionally keeps a FlashInfer-like interface (`plan` + `run`) but
implements the compute path in plain PyTorch so it can run without custom CUDA
kernels from the FlashInfer package.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple, Union, overload

import torch


def _check_kv_layout(kv_layout: str) -> None:
    if kv_layout not in ("NHD", "HND"):
        raise ValueError("kv_layout must be either 'NHD' or 'HND'.")


def _canonicalize_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype}")
    return mapping[key]


def _unpack_paged_kv_cache(
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_layout: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(paged_kv_cache, tuple):
        if len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache tuple must be (k_cache, v_cache).")
        return paged_kv_cache

    if not torch.is_tensor(paged_kv_cache):
        raise TypeError("paged_kv_cache must be a Tensor or (Tensor, Tensor).")
    if paged_kv_cache.dim() != 5:
        raise ValueError("single paged_kv_cache tensor must be rank-5.")

    if kv_layout == "NHD":
        # [num_pages, 2, page_size, num_kv_heads, head_dim]
        return paged_kv_cache[:, 0], paged_kv_cache[:, 1]
    # [num_pages, 2, num_kv_heads, page_size, head_dim]
    return paged_kv_cache[:, 0], paged_kv_cache[:, 1]


def _extract_page(kv_tensor: torch.Tensor, page_idx: int, kv_layout: str) -> torch.Tensor:
    page = kv_tensor[page_idx]
    if kv_layout == "NHD":
        # [page_size, num_kv_heads, head_dim]
        return page
    # HND -> [num_kv_heads, page_size, head_dim] -> [page_size, num_kv_heads, head_dim]
    return page.permute(1, 0, 2)


class BatchDecodeWithPagedKVCacheWrapper:
    """Standalone FlashInfer-like paged-KV decode wrapper."""

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
        _check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._use_cuda_graph = use_cuda_graph
        self._use_tensor_cores = use_tensor_cores
        self._backend = backend
        self._jit_args = jit_args

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._fixed_batch_size = 0

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError("paged_kv_indptr_buffer should be a Tensor in cuda graph mode.")
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError("paged_kv_indices_buffer should be a Tensor in cuda graph mode.")
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError("paged_kv_last_page_len_buffer should be a Tensor in cuda graph mode.")
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError("paged_kv_indptr_buffer size must be batch_size + 1.")

        self._planned = False
        self._pos_encoding_mode = "NONE"
        self._window_left = -1
        self._logits_soft_cap = 0.0
        self._sm_scale = None
        self._rope_scale = None
        self._rope_theta = None

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

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
        del block_tables, seq_lens, fixed_split_size, disable_split_kv
        if len(last_page_len) + 1 != len(indptr):
            raise ValueError("len(indptr) must be batch_size + 1.")
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("num_qo_heads must be a multiple of num_kv_heads.")
        if page_size < 1:
            raise ValueError("page_size must be >= 1.")

        batch_size = len(last_page_len)
        if self.is_cuda_graph_enabled and batch_size != self._fixed_batch_size:
            raise ValueError(
                f"batch size {batch_size} does not match fixed cuda graph batch size {self._fixed_batch_size}."
            )

        if self.is_cuda_graph_enabled:
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(last_page_len, non_blocking=non_blocking)
            self._paged_kv_indices_buf[: len(indices)].copy_(indices, non_blocking=non_blocking)
        else:
            self._paged_kv_indptr_buf = indptr.to(self.device, non_blocking=non_blocking)
            self._paged_kv_indices_buf = indices.to(self.device, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf = last_page_len.to(self.device, non_blocking=non_blocking)

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_dtype = _canonicalize_dtype(q_data_type)
        kv_dtype = _canonicalize_dtype(kv_data_type) or q_dtype
        o_dtype = _canonicalize_dtype(o_data_type) or q_dtype
        self._cached_q_data_type = q_dtype
        self._cached_kv_data_type = kv_dtype
        self._cached_o_data_type = o_dtype

        self._batch_size = batch_size
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = 0.0 if logits_soft_cap is None else logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._planned = True

    begin_forward = plan

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
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
        if q_len_per_req < 1:
            raise ValueError("q_len_per_req must be >= 1.")

        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        batch_size = self._paged_kv_last_page_len_buf.numel()
        expected_q = batch_size * q_len_per_req
        if q.shape[0] != expected_q:
            raise ValueError(
                f"q.shape[0] ({q.shape[0]}) must be batch_size * q_len_per_req ({expected_q})."
            )
        if q.shape[1] != self._num_qo_heads or q.shape[2] != self._head_dim:
            raise ValueError(
                "q must have shape [batch_size * q_len_per_req, num_qo_heads, head_dim] "
                "matching plan()."
            )

        local_window_left = self._window_left if window_left is None else window_left
        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        group_size = self._num_qo_heads // self._num_kv_heads

        out_dtype = self._cached_o_data_type or q.dtype
        if out is None:
            out = torch.empty_like(q, dtype=out_dtype)
        else:
            if out.shape != q.shape:
                raise ValueError(f"out shape {tuple(out.shape)} does not match q shape {tuple(q.shape)}.")

        lse_out: Optional[torch.Tensor] = None
        if return_lse:
            if lse is None:
                lse_out = torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
            else:
                if lse.shape != (q.shape[0], q.shape[1]):
                    raise ValueError("lse must have shape [batch_size * q_len_per_req, num_qo_heads].")
                lse_out = lse

        q_reshaped = q.view(batch_size, q_len_per_req, self._num_qo_heads, self._head_dim)
        out_reshaped = out.view(batch_size, q_len_per_req, self._num_qo_heads, self._head_dim)
        if lse_out is not None:
            lse_reshaped = lse_out.view(batch_size, q_len_per_req, self._num_qo_heads)

        indptr = self._paged_kv_indptr_buf
        indices = self._paged_kv_indices_buf
        last_page_len = self._paged_kv_last_page_len_buf

        for b in range(batch_size):
            start = int(indptr[b].item())
            end = int(indptr[b + 1].item())
            if end <= start:
                raise ValueError("Each request must have at least one page.")

            page_ids = indices[start:end].tolist()
            k_pages = []
            v_pages = []
            for local_i, page_id in enumerate(page_ids):
                k_page = _extract_page(k_cache, page_id, self._kv_layout)
                v_page = _extract_page(v_cache, page_id, self._kv_layout)
                if local_i == len(page_ids) - 1:
                    valid = int(last_page_len[b].item())
                    k_page = k_page[:valid]
                    v_page = v_page[:valid]
                k_pages.append(k_page)
                v_pages.append(v_page)

            k_seq = torch.cat(k_pages, dim=0)  # [L, kv_heads, D]
            v_seq = torch.cat(v_pages, dim=0)  # [L, kv_heads, D]
            k_seq = k_seq.repeat_interleave(group_size, dim=1)  # [L, q_heads, D]
            v_seq = v_seq.repeat_interleave(group_size, dim=1)  # [L, q_heads, D]

            q_b = q_reshaped[b]  # [q_len, q_heads, D]
            # [q_len, q_heads, L]
            scores = torch.einsum("qhd,lhd->qhl", q_b, k_seq).to(torch.float32) * sm_scale

            if self._logits_soft_cap > 0:
                cap = self._logits_soft_cap
                scores = cap * torch.tanh(scores / cap)

            if local_window_left is not None and local_window_left >= 0:
                l = k_seq.shape[0]
                left_bound = max(0, l - local_window_left)
                if left_bound > 0:
                    scores[..., :left_bound] = float("-inf")

            attn = torch.softmax(scores, dim=-1)
            out_b = torch.einsum("qhl,lhd->qhd", attn.to(v_seq.dtype), v_seq)
            out_reshaped[b].copy_(out_b.to(out_dtype))

            if lse_out is not None:
                # FlashInfer reports LSE in log2 domain.
                lse_reshaped[b].copy_(torch.logsumexp(scores, dim=-1) / math.log(2.0))

        is_float_one = isinstance(v_scale, float) and v_scale == 1.0
        if v_scale is not None and not is_float_one:
            out.mul_(v_scale)

        if return_lse:
            assert lse_out is not None
            return out, lse_out
        return out

    run_return_lse = lambda self, *args, **kwargs: self.run(*args, **kwargs, return_lse=True)

