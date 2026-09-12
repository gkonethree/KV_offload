"""
Sparse standalone BatchDecodeWithPagedKVCacheWrapper implementation.

This keeps the same plan/init behavior as `original` but augments `run(...)`
with sparse inputs:
  - sparse_len: (B, q_heads, 1)
  - sparse_idx: (B, q_heads, max_context_length)
  - sparse_weights: (B, q_heads, max_context_length)
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple, Union, overload

import torch

from original.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as _DenseWrapper,
)
from original.batch_decode_with_paged_kv_cache_wrapper import _extract_page, _unpack_paged_kv_cache


class BatchDecodeWithPagedKVCacheWrapper(_DenseWrapper):
    """Sparse version of standalone paged-KV decode wrapper."""

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

        if sparse_len.shape != (batch_size, self._num_qo_heads, 1):
            raise ValueError(
                f"sparse_len must have shape {(batch_size, self._num_qo_heads, 1)}, "
                f"got {tuple(sparse_len.shape)}."
            )
        if sparse_idx.dim() != 3:
            raise ValueError("sparse_idx must be rank-3: [B, q_heads, max_context_length].")
        if sparse_weights.dim() != 3:
            raise ValueError(
                "sparse_weights must be rank-3: [B, q_heads, max_context_length]."
            )
        if sparse_idx.shape != sparse_weights.shape:
            raise ValueError("sparse_idx and sparse_weights must have identical shapes.")
        if sparse_idx.shape[0] != batch_size or sparse_idx.shape[1] != self._num_qo_heads:
            raise ValueError(
                "sparse_idx/sparse_weights first two dims must be [B, q_heads] matching q."
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
        max_context_length = sparse_idx.shape[-1]

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
            seq_len = k_seq.shape[0]

            for qh in range(self._num_qo_heads):
                valid_sparse = int(sparse_len[b, qh, 0].item())
                if valid_sparse < 0 or valid_sparse > max_context_length:
                    raise ValueError(
                        f"sparse_len[{b},{qh}]={valid_sparse} is out of range [0, {max_context_length}]."
                    )

                if valid_sparse == 0:
                    out_reshaped[b, :, qh, :].zero_()
                    if lse_out is not None:
                        lse_reshaped[b, :, qh].fill_(float("-inf"))
                    continue

                token_idx = sparse_idx[b, qh, :valid_sparse].to(device=q.device, dtype=torch.long)
                if torch.any(token_idx < 0) or torch.any(token_idx >= seq_len):
                    raise ValueError(
                        f"sparse_idx has values outside [0, {seq_len - 1}] for batch={b}, head={qh}."
                    )

                token_weights = sparse_weights[b, qh, :valid_sparse].to(device=q.device, dtype=torch.float32)
                if torch.any(token_weights < 0):
                    raise ValueError("sparse_weights must be non-negative.")

                if local_window_left is not None and local_window_left >= 0:
                    left_bound = max(0, seq_len - local_window_left)
                    keep = token_idx >= left_bound
                    token_idx = token_idx[keep]
                    token_weights = token_weights[keep]
                    if token_idx.numel() == 0:
                        out_reshaped[b, :, qh, :].zero_()
                        if lse_out is not None:
                            lse_reshaped[b, :, qh].fill_(float("-inf"))
                        continue

                # Weighted sparse attention per head:
                # softmax(scores + log(weights + eps)) over sparse indices.
                k_sel = k_seq[token_idx, qh, :]  # [S, D]
                v_sel = v_seq[token_idx, qh, :]  # [S, D]
                q_sel = q_reshaped[b, :, qh, :]  # [q_len, D]

                scores = torch.einsum("qd,sd->qs", q_sel, k_sel).to(torch.float32) * sm_scale
                if self._logits_soft_cap > 0:
                    cap = self._logits_soft_cap
                    scores = cap * torch.tanh(scores / cap)

                weighted_scores = scores + torch.log(token_weights.clamp_min(1e-20)).unsqueeze(0)
                attn = torch.softmax(weighted_scores, dim=-1).to(v_sel.dtype)
                out_head = torch.einsum("qs,sd->qd", attn, v_sel)
                out_reshaped[b, :, qh, :].copy_(out_head.to(out_dtype))

                if lse_out is not None:
                    # Match FlashInfer convention: log2 domain.
                    lse_reshaped[b, :, qh].copy_(
                        torch.logsumexp(weighted_scores, dim=-1) / math.log(2.0)
                    )

        is_float_one = isinstance(v_scale, float) and v_scale == 1.0
        if v_scale is not None and not is_float_one:
            out.mul_(v_scale)

        if return_lse:
            assert lse_out is not None
            return out, lse_out
        return out

    run_return_lse = lambda self, *args, **kwargs: self.run(*args, **kwargs, return_lse=True)

