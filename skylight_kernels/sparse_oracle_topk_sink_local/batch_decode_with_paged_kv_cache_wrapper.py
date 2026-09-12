"""
Oracle top-k sparse standalone BatchDecodeWithPagedKVCacheWrapper
with attention-sink and local-window support.

Same plan/init behavior as `original`, but `run(...)` takes:
  - ``topk: float``      – fraction of *middle* KV positions to keep.
  - ``sink_size: int``   – number of leading tokens always kept (attention sinks).
  - ``local_size: int``  – number of trailing tokens always kept (local window).
  - ``channel_num: int`` – optional; restricts the selection dot-product to the
                           first N feature dims (default: full head_dim).

Per (batch, head), the selected token indices are the union of:
  1. Sink    – indices  ``[0, sink_size)``
  2. Top-k   – the ``k`` indices from ``[sink_size, seq_len - local_size)`` with
               the largest ``q[..., :channel_num] @ k[..., :channel_num]^T``
               scores, where ``k = max(1, min(mid_len, round(topk * mid_len)))``.
  3. Local   – indices  ``[seq_len - local_size, seq_len)``

When ``sink_size + local_size >= seq_len`` the middle range is empty and all
available tokens are used directly (sink and local are clamped to avoid overlap).
The full ``head_dim`` is always used for the actual sparse attention computation.

Selection scores are accumulated in fp32 and then rounded to ``q.dtype``
(fp16/bf16) before the top-k pick. This matches the optimized CUDA kernel
under ``sparse_oracle_topk_optimized/`` which writes its score tensor in
the same dtype to halve memory traffic, so this Python reference is
bit-for-set-equivalent with the optimized path on the same inputs.

Internally constructs per-batch sparse plan tensors with shapes:
  - sparse_len:     [q_heads]                        (uniform total_k per head)
  - sparse_idx:     [q_len, q_heads, total_k]
  - sparse_weights: [q_len, q_heads, total_k]        (all ones; plain softmax)

Currently only supports q_len_per_req == 1.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple, Union, overload

import torch

from original.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as _DenseWrapper,
)
from original.batch_decode_with_paged_kv_cache_wrapper import _extract_page, _unpack_paged_kv_cache


def _topk_fraction_to_k_eff(topk: float, n_keys: int) -> int:
    if n_keys <= 0:
        return 0
    k = int(round(float(topk) * float(n_keys)))
    k = max(1, k)
    return min(k, n_keys)


def _validate_topk_fraction(topk: object) -> float:
    if isinstance(topk, bool):
        raise TypeError(f"topk must be a float (not bool), got {topk!r}.")
    if not isinstance(topk, (int, float)):
        raise TypeError(
            f"topk must be a float (fraction of keys to keep), got {type(topk).__name__}."
        )
    v = float(topk)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(
            f"topk must be finite and > 0 (fraction of KV positions), got {topk!r}."
        )
    return v


class BatchDecodeWithPagedKVCacheWrapper(_DenseWrapper):
    """Sparse version of standalone paged-KV decode wrapper."""

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk: float,
        *args: Any,
        sink_size: int = 0,
        local_size: int = 0,
        channel_num: Optional[int] = None,
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
        topk: float,
        *args: Any,
        sink_size: int = 0,
        local_size: int = 0,
        channel_num: Optional[int] = None,
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
        topk: float,
        *args: Any,
        sink_size: int = 0,
        local_size: int = 0,
        channel_num: Optional[int] = None,
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
            raise ValueError(
                "sparse_oracle_topk currently supports q_len_per_req == 1 only."
            )
        topk_f = _validate_topk_fraction(topk)
        if channel_num is None:
            channel_num = self._head_dim
        if not isinstance(channel_num, int) or channel_num <= 0 or channel_num > self._head_dim:
            raise ValueError(
                f"channel_num must be in [1, {self._head_dim}], got {channel_num!r}."
            )

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
            seq_len = k_seq.shape[0]
            # Build the per-batch sparse plan via oracle top-k over q @ k^T.
            # sm_scale and logits_soft_cap are monotonic transforms, so omitting
            # them here does not change the top-k indices.
            # Clamp sink/local to the available sequence length so they never overlap.
            sink_eff = min(sink_size, seq_len)
            local_eff = min(local_size, max(0, seq_len - sink_eff))
            mid_start = sink_eff
            mid_end = seq_len - local_eff  # exclusive upper bound of the middle range
            mid_len = mid_end - mid_start   # >= 0

            # Top-k selection from the middle range [mid_start, mid_end).
            # We accumulate in fp32 (avoids spurious rank flips for tied
            # near-zero scores) and then round to q.dtype before topk.
            # The optimized CUDA kernel does exactly the same: fp32
            # accumulation, then a write that quantises to fp16/bf16.
            if mid_len > 0:
                k_eff_mid = _topk_fraction_to_k_eff(topk_f, mid_len)
                selection_scores = torch.einsum(
                    "qhd,lhd->qhl",
                    q_reshaped[b][..., :channel_num].float(),
                    k_seq[mid_start:mid_end][..., :channel_num].float(),
                ).to(q.dtype)  # [q_len, q_heads, mid_len]
                # Indices are relative to k_seq[mid_start:]; shift to global.
                topk_global = (
                    torch.topk(selection_scores, k=k_eff_mid, dim=-1).indices + mid_start
                )  # [q_len, q_heads, k_eff_mid]
            else:
                k_eff_mid = 0
                topk_global = torch.empty(
                    (q_len_per_req, self._num_qo_heads, 0), dtype=torch.long, device=q.device
                )

            total_k = sink_eff + k_eff_mid + local_eff
            sparse_len = torch.full(
                (self._num_qo_heads,), total_k, dtype=torch.int32, device=q.device
            )  # [q_heads]

            # Build combined index tensor: [sink | topk_mid | local] broadcast
            # over the q_len and q_heads dimensions.
            sink_idx_exp = (
                torch.arange(sink_eff, dtype=torch.long, device=q.device)
                .view(1, 1, -1)
                .expand(q_len_per_req, self._num_qo_heads, -1)
            )  # [q_len, q_heads, sink_eff]
            local_idx_exp = (
                torch.arange(mid_end, seq_len, dtype=torch.long, device=q.device)
                .view(1, 1, -1)
                .expand(q_len_per_req, self._num_qo_heads, -1)
            )  # [q_len, q_heads, local_eff]
            sparse_idx = torch.cat(
                [sink_idx_exp, topk_global, local_idx_exp], dim=-1
            )  # [q_len, q_heads, total_k]
            # All-ones weights => log(1) = 0 => weighted softmax reduces to plain softmax.
            sparse_weights = torch.ones_like(sparse_idx, dtype=k_seq.dtype)
            for qh in range(self._num_qo_heads):
                valid_sparse = int(sparse_len[qh].item())
                if valid_sparse == 0:
                    out_reshaped[b, :, qh, :].zero_()
                    if lse_out is not None:
                        lse_reshaped[b, :, qh].fill_(float("-inf"))
                    continue

                token_idx = sparse_idx[0, qh, :valid_sparse].to(device=q.device, dtype=torch.long)  # [k_eff]
                if torch.any(token_idx < 0) or torch.any(token_idx >= seq_len):
                    raise ValueError(
                        f"sparse_idx has values outside [0, {seq_len - 1}] for batch={b}, head={qh}."
                    )

                token_weights = sparse_weights[0, qh, :valid_sparse].to(device=q.device, dtype=torch.float32)  # [k_eff]
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

