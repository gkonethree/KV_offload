"""
Oracle top-k sparse standalone BatchDecodeWithPagedKVCacheWrapper.

Same plan/init behavior as `original`, but `run(...)` takes an extra
``topk: float`` argument (fraction of KV positions to keep per batch:
``k = max(1, min(seq_len, int(round(topk * seq_len))))``) and an optional
``channel_num: int``, and internally selects, per (batch, head), those
``k`` key positions with the largest
``q[..., :channel_num] @ k[..., :channel_num]^T`` scores. The full ``head_dim``
is still used for the actual sparse attention computation; ``channel_num``
only restricts the *selection* dot product. ``channel_num=None`` (default)
falls back to using the full head dim, recovering the original oracle.

Selection scores are accumulated in fp32 and then rounded to ``q.dtype``
(fp16/bf16) before the top-k pick. This matches the optimized CUDA kernel
under ``sparse_oracle_topk_optimized/`` which writes its score tensor in
the same dtype to halve memory traffic, so this Python reference is
bit-for-set-equivalent with the optimized path on the same inputs.

Internally constructs per-batch sparse plan tensors with shapes:
  - sparse_len:     [q_heads]              (uniform k per head, k from ``topk``)
  - sparse_idx:     [q_len, q_heads, k]
  - sparse_weights: [q_len, q_heads, k] (all ones; reduces to plain softmax)

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
            k_eff = _topk_fraction_to_k_eff(topk_f, seq_len)
            sparse_len = torch.full(
                (self._num_qo_heads,), k_eff, dtype=torch.int32, device=q.device
            )  # [q_heads]
            # Compute selection scores over only the first ``channel_num``
            # feature dims of Q and K. Restricting the dot product to a subset
            # of channels acts as a cheap proxy for the full-D similarity and
            # ~halves the K bandwidth needed for the selection pass when
            # channel_num <= 64 (see optimized kernel).
            #
            # We accumulate in fp32 (avoids spurious rank flips for tied
            # near-zero scores) and then round down to q.dtype before topk.
            # The optimized CUDA kernel does exactly the same thing: fp32
            # accumulation, then a final write that quantises to fp16/bf16
            # storage. Matching that here keeps the two paths
            # set-equivalent on the same inputs.
            selection_scores = torch.einsum(
                "qhd,lhd->qhl",
                q_reshaped[b][..., :channel_num].float(),
                k_seq[..., :channel_num].float(),
            ).to(q.dtype)  # [q_len, q_heads, L] in q.dtype
            sparse_idx = torch.topk(
                selection_scores, k=k_eff, dim=-1
            ).indices  # [q_len, q_heads, k_eff]
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

