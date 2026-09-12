"""
Optimized oracle top-k decode wrapper with attention-sink and local-window support.

Extends ``sparse_oracle_topk_optimized`` with ``sink_size`` and ``local_size``
parameters so each decode step selects tokens from three non-overlapping regions:

  1. **Sink**   – first ``sink_size`` tokens ``[0, sink_size)`` always kept.
  2. **Top-k**  – ``k = round(topk * mid_len)`` tokens selected by the highest
                  ``Q @ K^T`` score from the middle range
                  ``[sink_size, L_b - local_size)``.
  3. **Local**  – last ``local_size`` tokens ``[L_b - local_size, L_b)`` always kept.

``sink_size = 0`` and ``local_size = 0`` (the defaults) recover the original
``sparse_oracle_topk_optimized`` behaviour exactly.

Pipeline
--------
1. Compact score kernel:
     Dense ``Q @ K^T`` over only the middle range ``[sink_size, L_max-local_size)``,
     written in ``q.dtype`` (fp16/bf16).  Output is ``[B, H_q, L_mid]`` — no
     masking step needed since sink/local positions are never computed.
     Falls back to the full-range kernel when sink_size=0 and local_size=0.
2. Top-k selection (FlashInfer radix or ``torch.topk`` fallback):
     pick ``k_eff_mid`` indices from the compact scores.
3. Build combined index tensor ``[B, H_q, total_k]``:
       ``[sink_indices | topk_mid_indices | local_indices]``
4. Sparse decode kernel (reused from ``sparse_optimized``):
     full-head-dim paged-KV decode over the selected ``total_k`` indices.

All steps are device-only (no host–device syncs beyond the optional ``L_max``
resolution that the base class already does when ``max_seq_len`` is not supplied).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Optional, Tuple, Union, overload

import torch

from original_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as _DenseOptimizedWrapper,
    _unpack_paged_kv_cache,
)
from sparse_optimized.cuda_ops import get_sparse_decode_ops
from sparse_oracle_topk_optimized.cuda_ops import get_oracle_topk_ops
from .cuda_ops import get_compact_score_ops


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


def _resolve_topk_impl(prefer_flashinfer: bool = True) -> Callable:
    if prefer_flashinfer:
        try:
            import flashinfer  # noqa: WPS433
            fi_topk = getattr(flashinfer, "top_k", None)
            if callable(fi_topk):
                def _impl(scores_3d: torch.Tensor, k: int) -> torch.Tensor:
                    B, H_q, L = scores_3d.shape
                    scores_2d = scores_3d.reshape(B * H_q, L)
                    _, idx = fi_topk(scores_2d, k=k, sorted=False)
                    return idx.view(B, H_q, k)
                return _impl
        except Exception:  # noqa: BLE001
            pass

    def _torch_impl(scores_3d: torch.Tensor, k: int) -> torch.Tensor:
        return torch.topk(scores_3d, k=k, dim=-1, sorted=False).indices
    return _torch_impl


@dataclass
class SparsityStats:
    """Per-call sparsity counters maintained by the wrapper.

    ``last_*`` fields snapshot the most recent ``run()`` call. ``cumulative_*``
    fields aggregate across all ``run()`` calls since wrapper construction.
    All values come from Python-side state (the effective configuration the
    kernel ran with, after the per-request L_b cap); updating them is sync-free.

    Designed to be sampled cheaply after each request by downstream
    instrumentation (e.g. a ``prometheus_client.Gauge`` in the skylight
    inference backend) for silent-dense-fallback detection. If
    ``last_fraction`` stays at 1.0 when the operator configured a low topk,
    something — config-parsing, plugin registration, kernel dispatch —
    silently fell through to dense.
    """

    last_selected: int = 0
    """``min(total_k, n_keys)`` for the most recent run(): the per-row
    effective number of KV positions the kernel attended to."""

    last_total: int = 0
    """``n_keys`` (per-batch max context length) for the most recent run()."""

    last_fraction: float = 0.0
    """``last_selected / last_total`` for the most recent run() (0 if no calls)."""

    cumulative_selected: int = 0
    """Sum of ``last_selected`` across all run() calls since construction."""

    cumulative_total: int = 0
    """Sum of ``last_total`` across all run() calls since construction."""

    call_count: int = 0
    """Number of run() calls processed since construction."""


class BatchDecodeWithPagedKVCacheWrapper(_DenseOptimizedWrapper):
    """Optimized oracle top-k paged-KV decode with sink + local window support."""

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
        max_seq_len: Optional[int] = None,
        n_keys: Optional[int] = None,
        sink_size: int = 0,
        local_size: int = 0,
    ) -> None:
        """
        Args:
            topk: Default fraction of *middle* KV positions to keep.
                  Effective count ``k = max(1, min(mid_len, round(topk * mid_len)))``
                  where ``mid_len = n_keys - sink_size - local_size``.
            channel_num: Leading channels of Q/K used for the score dot-product.
                  Must be a multiple of 8 for fp16/bf16 and ``<= head_dim``.
                  ``None`` means use full ``head_dim``.
            max_seq_len: Engine-wide maximum context length (sync-free path).
            n_keys: True full-sequence length for topk-fraction scaling.
                  Defaults to ``L_max``.  Set to ``ctx`` when ``max_seq_len``
                  pads the score tensor beyond the true sequence length.
            sink_size: Number of leading tokens always included (attention sinks).
                  Stored as default; can be overridden per ``run()`` call.
            local_size: Number of trailing tokens always included (local window).
                  Stored as default; can be overridden per ``run()`` call.
        """
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
        self._oracle_ops = get_oracle_topk_ops()
        self._compact_ops = get_compact_score_ops()
        self._sparse_ops = get_sparse_decode_ops()
        self._k_split_cache: dict[int, int] = {}
        self._target_blocks_per_sm = 4
        self._topk_impl = _resolve_topk_impl(prefer_flashinfer=True)

        if topk is not None:
            _validate_topk_fraction(topk)
        if channel_num is not None and (
            not isinstance(channel_num, int) or channel_num <= 0
        ):
            raise ValueError(f"channel_num must be a positive int, got {channel_num!r}.")
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive int, got {max_seq_len!r}.")
        if n_keys is not None and n_keys <= 0:
            raise ValueError(f"n_keys must be a positive int, got {n_keys!r}.")
        if not isinstance(sink_size, int) or sink_size < 0:
            raise ValueError(f"sink_size must be a non-negative int, got {sink_size!r}.")
        if not isinstance(local_size, int) or local_size < 0:
            raise ValueError(f"local_size must be a non-negative int, got {local_size!r}.")

        self._topk_cfg: Optional[float] = float(topk) if topk is not None else None
        self._channel_num_cfg: Optional[int] = (
            int(channel_num) if channel_num is not None else None
        )
        self._max_seq_len_cfg: Optional[int] = (
            int(max_seq_len) if max_seq_len is not None else None
        )
        self._n_keys_cfg: Optional[int] = int(n_keys) if n_keys is not None else None
        self._sink_size_cfg: int = int(sink_size)
        self._local_size_cfg: int = int(local_size)

        # Per-call sparsity instrumentation (sync-free; see ``SparsityStats``).
        self._sparsity_stats = SparsityStats()

    # ------------------------------------------------------------------ setters

    def set_topk(self, topk: Optional[float]) -> None:
        if topk is not None:
            _validate_topk_fraction(topk)
        self._topk_cfg = float(topk) if topk is not None else None

    def set_channel_num(self, channel_num: Optional[int]) -> None:
        if channel_num is not None and (
            not isinstance(channel_num, int) or channel_num <= 0
        ):
            raise ValueError(f"channel_num must be a positive int, got {channel_num!r}.")
        self._channel_num_cfg = int(channel_num) if channel_num is not None else None

    def set_max_seq_len(self, max_seq_len: Optional[int]) -> None:
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be a positive int, got {max_seq_len!r}.")
        self._max_seq_len_cfg = int(max_seq_len) if max_seq_len is not None else None

    def set_n_keys(self, n_keys: Optional[int]) -> None:
        if n_keys is not None and n_keys <= 0:
            raise ValueError(f"n_keys must be a positive int, got {n_keys!r}.")
        self._n_keys_cfg = int(n_keys) if n_keys is not None else None

    def set_sink_size(self, sink_size: int) -> None:
        if not isinstance(sink_size, int) or sink_size < 0:
            raise ValueError(f"sink_size must be a non-negative int, got {sink_size!r}.")
        self._sink_size_cfg = int(sink_size)

    def set_local_size(self, local_size: int) -> None:
        if not isinstance(local_size, int) or local_size < 0:
            raise ValueError(f"local_size must be a non-negative int, got {local_size!r}.")
        self._local_size_cfg = int(local_size)

    # ------------------------------------------------------------------ instrumentation

    def get_sparsity_stats(self) -> SparsityStats:
        """Snapshot of the per-call sparsity counters.

        Returns a copy so callers can hold the value without worrying about
        subsequent ``run()`` calls mutating it. Safe to call from any thread
        (no GPU sync; pure Python attribute reads).
        """
        return replace(self._sparsity_stats)

    # ------------------------------------------------------------------ overloads

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk: Optional[float] = None,
        *args: Any,
        sink_size: Optional[int] = None,
        local_size: Optional[int] = None,
        channel_num: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        n_keys: Optional[int] = None,
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
        topk: Optional[float] = None,
        *args: Any,
        sink_size: Optional[int] = None,
        local_size: Optional[int] = None,
        channel_num: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        n_keys: Optional[int] = None,
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

    # ---------------------------------------------------------------------- run

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk: Optional[float] = None,
        *args: Any,
        sink_size: Optional[int] = None,
        local_size: Optional[int] = None,
        channel_num: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        n_keys: Optional[int] = None,
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
                "sink_local_optimized wrapper supports q_len_per_req == 1 only."
            )

        # Resolve topk
        if topk is None:
            topk = self._topk_cfg
        if topk is None:
            raise ValueError(
                "topk must be supplied via __init__(topk=...) / set_topk() or "
                "per-call as run(..., topk=...)."
            )
        topk_f = _validate_topk_fraction(topk)

        # Resolve channel_num
        if channel_num is None:
            channel_num = self._channel_num_cfg
        if channel_num is None:
            channel_num = self._head_dim
        if (
            not isinstance(channel_num, int)
            or channel_num <= 0
            or channel_num > self._head_dim
        ):
            raise ValueError(
                f"channel_num must be in [1, {self._head_dim}], got {channel_num!r}."
            )
        vec_lane = 8 if q.dtype in (torch.float16, torch.bfloat16) else 4
        if channel_num % vec_lane != 0:
            raise ValueError(
                f"channel_num must be a multiple of {vec_lane} for {q.dtype}, "
                f"got {channel_num}."
            )

        # Resolve sink_size / local_size
        sink_sz = self._sink_size_cfg if sink_size is None else int(sink_size)
        local_sz = self._local_size_cfg if local_size is None else int(local_size)
        if sink_sz < 0:
            raise ValueError(f"sink_size must be >= 0, got {sink_sz}.")
        if local_sz < 0:
            raise ValueError(f"local_size must be >= 0, got {local_sz}.")

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
        if self._kv_layout == "NHD":
            kv_stride_n = kv_stride_axis1
            kv_stride_h = kv_stride_axis2
        else:
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

        local_window_left = self._window_left if window_left is None else window_left
        if local_window_left is not None and local_window_left >= 0:
            raise NotImplementedError(
                "Sliding window (window_left) is not implemented for this wrapper; "
                "use local_size instead."
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

        q_in = q if q.is_contiguous() else q.contiguous()

        # ---- Paged-KV metadata ------------------------------------------------
        indptr_dev = self._kv_indptr_int32
        indices_dev = self._kv_indices_int32
        last_page_len_dev = self._kv_last_page_len_int32

        # ---- Resolve L_max (sync-free when max_seq_len is supplied) -----------
        max_seq_len_eff = (
            max_seq_len if max_seq_len is not None else self._max_seq_len_cfg
        )
        if max_seq_len_eff is not None:
            if max_seq_len_eff <= 0:
                raise ValueError(
                    f"max_seq_len must be a positive int, got {max_seq_len_eff!r}."
                )
            L_max = int(max_seq_len_eff)
        else:
            L_max = self._max_seq_len(
                indptr_dev, last_page_len_dev, page_size_actual
            )
            if L_max <= 0:
                raise ValueError("All sequences are empty.")

        n_keys_eff = n_keys if n_keys is not None else self._n_keys_cfg
        if n_keys_eff is None:
            n_keys_for_topk = int(L_max)
        else:
            n_keys_for_topk = int(n_keys_eff)
        if n_keys_for_topk <= 0:
            raise ValueError(f"n_keys must be positive, got {n_keys_for_topk!r}.")
        if n_keys_for_topk > L_max:
            raise ValueError(
                f"n_keys ({n_keys_for_topk}) must be <= L_max ({L_max})."
            )

        # Middle range length and effective k for top-k selection
        n_keys_mid = max(0, n_keys_for_topk - sink_sz - local_sz)
        k_eff_mid = _topk_fraction_to_k_eff(topk_f, n_keys_mid) if n_keys_mid > 0 else 0
        total_k = sink_sz + k_eff_mid + local_sz

        # Update sparsity instrumentation (sync-free; all Python ints).
        # selected is capped at n_keys_for_topk because sparse_len_per applies
        # the same per-request L_b cap below (step 6).
        _selected = min(total_k, n_keys_for_topk)
        self._sparsity_stats.last_selected = _selected
        self._sparsity_stats.last_total = n_keys_for_topk
        self._sparsity_stats.last_fraction = (
            _selected / n_keys_for_topk if n_keys_for_topk > 0 else 0.0
        )
        self._sparsity_stats.cumulative_selected += _selected
        self._sparsity_stats.cumulative_total += n_keys_for_topk
        self._sparsity_stats.call_count += 1

        # ---- 1. Compute per-batch sequence lengths on device (no sync) --------
        pages_count = indptr_dev[1:] - indptr_dev[:-1]   # [B], int32
        L_per = (
            (pages_count - 1).clamp(min=0).to(torch.int64) * int(page_size_actual)
            + last_page_len_dev.to(torch.int64)
        )  # [B], int64

        # ---- 2+3. Compact score kernel: Q @ K^T over [sink_sz, L_max-local_sz) only.
        # Gate on k_eff_mid (NOT on sink/local being nonzero): when L_max
        # <= sink_sz + local_sz (e.g. vLLM's memory-profile pass uses
        # seq_len=1 per request, far smaller than any reasonable sink+local
        # window), n_keys_mid is already 0 from line 422 and k_eff_mid from
        # line 423 — so there is no top-K work to do and the middle range
        # is empty or negative. Calling compute_scores_compact with
        # token_end <= token_start used to crash with "token_end must be >
        # token_start"; we now skip the scoring step entirely. Sink + local
        # together cover all available keys, which is exactly what the
        # combined sparse_idx in step 5 ends up encoding.
        scores = None
        mid_offset = sink_sz
        if k_eff_mid > 0:
            if sink_sz == 0 and local_sz == 0:
                # Fast path: full-range kernel, same as oracle_topk_optimized.
                scores = self._oracle_ops.oracle_topk_compute_scores(
                    q_in, k_cache,
                    int(kv_stride_page), int(kv_stride_n), int(kv_stride_h),
                    indptr_dev, indices_dev, last_page_len_dev,
                    int(num_kv_heads_actual), int(page_size_actual),
                    int(L_max), int(channel_num),
                )  # [B, H_q, L_max]
                mid_offset = 0
            else:
                token_start = sink_sz
                token_end = L_max - local_sz
                # k_eff_mid > 0 implies n_keys_mid > 0 (from line 423) which
                # implies token_end > token_start. Defensive assert in case
                # the invariant ever drifts.
                assert token_end > token_start, (
                    f"internal invariant violated: k_eff_mid={k_eff_mid} > 0 "
                    f"but middle range is empty (token_start={token_start}, "
                    f"token_end={token_end}, L_max={L_max}, sink={sink_sz}, "
                    f"local={local_sz})"
                )
                scores = self._compact_ops.oracle_topk_compute_scores_compact(
                    q_in, k_cache,
                    int(kv_stride_page), int(kv_stride_n), int(kv_stride_h),
                    indptr_dev, indices_dev, last_page_len_dev,
                    int(num_kv_heads_actual), int(page_size_actual),
                    int(token_start), int(token_end), int(channel_num),
                )  # [B, H_q, L_mid]
                mid_offset = token_start

        # ---- 4. Top-k selection ------------------------------------------------
        if k_eff_mid > 0:
            topk_idx = self._topk_impl(scores, k_eff_mid) + mid_offset  # [B, H_q, k_eff_mid]
        else:
            topk_idx = torch.empty(
                (batch_size, self._num_qo_heads, 0),
                dtype=torch.int64, device=q.device,
            )
        if scores is not None:
            del scores

        # ---- 5. Build combined index tensor [B, H_q, total_k] -----------------
        if sink_sz == 0 and local_sz == 0:
            sparse_idx = topk_idx.contiguous()
        else:
            parts: list[torch.Tensor] = []
            if sink_sz > 0:
                sink_idx = (
                    torch.arange(sink_sz, dtype=torch.int64, device=q.device)
                    .view(1, 1, -1)
                    .expand(batch_size, self._num_qo_heads, -1)
                )
                parts.append(sink_idx)
            if k_eff_mid > 0:
                parts.append(topk_idx)
            if local_sz > 0:
                # Per-batch local indices: L_b - local_sz, ..., L_b - 1
                local_start_idx = (L_per - local_sz).clamp(min=sink_sz)  # [B], int64
                local_offsets = torch.arange(local_sz, dtype=torch.int64, device=q.device)
                local_idx = (
                    local_start_idx.view(batch_size, 1, 1) + local_offsets.view(1, 1, -1)
                ).expand(batch_size, self._num_qo_heads, -1)  # [B, H_q, local_sz]
                parts.append(local_idx)
            sparse_idx = torch.cat(parts, dim=-1).contiguous()  # [B, H_q, total_k]

        # ---- 6. Sparse decode over the selected total_k tokens ----------------
        # Cap sparse_len per request at L_b. When L_b < total_k (short context
        # with non-trivial sink+local — e.g. a 50-token prompt with sink=128
        # local=256, or vLLM's profile_run with L=1), entries beyond the
        # request's allocated pages would otherwise read neighboring requests'
        # data (kv_indices[indptr[b]+page] indexes off the end of request b's
        # page slice). The first L_b entries of sparse_idx are by construction
        # valid positions [0..L_b-1] (sink covers [0..sink), local clamped to
        # start at sink covers [sink..L_b) for L_b <= sink+local), so the
        # kernel reads only valid positions when sparse_len is capped.
        #
        # Common non-degenerate case (L_b >= total_k): clamp is a no-op and
        # the per-request sparse_len equals the constant total_k as before.
        sparse_len_per = L_per.to(torch.int32).clamp(max=total_k)  # [B], int32
        sparse_len = (
            sparse_len_per.view(batch_size, 1, 1)
            .expand(batch_size, self._num_qo_heads, 1)
            .contiguous()
        )
        sparse_weights = torch.ones(
            (batch_size, self._num_qo_heads, total_k),
            dtype=torch.float32, device=q.device,
        )

        k_split = self._k_split_cache.get(total_k)
        if k_split is None:
            k_split = int(self._sparse_ops.pick_k_split(
                int(batch_size), int(self._num_qo_heads), int(total_k),
                int(self._target_blocks_per_sm),
            ))
            self._k_split_cache[total_k] = k_split

        out_t, lse_t = self._sparse_ops.sparse_decode_run(
            q_in,
            k_cache,
            v_cache,
            int(kv_stride_page),
            int(kv_stride_n),
            int(kv_stride_h),
            int(kv_v_offset_elem),
            indptr_dev,
            indices_dev,
            sparse_len,
            sparse_idx,
            sparse_weights,
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
                    raise ValueError("lse must have shape [batch_size, num_qo_heads].")
                lse.copy_(lse_t)
                return out_buf, lse
            return out_buf, lse_t
        return out_buf

    # ------------------------------------------------------------------ helpers

    def _max_seq_len(
        self,
        indptr_dev: torch.Tensor,
        last_page_len_dev: torch.Tensor,
        page_size: int,
    ) -> int:
        """Return max over batches; cached to avoid repeated host syncs."""
        key = (indptr_dev.data_ptr(), last_page_len_dev.data_ptr())
        cached = getattr(self, "_lmax_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        pages = indptr_dev[1:] - indptr_dev[:-1]
        L_per = (pages - 1).clamp(min=0) * page_size + last_page_len_dev.to(torch.int32)
        L_max = int(L_per.max().item()) if L_per.numel() > 0 else 0
        self._lmax_cache = (key, L_max)
        return L_max

    run_return_lse = lambda self, *args, **kwargs: self.run(  # noqa: E731
        *args, **kwargs, return_lse=True
    )
