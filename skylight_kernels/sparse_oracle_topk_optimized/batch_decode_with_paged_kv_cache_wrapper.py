"""
Optimized standalone oracle top-k decode wrapper backed by two custom CUDA
kernels.

Pipeline:
  1. Score kernel (this package, `csrc/oracle_topk_score_kernel.cu`):
       For every (b, q_head, t) compute the dense Q @ K^T score directly
       from the paged KV cache, using only the first ``channel_num`` feature
       dims of Q and K (``channel_num <= head_dim``; defaults to head_dim,
       must be a multiple of the vec lane width = 8 for fp16/bf16). K reads
       are shared across the ``GROUP_SIZE = H_q / H_kv`` query heads that map
       to each KV head. With ``channel_num <= 64`` the kernel reads ~half as
       many K cache lines as the full-head-dim variant.
       Output: ``scores[B, H_q, L_max]`` in fp32, with positions beyond the
       per-batch sequence length filled with ``-inf``.
  2. Top-k selection over the last dim picks the per-(b, q_head) top-k token
     indices. Uses FlashInfer's radix-based ``flashinfer.top_k`` when
     available (1.3-3.1x faster than ``torch.topk`` at L=128K), and falls
     back to ``torch.topk`` otherwise.
  3. Sparse decode kernel (reused from `sparse_optimized`): runs paged-KV
     decode over the FULL head_dim using only the selected indices, with
     all-ones weights so the weighted softmax reduces to a plain softmax.

API mirrors `original_optimized.BatchDecodeWithPagedKVCacheWrapper.run` but
``run(...)`` takes additional ``topk: float`` (fraction of KV positions to
keep: ``k = round(topk * n_keys)``, clamped to ``[1, n_keys]``) and
``channel_num: int``, and constructs the sparse plan internally. The integer
``n_keys`` defaults to ``L_max`` (the score tensor's last-dim length). When
``max_seq_len`` is larger than the true per-batch max length, pass
``run(..., n_keys=...)`` (or ``__init__(n_keys=...)``) with the host-known max
sequence length so ``k`` matches ``round(topk * true_L)`` without any
device-to-host sync. Currently supports ``q_len_per_req == 1``.

Sync-free fast path: callers that know the engine's ``max_seq_len`` (e.g.
vLLM's ``max_model_len``) can supply it via ``__init__(max_seq_len=...)``
or per-call ``run(..., max_seq_len=...)``. When set, the wrapper skips the
device-side per-batch ``L_max`` computation (which otherwise costs one
``.max().item()`` host-device sync on the first call after each
``plan()``) and ``run()`` becomes end-to-end async / CUDA-Graph friendly.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, Optional, Tuple, Union, overload

import torch

from original_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as _DenseOptimizedWrapper,
    _unpack_paged_kv_cache,
)
from sparse_optimized.cuda_ops import get_sparse_decode_ops

from .cuda_ops import get_oracle_topk_ops


def _topk_fraction_to_k_eff(topk: float, n_keys: int) -> int:
    """Convert fractional ``topk`` to an integer k for ``torch.topk`` / FlashInfer.

    ``k_eff = max(1, min(n_keys, int(round(topk * n_keys))))`` where ``n_keys`` is
    the number of valid KV positions used for scaling (typically the true max
    sequence length over the batch). It may be smaller than ``L_max`` when the
    score tensor is padded to ``max_seq_len >`` that max. Passing ``topk =
    sparsity`` with uniform context length ``ctx`` and ``n_keys = ctx``
    reproduces the historical ``max(1, int(round(sparsity * ctx)))`` behaviour
    from the profiling scripts.
    """
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
    """Return a callable ``(scores_3d, k) -> indices_3d (int64)``.

    FlashInfer's ``flashinfer.top_k`` is a radix-based selection that is
    materially faster than ``torch.topk`` once the inner-dim is large
    (>10K). At our long-context shapes (L_max = 128K) it cuts the topk
    stage cost from ~0.9 ms to ~0.3 ms at 1% sparsity.
    """
    if prefer_flashinfer:
        try:
            import flashinfer  # noqa: WPS433  (intentional optional import)
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


class BatchDecodeWithPagedKVCacheWrapper(_DenseOptimizedWrapper):
    """Optimized oracle top-k paged-KV decode wrapper."""

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
    ) -> None:
        """
        Args:
            topk: Default fraction of KV positions to keep per ``(batch,
                query_head)``. Effective count is
                ``max(1, min(n_keys, int(round(topk * n_keys))))`` where
                ``n_keys`` defaults to the score-axis length ``L_max`` unless
                overridden by ``n_keys=`` here or per ``run(..., n_keys=...)``.
                When ``max_seq_len`` pads the score tensor beyond the true batch
                max, set ``n_keys`` to that true max (host int, no GPU read) so
                ``k`` stays correct and ``run()`` stays sync-free. When set,
                callers may omit ``topk`` from ``run()``. Override per call via
                ``run(..., topk=...)`` or :meth:`set_topk`.
            channel_num: Default number of leading head channels used for
                the dense ``Q @ K^T`` selection score (must be a multiple
                of the kernel's vec lane width: 8 for fp16/bf16, 4 for
                fp32; ``<= head_dim``). When ``None`` (default), the
                kernel uses the full ``head_dim``. Can be overridden per
                call via ``run(..., channel_num=...)`` or updated later
                via :meth:`set_channel_num`.
            max_seq_len: Engine-wide maximum context length (vLLM's
                ``max_model_len``). When provided, ``run()`` uses this value
                as the score-tensor's last-dim bound and **skips** the
                device-side ``L_per.max().item()`` host-device sync. The
                score kernel still pads positions ``[L_b, max_seq_len)``
                with ``-inf`` per request, so top-k correctness is preserved
                even when ``max_seq_len`` is strictly larger than the
                longest active sequence in the batch. Trade-off: the score
                tensor and the score kernel both touch
                ``B * H_q * max_seq_len`` slots regardless of the actual
                per-request lengths, which adds ``(max_seq_len - L_max) *
                B * H_q`` padding writes in the kernel and a corresponding
                top-k read pass over them. For long-context serving
                (``L_max ~= max_seq_len``) the overhead is negligible.

                Can also be supplied / overridden per-call via
                ``run(..., max_seq_len=...)``.
            n_keys: Optional default for ``run(..., n_keys=...)`` (see
                :meth:`run`). When ``max_seq_len`` is a padded upper bound,
                set this to the current batch's true max sequence length so
                ``round(topk * n_keys)`` does not use the padded axis length.
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
        self._sparse_ops = get_sparse_decode_ops()
        # Cached choice of K_SPLIT for the sparse decode kernel, keyed by k_eff.
        self._k_split_cache: dict[int, int] = {}
        # Heuristic knob (mirrors sparse_optimized): aim for this many CTAs/SM.
        self._target_blocks_per_sm = 4
        # Resolve top-k impl once: prefer FlashInfer's radix top-k when present,
        # fall back to ``torch.topk`` otherwise.
        self._topk_impl = _resolve_topk_impl(prefer_flashinfer=True)
        if topk is not None:
            _validate_topk_fraction(topk)
        if channel_num is not None and (
            not isinstance(channel_num, int) or channel_num <= 0
        ):
            raise ValueError(
                f"channel_num must be a positive int, got {channel_num!r}."
            )
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive int, got {max_seq_len!r}."
            )
        if n_keys is not None and n_keys <= 0:
            raise ValueError(f"n_keys must be a positive int, got {n_keys!r}.")
        self._topk_cfg: Optional[float] = float(topk) if topk is not None else None
        self._channel_num_cfg: Optional[int] = (
            int(channel_num) if channel_num is not None else None
        )
        self._max_seq_len_cfg: Optional[int] = (
            int(max_seq_len) if max_seq_len is not None else None
        )
        self._n_keys_cfg: Optional[int] = (
            int(n_keys) if n_keys is not None else None
        )
        # Store last top-k indices for debugging/inspection
        self._last_topk_idx: Optional[torch.Tensor] = None

    def get_last_topk_indices(self) -> Optional[torch.Tensor]:
        """Return the top-k token indices selected in the last run().
        
        Returns tensor of shape [batch_size, num_qo_heads, k_eff] with the
        selected token indices for each (batch, query_head) pair.
        """
        return self._last_topk_idx

    def get_last_topk_indices_for_batch(self, batch_idx: int, head_idx: int = 0) -> Optional[list[int]]:
        """Return top-k indices for a specific batch and head as a list."""
        if self._last_topk_idx is None:
            return None
        if batch_idx >= self._last_topk_idx.shape[0]:
            return None
        return self._last_topk_idx[batch_idx, head_idx].tolist()

    def set_topk(self, topk: Optional[float]) -> None:
        """Update the default fractional ``topk`` after construction.

        Pass ``None`` to require ``topk`` to be supplied per ``run()`` call.
        """
        if topk is not None:
            _validate_topk_fraction(topk)
        self._topk_cfg = float(topk) if topk is not None else None

    def set_channel_num(self, channel_num: Optional[int]) -> None:
        """Update the default ``channel_num`` after construction.

        Pass ``None`` to revert to using the full ``head_dim`` per call.
        """
        if channel_num is not None and (
            not isinstance(channel_num, int) or channel_num <= 0
        ):
            raise ValueError(
                f"channel_num must be a positive int, got {channel_num!r}."
            )
        self._channel_num_cfg = (
            int(channel_num) if channel_num is not None else None
        )

    def set_max_seq_len(self, max_seq_len: Optional[int]) -> None:
        """Update the engine-wide ``max_seq_len`` after construction.

        Pass ``None`` to revert to the device-side per-call computation.
        """
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive int, got {max_seq_len!r}."
            )
        self._max_seq_len_cfg = (
            int(max_seq_len) if max_seq_len is not None else None
        )

    def set_n_keys(self, n_keys: Optional[int]) -> None:
        """Update the default ``n_keys`` for top-k scaling after construction.

        Pass ``None`` to use ``L_max`` (the score tensor length) when
        ``run(..., n_keys=...)`` is omitted.
        """
        if n_keys is not None and n_keys <= 0:
            raise ValueError(f"n_keys must be a positive int, got {n_keys!r}.")
        self._n_keys_cfg = int(n_keys) if n_keys is not None else None

    # ------------------------------------------------------------------ overloads
    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk: Optional[float] = None,
        *args: Any,
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
        print(f"[SPARSE KERNEL DEBUG] run() CALLED! q.shape={q.shape if hasattr(q, 'shape') else type(q)}", flush=True)
        import sys
        sys.stdout.flush()
        del args, enable_pdl, sinks, skip_softmax_threshold_scale_factor, kv_cache_sf
        if not self._planned:
            raise RuntimeError("plan() must be called before run().")
        if q_len_per_req is None:
            q_len_per_req = 1
        if q_len_per_req != 1:
            raise NotImplementedError(
                "Oracle top-k optimized wrapper supports q_len_per_req == 1 only."
            )
        # Per-call args override constructor defaults.
        if topk is None:
            topk = self._topk_cfg
        if topk is None:
            raise ValueError(
                "topk must be supplied either via __init__(topk=...) / "
                "set_topk() or per-call as run(..., topk=...)."
            )
        topk_f = _validate_topk_fraction(topk)
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
        # The CUDA kernel skips loads at vector-lane granularity, so
        # channel_num must be a multiple of the vec lane width.
        vec_lane = 8 if q.dtype in (torch.float16, torch.bfloat16) else 4
        if channel_num % vec_lane != 0:
            raise ValueError(
                f"channel_num must be a multiple of {vec_lane} for {q.dtype}, "
                f"got {channel_num}."
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
                "Sliding window not implemented for oracle top-k wrapper."
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

        # ---- 1. Per-batch sequence-length info for the score kernel ----------
        # _kv_indptr_int32 / _kv_indices_int32 are populated by parent's plan().
        indptr_dev = self._kv_indptr_int32
        indices_dev = self._kv_indices_int32
        last_page_len_dev = self._kv_last_page_len_int32

        # Resolve L_max (the score-tensor's last-dim bound).
        #
        # Preferred (and sync-free) path: caller supplied an upper bound via
        # ``run(max_seq_len=...)`` or the constructor's ``max_seq_len=``,
        # typically the engine's ``max_model_len``. The score kernel still pads
        # positions ``[L_b, max_seq_len)`` with -inf per request, so a bound
        # >= the true per-batch max is always safe. This also makes ``run()``
        # CUDA-Graph capturable end-to-end.
        #
        # Fallback: derive L_max from the device-side indptr/last_page_len. This
        # incurs a ``.max().item()`` host-device sync on the first call per
        # ``plan()`` (cached thereafter).
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
            raise ValueError(
                f"n_keys must be positive, got {n_keys_for_topk!r}."
            )
        if n_keys_for_topk > L_max:
            raise ValueError(
                f"n_keys ({n_keys_for_topk}) must be <= L_max ({L_max}), "
                "the score tensor's last-dim length."
            )
        k_eff = _topk_fraction_to_k_eff(topk_f, n_keys_for_topk)

        # ---- 2. Compute dense Q @ K^T scores per (b, qh) ---------------------
        # Selection uses only the first ``channel_num`` channels of Q and K.
        scores = self._oracle_ops.oracle_topk_compute_scores(
            q_in,
            k_cache,
            int(kv_stride_page),
            int(kv_stride_n),
            int(kv_stride_h),
            indptr_dev,
            indices_dev,
            last_page_len_dev,
            int(num_kv_heads_actual),
            int(page_size_actual),
            int(L_max),
            int(channel_num),
        )  # [B, H_q, L_max], float32, with -inf at padded positions

        # ---- 3. Top-k selection ---------------------------------------------
        # Uses ``flashinfer.top_k`` (radix-based, ~3x faster at L=128K) when
        # available, or ``torch.topk`` as a fallback. Result is int64 in both
        # paths, shape [B, H_q, k_eff].
        topk_idx = self._topk_impl(scores, k_eff)
        # Store for inspection
        self._last_topk_idx = topk_idx.detach().cpu()
        # Also store the scores for debugging
        self._last_topk_scores = scores.detach().cpu() if 'scores' in locals() else None
        print(f"[SPARSE KERNEL DEBUG] run() called: batch_size={batch_size}, H_q={num_qo_heads}, L_max={L_max}, k_eff={k_eff}, n_keys_for_topk={n_keys_for_topk}", flush=True)
        print(f"[SPARSE KERNEL DEBUG] topk_idx shape: {topk_idx.shape}, first batch/head tokens: {topk_idx[0, 0].tolist()[:10]}", flush=True)
        del scores  # free 4 * B * H_q * L_max bytes early

        # ---- 4. Build sparse plan (uniform top_k across all heads) ----------
        sparse_len = torch.full(
            (batch_size, self._num_qo_heads, 1),
            k_eff, dtype=torch.int32, device=q.device,
        )
        sparse_weights = torch.ones(
            (batch_size, self._num_qo_heads, k_eff),
            dtype=torch.float32, device=q.device,
        )
        sparse_idx_in = topk_idx.contiguous() if not topk_idx.is_contiguous() else topk_idx
        # ---- 5. Reuse the sparse decode kernel ------------------------------
        max_S = k_eff
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
            indptr_dev,
            indices_dev,
            sparse_len,
            sparse_idx_in,
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
                    raise ValueError(
                        "lse must have shape [batch_size, num_qo_heads]."
                    )
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
        """Return max over batches of L_b = (pages_b - 1) * page_size + last_pl_b.

        Forces one device->host sync via ``.item()`` and is therefore *not*
        CUDA-graph capturable; callers that need graph capture must pass
        ``max_seq_len=`` to ``__init__`` / ``run()`` so this fallback is
        skipped entirely.

        Cached by ``(data_ptr_indptr, data_ptr_last_page_len)`` so repeated
        calls against the same persistent buffers avoid the host sync. The
        cache key intentionally does *not* include ``tensor._version``: that
        attribute is unavailable on torch ``inference_mode`` tensors and
        accessing it raises ``RuntimeError: Inference tensors do not track
        version counter``. The trade-off is that in-place edits to the same
        buffers are not detected by the cache; long-running engines should
        rely on the sync-free ``max_seq_len=`` path instead.
        """
        key = (indptr_dev.data_ptr(), last_page_len_dev.data_ptr())
        cached = getattr(self, "_lmax_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        # Compute on device, then sync once.
        pages = indptr_dev[1:] - indptr_dev[:-1]
        L_per = (pages - 1).clamp(min=0) * page_size + last_page_len_dev.to(torch.int32)
        L_max = int(L_per.max().item()) if L_per.numel() > 0 else 0
        self._lmax_cache = (key, L_max)
        return L_max

    run_return_lse = lambda self, *args, **kwargs: self.run(  # noqa: E731
        *args, **kwargs, return_lse=True
    )
