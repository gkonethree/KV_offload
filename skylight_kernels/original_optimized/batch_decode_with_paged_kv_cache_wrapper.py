"""
Standalone BatchDecodeWithPagedKVCacheWrapper backed by a custom CUDA kernel.

This wrapper exposes the same constructor / plan / run API as
`flashinfer.BatchDecodeWithPagedKVCacheWrapper`, but does NOT depend on the
FlashInfer library. The compute path is implemented in
`csrc/paged_decode_kernel.cu` and loaded via torch's JIT cpp extension.

The kernel implements a split-K paged decode similar in spirit to
FlashInfer's `BatchDecodeWithPagedKVCacheKernel`:

  * Each CTA handles one (request, kv_head, kv_chunk) tile.
  * Online softmax accumulates m / l / o in registers across the chunk.
  * A small merge kernel reduces partial chunks for each (request, head).
  * BDX vectorized 16-byte fp16/bf16 loads cooperate to load Q, K and V tiles.
  * BDY = group_size lays out one Q head per row (GQA without redundant K loads).
  * BDZ runs multiple KV positions in parallel inside a single CTA.

Supported configs: head_dim in {64, 128, 256}; group_size in {1,2,4,6,8,16};
fp16 / bf16; NHD or HND layouts; logits_soft_cap; sliding window; q_len_per_req=1.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Tuple, Union, overload

import torch

from .cuda_ops import get_paged_decode_ops


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
    key = str(dtype).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype}")
    return mapping[key]


def _unpack_paged_kv_cache(
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_layout: str,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int]:
    """
    Returns (k_tensor, v_tensor, kv_stride_page, kv_stride_n, kv_stride_h,
             v_offset_elements, page_size, num_kv_heads).
    Strides are in elements (not bytes), matching `paged_kv_t` semantics in
    flashinfer.

    For the 5D combined tensor we deliberately do NOT call .contiguous() on a
    slice (which would copy the entire cache); instead we pass the raw tensor
    with stride info so the kernel can index correctly.
    """
    if isinstance(paged_kv_cache, tuple):
        if len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache tuple must be (k_cache, v_cache).")
        k, v = paged_kv_cache
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError("k_cache and v_cache must be rank-4.")
        if kv_layout == "NHD":
            page_size = int(k.shape[1])
            num_kv_heads = int(k.shape[2])
        else:
            num_kv_heads = int(k.shape[1])
            page_size = int(k.shape[2])
        head_dim = int(k.shape[3])
        # FlashInfer paged_kv_t expects kv_strides as the *axis* strides of
        # the 4D K (or V) tensor:
        #   NHD shape [P, page_size, H, D] -> strides
        #     (page_size*H*D, H*D, D, 1)
        #   HND shape [P, H, page_size, D] -> strides
        #     (H*page_size*D, page_size*D, D, 1)
        # We pass kv_strides[0..2] (axes page, axis-1, axis-2). The kernel's
        # constructor maps stride_n/stride_h based on layout internally.
        if kv_layout == "NHD":
            kv_strides_arr = [page_size * num_kv_heads * head_dim,
                              num_kv_heads * head_dim, head_dim]
        else:
            kv_strides_arr = [num_kv_heads * page_size * head_dim,
                              page_size * head_dim, head_dim]
        return (k, v, kv_strides_arr[0], kv_strides_arr[1], kv_strides_arr[2],
                0, page_size, num_kv_heads)

    if not torch.is_tensor(paged_kv_cache):
        raise TypeError("paged_kv_cache must be a Tensor or (Tensor, Tensor).")
    if paged_kv_cache.dim() != 5:
        raise ValueError("single paged_kv_cache tensor must be rank-5.")
    if kv_layout == "NHD":
        page_size = int(paged_kv_cache.shape[2])
        num_kv_heads = int(paged_kv_cache.shape[3])
    else:
        num_kv_heads = int(paged_kv_cache.shape[2])
        page_size = int(paged_kv_cache.shape[3])
    head_dim = int(paged_kv_cache.shape[4])
    # Combined 5D cache: outer page stride includes the leading "2" dim.
    # axis-1 / axis-2 strides are taken from the 4D K-slice layout (axes
    # 1 and 2 of paged_kv_cache after dropping the K/V dim).
    if kv_layout == "NHD":
        stride_page = 2 * page_size * num_kv_heads * head_dim
        kv_strides_arr = [stride_page, num_kv_heads * head_dim, head_dim]
    else:
        stride_page = 2 * num_kv_heads * page_size * head_dim
        kv_strides_arr = [stride_page, page_size * head_dim, head_dim]
    v_offset = page_size * num_kv_heads * head_dim
    return (paged_kv_cache, paged_kv_cache, kv_strides_arr[0],
            kv_strides_arr[1], kv_strides_arr[2], v_offset, page_size,
            num_kv_heads)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _partition_paged_kv_binsearch(
    *, max_grid_size: int, gdy: int, num_pages: list, page_size: int
) -> Tuple[int, int]:
    """Direct port of `PartitionPagedKVCacheBinarySearchMinNumPagePerBatch`
    from `flashinfer/attention/scheduler.cuh`.

    Returns (kv_chunk_size_in_pages, new_batch_size).
    """
    min_num_pages = max(128 // max(page_size, 1), 1)
    if not num_pages:
        return min_num_pages, 0
    high = max(num_pages)
    low = min_num_pages
    while low < high:
        mid = (low + high) // 2
        new_batch_size = sum(_ceil_div(p, mid) for p in num_pages)
        if new_batch_size * gdy > max_grid_size:
            low = mid + 1
        else:
            high = mid
    new_batch_size = sum(_ceil_div(max(p, 1), low) for p in num_pages)
    return low, new_batch_size


def _decode_split_kv_indptr(
    *, indptr_h: list, batch_size: int, kv_chunk_size_in_pages: int
) -> Tuple[list, list, list]:
    """Direct port of `DecodeSplitKVIndptr` from
    `flashinfer/attention/scheduler.cuh`."""
    request_indices, kv_tile_indices, o_indptr = [], [], [0]
    for b in range(batch_size):
        n_pages = max(indptr_h[b + 1] - indptr_h[b], 1)
        n_chunks = _ceil_div(n_pages, kv_chunk_size_in_pages)
        for k in range(n_chunks):
            request_indices.append(b)
            kv_tile_indices.append(k)
        o_indptr.append(o_indptr[-1] + n_chunks)
    return request_indices, kv_tile_indices, o_indptr


def _decode_plan(
    *,
    indptr_h: list,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    sm_count: int,
    ops_module,
    device: torch.device,
    enable_cuda_graph: bool = False,
):
    """Plan the split-KV decode bookkeeping. Mirrors `DecodePlan` and
    `BatchDecodeWithPagedKVCacheWorkEstimationDispatched` from
    `flashinfer/attention/scheduler.cuh` exactly, but allocates the
    bookkeeping tensors directly on the GPU via PyTorch (i.e. no
    AlignedAllocator slicing of a single workspace buffer).

    Returns a dict of pre-built device tensors used at run() time.
    """
    group_size = num_qo_heads // num_kv_heads
    gdy = num_kv_heads
    # Query the kernel's max grid size for this dtype/D/G triplet.
    scalar_dtype_int = int(q_dtype.itemsize)  # placeholder, we use ScalarType id
    # PyTorch ScalarType integer ids: kHalf=5, kBFloat16=15. Pass via the C++
    # API which accepts c10::ScalarType cast from int64.
    if q_dtype == torch.float16:
        scalar_dtype_int = 5
    elif q_dtype == torch.bfloat16:
        scalar_dtype_int = 15
    else:
        raise NotImplementedError(f"Unsupported q dtype {q_dtype}")
    max_grid_size = int(
        ops_module.query_max_grid_size(scalar_dtype_int, head_dim, group_size)
    )

    if max_grid_size <= 0:
        # Fallback if occupancy query failed for some reason.
        max_grid_size = sm_count * 8

    num_pages = [int(indptr_h[i + 1] - indptr_h[i]) for i in range(batch_size)]

    if batch_size * gdy >= max_grid_size:
        split_kv = False
        kv_chunk_size_in_pages = 1
        for n in num_pages:
            if n > kv_chunk_size_in_pages:
                kv_chunk_size_in_pages = n
        new_batch_size = batch_size
    else:
        kv_chunk_size_in_pages, new_batch_size = _partition_paged_kv_binsearch(
            max_grid_size=max_grid_size, gdy=gdy, num_pages=num_pages, page_size=page_size
        )
        split_kv = new_batch_size != batch_size

    if enable_cuda_graph:
        padded_batch_size = max_grid_size // gdy if split_kv else batch_size
    else:
        padded_batch_size = new_batch_size

    request_indices, kv_tile_indices, o_indptr = _decode_split_kv_indptr(
        indptr_h=indptr_h,
        batch_size=batch_size,
        kv_chunk_size_in_pages=kv_chunk_size_in_pages,
    )

    request_indices_t = torch.tensor(request_indices, dtype=torch.int32, device=device)
    kv_tile_indices_t = torch.tensor(kv_tile_indices, dtype=torch.int32, device=device)
    o_indptr_t = torch.tensor(o_indptr, dtype=torch.int32, device=device)
    kv_chunk_size_ptr_t = torch.tensor(
        [kv_chunk_size_in_pages * page_size], dtype=torch.int32, device=device
    )

    block_valid_mask_t = None
    tmp_v_t = None
    tmp_s_t = None
    if split_kv:
        if enable_cuda_graph:
            mask_vals = [1 if i < new_batch_size else 0 for i in range(padded_batch_size)]
            block_valid_mask_t = torch.tensor(mask_vals, dtype=torch.bool, device=device)
        # tmp buffers: shape [padded_batch_size, num_qo_heads, head_dim] for v
        # and [padded_batch_size, num_qo_heads] for s (logsumexp).
        tmp_v_t = torch.empty(
            (padded_batch_size, num_qo_heads, head_dim), dtype=q_dtype, device=device
        )
        tmp_s_t = torch.empty(
            (padded_batch_size, num_qo_heads), dtype=torch.float32, device=device
        )

    return {
        "request_indices": request_indices_t,
        "kv_tile_indices": kv_tile_indices_t,
        "o_indptr": o_indptr_t,
        "kv_chunk_size_ptr": kv_chunk_size_ptr_t,
        "block_valid_mask": block_valid_mask_t,
        "tmp_v": tmp_v_t,
        "tmp_s": tmp_s_t,
        "padded_batch_size": padded_batch_size,
        "split_kv": split_kv,
    }


class BatchDecodeWithPagedKVCacheWrapper:
    """Standalone paged-KV decode wrapper with a custom CUDA kernel backend."""

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
        del jit_args
        _check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._use_cuda_graph = use_cuda_graph
        self._use_tensor_cores = use_tensor_cores
        self._backend = backend

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._fixed_batch_size = 0

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a Tensor in cuda graph mode."
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a Tensor in cuda graph mode."
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a Tensor in cuda graph mode."
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError("paged_kv_indptr_buffer size must be batch_size + 1.")

        # Number of SMs on this device (used by the planner to pick chunk size).
        try:
            props = torch.cuda.get_device_properties(self.device)
            self._sm_count = int(props.multi_processor_count)
        except Exception:
            self._sm_count = 132

        # Eagerly compile the CUDA extension so the first run() is fast.
        self._ops = get_paged_decode_ops()

        self._planned = False
        self._pos_encoding_mode = "NONE"
        self._window_left = -1
        self._logits_soft_cap = 0.0
        self._sm_scale: Optional[float] = None
        self._rope_scale: Optional[float] = None
        self._rope_theta: Optional[float] = None

    # ----- public properties -------------------------------------------------
    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    # ----- plan --------------------------------------------------------------
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
        del block_tables, seq_lens, fixed_split_size
        if pos_encoding_mode != "NONE":
            raise NotImplementedError(
                "Standalone wrapper only supports pos_encoding_mode='NONE'."
            )
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("num_qo_heads must be a multiple of num_kv_heads.")
        if page_size < 1:
            raise ValueError("page_size must be >= 1.")
        if head_dim not in (64, 128, 256):
            raise NotImplementedError(
                f"Unsupported head_dim {head_dim}; supported: 64, 128, 256."
            )
        group_size = num_qo_heads // num_kv_heads
        if group_size not in (1, 2, 4, 6, 8, 16):
            raise NotImplementedError(
                f"Unsupported group_size {group_size}; supported: 1, 2, 4, 6, 8, 16."
            )

        batch_size = len(last_page_len)
        if len(indptr) != batch_size + 1:
            raise ValueError("len(indptr) must be batch_size + 1.")

        if self.is_cuda_graph_enabled and batch_size != self._fixed_batch_size:
            raise ValueError(
                f"batch size {batch_size} does not match cuda graph batch size {self._fixed_batch_size}."
            )

        if self.is_cuda_graph_enabled:
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(last_page_len, non_blocking=non_blocking)
            self._paged_kv_indices_buf[: len(indices)].copy_(indices, non_blocking=non_blocking)
        else:
            self._paged_kv_indptr_buf = indptr.to(self.device, non_blocking=non_blocking).contiguous()
            self._paged_kv_indices_buf = indices.to(self.device, non_blocking=non_blocking).contiguous()
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            ).contiguous()

        # Always keep int32 contiguous copies on-device for the kernel.
        self._kv_indptr_int32 = self._paged_kv_indptr_buf.to(torch.int32).contiguous()
        self._kv_indices_int32 = self._paged_kv_indices_buf.to(torch.int32).contiguous()
        self._kv_last_page_len_int32 = self._paged_kv_last_page_len_buf.to(torch.int32).contiguous()

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
        self._group_size = group_size
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = 0.0 if logits_soft_cap is None else float(logits_soft_cap)
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

        # Materialize indptr on host once for the scheduler logic. Use the
        # original (pre-int32) indptr if it's already on CPU, otherwise copy
        # the small int32 device tensor.
        if indptr.is_cuda:
            indptr_h = indptr.detach().to("cpu", torch.int64).tolist()
        else:
            indptr_h = indptr.detach().to(torch.int64).tolist()

        plan_dict = _decode_plan(
            indptr_h=indptr_h,
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_dtype=q_dtype,
            sm_count=self._sm_count,
            ops_module=self._ops,
            device=self.device,
            enable_cuda_graph=self.is_cuda_graph_enabled,
        )
        if disable_split_kv:
            # Force a single-CTA-per-request layout: rebuild with chunk size
            # = max(num_pages_per_req).
            self._plan_dict = _decode_plan(
                indptr_h=indptr_h,
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                q_dtype=q_dtype,
                sm_count=self._sm_count * 1024,  # huge max_grid -> no split
                ops_module=self._ops,
                device=self.device,
                enable_cuda_graph=self.is_cuda_graph_enabled,
            )
        else:
            self._plan_dict = plan_dict

        self._planned = True

    begin_forward = plan

    # ----- run ---------------------------------------------------------------
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
        if q_len_per_req != 1:
            raise NotImplementedError(
                "Standalone wrapper only supports q_len_per_req == 1."
            )

        (
            k_cache,
            v_cache,
            kv_stride_page,
            kv_stride_n,
            kv_stride_h,
            kv_v_offset_elem,
            page_size_actual,
            num_kv_heads_actual,
        ) = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
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
        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale

        out_dtype = self._cached_o_data_type or q.dtype
        if out is None:
            out_buf = None
        else:
            if out.shape != q.shape:
                raise ValueError(
                    f"out shape {tuple(out.shape)} does not match q shape {tuple(q.shape)}."
                )
            out_buf = out

        q_in = q if q.is_contiguous() else q.contiguous()

        kv_layout_nhd = (self._kv_layout == "NHD")

        plan = self._plan_dict
        out_t, lse_t = self._ops.paged_decode_run(
            q_in,
            k_cache,
            v_cache,
            int(kv_stride_page),
            int(kv_stride_n),
            int(kv_stride_h),
            int(kv_v_offset_elem),
            self._kv_indptr_int32,
            self._kv_indices_int32,
            self._kv_last_page_len_int32,
            int(page_size_actual),
            int(num_kv_heads_actual),
            float(sm_scale),
            float(self._logits_soft_cap),
            int(local_window_left if local_window_left is not None else -1),
            bool(kv_layout_nhd),
            bool(return_lse),
            plan["request_indices"],
            plan["kv_tile_indices"],
            plan["o_indptr"],
            plan["kv_chunk_size_ptr"],
            plan["block_valid_mask"],
            plan["tmp_v"],
            plan["tmp_s"],
            int(plan["padded_batch_size"]),
            bool(plan["split_kv"]),
        )

        if out_buf is not None:
            if out_t.dtype != out_buf.dtype:
                out_buf.copy_(out_t.to(out_buf.dtype))
            else:
                out_buf.copy_(out_t)
        else:
            if out_t.dtype != out_dtype:
                out_buf = out_t.to(out_dtype)
            else:
                out_buf = out_t

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

    run_return_lse = lambda self, *args, **kwargs: self.run(*args, **kwargs, return_lse=True)
