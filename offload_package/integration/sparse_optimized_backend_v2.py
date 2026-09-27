"""SparseOptimizedBackendV2 — vLLM AttentionBackend using sparse_optimized kernel."""

from __future__ import annotations

import os
from typing import Any, ClassVar

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.flashinfer import (
    FIDecode,
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.utils import get_flashinfer_layout_string
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class _NoFastPlanDecodeWrapper:
    """Decode wrapper that suppresses vLLM's fast-plan-decode optimization."""

    is_cuda_graph_enabled: ClassVar[bool] = False

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _import_sparse_optimized_wrapper_cls():
    """Lazy import of sparse_optimized wrapper."""
    try:
        from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper as cls
    except ImportError as exc:
        raise ImportError(
            "SparseOptimizedBackendV2 requires the sparse_optimized kernel. "
            "Install it via `pip install -e skylight_kernels/sparse_optimized`."
        ) from exc
    return cls


class SparseOptimizedBackendV2(FlashInferBackend):
    """FlashInfer backend with sparse_optimized sparse decode wrapper.

    Identical to FLASHINFER for prefill / cascade attention / KV-cache
    update; only the per-step decode wrapper is swapped for the
    sparse_optimized kernel from skylight_kernels.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return DeviceCapability(7, 5) <= capability <= DeviceCapability(12, 1)

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SparseOptimizedImplV2"]:
        return SparseOptimizedImplV2

    @staticmethod
    def get_builder_cls() -> type["SparseOptimizedMetadataBuilderV2"]:
        return SparseOptimizedMetadataBuilderV2

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in ("DECODER", "decoder")

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_sparse(cls) -> bool:
        return True


class SparseOptimizedMetadataBuilderV2(FlashInferMetadataBuilder):
    """Metadata builder with sparse_optimized decode wrapper."""

    _offload_orchestrator: object | None = None
    _last_decode_wrapper: object | None = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self._sparse_wrapper_cls: type[Any] | None = None
        logger.info("SparseOptimizedBackendV2: using sparse_optimized sparse decode kernel")

    def _make_sparse_decode_wrapper(
        self,
        use_cudagraph: bool,
        paged_kv_indptr: torch.Tensor | None,
        paged_kv_indices: torch.Tensor | None,
        paged_kv_last_page_len: torch.Tensor | None,
    ) -> Any:
        if self._sparse_wrapper_cls is None:
            self._sparse_wrapper_cls = _import_sparse_optimized_wrapper_cls()

        return self._sparse_wrapper_cls(
            self._get_workspace_buffer(),
            get_flashinfer_layout_string(self.kv_cache_layout),
            use_cuda_graph=use_cudagraph,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            use_tensor_cores=True,
        )

    @override
    def _get_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool = False
    ) -> Any:
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None

            inner = self._make_sparse_decode_wrapper(
                use_cudagraph=use_cudagraph,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
            )
            decode_wrapper = _NoFastPlanDecodeWrapper(inner)

            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        SparseOptimizedMetadataBuilderV2._last_decode_wrapper = decode_wrapper
        return decode_wrapper

    @override
    @classmethod
    def get_cudagraph_support(
        cls: type["SparseOptimizedMetadataBuilderV2"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


class SparseOptimizedImplV2(FlashInferImpl):
    """Identical to FlashInferImpl but with sparse_optimized decode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_quant_query_input = False