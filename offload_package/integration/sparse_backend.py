"""Sparse Attention Backend using sparse_oracle_topk_sink_local kernel.

This backend wraps the sparse_oracle_topk_sink_local kernel for use with 
standard decoder attention (DECODER attention type) in vLLM.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Optional

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.flashinfer import (
    get_flashinfer_layout_string,
    FIDecode,
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from skylight.config import SkylightSparseConfig

logger = init_logger(__name__)


class _NoFastPlanDecodeWrapper:
    """Decode wrapper that suppresses vllm's fast-plan-decode optimization.

    vllm's ``fast_plan_decode`` short-circuits to flashinfer-internal
    ``flashinfer.decode.fast_decode_plan`` when the wrapper reports
    ``is_cuda_graph_enabled=True``. That helper probes private flashinfer
    attributes which our sparse wrapper does not have. Reporting ``False`` makes
    vllm fall back to calling our wrapper's ``plan(...)`` directly, which
    forwards to the inner sparse wrapper's own cudagraph-aware ``plan()``.
    """

    is_cuda_graph_enabled: ClassVar[bool] = False

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        # Extract required sparse kernel parameters from the config
        from skylight.config import SkylightSparseConfig
        cfg = SkylightSparseConfig.from_env()
        # Add sparse kernel specific arguments
        kwargs['topk'] = cfg.topk
        kwargs['sink_size'] = cfg.sink_size
        kwargs['local_size'] = cfg.local_size
        # Handle channel_num = -1 (use full head_dim)
        channel_num = cfg.channel_num
        if channel_num == -1:
            # Get head_dim from the first argument (query tensor)
            if args:
                query = args[0]
                # query shape: [batch, num_heads, head_dim] or [num_heads, head_dim]
                head_dim = query.shape[-1]
                channel_num = head_dim
        kwargs['channel_num'] = channel_num
        kwargs['topk'] = cfg.topk
        kwargs['sink_size'] = cfg.sink_size
        kwargs['local_size'] = cfg.local_size
        return self._inner.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _import_sparse_wrapper_cls():
    """Lazy import: ``import sparse_backend`` doesn't require the kernel package.
    
    The import only fires when the backend's metadata builder actually
    constructs its first decode wrapper.
    """
    try:
        from sparse_oracle_topk_sink_local import (
            BatchDecodeWithPagedKVCacheWrapper as cls,
        )
    except ImportError as exc:
        raise ImportError(
            "SparseAttentionBackend requires the skylight-kernels package "
            "with sparse_oracle_topk_sink_local. Install it editable in the "
            "workspace via `uv sync`."
        ) from exc
    return cls


class _NoFastPlanDecodeWrapper:
    """Decode wrapper that suppresses vllm's fast-plan-decode optimization.

    vllm's ``fast_plan_decode`` short-circuits to flashinfer-internal
    ``flashinfer.decode.fast_decode_plan`` when the wrapper reports
    ``is_cuda_graph_enabled=True``. That helper probes private flashinfer
    attributes which our sparse wrapper does not have. Reporting ``False`` makes
    vllm fall back to calling our wrapper's ``plan(...)`` directly, which
    forwards to the inner sparse wrapper's own cudagraph-aware ``plan()``.
    """

    is_cuda_graph_enabled: ClassVar[bool] = False

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        # Extract required sparse kernel parameters from the config
        from skylight.config import SkylightSparseConfig
        cfg = SkylightSparseConfig.from_env()
        # Add sparse kernel specific arguments
        kwargs['topk'] = cfg.topk
        kwargs['sink_size'] = cfg.sink_size
        kwargs['local_size'] = cfg.local_size
        # Handle channel_num = -1 (use full head_dim)
        channel_num = cfg.channel_num
        if channel_num == -1:
            # Get head_dim from the first argument (query tensor)
            if args:
                query = args[0]
                # query shape: [batch, num_heads, head_dim] or [num_heads, head_dim]
                head_dim = query.shape[-1]
                channel_num = head_dim
        kwargs['channel_num'] = channel_num
        kwargs['topk'] = cfg.topk
        kwargs['sink_size'] = cfg.sink_size
        kwargs['local_size'] = cfg.local_size
        return self._inner.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _import_sparse_wrapper_cls():
    """Lazy import: ``import sparse_backend`` doesn't require the kernel package.
    
    The import only fires when the backend's metadata builder actually
    constructs its first decode wrapper.
    """
    try:
        from sparse_oracle_topk_sink_local import (
            BatchDecodeWithPagedKVCacheWrapper as cls,
        )
    except ImportError as exc:
        raise ImportError(
            "SparseAttentionBackend requires the skylight-kernels package "
            "with sparse_oracle_topk_sink_local. Install it editable in the "
            "workspace via `uv sync`."
        ) from exc
    return cls


class _NoFastPlanDecodeWrapper:
    """Decode wrapper that suppresses vllm's fast-plan-decode optimization.

    vllm's ``fast_plan_decode`` short-circuits to flashinfer-internal
    ``flashinfer.decode.fast_decode_plan`` when the wrapper reports
    ``is_cuda_graph_enabled=True``. That helper probes private flashinfer
    attributes which our sparse wrapper does not have. Reporting ``False`` makes
    vllm fall back to calling our wrapper's ``plan(...)`` directly, which
    forwards to the inner sparse wrapper's own cudagraph-aware ``plan()``.
    """

    is_cuda_graph_enabled: ClassVar[bool] = False

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        # Extract required sparse kernel parameters from the config
        from skylight.config import SkylightSparseConfig
        cfg = SkylightSparseConfig.from_env()
        # Add sparse kernel specific arguments
        kwargs['topk'] = cfg.topk
        kwargs['sink_size'] = cfg.sink_size
        kwargs['local_size'] = cfg.local_size
        # Handle channel_num = -1 (use full head_dim)
        channel_num = cfg.channel_num
        if channel_num == -1:
            # Get head_dim from the first argument (query tensor)
            if args:
                query = args[0]
                # query shape: [batch, num_heads, head_dim] or [num_heads, head_dim]
                head_dim = query.shape[-1]
                channel_num = head_dim
        kwargs['channel_num'] = channel_num
        kwargs['topk'] = cfg.topk
        kwargs['sink_size'] = cfg.sink_size
        kwargs['local_size'] = cfg.local_size
        return self._inner.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _import_sparse_wrapper_cls():
    """Lazy import: ``import sparse_backend`` doesn't require the kernel package.
    
    The import only fires when the backend's metadata builder actually
    constructs its first decode wrapper.
    """
    try:
        from sparse_oracle_topk_sink_local import (
            BatchDecodeWithPagedKVCacheWrapper as cls,
        )
    except ImportError as exc:
        raise ImportError(
            "SparseAttentionBackend requires the skylight-kernels package "
            "with sparse_oracle_topk_sink_local. Install it editable in the "
            "workspace via `uv sync`."
        ) from exc
    return cls


class SparseAttentionBackend(FlashInferBackend):
    """FlashInfer backend with sparse decode using oracle top-k with sink/local.

    Identical to FLASHINFER for prefill / cascade attention / KV-cache
    update; only the per-step decode wrapper is swapped for the configured
    sparse oracle top-k with sink/local from skylight-kernels.
    """

    # KV cache types: sparse decode kernel handles unquantized fp16/bf16 only.
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
    def get_impl_cls() -> type["SparseAttentionImpl"]:
        return SparseAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SparseAttentionMetadataBuilder"]:
        return SparseAttentionMetadataBuilder

    # Override to support standard decoder attention
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in ("DECODER", "decoder")

    @classmethod
    def is_sparse(cls) -> bool:
        return False

    @classmethod
    def supports_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in ("DECODER", "decoder")


class SparseAttentionMetadataBuilder(FlashInferMetadataBuilder):
    """Parent's metadata, with the per-step decode wrapper swapped to sparse."""

    # Class-level reference to orchestrator (set by worker connector)
    _offload_orchestrator: object | None = None
    # Cached sparse pattern from previous step
    _cached_sparse_idx: torch.Tensor | None = None
    _cached_sparse_len: torch.Tensor | None = None
    _cached_request_ids: list[str] | None = None
    # Class-level reference to last decode wrapper created (for debugging)
    _last_decode_wrapper: object | None = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        cfg = SkylightSparseConfig.from_env()
        self._sparse_topk: float = cfg.topk
        self._sparse_sink_size: int = cfg.sink_size
        self._sparse_local_size: int = cfg.local_size
        # ``-1`` is the sentinel "use full head_dim"; resolve it now that
        # head_dim is known so the wrapper receives a concrete int.
        self._sparse_channel_num: int = (
            int(self.head_dim) if cfg.channel_num == -1 else int(cfg.channel_num)
        )

        # Force the FI-native decode path; our sparse wrapper replaces dense
        # FlashInfer decode, TRTLLM has no equivalent. This is an INSTANCE
        # attribute, NOT a monkey-patch of vllm.utils.flashinfer.
        self.use_trtllm_decode_attention = False

        # The sparse decode kernel only handles ``q_len_per_req == 1``.
        # Reset the reorder threshold so spec-decode isn't bundled in.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

        # Engine-wide upper bound on per-request length; passed once to every
        # sparse wrapper so it can use a static score-tensor last dim and skip
        # the device-side L_max computation. That's what makes the decode
        # end-to-end CUDA-graph capturable.
        self._sparse_max_seq_len: int = int(self.model_config.max_model_len)

        # Lazy-imported kernel class; populated on first wrapper construction.
        self._sparse_wrapper_cls: type[Any] | None = None

        logger.info(
            "SparseAttentionBackend: topk=%.4f sink=%d local=%d channel_num=%d max_seq_len=%d",
            self._sparse_topk,
            self._sparse_sink_size,
            self._sparse_local_size,
            self._sparse_channel_num,
            self._sparse_max_seq_len,
        )

    def _make_sparse_decode_wrapper(
        self,
        use_cudagraph: bool,
        paged_kv_indptr: torch.Tensor | None,
        paged_kv_indices: torch.Tensor | None,
        paged_kv_last_page_len: torch.Tensor | None,
    ) -> Any:
        from skylight.config import SkylightSparseConfig
        cfg = SkylightSparseConfig.from_env()
        
        if self._sparse_wrapper_cls is None:
            self._sparse_wrapper_cls = self._import_sparse_wrapper_cls()
        print(f"[SparseBackend DEBUG] _make_sparse_decode_wrapper called: use_cudagraph={use_cudagraph}", flush=True)
        return self._sparse_wrapper_cls(
            self._get_workspace_buffer(),
            get_flashinfer_layout_string(self.kv_cache_layout),
            use_cuda_graph=use_cudagraph,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            use_tensor_cores=True,
        )

    @staticmethod
    def _import_sparse_wrapper_cls():
        """Lazy import: ``import sparse_backend`` doesn't require the kernel package.
        
        The import only fires when the backend's metadata builder actually
        constructs its first decode wrapper.
        """
        try:
            from sparse_oracle_topk_sink_local import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        except ImportError as exc:
            raise ImportError(
                "SparseAttentionBackend requires the skylight-kernels package "
                "with sparse_oracle_topk_sink_local. Install it editable in the "
                "workspace via `uv sync`."
            ) from exc
        return cls

    @override  # type: ignore[misc]
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        print(f"[SparseBackend DEBUG] build() called: num_decodes={getattr(common_attn_metadata, 'num_decodes', 'N/A')}", flush=True)
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        
        # Replace the decode wrapper with sparse wrapper
        if attn_metadata.decode is not None:
            print(f"[SparseBackend DEBUG] Replacing decode wrapper with sparse wrapper", flush=True)
            batch_size = getattr(common_attn_metadata, 'num_decodes', 0)
            if batch_size > 0:
                sparse_wrapper = self._get_sparse_decode_wrapper(
                    batch_size=0,  # Will be determined by batch_size
                    use_cudagraph=False,
                )
                attn_metadata.decode = type(attn_metadata.decode)(wrapper=sparse_wrapper)
                print(f"[SparseBackend DEBUG] Replaced decode wrapper with sparse wrapper", flush=True)
        
        return attn_metadata

    def _get_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool = False
    ) -> Any:
        print(f"[SparseBackend DEBUG] _get_decode_wrapper called: batch_size={batch_size}, use_cudagraph={use_cudagraph}", flush=True)
        # Mirror ``FlashInferMetadataBuilder._get_decode_wrapper`` but build
        # the sparse wrapper. When ``use_cudagraph`` is requested, slice the
        # persistent paged-KV buffers and cache one wrapper per captured
        # batch size, exactly like the parent.
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(self._batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: self._batch_size + 1]
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
                self._decode_wrappers_cudagraph[self._batch_size] = decode_wrapper  # type: ignore[assignment]
            else:
                self._decode_wrapper = decode_wrapper  # type: ignore[assignment]

        print(f"[SparseBackend DEBUG] _get_decode_wrapper returning wrapper: {type(decode_wrapper)}", flush=True)
        return decode_wrapper

    def _make_sparse_decode_wrapper(
        self,
        use_cudagraph: bool,
        paged_kv_indptr: torch.Tensor | None,
        paged_kv_indices: torch.Tensor | None,
        paged_kv_last_page_len: torch.Tensor | None,
    ) -> Any:
        from skylight.config import SkylightSparseConfig
        cfg = SkylightSparseConfig.from_env()
        
        if self._sparse_wrapper_cls is None:
            self._sparse_wrapper_cls = self._import_sparse_wrapper_cls()
        print(f"[SparseBackend DEBUG] _make_sparse_decode_wrapper called: use_cudagraph={use_cudagraph}", flush=True)
        return self._sparse_wrapper_cls(
            self._get_workspace_buffer(),
            get_flashinfer_layout_string(self.kv_cache_layout),
            use_cuda_graph=use_cudagraph,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            use_tensor_cores=True,
        )

    @staticmethod
    def _import_sparse_wrapper_cls():
        """Lazy import: ``import sparse_backend`` doesn't require the kernel package.
        
        The import only fires when the backend's metadata builder actually
        constructs its first decode wrapper.
        """
        try:
            from sparse_oracle_topk_sink_local import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        except ImportError as exc:
            raise ImportError(
                "SparseAttentionBackend requires the skylight-kernels package "
                "with sparse_oracle_topk_sink_local. Install it editable in the "
                "workspace via `uv sync`."
            ) from exc
        return cls

    @override  # type: ignore[misc]
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        print(f"[SparseBackend DEBUG] build() called: num_decodes={getattr(common_attn_metadata, 'num_decodes', 'N/A')}", flush=True)
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        
        # Replace the decode wrapper with sparse wrapper
        if attn_metadata.decode is not None:
            print(f"[SparseBackend DEBUG] Replacing decode wrapper with sparse wrapper", flush=True)
            batch_size = getattr(common_attn_metadata, 'num_decodes', 0)
            if batch_size > 0:
                sparse_wrapper = self._get_sparse_decode_wrapper(
                    batch_size=0,  # Will be determined by batch_size
                    use_cudagraph=False,
                )
                attn_metadata.decode = type(attn_metadata.decode)(wrapper=sparse_wrapper)
                print(f"[SparseBackend DEBUG] Replaced decode wrapper with sparse wrapper", flush=True)
        
        return attn_metadata

    def _get_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool = False
    ) -> Any:
        print(f"[SparseBackend DEBUG] _get_decode_wrapper called: batch_size={batch_size}, use_cudagraph={use_cudagraph}", flush=True)
        # Mirror ``FlashInferMetadataBuilder._get_decode_wrapper`` but build
        # the sparse wrapper. When ``use_cudagraph`` is requested, slice the
        # persistent paged-KV buffers and cache one wrapper per captured
        # batch size, exactly like the parent.
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(self._batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: self._batch_size + 1]
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
                self._decode_wrappers_cudagraph[self._batch_size] = decode_wrapper  # type: ignore[assignment]
            else:
                self._decode_wrapper = decode_wrapper  # type: ignore[assignment]

        print(f"[SparseBackend DEBUG] _get_decode_wrapper returning wrapper: {type(decode_wrapper)}", flush=True)
        return decode_wrapper

    def _make_sparse_decode_wrapper(
        self,
        use_cudagraph: bool,
        paged_kv_indptr: torch.Tensor | None,
        paged_kv_indices: torch.Tensor | None,
        paged_kv_last_page_len: torch.Tensor | None,
    ) -> Any:
        from skylight.config import SkylightSparseConfig
        cfg = SkylightSparseConfig.from_env()
        
        if self._sparse_wrapper_cls is None:
            self._sparse_wrapper_cls = self._import_sparse_wrapper_cls()
        print(f"[SparseBackend DEBUG] _make_sparse_decode_wrapper called: use_cudagraph={use_cudagraph}", flush=True)
        return self._sparse_wrapper_cls(
            self._get_workspace_buffer(),
            get_flashinfer_layout_string(self.kv_cache_layout),
            use_cuda_graph=use_cudagraph,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            use_tensor_cores=True,
        )

    @staticmethod
    def _import_sparse_wrapper_cls():
        """Lazy import: ``import sparse_backend`` doesn't require the kernel package.
        
        The import only fires when the backend's metadata builder actually
        constructs its first decode wrapper.
        """
        try:
            from sparse_oracle_topk_sink_local import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        except ImportError as exc:
            raise ImportError(
                "SparseAttentionBackend requires the skylight-kernels package "
                "with sparse_oracle_topk_sink_local. Install it editable in the "
                "workspace via `uv sync`."
            ) from exc
        return cls

    @override  # type: ignore[misc]
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        print(f"[SparseBackend DEBUG] build() called: num_decodes={getattr(common_attn_metadata, 'num_decodes', 'N/A')}", flush=True)
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        
        # Replace the decode wrapper with sparse wrapper
        if attn_metadata.decode is not None:
            print(f"[SparseBackend DEBUG] Replacing decode wrapper with sparse wrapper", flush=True)
            batch_size = getattr(common_attn_metadata, 'num_decodes', 0)
            if batch_size > 0:
                sparse_wrapper = self._get_sparse_decode_wrapper(
                    batch_size=0,  # Will be determined by batch_size
                    use_cudagraph=False,
                )
                attn_metadata.decode = type(attn_metadata.decode)(wrapper=sparse_wrapper)
                print(f"[SparseBackend DEBUG] Replaced decode wrapper with sparse wrapper", flush=True)
        
        return attn_metadata

    def _get_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool = False
    ) -> Any:
        print(f"[SparseBackend DEBUG] _get_decode_wrapper called: batch_size={batch_size}, use_cudagraph={use_cudagraph}", flush=True)
        # Mirror ``FlashInferMetadataBuilder._get_decode_wrapper`` but build
        # the sparse wrapper. When ``use_cudagraph`` is requested, slice the
        # persistent paged-KV buffers and cache one wrapper per captured
        # batch size, exactly like the parent.
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(self._batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: self._batch_size + 1]
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
                self._decode_wrappers_cudagraph[self._batch_size] = decode_wrapper  # type: ignore[assignment]
            else:
                self._decode_wrapper = decode_wrapper  # type: ignore[assignment]

        # Store the wrapper in class attribute for debugging access
        SparseAttentionMetadataBuilder._last_decode_wrapper = decode_wrapper

        print(f"[SparseBackend DEBUG] _get_decode_wrapper returning wrapper: {type(decode_wrapper)}", flush=True)
        return decode_wrapper

    def _make_sparse_decode_wrapper(
        self,
        use_cudagraph: bool,
        paged_kv_indptr: torch.Tensor | None,
        paged_kv_indices: torch.Tensor | None,
        paged_kv_last_page_len: torch.Tensor | None,
    ) -> Any:
        from skylight.config import SkylightSparseConfig
        cfg = SkylightSparseConfig.from_env()
        
        if self._sparse_wrapper_cls is None:
            self._sparse_wrapper_cls = self._import_sparse_wrapper_cls()
        print(f"[SparseBackend DEBUG] _make_sparse_decode_wrapper called: use_cudagraph={use_cudagraph}", flush=True)
        return self._sparse_wrapper_cls(
            self._get_workspace_buffer(),
            get_flashinfer_layout_string(self.kv_cache_layout),
            use_cuda_graph=use_cudagraph,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            use_tensor_cores=True,
        )

    

class SparseAttentionImpl(FlashInferImpl):
    """Implementation that uses sparse decode wrapper from metadata."""

    def _get_decode_wrapper(self, use_cudagraph: bool = False):
        """Override to use sparse wrapper from metadata instead of FlashInfer's."""
        # Get the sparse wrapper from the metadata
        if self.attn_metadata is not None and self.attn_metadata.decode is not None:
            return self.attn_metadata.decode.wrapper
        # Fallback to parent implementation
        return super()._get_decode_wrapper(use_cudagraph)