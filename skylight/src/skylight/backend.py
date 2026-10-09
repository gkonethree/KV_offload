"""SkylightSparseBackend — vllm AttentionBackend integration for skylight-kernels.

Pattern follows apd10's ``flashinfer_sparse.py``: subclass FlashInferBackend +
FlashInferMetadataBuilder + FlashInferImpl, override only the decode wrapper
construction. Prefill, cascade attention, and the KV-cache update path
inherit from FlashInfer unchanged.

Registered under :attr:`AttentionBackendEnum.CUSTOM` by
:func:`skylight.plugin.install_plugin`; selected at runtime via
``VLLM_ATTENTION_BACKEND=CUSTOM``.
"""
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

from vllm.v1.attention.backends.utils import (
    get_flashinfer_layout_string,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from skylight.config import SkylightSparseConfig

logger = init_logger(__name__)
import os as _os
from offload_package import timing as _timing
_DBG = _os.environ.get("OFFLOAD_PACKAGE_DEBUG", "0") == "1"
def _dbg(msg: str) -> None:
    if _DBG:
        print(msg, flush=True)

# Module-level cache for sparse pattern (shared across class instances)
_sparse_pattern_cache = {
    'sparse_idx': None,
    'sparse_len': None,
    'request_ids': None,
}


class _NoFastPlanDecodeWrapper:
    """Decode wrapper that suppresses vllm's fast-plan-decode optimization.

    Naming follows vllm's ``CUDAGraphWrapper`` / ``TorchCompileWithNoGuardsWrapper``
    convention: ``<Behavior>Wrapper``. Here the behavior is "no fast plan".

    vllm's ``fast_plan_decode`` short-circuits to flashinfer-internal
    ``flashinfer.decode.fast_decode_plan`` when the wrapper reports
    ``is_cuda_graph_enabled=True``. That helper probes private flashinfer
    attributes (``_int_workspace_buffer``, ``_cached_module``) which our
    sparse wrapper does not have (it inherits from ``original_optimized``,
    not from flashinfer's wrapper directly). Reporting ``False`` makes vllm
    fall back to calling our wrapper's ``plan(...)`` directly, which
    forwards to the inner sparse wrapper's own cudagraph-aware ``plan()``.
    The inner wrapper still uses its persistent paged-KV buffers because
    it was constructed with ``use_cuda_graph=True``.

    Most attributes (``plan``, ``_window_left``, ``_sm_scale``,
    ``_logits_soft_cap``, ...) forward transparently to the inner wrapper
    via ``__getattr__``. ``run()`` is overridden explicitly so we can
    update the ``skylight_effective_sparsity_fraction`` Prometheus Gauge
    after each batch — operators can ``curl /metrics`` to confirm sparse
    is actually firing (1.0 means dense fallback).
    """

    is_cuda_graph_enabled: ClassVar[bool] = False

    def __init__(self, inner: Any) -> None:
        # Bypass __setattr__ to write to the slot directly (no fwd to inner).
        object.__setattr__(self, "_inner", inner)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to inner ``run`` then update the sparsity-fraction Gauge."""
        _dbg(f"[_NoFastPlanDecodeWrapper DEBUG] run() CALLED, inner type={type(self._inner).__name__}, has_run={hasattr(self._inner, 'run')}")
        import sys
        sys.stdout.flush()
        result = self._inner.run(*args, **kwargs)
        get_stats = getattr(self._inner, "get_sparsity_stats", None)
        if callable(get_stats):
            # Lazy import to avoid a circular dep at module-load time
            # (metrics.py is registered by plugin.install_plugin()).
            from skylight.metrics import SPARSE_EFFECTIVE_FRACTION
            SPARSE_EFFECTIVE_FRACTION.set(get_stats().last_fraction)
        return result

    def __getattr__(self, name: str) -> Any:
        # __getattr__ runs only on miss; class attrs (like is_cuda_graph_enabled)
        # and explicit methods (like run) short-circuit before we get here.
        return getattr(self._inner, name)


def _import_sparse_wrapper_cls():
    """Lazy import: ``import skylight.backend`` doesn't require skylight-kernels.

    The import only fires when the backend's metadata builder actually
    constructs its first decode wrapper. That keeps test collection cheap
    and lets us ship the backend module even on hosts without the kernel
    package installed.
    """
    method = os.environ.get("SKYLIGHT_SPARSE_METHOD", "oracle").strip().lower()
    try:
        if method == "block_minmax":
            from block_minmax_incr_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method == "sparse_optimized":
            from offload_package.integration.sparse_optimized_adapter import (
                SparseOptimizedAdapterWrapper as cls,
            )
        elif method == "offload":
            from offload_package.integration.sparse_offload_adapter import (
                SparseOffloadAdapterWrapper as cls,
            )
        elif method in ("pq", "pq_optimized"):
            from pq_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method in ("pqrs", "pqrs_optimized"):
            from pqrs_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method in ("pqis", "pqis_optimized"):
            from pqis_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method in ("oracle", ""):
            from sparse_oracle_topk_sink_local_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        else:
            raise ValueError(
                f"SKYLIGHT_SPARSE_METHOD={method!r} not recognized "
                "(expected 'oracle', 'block_minmax', 'sparse_optimized', "
                "'pq', 'pqrs', or 'offload')."
            )
    except ImportError as exc:
        raise ImportError(
            "SkylightSparseBackend requires the skylight-kernels package. "
            "Install it editable in the workspace via `uv sync`."
        ) from exc
    return cls


class SkylightSparseBackend(FlashInferBackend):
    """FlashInfer backend with selectable sparse decode.

    Identical to FLASHINFER for prefill / cascade attention / KV-cache
    update; only the per-step decode wrapper is swapped for the configured
    oracle or block-minmax wrapper from skylight-kernels.
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
        # Mirror FLASHINFER's accepted page sizes.
        return [16, 32, 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Kernel templates are instantiated for {64, 128, 256}.
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Turing (sm_75) through ~Blackwell. The kernel uses standard CUDA
        # features available on this range.
        return DeviceCapability(7, 5) <= capability <= DeviceCapability(12, 1)

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SkylightSparseImpl"]:
        return SkylightSparseImpl

    @staticmethod
    def get_builder_cls() -> type["SkylightSparseMetadataBuilder"]:
        return SkylightSparseMetadataBuilder

    # Override to support standard decoder attention
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in ("DECODER", "decoder")

    @classmethod
    def is_sparse(cls) -> bool:
        return False  # Dense decoder interface with sparse kernels internally

    @classmethod
    def supports_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in ("DECODER", "decoder")


class SkylightSparseMetadataBuilder(FlashInferMetadataBuilder):
    """Parent's metadata, with the per-step decode wrapper swapped to sparse."""

    # Class-level reference to orchestrator (set by worker connector)
    _offload_orchestrator: object | None = None
    # Class-level reference to the GPU model runner (set by worker connector).
    # Used to derive the CURRENT step's decode request ids (post-reorder,
    # decode rows first) at build time -- the scheduler-bound ids arrive one
    # step late (bind runs after build) and their order can differ from the
    # runner's row order in mixed batches.
    _offload_runner: object | None = None
    # Last bail reason recorded by _patch_indices_for_staging for this build.
    _patch_bail_reason: str | None = None
    # Engine-step serial (bumped per build) and the step whose pattern was
    # last D2H-cached; together they limit pattern caching to once per step.
    _step_serial: int = 0
    _pattern_cached_serial: int = -1
    # Cached sparse pattern from previous step
    _cached_sparse_idx: torch.Tensor | None = None
    _cached_sparse_len: torch.Tensor | None = None
    _cached_request_ids: list[str] | None = None
    # Debug: last patched paged indices (set by _patch_indices_for_staging)
    _last_patched_indices: torch.Tensor | None = None
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

        # --- block_minmax FULL-cudagraph hook (gated; default path unchanged). Persistent
        # per-request slot index buffers + a shared (cross-layer) summary maintainer, mirroring
        # vLLM's Mamba state_indices pattern: build() fills slots from block_table[:,0],
        # Impl.forward inits new summaries from prompt K (eager) and the captured decode folds.
        self._sk_fullcg = (
            os.environ.get("SKYLIGHT_INCR_FULLCG") == "1"
            and os.environ.get("SKYLIGHT_INCR_SLOT") == "1"
        )
        self._sk_maint = None
        self._sk_syncfree = False
        self._sk_slots_d = None
        self._sk_seqlens_d = None
        self._sk_sub_page = int(os.environ.get("SKYLIGHT_SPARSE_SUB_PAGE", "8"))
        self._sk_ptotal_fixed = 0
        if self._sk_fullcg:
            from block_minmax_incr_optimized.summary_maintain import SlotSummaryMaintainer
            # n_slots MUST cover vLLM's largest captured decode batch (it profiles/captures up
            # to decode_cudagraph_max_bs); under-sizing crashes capture. Bound serving memory by
            # lowering --max-num-seqs, not this.
            max_bs = int(vllm_config.scheduler_config.max_num_seqs)
            cap = getattr(vllm_config.compilation_config, "max_cudagraph_capture_size", None)
            if cap:
                max_bs = min(max_bs, int(cap))
            self._sk_maint = SlotSummaryMaintainer(
                self._sk_sub_page, int(self.kv_cache_spec.num_kv_heads), int(self.head_dim),
                int(self.kv_cache_spec.block_size), max_bs, get_flashinfer_layout_string(self.kv_cache_layout))
            self._sk_maxbs = max_bs
            self._sk_owner = None     # GPU [max_bs] first-page id occupying each slot/position
            self._sk_slots_d = torch.zeros(max_bs, dtype=torch.int32, device=device)
            self._sk_seqlens_d = torch.zeros(max_bs, dtype=torch.int32, device=device)
            self._sk_pos = torch.arange(max_bs, dtype=torch.int32, device=device)  # slot=position
            # FlashInfer-style sync-free decode: detect (re)init from CPU seq_lens, no GPU readback.
            self._sk_syncfree = os.environ.get("SKYLIGHT_INCR_SYNCFREE") == "1"
            self._sk_prev_seqlen = torch.full((max_bs,), -1, dtype=torch.int64)  # CPU
            if self._sk_syncfree:
                logger.info("SkylightSparseBackend: SYNC-FREE decode path ON")
            self._sk_ptotal_fixed = (
                self._sparse_max_seq_len + self._sk_sub_page - 1) // self._sk_sub_page
            logger.info("SkylightSparseBackend: FULL-cudagraph slot path ON (max_bs=%d, P=%d)",
                        max_bs, self._sk_ptotal_fixed)

        # --- PQ slot-codes hook (SKYLIGHT_SPARSE_METHOD=pq|pqrs). GPU-resident
        # per-token K codes (per-request slot buffer) so selection can score
        # pages WITHOUT reading the KV cache -- which is what keeps offloaded
        # (CPU-resident) blocks selectable for fetch-back. Mirrors the
        # block_minmax FULL-cudagraph hook; the maintainer exposes the same
        # get_buf/init_layer call shape as SlotSummaryMaintainer.
        self._sk_pq = (
            not self._sk_fullcg
            and os.environ.get("SKYLIGHT_SPARSE_METHOD", "oracle").strip().lower()
            in ("pq", "pq_optimized", "pqrs", "pqrs_optimized", "pqis", "pqis_optimized")
        )
        if self._sk_pq:
            from pq_optimized.slot_codes import SlotCodeMaintainer
            max_bs = int(vllm_config.scheduler_config.max_num_seqs)
            cap = getattr(vllm_config.compilation_config, "max_cudagraph_capture_size", None)
            if cap:
                max_bs = min(max_bs, int(cap))
            # Codes buffer is [n_slots+1, H_kv, L_max, M] per layer; slot IS
            # the batch position, so n_slots must cover the largest batch.
            n_slots = max(int(os.environ.get("SKYLIGHT_PQ_SLOTS", "0")), max_bs)
            self._sk_maint = SlotCodeMaintainer(
                h_kv=int(self.kv_cache_spec.num_kv_heads),
                m=int(os.environ.get("SKYLIGHT_PQ_SUBSPACES", "16")),
                page=int(self.kv_cache_spec.block_size),
                n_slots=n_slots,
                l_max=self._sparse_max_seq_len,
                max_batch=max_bs + 1,
            )
            # Selection channels must be fixed BEFORE get_buf() sizes the
            # codebook; otherwise the combined K||V last dim (2*head_dim) would
            # be mistaken for the key width.
            self._sk_maint._ch = self._sparse_channel_num
            self._sk_maxbs = max_bs
            self._sk_owner = None     # GPU [max_bs] first-page id occupying each slot/position
            self._sk_slots_d = torch.zeros(max_bs, dtype=torch.int32, device=device)
            self._sk_seqlens_d = torch.zeros(max_bs, dtype=torch.int32, device=device)
            self._sk_pos = torch.arange(max_bs, dtype=torch.int32, device=device)  # slot=position
            self._sk_prev_sl = None
            # For PQ the second get_buf arg is the token-level max length.
            self._sk_ptotal_fixed = self._sparse_max_seq_len
            logger.info(
                "SkylightSparseBackend: PQ slot codes ON (n_slots=%d, L_max=%d, M=%d, ch=%d)",
                n_slots, self._sparse_max_seq_len,
                self._sk_maint.m, self._sk_maint._ch,
            )

        # Sync staging: if enabled, compute sparse pattern in build() and use for
        # current step's staging (0 lag). Default is 1-step lag (prev step's pattern).
        self._sk_sync_staging = os.environ.get("SKYLIGHT_SYNC_STAGING") == "1"
        if self._sk_sync_staging:
            logger.info("SkylightSparseBackend: SYNC staging enabled (0 lag)")

        logger.info(
            "SkylightSparseBackend: topk=%.4f sink=%d local=%d channel_num=%d max_seq_len=%d",
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
        if self._sparse_wrapper_cls is None:
            self._sparse_wrapper_cls = _import_sparse_wrapper_cls()
        _dbg(f"[SKYLIGHT DEBUG] _make_sparse_decode_wrapper called: use_cudagraph={use_cudagraph}, sync_staging={self._sk_sync_staging}")
        wrapper = self._sparse_wrapper_cls(
            self._get_workspace_buffer(),
            get_flashinfer_layout_string(self.kv_cache_layout),
            use_cuda_graph=use_cudagraph,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            use_tensor_cores=True,
            topk=self._sparse_topk,
            channel_num=self._sparse_channel_num,
            sink_size=self._sparse_sink_size,
            local_size=self._sparse_local_size,
            max_seq_len=self._sparse_max_seq_len,
        )
        # Pass orchestrator to adapter for sync staging
        orchestrator = SkylightSparseMetadataBuilder._offload_orchestrator
        if self._sk_sync_staging and orchestrator is not None:
            if hasattr(wrapper, 'set_orchestrator'):
                wrapper.set_orchestrator(orchestrator)
                # Also cache request_ids for staging
                if hasattr(self, '_cached_request_ids'):
                    wrapper._cached_request_ids = self._cached_request_ids
        return wrapper

    @staticmethod
    def _set_n_keys_on_decode(decode: FIDecode | None, n_keys: int) -> None:
        """Push the true per-batch max context length into the inner wrapper.

        Sync-free: a Python attribute set on the inner wrapper's
        ``_n_keys_cfg`` field via its ``set_n_keys`` setter. The kernel reads
        it on the next launch. Required because the score tensor's last
        dim is padded to ``max_model_len`` (for CUDA-graph capturability),
        but fractional topk must scale against the batch's real max sequence
        length, not the padded axis length.
        """
        if decode is None:
            return
        w = decode.wrapper
        inner = getattr(w, "_inner", w)
        setter = getattr(inner, "set_n_keys", None)
        if callable(setter):
            setter(int(n_keys))

    @override  # type: ignore[misc]
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        _dbg(f"[SKYLIGHT DEBUG] build() called: num_decodes={getattr(common_attn_metadata, 'num_decodes', 'N/A')}")
        cls = type(self)
        self._patch_bail_reason = None
        num_decodes = 0
        # One pattern cache per engine step: _cache_sparse_pattern (called
        # after EVERY layer's forward) uses this serial to cache only the
        # first layer's pattern per step instead of 22 near-identical D2H
        # copies of which only the last was ever consumed.
        cls._step_serial = getattr(cls, "_step_serial", 0) + 1

        # Derive THIS step's decode request ids from the runner's input batch.
        # reorder_batch_to_split_decodes_and_prefills has already run by now,
        # so req_ids[:num_decodes] is the true decode row order -- including
        # mixed batches (decode + prefill-continuation rows), where the
        # scheduler-side id order is not guaranteed to match row order.
        # num_decodes is computed whenever the offload machinery is present
        # (the fail-loud guard below needs it); only the id STASH is subject
        # to the SKYLIGHT_OFFLOAD_DISABLE_RUNNER_IDS fault-injection switch.
        if (
            cls._offload_runner is not None
            and cls._offload_orchestrator is not None
            and common_attn_metadata is not None
            and not torch.cuda.is_current_stream_capturing()
        ):
            try:
                num_decodes, _, _, _ = split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                )
            except Exception as e:
                logger.warning("split_decodes_and_prefills failed: %s", e)
                num_decodes = 0
            if num_decodes > 0 and os.environ.get(
                "SKYLIGHT_OFFLOAD_DISABLE_RUNNER_IDS", "0"
            ) != "1":
                ib = getattr(cls._offload_runner, "input_batch", None)
                all_ids = list(getattr(ib, "req_ids", []) or [])
                ids = [str(r) for r in all_ids[:num_decodes]]
                if ids and not ids[0].startswith(("_dummy", "_warmup")):
                    cls._cached_request_ids = ids
                    _sparse_pattern_cache["request_ids"] = list(ids)

        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )

        # Fail-loud backstop: under true freeing, offloaded blocks' original
        # GPU pages are freed, and the index patcher + in-run fetch are the
        # only things standing between the kernel and those pages. A bail in
        # either path on a step with decode rows means selected cold blocks
        # would be read from stale/null pages with weight 1. Raise instead.
        orch = cls._offload_orchestrator
        if (
            self._patch_bail_reason is not None
            and num_decodes > 0
            and orch is not None
            and bool(getattr(orch, "true_freeing", False))
            and getattr(orch.residency, "num_cpu_blocks", 0) > 0
        ):
            raise RuntimeError(
                f"offload index patch bailed ({self._patch_bail_reason}) on a "
                f"decode step with {orch.residency.num_cpu_blocks} "
                "CPU-resident blocks under true freeing; refusing to let the "
                "kernel read freed pages."
            )

        # Hand the offload orchestrator + this step's decode request order to
        # the sparse wrapper every step (wrapper instances are cached per batch
        # size; class-level injection happens once in _build_orchestrator).
        if self._sk_pq and attn_metadata.decode is not None:
            inner = getattr(getattr(attn_metadata.decode, "wrapper", None), "_inner", None)
            inner = inner if inner is not None else getattr(
                attn_metadata.decode, "wrapper", None
            )
            cls = type(self)
            if inner is not None and hasattr(inner, "set_offload_orchestrator"):
                inner.set_offload_orchestrator(cls._offload_orchestrator)
                inner.set_offload_request_ids(cls._cached_request_ids)

        if (self._sk_fullcg or self._sk_pq) and not torch.cuda.is_current_stream_capturing():
            # Under FULL cudagraph the selected-set size (total_k) MUST be fixed across replays,
            # so do NOT push the dynamic per-batch length via set_n_keys (that would bake a
            # capture-time total_k into the graph). Leaving n_keys=None makes the wrapper size
            # total_k from the fixed max_seq_len; per-request L_per still masks invalid tokens.
            self._sk_fill_slots(attn_metadata, common_attn_metadata)
        else:
            self._set_n_keys_on_decode(
                attn_metadata.decode
                if isinstance(attn_metadata.decode, FIDecode)
                else None,
                int(common_attn_metadata.max_seq_len),
            )
        return attn_metadata

    @override  # type: ignore[misc]
    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np,
        seq_lens_np,
        block_table_tensor,
        num_reqs,
        page_size,
    ):
        """Fill persistent FI buffers, then patch paged_kv_indices for staging.

        Runs before the decode plan() inside super().build(), so the kernel sees
        staging slots / null-page redirections for offloaded blocks. Copies for
        the staging fetch are stream-ordered before the kernel launch.
        """
        paged_kv_indices = super()._compute_flashinfer_kv_metadata(
            num_blocks_np, seq_lens_np, block_table_tensor, num_reqs, page_size
        )
        self._patch_indices_for_staging(paged_kv_indices, num_reqs)
        return paged_kv_indices

    def _patch_indices_for_staging(self, paged_kv_indices, num_reqs: int) -> None:
        """Stage CPU blocks per cached sparse pattern and patch index content.

        Records ``self._patch_bail_reason`` for every early return that leaves
        offloaded blocks' index entries pointing at their original (possibly
        freed) GPU pages; ``build()`` turns hazardous bails into errors under
        true freeing. A mismatch between the cached sparse pattern and the
        current batch only skips the 1-step-lag prefetch -- the residency
        remediation (null-fill / staging re-point) still runs, so the
        in-run fetch path stays sound on composition changes.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        cls = type(self)
        orchestrator = cls._offload_orchestrator
        if orchestrator is None:
            _dbg("[PATCH DEBUG] _patch_indices_for_staging: no orchestrator")
            return
        request_ids = cls._cached_request_ids or []
        if len(request_ids) == 0:
            self._patch_bail_reason = "ids_empty"
            _dbg("[PATCH DEBUG] _patch_indices_for_staging: no request ids")
            return
        if len(request_ids) > num_reqs:
            self._patch_bail_reason = "ids_gt_reqs"
            _dbg(f"[PATCH DEBUG] _patch_indices_for_staging: too many ids ({len(request_ids)}) for {num_reqs} rows")
            return
        try:
            sparse_idx = cls._cached_sparse_idx
            sparse_len = cls._cached_sparse_len

            staging_map = {}
            patch_sparse_idx = sparse_idx
            patch_sparse_len = sparse_len
            if sparse_idx is not None and sparse_len is not None:
                # Pattern-cache batch must equal the current decode row count,
                # else the cached pattern belongs to a previous batch
                # composition and staging it would fetch for the wrong rows.
                # Skip ONLY the lag prefetch here; the remediation below still
                # runs and the same-step in-run fetch covers this step's
                # selection.
                if sparse_idx.shape[0] != len(request_ids):
                    _dbg(f"[PATCH DEBUG] _patch_indices_for_staging: lag pattern batch mismatch ({sparse_idx.shape[0]} vs {len(request_ids)}); skipping prefetch only")
                    patch_sparse_idx = None
                    patch_sparse_len = None
                else:
                    # Fetch CPU blocks needed by the previous step's top-k.
                    _dbg(f"[PATCH DEBUG] _patch_indices_for_staging: calling prepare_staging with sparse_idx.shape={sparse_idx.shape}")
                    staging_map = orchestrator.prepare_staging(
                        request_ids, sparse_idx, sparse_len
                    )
                    _dbg(f"[PATCH DEBUG] prepare_staging returned: {staging_map}")

            # Defensive alignment check: per-row page counts in the current
            # batch must equal the request's tracked block count, otherwise the
            # request order does not match decode row order and we must not
            # patch (row-b is ReqX's span, and patching assumes request_ids[b]).
            indptr_gpu = self.paged_kv_indptr.gpu[: len(request_ids) + 1]
            indptr_list = indptr_gpu.detach().cpu().tolist()
            for b, rid in enumerate(request_ids):
                row_pages = int(indptr_list[b + 1]) - int(indptr_list[b])
                tracked = len(orchestrator.residency.refs_for_request(rid))
                if tracked == 0 or row_pages != tracked:
                    self._patch_bail_reason = (
                        f"row_alignment b={b} rid={rid} "
                        f"tracked={tracked} row_pages={row_pages}"
                    )
                    _dbg(f"[PATCH DEBUG] bail b={b} rid={rid} tracked={tracked} row_pages={row_pages}")
                    return

            from offload_package.selection.adapter import (
                patch_paged_kv_indices_with_staging,
            )
            null_id = 0 if getattr(orchestrator, "true_freeing", False) else None
            _dbg(f"[PATCH DEBUG] calling patch_paged_kv_indices_with_staging with staging_map={staging_map}")
            _timing.start("build.patch")
            patch_paged_kv_indices_with_staging(
                paged_kv_indices,
                indptr_gpu,
                request_ids,
                patch_sparse_idx,
                patch_sparse_len,
                self.kv_cache_spec.block_size,
                orchestrator.staging_base_page_idx,
                orchestrator.residency,
                staging_map,
                null_block_id=null_id,
                indptr_host=indptr_list,
                inplace=True,
            )
            _timing.end("build.patch")
            cls._last_patched_indices = paged_kv_indices
        except Exception as e:
            _dbg(f"[SKYLIGHT DEBUG] Failed to patch staging indices: {e}")
            import traceback
            traceback.print_exc()
            if (
                getattr(orchestrator, "true_freeing", False)
                and getattr(orchestrator.residency, "num_cpu_blocks", 0) > 0
            ):
                # A swallowed exception here leaves offloaded blocks' entries
                # pointing at freed pages; fail loudly under true freeing.
                raise

    def _sk_fill_slots(self, m, cm) -> None:
        """Eager, runs every build() (incl. the cudagraph-capture build): assign each DECODE
        request a stable dense slot (by first physical page id), write it into the PERSISTENT
        slots_d/seqlens_d buffers in-place (so the captured graph reads fresh values), and
        attach the slot tensors + the pending-init list to the metadata for Impl.forward."""
        if self._sk_syncfree and not getattr(m, "_sk_in_fallback", False):
            return self._sk_fill_slots_syncfree(m, cm)
        m.sk_maint = None
        m.sk_pending = []
        m.sk_pages = {}
        nd = int(getattr(m, "num_decodes", 0))
        if nd <= 0:
            return
        dev = self._sk_slots_d.device
        bt = cm.block_table_tensor                        # [num_reqs, max_blocks], decodes first
        first_pages = bt[:nd, 0]                           # [nd] GPU
        # slot = batch position (stable within a captured uniform-decode run); GPU->GPU, NO sync.
        self._sk_slots_d[:nd].copy_(self._sk_pos[:nd])
        self._sk_seqlens_d[:nd].copy_(cm.seq_lens[:nd].to(torch.int32))
        if self._sk_owner is None:
            self._sk_owner = torch.full((self._sk_maxbs,), -1, dtype=first_pages.dtype, device=dev)
        # Rebuild a slot's summary when a NEW request occupies it. first-page change alone is
        # WRONG: with prefix caching, sequential requests share the first physical block so the
        # proxy never fires and requests #2+ reuse a stale summary. Also rebuild when the slot's
        # seqlen breaks the steady-decode +1 progression (== a new prompt / prefill->decode).
        _page_changed = self._sk_owner[:nd] != first_pages
        self._sk_owner[:nd] = first_pages
        _cur_sl = cm.seq_lens[:nd]
        if getattr(self, '_sk_prev_sl', None) is None or self._sk_prev_sl.shape[0] < self._sk_maxbs:
            self._sk_prev_sl = torch.zeros(self._sk_maxbs, dtype=_cur_sl.dtype, device=dev)
        _prev_sl = self._sk_prev_sl[:nd]
        _seq_discont = (_cur_sl != _prev_sl + 1) & (_cur_sl != _prev_sl)
        self._sk_prev_sl[:nd] = _cur_sl
        changed = _page_changed | _seq_discont
        # The ONLY potential sync: detect a position whose occupant changed (needs (re)init).
        # Fires on batch-composition change; steady decode -> all False -> no host work.
        if bool(changed.any()):
            idx = changed.nonzero(as_tuple=True)[0].tolist()
            seqs = cm.seq_lens[:nd].to("cpu").tolist()
            page = int(self.kv_cache_spec.block_size)
            for b in idx:
                L = int(seqs[b]); npg = (L + page - 1) // page
                m.sk_pending.append((b, b, L)); m.sk_pages[b] = bt[b, :npg]
        m.sk_maint = self._sk_maint
        m.sk_slots = self._sk_slots_d[:nd]
        m.sk_seqlens = self._sk_seqlens_d[:nd]
        m.sk_ptotal = int(self._sk_ptotal_fixed)
        # EAGER plan-phase summary build. Under FULL cudagraph the decode forward is
        # REPLAYED, so Impl.forward.init_layer never runs for real requests (at max_num_seqs=1
        # there is no piecewise/mixed eager decode step) -> summary stays dummy -> 0/9. This
        # build() runs EAGER every step (prepare_inputs, before any capture), so init here is
        # alloc-safe AND actually executes. Populate each layer's persistent slot buffer that
        # the captured scorer reads; the captured append maintains it per token.
        if m.sk_pending and self._sk_maint is not None:
            _pp = [(b, slot, int(L) - 1) for (b, slot, L) in m.sk_pending]
            _sfc = self.compilation_config.static_forward_context
            for _ln in self.layer_names:
                _lyr = _sfc.get(_ln)
                if _lyr is None:
                    continue
                _kvc = _lyr.kv_cache
                if isinstance(_kvc, (list, tuple)):
                    _kvc = _kvc[0] if len(_kvc) else None
                if _kvc is None or _kvc.numel() == 0:
                    continue
                _buf = self._sk_maint.get_buf(_kvc, m.sk_ptotal)
                self._sk_maint._k_scale = float(getattr(_lyr, '_k_scale_float', 1.0) or 1.0)
                self._sk_maint.init_layer(_buf, _kvc, m.sk_pages, _pp, _kvc.dtype)

    def _sk_fill_slots_syncfree(self, m, cm) -> None:
        """FlashInfer-style: detect summary (re)init purely from CPU seq_lens (no GPU
        readback). slot == batch position (fixed arange). Per-request seqlen is NOT
        copied to a buffer; the wrapper derives L_per from vLLM indptr/last_page_len."""
        m.sk_maint = None
        m.sk_pending = []
        m.sk_pages = {}
        nd = int(getattr(m, "num_decodes", 0))
        if nd <= 0:
            return
        cur = cm._seq_lens_cpu
        if cur is None:
            if not getattr(type(self), "_sk_syncfree_warned", False):
                type(self)._sk_syncfree_warned = True
                logger.warning("SYNC-FREE: _seq_lens_cpu is None; falling back to synced path")
            m._sk_in_fallback = True
            return self._sk_fill_slots(m, cm)
        cur = cur[:nd].to(torch.int64)
        prev = self._sk_prev_seqlen
        reinit = (cur != prev[:nd] + 1) & (cur != prev[:nd])  # +1=steady, ==prev=no-op (capture)
        prev[:nd] = cur
        if bool(reinit.any()):
            bt = cm.block_table_tensor
            page = int(self.kv_cache_spec.block_size)
            for b in reinit.nonzero(as_tuple=True)[0].tolist():
                L = int(cur[b]); npg = (L + page - 1) // page
                m.sk_pending.append((b, b, L))
                m.sk_pages[b] = bt[b, :npg]
        m.sk_maint = self._sk_maint
        m.sk_slots = self._sk_pos[:nd]
        m.sk_ptotal = int(self._sk_ptotal_fixed)

    @override  # type: ignore[misc]
    def _get_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool = False
    ) -> Any:
        _dbg(f"[SKYLIGHT DEBUG] _get_decode_wrapper called: batch_size={batch_size}, use_cudagraph={use_cudagraph}")
        # Mirror ``FlashInferMetadataBuilder._get_decode_wrapper`` but build
        # the sparse wrapper. When ``use_cudagraph`` is requested, slice the
        # persistent paged-KV buffers and cache one wrapper per captured
        # batch size, exactly like the parent.
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                # paged_kv_indices is a plain GPU tensor in this fork (only
                # indptr / last_page_len are CpuGpuBuffer).
                paged_kv_indices = self.paged_kv_indices
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
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper  # type: ignore[assignment]
            else:
                self._decode_wrapper = decode_wrapper  # type: ignore[assignment]

        # Store the wrapper in class attribute for debugging access
        SkylightSparseMetadataBuilder._last_decode_wrapper = decode_wrapper

        _dbg(f"[SKYLIGHT DEBUG] _get_decode_wrapper returning wrapper: {type(decode_wrapper)}")
        return decode_wrapper

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["SkylightSparseMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # FI-native decode forced, q_len_per_req == 1 → single-token decode CG.
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


class SkylightSparseImpl(FlashInferImpl):
    """Identical to :class:`FlashInferImpl` except for the block_minmax FULL-cudagraph hook:
    before the (capturable) decode runs, init any NEW request's summary from its prompt K
    (eager; this step is prefill/mixed -> piecewise) and wire the persistent per-request slot
    tensors onto the decode wrapper so the captured append/scorer read them. The dispatch in
    ``FlashInferImpl.forward()`` is otherwise reused unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sparse decode keeps Q in bf16: the kernel dequantizes fp8 K with k_scale and
        # selects top-k by precise q.k scores. Decode is KV-bandwidth-bound, so quantizing
        # Q buys no speed and only degrades selection fidelity; vLLM's fp8 query path also
        # routes to a dense trtllm kernel that does not implement our sparse selection.
        self.supports_quant_query_input = False

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        m = attn_metadata
        if os.environ.get("SKYLIGHT_INCR_DBGCNT"):
            cls = SkylightSparseImpl
            cls._fwd_calls = getattr(cls, "_fwd_calls", 0) + 1
            if cls._fwd_calls % 520 == 0:
                logger.info("SkylightSparseImpl.forward eager-call count=%d "
                            "(if climbing during steady decode -> NOT graph-replayed)",
                            cls._fwd_calls)
        maint = getattr(m, "sk_maint", None)
        if maint is not None and int(getattr(m, "num_decode_tokens", 0)) > 0:
            buf = maint.get_buf(kv_cache, m.sk_ptotal)
            # fp8 KV: dequant scale for the summary K reads (1.0 for bf16/fp16 -> no-op)
            maint._k_scale = float(getattr(layer, "_k_scale_float", 1.0) or 1.0)
            # Init NEW reqs' summary from PROMPT K (length L-1) BEFORE super(): reshape_and_cache
            # has already written prompt K (from prefill); the captured append inside super()
            # then folds the newest token. (Order matters: init must precede the append.)
            # init_layer hoisted to the EAGER metadata build (_sk_fill_slots) so it runs even
            # when this forward is cudagraph-REPLAYED. The per-layer buf (keyed by
            # kv_cache.data_ptr) is already populated there; the captured append maintains it.
            w = getattr(m.decode, "wrapper", None)
            inner = getattr(w, "_inner", w)
            if inner is not None:
                inner._slot_buf = buf
                inner._slot_slots = m.sk_slots
                inner._slot_seqlens = getattr(m, "sk_seqlens", None)
                inner._slot_ptotal = int(m.sk_ptotal)
                inner._slot_maint = maint
                inner._slot_summary = True
                inner._slot_external = True
                inner._sk_meta = m
        out = super().forward(layer, query, key, value, kv_cache, attn_metadata, output,
                              *args, **kwargs)
        
        # After forward, cache the sparse pattern for next step's staging
        _dbg(f"[IMPL DEBUG] Calling _cache_sparse_pattern")
        self._cache_sparse_pattern(m, layer)

        self._log_layer_sparsity(m, layer)
        return out

    def _cache_sparse_pattern(self, attn_metadata, layer=None):
        """Cache sparse_idx and sparse_len from the kernel for next step's staging.

        Only the FIRST layer of each step caches: the lag prefetch consumes
        one pattern per step (the last writer wins), so caching per layer did
        22 D2H copies/step of which 21 were immediately overwritten. Layer-0's
        pattern is an equally good one-step predictor. The gate is the build
        serial (models don't reliably expose layer indices on the attention
        module).
        """
        if attn_metadata is None:
            return
        if torch.cuda.is_current_stream_capturing():
            # D2H syncs are illegal during graph capture; the pattern cache
            # only feeds the *next* step's eager build()-time staging fetch, so
            # skipping during capture is safe.
            return
        builder_cls = SkylightSparseMetadataBuilder
        if getattr(builder_cls, "_pattern_cached_serial", -1) == getattr(
            builder_cls, "_step_serial", -1
        ):
            return  # already cached this step
        try:
            # Get the decode wrapper
            w = getattr(attn_metadata, "decode", None)
            if w is None:
                return
            w = getattr(w, "wrapper", None)
            if w is None:
                return
            inner = getattr(w, "_inner", w)
            if inner is None:
                return

            # Get the sparse indices from the kernel (stored in _last_topk_idx)
            sparse_idx = getattr(inner, "_last_topk_idx", None)
            if sparse_idx is None:
                return

            # Cache sparse pattern (request_ids will be filled by worker connector from model runner)
            batch_size = sparse_idx.shape[0]
            sparse_len = torch.full(
                (batch_size, sparse_idx.shape[1], 1),
                sparse_idx.shape[2], dtype=torch.int32, device=sparse_idx.device,
            )

            # Cache for next step (move to CPU to avoid GPU memory pressure)
            _timing.start("build.pattern_d2h")
            global _sparse_pattern_cache
            _sparse_pattern_cache['sparse_idx'] = sparse_idx.detach().cpu()
            _sparse_pattern_cache['sparse_len'] = sparse_len.detach().cpu()
            _timing.end("build.pattern_d2h")
            # Feed the same host-side pattern to the LRU scorer (no-op for
            # scorers without a note_usage hook): recency by top-k selection.
            orch = builder_cls._offload_orchestrator
            if orch is not None:
                ids = builder_cls._cached_request_ids or []
                if ids:
                    orch.note_selection(
                        ids, sparse_idx, int(getattr(orch, "page_size", 16)),
                    )
            # Also populate the class attrs read by build()'s staging-prep +
            # index patching path (_prepare_staging_for_sparse_attention).
            # request_ids will be filled by worker connector from model runner.
            builder_cls._cached_sparse_idx = _sparse_pattern_cache['sparse_idx']
            builder_cls._cached_sparse_len = _sparse_pattern_cache['sparse_len']
            builder_cls._pattern_cached_serial = getattr(
                builder_cls, "_step_serial", -1
            )
            _dbg(f"[SKYLIGHT DEBUG] SET _sparse_pattern_cache: sparse_idx id={id(_sparse_pattern_cache['sparse_idx'])}")
            _dbg(f"[SKYLIGHT DEBUG] Cached sparse pattern: batch={batch_size}, k={sparse_idx.shape[2]}")
        except Exception as e:
            _dbg(f"[SKYLIGHT DEBUG] Failed to cache sparse pattern: {e}")

    def _log_layer_sparsity(self, attn_metadata, layer) -> None:
        """Sample per-layer sparsity into micro_metrics.jsonl when enabled."""
        m = attn_metadata
        if int(getattr(m, "num_decode_tokens", 0)) <= 0:
            return
        w = getattr(getattr(m, "decode", None), "wrapper", None)
        inner = getattr(w, "_inner", w)
        get_stats = getattr(inner, "get_sparsity_stats", None)
        if not callable(get_stats):
            return
        stats = get_stats()
        fraction = getattr(stats, "last_fraction", None)
        if fraction is None:
            return
        layer_idx = getattr(layer, "layer_idx", None)
        if layer_idx is None:
            layer_idx = getattr(layer, "layer_id", None)
        layer_name = (
            getattr(layer, "layer_name", None)
            or getattr(layer, "prefix", None)
            or type(layer).__name__
        )
        try:
            from skylight.bench.micro_metrics import get_logger
            get_logger().log(
                "skylight_sparsity_fraction",
                float(fraction),
                metadata={"layer_idx": layer_idx, "layer_name": str(layer_name)},
                location="SkylightSparseImpl.forward",
            )
        except Exception:
            pass
