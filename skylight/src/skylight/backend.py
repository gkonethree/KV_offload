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
    get_flashinfer_layout_string,  # this vLLM fork has no get_kv_cache_layout
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from skylight.config import SkylightSparseConfig

logger = init_logger(__name__)

try:  # KV-offload timing (offload_package); Skylight also runs without it
    from offload_package import timing as _timing
except ImportError:
    class _timing:  # no-op stand-in for offload_package.timing
        @staticmethod
        def start(stage: str) -> None:
            pass

        end = start


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
        elif method == "pq":
            # Pure-torch selection (no CUDA kernel yet): correct, but far
            # slower per step than the other two. Meant for accuracy studies,
            # not throughput numbers.
            from pq import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method == "pq_optimized":
            # Graph-safe production path: slot-buffer codes, a CUDA ADC
            # scorer, and sparse_optimized's decode. Same selection as 'pq',
            # arranged so a cudagraph can hold it.
            from pq_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method == "pqrs_optimized":
            # pq_optimized plus vAttention's weighted sample of the residual.
            # No extra kernel: sparse_decode already applies gather weights as
            # a log-shift, which is the estimator.
            from pqrs_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method == "pqis_optimized":
            # vAttention++ Algorithm 1: Gumbel-top-k over the PQ scores draws a
            # sample from softmax(s~), and inverse inclusion probabilities
            # correct it. Same budget as pq_optimized -- the sample replaces the
            # top-k block rather than extending it -- so the two differ only in
            # the estimator.
            from pqis_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method == "qis_optimized":
            # Quest bounds + importance sampling. block_minmax's per-sub-page
            # min/max score, stratified the way pqis stratifies PQ scores, and
            # selected PER QUERY HEAD -- which is why it decodes through
            # sparse_optimized rather than FlashInfer's paged-sparse kernel
            # (that one takes a single page list per request, so block_minmax
            # collapses its per-head bounds with amax).
            #
            # It needs the SUMMARY maintainer, not the PQ code maintainer, and
            # gets it by NOT being listed in the pq_optimized branch below --
            # the fullcg fallback installs SlotSummaryMaintainer. Adding
            # 'qis_optimized' to that list would hand it PQ codes it cannot
            # read.
            from qis_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method == "pqrs":
            # pq plus vAttention's weighted sampling of the residual. Sample
            # size comes from SKYLIGHT_PQRS_NSAMPLE; 0 makes it identical to
            # 'pq'. Same pure-torch caveat.
            from pqrs import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        elif method in ("oracle", ""):
            from sparse_oracle_topk_sink_local_optimized import (
                BatchDecodeWithPagedKVCacheWrapper as cls,
            )
        else:
            raise ValueError(
                f"SKYLIGHT_SPARSE_METHOD={method!r} not recognized "
                "(expected 'oracle', 'block_minmax', 'pq', 'pqrs', 'pq_optimized', "
                "'pqrs_optimized', 'pqis_optimized' or 'qis_optimized')."
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
        # Must equal an :class:`AttentionBackendEnum` member name. vllm
        # internally does ``AttentionBackendEnum[backend.get_name()]`` (see
        # ``vllm/model_executor/layers/attention/attention.py``) and would
        # ValueError on any string not in the enum. We register under the
        # ``CUSTOM`` placeholder slot (the only enum entry meant for
        # third-party backends), so this must return ``"CUSTOM"``.
        #
        # Without forking vllm we can't add a ``SKYLIGHT_SPARSE`` enum
        # member; the user-facing selection name is ``CUSTOM`` everywhere.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SkylightSparseImpl"]:
        return SkylightSparseImpl

    @staticmethod
    def get_builder_cls() -> type["SkylightSparseMetadataBuilder"]:
        return SkylightSparseMetadataBuilder


class SkylightSparseMetadataBuilder(FlashInferMetadataBuilder):
    """Parent's metadata, with the per-step decode wrapper swapped to sparse."""

    # ---- KV offload hooks (offload_package) ---------------------------------
    # Class-level references set by the offload worker connector
    # (offload_package/integration/worker_connector.py).
    _offload_orchestrator: object | None = None
    # The GPU model runner: used to derive the CURRENT step's decode request
    # ids at build time (post-reorder, decode rows first) -- the
    # scheduler-bound ids arrive one step late and their order can differ from
    # the runner's row order in mixed batches.
    _offload_runner: object | None = None
    # Last bail reason recorded by _patch_indices_for_staging for this build.
    _patch_bail_reason: str | None = None
    # Engine-step serial (bumped per build) and the step whose selection was
    # last noted; together they limit selection noting to once per step.
    _step_serial: int = 0
    _pattern_cached_serial: int = -1
    _cached_request_ids: list[str] | None = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # The offload hooks are CLASS attributes injected by the offload
        # connector. An engine without that connector (e.g. a plain pq run
        # after an offload run in the same process) must not inherit the
        # previous engine's orchestrator/runner: it would patch indices and
        # move codes against a dead engine's residency table.
        _kvt = getattr(vllm_config, "kv_transfer_config", None)
        self._sk_offload_cfg = None
        if _kvt is not None and getattr(_kvt, "kv_connector", None) == "OffloadPackageConnector":
            self._sk_offload_cfg = dict(getattr(_kvt, "kv_connector_extra_config", None) or {})
        else:
            cls_ = SkylightSparseMetadataBuilder
            cls_._offload_orchestrator = None
            cls_._offload_runner = None
            cls_._cached_request_ids = []

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
        # SKYLIGHT_SPARSE_DYNAMIC_TOPK=1: size the selected set from the batch's
        # ACTUAL max_seq_len every step, so topk means x% of the live context
        # rather than x% of max_model_len. Without it, methods on the FULL-cg
        # path (block_minmax, pq_optimized, pqrs_optimized) use a fixed token
        # count derived from max_model_len while 'oracle', which is not on that
        # path, already gets the dynamic budget -- so the two are not
        # budget-matched at the same nominal topk.
        #
        # REQUIRES EAGER DECODE. total_k = sink + k_mid + local is the captured
        # shape of sparse_idx, so a per-batch n_keys changes it between replays.
        # serve_and_eval_aime forces --enforce-eager when this is set; nothing
        # here can detect capture reliably across vllm versions, so it is the
        # caller's contract.
        # SKYLIGHT_SPARSE_BUDGET_MODE = auto | dynamic | fixed
        #   auto    (default) whatever each path does today: 'oracle' dynamic,
        #           the FULL-cg methods fixed. NOT budget-matched.
        #   dynamic every method sizes topk from the live context. Honest
        #           labels, but requires eager decode.
        #   fixed   no method pushes n_keys, so 'oracle' joins the FULL-cg
        #           methods on a fixed count from max_model_len. Matched, stays
        #           captured, and makes k_mid a known constant -- which is what
        #           lets pqrs's top-k/sample split be set exactly.
        _mode = os.environ.get("SKYLIGHT_SPARSE_BUDGET_MODE", "auto").strip().lower()
        if _mode not in ("auto", "dynamic", "fixed"):
            raise ValueError(
                f"SKYLIGHT_SPARSE_BUDGET_MODE={_mode!r} not recognized "
                "(expected 'auto', 'dynamic' or 'fixed')."
            )
        # Mirrors SlotCodeMaintainer's own read; the backend needs it to
        # know when to force a deferred codebook fit.
        _du = os.environ.get("SKYLIGHT_SPARSE_DENSE_UNTIL", "").strip()
        self._sk_dense_until = int(_du) if _du else 0
        self._sk_fixed_topk = _mode == "fixed"
        self._sk_dynamic_topk = (
            _mode == "dynamic"
            or os.environ.get("SKYLIGHT_SPARSE_DYNAMIC_TOPK") == "1"
        )
        if self._sk_fixed_topk and self._sk_dynamic_topk:
            raise ValueError(
                "SKYLIGHT_SPARSE_BUDGET_MODE=fixed conflicts with "
                "SKYLIGHT_SPARSE_DYNAMIC_TOPK=1; pick one."
            )
        if self._sk_dynamic_topk:
            logger.info("SKYLIGHT_SPARSE_DYNAMIC_TOPK=1: topk is x%% of the live "
                        "context; REQUIRES eager decode (uncaptured shapes)")
        self._sk_maint = None
        self._sk_syncfree = False
        self._sk_slots_d = None
        self._sk_seqlens_d = None
        self._sk_sub_page = int(os.environ.get("SKYLIGHT_SPARSE_SUB_PAGE", "8"))
        self._sk_ptotal_fixed = 0
        # Which maintainer backs the FULL-cudagraph slot path. The hook below
        # (_sk_fill_slots) is method-agnostic: slot assignment, the persistent
        # buffers and change detection are the same problem for any
        # incrementally maintained per-request summary. Only the CONTENTS
        # differ -- per-sub-page min/max bounds for block_minmax, per-token PQ
        # codes for pq_optimized -- so both are driven through the same three
        # calls: get_buf, init_layer, and the captured append.
        _sk_method = os.environ.get("SKYLIGHT_SPARSE_METHOD", "oracle").strip().lower()
        if self._sk_fullcg and _sk_method in ("pq_optimized", "pqrs_optimized",
                                              "pqis_optimized"):
            from pq_optimized.slot_codes import SlotCodeMaintainer

            max_bs = int(vllm_config.scheduler_config.max_num_seqs)
            cap = getattr(vllm_config.compilation_config, "max_cudagraph_capture_size", None)
            if cap:
                max_bs = min(max_bs, int(cap))
            self._sk_maint = SlotCodeMaintainer(
                h_kv=int(self.kv_cache_spec.num_kv_heads),
                m=int(os.environ.get("SKYLIGHT_PQ_SUBSPACES", "16")),
                page=int(self.kv_cache_spec.block_size),
                n_slots=max_bs,
                l_max=int(self._sparse_max_seq_len),
                max_batch=max_bs,
            )
            self._sk_maint._codebook_size = int(
                os.environ.get("SKYLIGHT_PQ_CODEBOOK", "256"))
            self._sk_maint._fit_sample = int(
                os.environ.get("SKYLIGHT_PQ_FIT_SAMPLE", "4096"))
            self._sk_maint._ch = (
                int(self.head_dim) if self._sparse_channel_num == -1
                else int(self._sparse_channel_num))
            self._sk_maxbs = max_bs
            self._sk_owner = None
            self._sk_slots_d = torch.zeros(max_bs, dtype=torch.int32, device=device)
            self._sk_seqlens_d = torch.zeros(max_bs, dtype=torch.int32, device=device)
            self._sk_pos = torch.arange(max_bs, dtype=torch.int32, device=device)
            self._sk_prev_seqlen = torch.full((max_bs,), -1, dtype=torch.int64)
            # For PQ the "summary width" is the token axis, not a sub-page count.
            self._sk_ptotal_fixed = int(self._sparse_max_seq_len)
            logger.info(
                "SkylightSparseBackend: FULL-cudagraph PQ slot path ON "
                "(max_bs=%d, L_max=%d)", max_bs, self._sk_ptotal_fixed)
        elif self._sk_fullcg:
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
                # The sync-free path detects a slot's new occupant from CPU
                # seq-len continuity alone and rebuilds summaries from the
                # block table: no request ownership, physical lengths. Under KV
                # offload that can rebuild from freed (true freeing) or
                # compacted (token-level offload) pages. Refuse the combo.
                _x = self._sk_offload_cfg
                if _x is not None and _x.get("enabled", True) and (
                        _x.get("true_free", True)
                        or _x.get("offload_granularity", "block") == "token"):
                    raise ValueError(
                        "SKYLIGHT_INCR_SYNCFREE=1 is not supported with "
                        "OffloadPackageConnector true freeing or token-level "
                        "offloading; disable SKYLIGHT_INCR_SYNCFREE.")
            self._sk_ptotal_fixed = (
                self._sparse_max_seq_len + self._sk_sub_page - 1) // self._sk_sub_page
            logger.info("SkylightSparseBackend: FULL-cudagraph slot path ON (max_bs=%d, P=%d)",
                        max_bs, self._sk_ptotal_fixed)

        # KV offload + PQ: codes must be maintained per request slot (the
        # FULL-cg slot path above). Without it pq_optimized encodes from the
        # block table, which under offload holds freed / compacted pages.
        if (self._sk_offload_cfg is not None and self._sk_offload_cfg.get("enabled", True)
                and _sk_method in ("pq_optimized", "pqrs_optimized", "pqis_optimized")
                and not self._sk_fullcg):
            raise ValueError(
                f"SKYLIGHT_SPARSE_METHOD={_sk_method} with OffloadPackageConnector "
                "needs the PQ slot path: set SKYLIGHT_INCR_FULLCG=1 and "
                "SKYLIGHT_INCR_SLOT=1 (eager is fine; add "
                "SKYLIGHT_SPARSE_DYNAMIC_TOPK=1 for top-k over the live context).")

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
        return self._sparse_wrapper_cls(
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
        num_decodes = self._sk_offload_prebuild(common_attn_metadata)
        attn_metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build=fast_build,
        )
        self._sk_offload_postbuild(attn_metadata, num_decodes)
        # Token-level KV offload makes seq_lens PHYSICAL (logical - offloaded);
        # top-k must scale with the LOGICAL context, so add the offset back.
        _tok_off = self._sk_tok_offsets(int(getattr(attn_metadata, "num_decodes", 0) or 0))
        _max_logical = int(common_attn_metadata.max_seq_len) + (
            int(_tok_off.max()) if _tok_off is not None else 0)
        if self._sk_fullcg:
            # Under FULL cudagraph the selected-set size (total_k) MUST be fixed across replays,
            # so do NOT push the dynamic per-batch length via set_n_keys (that would bake a
            # capture-time total_k into the graph). Leaving n_keys=None makes the wrapper size
            # total_k from the fixed max_seq_len; per-request L_per still masks invalid tokens.
            self._sk_fill_slots(attn_metadata, common_attn_metadata)
            if self._sk_dynamic_topk:
                # Opt-in: match 'oracle', which gets this by virtue of not being
                # on this path. Only valid eager -- see the flag's comment.
                self._set_n_keys_on_decode(
                    attn_metadata.decode
                    if isinstance(attn_metadata.decode, FIDecode)
                    else None,
                    _max_logical,
                )
        elif not self._sk_fixed_topk:
            self._set_n_keys_on_decode(
                attn_metadata.decode
                if isinstance(attn_metadata.decode, FIDecode)
                else None,
                _max_logical,
            )
        self._log_budget_once(common_attn_metadata)
        return attn_metadata

    # ---- KV offload: per-build hooks ----------------------------------------

    def _sk_offload_prebuild(self, cm) -> int:
        """Before the parent build: bump the step serial and record THIS
        step's decode request ids from the runner's input batch. Returns the
        number of decode rows (0 when offload is not wired)."""
        cls = type(self)
        self._patch_bail_reason = None
        cls._step_serial = getattr(cls, "_step_serial", 0) + 1
        num_decodes = 0
        # reorder_batch_to_split_decodes_and_prefills has already run by now,
        # so req_ids[:num_decodes] is the true decode row order -- including
        # mixed batches, where the scheduler-side id order is not guaranteed
        # to match row order. Only the id STASH is subject to the
        # SKYLIGHT_OFFLOAD_DISABLE_RUNNER_IDS fault-injection switch.
        if (
            cls._offload_runner is not None
            and cls._offload_orchestrator is not None
            and cm is not None
            and not torch.cuda.is_current_stream_capturing()
        ):
            try:
                num_decodes, _, _, _ = split_decodes_and_prefills(
                    cm, decode_threshold=self.reorder_batch_threshold,
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
        return num_decodes

    def _sk_offload_postbuild(self, attn_metadata, num_decodes: int) -> None:
        """After the parent build: fail loudly on a hazardous index-patch
        bail, and hand the orchestrator + this step's decode request order to
        the sparse wrapper (wrapper instances are cached per batch size)."""
        cls = type(self)
        orch = cls._offload_orchestrator
        # Under true freeing, offloaded blocks' original GPU pages are freed;
        # the index patcher + in-run fetch are all that stand between the
        # kernel and those pages, so a bail on a decode step is an error.
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
        if attn_metadata.decode is not None:
            w = getattr(attn_metadata.decode, "wrapper", None)
            inner = getattr(w, "_inner", w)
            if inner is not None and hasattr(inner, "set_offload_orchestrator"):
                inner.set_offload_orchestrator(orch)
                inner.set_offload_request_ids(cls._cached_request_ids)

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
        """Re-point offloaded blocks' index entries at build time: STAGING
        blocks to their staging slot, CPU blocks to the null page (under true
        freeing). The PQ wrapper's in-run fetch then stages what this step
        selects.

        Records ``self._patch_bail_reason`` for every early return that leaves
        offloaded blocks' index entries pointing at their original (possibly
        freed) GPU pages; ``build()`` turns hazardous bails into errors under
        true freeing.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        cls = type(self)
        orchestrator = cls._offload_orchestrator
        if orchestrator is None:
            return
        request_ids = cls._cached_request_ids or []
        if len(request_ids) == 0:
            self._patch_bail_reason = "ids_empty"
            return
        if len(request_ids) > num_reqs:
            self._patch_bail_reason = "ids_gt_reqs"
            return
        try:
            # Defensive alignment check: per-row page counts in the current
            # batch must equal the request's tracked block count, otherwise the
            # request order does not match decode row order and we must not
            # patch (row-b is ReqX's span, and patching assumes request_ids[b]).
            # The parent filled the host side of paged_kv_indptr: no GPU sync.
            n = len(request_ids)
            indptr_gpu = self.paged_kv_indptr.gpu[: n + 1]
            indptr_list = self.paged_kv_indptr.np[: n + 1].tolist()
            for b, rid in enumerate(request_ids):
                row_pages = int(indptr_list[b + 1]) - int(indptr_list[b])
                tracked = len(orchestrator.residency.refs_for_request(rid))
                if tracked == 0 or row_pages != tracked:
                    self._patch_bail_reason = (
                        f"row_alignment b={b} rid={rid} "
                        f"tracked={tracked} row_pages={row_pages}"
                    )
                    return

            from offload_package.selection.adapter import (
                patch_paged_kv_indices_with_staging,
            )
            null_id = 0 if getattr(orchestrator, "true_freeing", False) else None
            _timing.start("build.patch")
            patch_paged_kv_indices_with_staging(
                paged_kv_indices,
                indptr_gpu,
                request_ids,
                None,
                None,
                self.kv_cache_spec.block_size,
                orchestrator.staging_base_page_idx,
                orchestrator.residency,
                {},
                null_block_id=null_id,
                indptr_host=indptr_list,
                inplace=True,
            )
            _timing.end("build.patch")
        except Exception:
            logger.exception("SkylightSparseBackend: patching staging indices failed")
            if (
                getattr(orchestrator, "true_freeing", False)
                and getattr(orchestrator.residency, "num_cpu_blocks", 0) > 0
            ):
                # A swallowed exception here leaves offloaded blocks' entries
                # pointing at freed pages; fail loudly under true freeing.
                raise

    def _log_budget_once(self, cm) -> None:
        """SKYLIGHT_BUDGET_DBG=1: print the budget each method ACTUALLY uses.

        The arms are not obviously comparable and the difference is invisible at
        runtime. 'oracle' is off the FULL-cg path, so set_n_keys pushes the live
        max_seq_len and its topk is a true fraction of the live context.
        block_minmax and pq_optimized are on that path, where n_keys is withheld
        (total_k is the captured shape of sparse_idx) and topk is a fixed token
        count from max_model_len. This prints both, plus what an honest x% would
        have selected, so the ratio is a measurement rather than an argument.

        Lives in build(), which is EAGER every step for every method -- a print
        inside the wrapper's run() would miss the captured methods entirely,
        since their run() only executes during capture warmup.
        """
        if os.environ.get("SKYLIGHT_BUDGET_DBG") != "1":
            return
        if getattr(type(self), "_budget_dbg_done", False):
            return
        live = int(getattr(cm, "max_seq_len", 0) or 0)
        if live <= self._sparse_sink_size + self._sparse_local_size:
            return                      # middle is empty; nothing to report yet
        setattr(type(self), "_budget_dbg_done", True)
        sink, local = int(self._sparse_sink_size), int(self._sparse_local_size)
        dynamic = self._sk_dynamic_topk or (
            (not self._sk_fullcg) and not self._sk_fixed_topk)
        n_keys = live if dynamic else int(self._sparse_max_seq_len)

        def k_of(n):
            mid = max(0, n - sink - local)
            if mid <= 0:
                return 0
            return min(max(1, round(float(self._sparse_topk) * mid)), mid)

        k_used, k_honest = k_of(n_keys), k_of(live)
        tot_used, tot_honest = sink + k_used + local, sink + k_honest + local
        # stderr, not logger.info: skylight's logger records do not reliably
        # reach the captured server.log, which is why block_minmax's own
        # SKYLIGHT_BMM_DBG writes to stderr too. A diagnostic you cannot read
        # is worse than none, since its silence looks like a negative result.
        import sys as _sys
        print(
            "[BUDGET] method=%s fullcg=%s dyn=%s fixed=%s topk=%.4g | live_L=%d "
            "n_keys_used=%d -> k_mid=%d total_k=%d | honest x%%: k_mid=%d "
            "total_k=%d | reading %.2fx honest (%.2f%% of live vs %.2f%% nominal)"
            % (os.environ.get("SKYLIGHT_SPARSE_METHOD", "?"), self._sk_fullcg,
               self._sk_dynamic_topk, self._sk_fixed_topk,
               float(self._sparse_topk), live,
               n_keys, k_used, tot_used, k_honest, tot_honest,
               (tot_used / tot_honest) if tot_honest else float("nan"),
               100.0 * tot_used / live, 100.0 * float(self._sparse_topk)),
            file=_sys.stderr, flush=True,
        )
        logger.info("[BUDGET] (also on stderr) method=%s n_keys_used=%d total_k=%d",
                    os.environ.get("SKYLIGHT_SPARSE_METHOD", "?"), n_keys, tot_used)

    # ---- KV offload changes in _sk_fill_slots ------------------------------
    # Upstream detects a slot's new occupant (first-page change or a break in
    # the +1 seq-len progression) and re-encodes that row's PQ codes from the
    # block table. Under KV offload that re-encode can read the wrong bytes:
    #   * token-level offload compacts KV, so cm.seq_lens is PHYSICAL and
    #     drops after every offload round -- read as a "discontinuity" -- and
    #     the pages now hold packed / compacted tokens, not positions 0..L-1;
    #   * block offload with true freeing frees the pages of offloaded blocks,
    #     which may already hold another request's KV.
    # The minimum changes, all no-ops without offload:
    #   1. lengths are LOGICAL (seq_lens + per-row token-offload offset);
    #   2. a row whose request just moved (vLLM condensed the batch) gets its
    #      codes COPIED from the request's old row -- exact -- instead of a
    #      re-encode (_sk_plan_row_moves; rows owned by request id);
    #   3. any remaining re-encode is refused when the request's pages were
    #      compacted or freed (_sk_assert_safe_reencode).
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
        _logical_sl = cm.seq_lens[:nd]                     # offload change (1)
        _tok_off = self._sk_tok_offsets(nd)
        if _tok_off is not None:
            _logical_sl = _logical_sl + _tok_off.to(_logical_sl.dtype)
        self._sk_seqlens_d[:nd].copy_(_logical_sl.to(torch.int32))
        if self._sk_owner is None:
            self._sk_owner = torch.full((self._sk_maxbs,), -1, dtype=first_pages.dtype, device=dev)
        # Rebuild a slot's summary when a NEW request occupies it. first-page change alone is
        # WRONG: with prefix caching, sequential requests share the first physical block so the
        # proxy never fires and requests #2+ reuse a stale summary. Also rebuild when the slot's
        # seqlen breaks the steady-decode +1 progression (== a new prompt / prefill->decode).
        _page_changed = self._sk_owner[:nd] != first_pages
        self._sk_owner[:nd] = first_pages
        _cur_sl = _logical_sl
        if getattr(self, '_sk_prev_sl', None) is None or self._sk_prev_sl.shape[0] < self._sk_maxbs:
            self._sk_prev_sl = torch.zeros(self._sk_maxbs, dtype=_cur_sl.dtype, device=dev)
        _prev_sl = self._sk_prev_sl[:nd]
        _seq_discont = (_cur_sl != _prev_sl + 1) & (_cur_sl != _prev_sl)
        # Every row's previous length (incl. rows a shrinking batch just
        # vacated), for the row-move lookup below (offload change (2)).
        _prev_all = self._sk_prev_sl.clone()
        self._sk_prev_sl[:nd] = _cur_sl
        changed = _page_changed | _seq_discont
        # Deferred codebook fit. With SKYLIGHT_SPARSE_DENSE_UNTIL set, the PQ
        # codebook deliberately does NOT fit at prefill -- centroids learned
        # from a ~100-token prompt and then used for 32k of generation are
        # close to worthless. It fits once a request has that many keys, and
        # nothing else would ever bring the request back here: the two terms
        # above are exactly false during steady decode, by design. So the
        # crossing is detected explicitly, and only while a layer is still
        # waiting -- one small D2H copy per step until the fit lands, then
        # never again.
        # Retry an unfitted codebook whenever enough keys exist -- NOT only
        # when DENSE_UNTIL asked for a deferral. Measured: the AIME prompt is
        # 255 tokens against a 256-key floor, so init_layer skipped the fit,
        # nothing ever retried, and PQ ran on ZERO centroids for every result
        # in the project. A codebook that silently stays zeros forever is the
        # failure mode this guards.
        # Only the PQ code maintainer has a codebook to be unfitted. The
        # summary maintainer (block_minmax, oracle, qis) is exact from the
        # first token and exposes neither method, so an unguarded call here
        # crashed engine init with
        #   AttributeError: 'SlotSummaryMaintainer' has no attribute 'any_unfitted'
        # for every non-PQ method on the fullcg path.
        if (self._sk_maint is not None
                and hasattr(self._sk_maint, "any_unfitted")
                and self._sk_maint.any_unfitted()
                and int(_cur_sl.max().item()) >= self._sk_maint.fit_threshold()):
            changed = changed | torch.ones_like(changed)
        # The ONLY potential sync: detect a position whose occupant changed (needs (re)init).
        # Fires on batch-composition change; steady decode -> all False -> no host work.
        moves: list[tuple[int, int]] = []    # (src_row, dst_row), offload change (2)
        if bool(changed.any()):
            idx = changed.nonzero(as_tuple=True)[0].tolist()
            seqs = _logical_sl.to("cpu").tolist()
            page = int(self.kv_cache_spec.block_size)
            moves = self._sk_plan_row_moves(idx, seqs, _prev_all.tolist(), nd)
            moved_dst = {d for _, d in moves}
            for b in idx:
                if b in moved_dst:
                    continue
                L = int(seqs[b]); npg = (L + page - 1) // page
                m.sk_pending.append((b, b, L)); m.sk_pages[b] = bt[b, :npg]
        self._sk_record_row_owners(nd)
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
        if (moves or m.sk_pending) and self._sk_maint is not None:
            _pp = [(b, slot, int(L) - 1) for (b, slot, L) in m.sk_pending]
            _sfc = self.compilation_config.static_forward_context
            if moves:
                _src = torch.tensor([a for a, _ in moves], dtype=torch.long, device=dev)
                _dst = torch.tensor([b for _, b in moves], dtype=torch.long, device=dev)
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
                if moves:
                    # Advanced-index RHS is gathered before the write, so
                    # permutations (A<->B) are safe. Moves run before inits:
                    # a vacated source row may be re-initialized below.
                    _buf[_dst] = _buf[_src]
                if _pp:
                    self._sk_maint._k_scale = float(getattr(_lyr, '_k_scale_float', 1.0) or 1.0)
                    self._sk_maint.init_layer(_buf, _kvc, m.sk_pages, _pp, _kvc.dtype)

    def _sk_current_row_ids(self, nd: int):
        """Request id per decode row for THIS step, or None if unknown."""
        runner = type(self)._offload_runner
        ib = getattr(runner, "input_batch", None) if runner is not None else None
        ids = list(getattr(ib, "req_ids", []) or [])[:nd]
        if len(ids) != nd or any(i is None for i in ids):
            return None
        return [str(i) for i in ids]

    def _sk_plan_row_moves(self, changed_rows, seqs, prev_all, nd):
        """For each changed row, find the row its request occupied last step.

        Slot == batch position, so when vLLM condenses the batch (a request
        finished) a continuing request lands in a new row. Its codes already
        exist in its OLD row; moving them is exact. Re-encoding from the
        block table is NOT: under offload with true freeing the pages of its
        offloaded blocks have been freed and possibly reused by other
        requests, so the re-encoded codes would be garbage.
        """
        ids = self._sk_current_row_ids(nd)
        owners = getattr(self, "_sk_row_owner", None)
        moves: list[tuple[int, int]] = []
        if ids is None or owners is None:
            self._sk_assert_safe_reencode(changed_rows, ids)
            return moves
        old_row = {rid: r for r, rid in enumerate(owners) if rid is not None}
        need_init = []
        for b in changed_rows:
            rid = ids[b]
            src = old_row.get(rid)
            if (src is not None and src != b
                    and int(seqs[b]) in (int(prev_all[src]) + 1, int(prev_all[src]))):
                moves.append((src, b))
            else:
                need_init.append(b)
        self._sk_assert_safe_reencode(need_init, ids)
        return moves

    def _sk_tok_offsets(self, nd: int):
        """Per decode row: tokens offloaded by token-level offloading
        (logical - physical length), on device; None when all zero."""
        orch = type(self)._offload_orchestrator
        eng = getattr(orch, "token_offload", None) if orch is not None else None
        if eng is None or nd <= 0:
            return None
        ids = self._sk_current_row_ids(nd)
        if ids is None:
            return None
        off = [eng.offset(r) for r in ids]
        if not any(off):
            return None
        return torch.tensor(off, dtype=torch.int64, device=self.device)

    def _sk_assert_safe_reencode(self, rows, ids) -> None:
        """Re-encoding reads K from the block table; refuse when any of the
        request's blocks is no longer on its original GPU page."""
        orch = type(self)._offload_orchestrator
        eng = getattr(orch, "token_offload", None) if orch is not None else None
        if eng is not None and rows:
            for b in rows:
                if ids is None or eng.remapped(ids[b]):
                    raise RuntimeError(
                        "PQ codes would be re-encoded from compacted pages "
                        "(token-level offloading); refusing -- codes must move.")
        if orch is None or not getattr(orch, "true_freeing", False) or not rows:
            return
        if ids is None:
            if getattr(orch, "freed_pages", None):
                raise RuntimeError(
                    "PQ codes must be re-encoded for a changed decode row but "
                    "request ids are unavailable while blocks are offloaded "
                    "under true freeing; refusing to encode freed pages.")
            return
        freed = getattr(orch, "freed_pages", set())
        for b in rows:
            for ref in orch.residency.refs_for_request(ids[b]):
                if ref in freed:
                    raise RuntimeError(
                        f"PQ codes for request {ids[b]} (row {b}) would be "
                        "re-encoded from the block table, but its block "
                        f"{ref.block_idx} is offloaded (pages freed); refusing.")

    def _sk_record_row_owners(self, nd: int) -> None:
        ids = self._sk_current_row_ids(nd)
        owners = [None] * int(self._sk_maxbs)
        if ids is not None:
            owners[:nd] = ids
        self._sk_row_owner = owners

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
        # KV offload: feed this step's selection to the orchestrator (LRU).
        self._note_selection(m)
        self._log_layer_sparsity(m, layer)
        return out

    def _note_selection(self, attn_metadata) -> None:
        """Hand this step's top-k selection (the PQ wrapper's GPU
        ``_last_topk_idx``) to the offload orchestrator's ``note_selection``
        (LRU recency for blocks). Once per engine step -- the first layer;
        the build serial gates it (attention modules don't reliably expose
        layer indices)."""
        if attn_metadata is None or torch.cuda.is_current_stream_capturing():
            return
        builder_cls = SkylightSparseMetadataBuilder
        if getattr(builder_cls, "_pattern_cached_serial", -1) == getattr(
            builder_cls, "_step_serial", -1
        ):
            return  # already noted this step
        try:
            w = getattr(getattr(attn_metadata, "decode", None), "wrapper", None)
            inner = getattr(w, "_inner", w)
            sparse_idx = getattr(inner, "_last_topk_idx", None)
            if sparse_idx is None:
                return
            orch = builder_cls._offload_orchestrator
            ids = builder_cls._cached_request_ids or []
            if orch is not None and ids:
                orch.note_selection(
                    ids, sparse_idx, int(getattr(orch, "page_size", 16)),
                )
            builder_cls._pattern_cached_serial = getattr(
                builder_cls, "_step_serial", -1
            )
        except Exception:
            logger.debug("SkylightSparseBackend: note_selection failed", exc_info=True)

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
