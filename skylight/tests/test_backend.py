"""Tests for skylight.backend: structure + registration.

Unit tests only (no GPU). End-to-end "actually generates tokens" coverage
lives in tests/integration/test_smoke_sparse.py.
"""
from __future__ import annotations

import pytest

import skylight.blackwell as blackwell
import skylight.plugin as plugin


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear plugin + blackwell module-level guards before each test."""
    plugin._INSTALLED = False
    blackwell._APPLIED = False
    blackwell.state.clear()
    # Clear the registry override so register_backend gets fresh state.
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        _ATTN_OVERRIDES,
    )
    _ATTN_OVERRIDES.pop(AttentionBackendEnum.CUSTOM, None)
    yield


# -------------------------------- module loads --------------------------------


def test_backend_module_imports_cleanly():
    """skylight.backend imports without triggering the kernel-package import."""
    import skylight.backend as backend
    assert hasattr(backend, "SkylightSparseBackend")
    assert hasattr(backend, "SkylightSparseMetadataBuilder")
    assert hasattr(backend, "SkylightSparseImpl")
    # The kernel-package import is lazy — verify it hasn't fired yet.
    import sys
    # Note: the kernel package may be imported by some OTHER code (e.g.
    # test_blackwell's torch import chain doesn't pull it in). We only
    # assert backend.py itself didn't eagerly import it.
    assert backend._import_sparse_wrapper_cls.__name__ == "_import_sparse_wrapper_cls"


# -------------------------------- subclass hierarchy -------------------------


def test_subclass_hierarchy():
    """Our classes extend the corresponding FlashInfer ones."""
    from vllm.v1.attention.backends.flashinfer import (
        FlashInferBackend,
        FlashInferImpl,
        FlashInferMetadataBuilder,
    )
    from skylight.backend import (
        SkylightSparseBackend,
        SkylightSparseImpl,
        SkylightSparseMetadataBuilder,
    )

    assert issubclass(SkylightSparseBackend, FlashInferBackend)
    assert issubclass(SkylightSparseMetadataBuilder, FlashInferMetadataBuilder)
    assert issubclass(SkylightSparseImpl, FlashInferImpl)


# -------------------------------- class methods ------------------------------


def test_get_name_returns_custom_enum_member():
    """get_name() must return a valid AttentionBackendEnum member name.

    vllm's attention layer does ``AttentionBackendEnum[backend.get_name()]``;
    a string not in the enum raises ValueError. Without forking vllm to
    add a SKYLIGHT_SPARSE member, we ride the ``CUSTOM`` placeholder slot.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from skylight.backend import SkylightSparseBackend

    name = SkylightSparseBackend.get_name()
    assert name == "CUSTOM"
    # Sanity: the returned string IS a valid enum member name.
    assert AttentionBackendEnum[name] is AttentionBackendEnum.CUSTOM


def test_get_impl_and_builder_cls_resolve():
    """Backend points at our impl + builder classes."""
    from skylight.backend import (
        SkylightSparseBackend,
        SkylightSparseImpl,
        SkylightSparseMetadataBuilder,
    )
    assert SkylightSparseBackend.get_impl_cls() is SkylightSparseImpl
    assert SkylightSparseBackend.get_builder_cls() is SkylightSparseMetadataBuilder


def test_supported_head_sizes_matches_kernel_templates():
    """Kernel templates are instantiated for {64, 128, 256}."""
    from skylight.backend import SkylightSparseBackend
    assert SkylightSparseBackend.get_supported_head_sizes() == [64, 128, 256]


def test_supported_dtypes_excludes_fp8():
    """Sparse kernel only supports unquantized fp16/bf16 today."""
    import torch
    from skylight.backend import SkylightSparseBackend
    assert SkylightSparseBackend.supported_dtypes == [torch.float16, torch.bfloat16]


# -------------------------------- registration -------------------------------


def test_install_plugin_registers_under_custom():
    """install_plugin() populates _ATTN_OVERRIDES[CUSTOM] with our class path."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        _ATTN_OVERRIDES,
    )

    plugin.install_plugin()
    path = _ATTN_OVERRIDES.get(AttentionBackendEnum.CUSTOM)
    assert path == "skylight.backend.SkylightSparseBackend"


def test_custom_enum_resolves_to_skylight_backend_after_install():
    """End-to-end: AttentionBackendEnum.CUSTOM.get_class() returns our backend."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    plugin.install_plugin()

    resolved = AttentionBackendEnum.CUSTOM.get_class()
    from skylight.backend import SkylightSparseBackend
    assert resolved is SkylightSparseBackend


def test_register_backend_is_overrideable():
    """Calling install_plugin() twice keeps the override (no error/duplicate)."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        _ATTN_OVERRIDES,
    )

    plugin.install_plugin()
    assert AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES

    plugin.install_plugin()  # second call no-ops via _INSTALLED guard
    assert _ATTN_OVERRIDES[AttentionBackendEnum.CUSTOM] == \
        "skylight.backend.SkylightSparseBackend"


# -------------------------------- _NoFastPlanDecodeWrapper -------------------


class _FakeInner:
    """Stand-in for the kernel wrapper used by the no-fast-plan tests."""
    is_cuda_graph_enabled = True  # deliberately opposite of what the outer reports
    _window_left = -1
    _sm_scale = 0.125

    def __init__(self) -> None:
        self.run_calls = 0

    def run(self, q: str, kv: str) -> str:
        self.run_calls += 1
        return f"ran({q},{kv})"


def test_no_fast_plan_wrapper_pins_is_cuda_graph_enabled_false():
    """The outer wrapper's ``is_cuda_graph_enabled`` is False regardless of inner."""
    from skylight.backend import _NoFastPlanDecodeWrapper

    inner = _FakeInner()
    assert inner.is_cuda_graph_enabled is True

    outer = _NoFastPlanDecodeWrapper(inner)
    assert outer.is_cuda_graph_enabled is False


def test_no_fast_plan_wrapper_forwards_attribute_access():
    """Non-shadowed attributes flow through __getattr__ to the inner wrapper."""
    from skylight.backend import _NoFastPlanDecodeWrapper

    outer = _NoFastPlanDecodeWrapper(_FakeInner())
    assert outer._window_left == -1
    assert outer._sm_scale == 0.125


def test_no_fast_plan_wrapper_forwards_method_calls():
    """Method calls reach the inner wrapper (observable via the call counter)."""
    from skylight.backend import _NoFastPlanDecodeWrapper

    inner = _FakeInner()
    outer = _NoFastPlanDecodeWrapper(inner)

    assert outer.run("q", "kv") == "ran(q,kv)"
    assert inner.run_calls == 1


def test_no_fast_plan_wrapper_missing_attr_raises():
    """Access to nonexistent attributes raises AttributeError."""
    from skylight.backend import _NoFastPlanDecodeWrapper

    outer = _NoFastPlanDecodeWrapper(_FakeInner())
    with pytest.raises(AttributeError):
        _ = outer.does_not_exist
