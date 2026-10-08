# SPDX-License-Identifier: Apache-2.0
"""Fused per-block K/V norm scorer (native CUDA, single launch).

Same contract as offload_package/offload/triton_score.py:
    k_norm, v_norm = score_blocks_fused(gpu_caches, gpu_block_ids)
each [L, B, H] fp32, per-head mean-over-token norm. The CUDA extension is
JIT-built from csrc/kv_norm_kernel.cu on first use and cached by torch's
extension cache (~/.cache/torch_extensions). preferred over the Triton path;
the manager falls back to Triton / the PyTorch loop when this is unavailable.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading

import torch

_MODULE = None
_STATE: str | None = None  # "ok" | None=unknown | error string
_LOCK = threading.Lock()


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    # venv's nvidia cu13 wheel ships nvcc
    for base in sys.path:
        nvcc = os.path.join(base, "nvidia", "cu13", "bin", "nvcc")
        if os.path.exists(nvcc):
            os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))
            return
    nvcc = shutil.which("nvcc")
    if nvcc:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))


def _build():
    global _MODULE
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "csrc", "kv_norm_kernel.cu")
    _ensure_cuda_home()
    if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    _MODULE = load(
        name="offload_fused_score_cuda",
        sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("OFFLOAD_PACKAGE_BUILD_VERBOSE", "0") == "1",
    )
    return _MODULE


def available() -> bool:
    """True when the extension built (or could build) successfully."""
    global _STATE
    if _STATE is not None:
        return _STATE == "ok"
    with _LOCK:
        if _STATE is not None:
            return _STATE == "ok"
        if not torch.cuda.is_available():
            _STATE = "no-cuda"
        else:
            try:
                _build()
                _STATE = "ok"
            except Exception as exc:  # pragma: no cover - env dependent
                _STATE = f"build-failed: {exc!r}"
        return _STATE == "ok"


def availability_error() -> str | None:
    return None if _STATE == "ok" else (_STATE or "not-probed")


def score_blocks_fused(
    gpu_caches: list[torch.Tensor],
    gpu_block_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (k_norm, v_norm) each [L, B, H] — per-head mean-over-token norms.

    Same contract/raises as triton_score.score_blocks_fused; valid only for the
    combined [P, H, N, 2C] fp16/bf16 per-layer layout with contiguous last dim.
    """
    if not available():
        raise RuntimeError(f"cuda fused scorer unavailable: {availability_error()}")
    if not gpu_caches:
        raise ValueError("no layers")
    first = gpu_caches[0]
    L = len(gpu_caches)
    H = first.shape[1]
    dev = first.device
    if not gpu_block_ids:
        empty = torch.empty((L, 0, H), dtype=torch.float32, device=dev)
        return empty, empty
    if first.dim() != 4 or first.size(3) % 2 != 0:
        raise ValueError(f"expected combined [P,H,N,2C] layout, got {tuple(first.shape)}")
    if first.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype for cuda scorer: {first.dtype}")
    if first.stride(3) != 1:
        raise ValueError("last dim must be contiguous (stride 1 in elements)")

    ids = torch.tensor(gpu_block_ids, dtype=torch.int32, device=dev)
    k_out, v_out = _MODULE.kv_norm_fused(list(gpu_caches), ids)
    return k_out, v_out
