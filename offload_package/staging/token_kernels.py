# SPDX-License-Identifier: Apache-2.0
"""JIT loader for the token-staging CUDA kernels (csrc/token_store_kernels.cu).

``get()`` returns the compiled module, or None (with one loud warning) when it
cannot be built or ``OFFLOAD_PACKAGE_TOKEN_KERNELS=0``; callers then use the
torch path in token_store.py.
"""
from __future__ import annotations

import os
import sys

import torch

_MODULE = None
_ERROR: str | None = None
_WARNED = False


def _build():
    import torch.utils.cpp_extension as cpp_ext
    from torch.utils.cpp_extension import load

    from offload_package.scoring.cuda_norms import _ensure_cuda_home

    _ensure_cuda_home()
    if os.environ.get("CUDA_HOME"):
        cpp_ext.CUDA_HOME = os.environ["CUDA_HOME"]
        binp = os.path.join(os.environ["CUDA_HOME"], "bin")
        if binp not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = binp + os.pathsep + os.environ.get("PATH", "")
    bindir = os.path.dirname(sys.executable)          # ninja lives next to the venv python
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc",
                       "token_store_kernels.cu")
    return load(
        name="offload_token_store_kernels",
        sources=[src],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("OFFLOAD_PACKAGE_BUILD_VERBOSE", "0") == "1",
    )


def get():
    """The kernel module, or None (torch fallback)."""
    global _MODULE, _ERROR, _WARNED
    if os.environ.get("OFFLOAD_PACKAGE_TOKEN_KERNELS", "1") == "0":
        return None
    if _MODULE is None and _ERROR is None:
        if not torch.cuda.is_available():
            _ERROR = "no CUDA device"
        else:
            try:
                _MODULE = _build()
            except Exception as exc:                 # noqa: BLE001
                _ERROR = f"{type(exc).__name__}: {exc}"
    if _MODULE is None and not _WARNED:
        _WARNED = True
        print(f"[TOKEN KERNELS] FALLBACK to torch token staging: {_ERROR}", flush=True)
    return _MODULE


def why_unavailable() -> str | None:
    get()
    return _ERROR
