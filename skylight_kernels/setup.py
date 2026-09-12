"""Ahead-of-time CUDA build for skylight_kernels' four optimized extensions.

Replaces the previous torch.utils.cpp_extension.load() JIT path. After
``pip install`` / ``uv sync`` of this package, each kernel's compiled
``.so`` is importable as ``<pkg>._cuda_ops`` and no compilation work
runs at runtime — so a fresh skylight server / smoke test starts in
~30 s instead of waiting on a 5-10 min cold build.

Progress during the AOT build:
  - ninja is the default builder via BuildExtension. We set
    NINJA_STATUS=[%f/%t %e elapsed] %f so each compile step prints
    "[X/Y NNs elapsed] nvcc ..."
  - BuildExtension prints every nvcc/cxx invocation when invoked with
    ``-v``. To see the full stream during install, run
    ``uv sync --verbose`` (or ``pip install -v``).

To target a specific GPU arch, export ``TORCH_CUDA_ARCH_LIST`` before
the build. The default auto-detects from the host GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

# Make ninja's per-file progress visible whenever stdout is being shown.
# Format: "[3/8 12.5s elapsed] /usr/local/cuda/bin/nvcc -O3 ... foo.cu"
os.environ.setdefault("NINJA_STATUS", "[%f/%t %e elapsed] ")

# Shared compile flags. Mirrors the original cuda_ops.py JIT settings exactly
# so the AOT build produces byte-equivalent kernels.
NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]
CXX_FLAGS = ["-O3", "-std=c++17"]

# All four kernels include FlashInfer headers extracted under
# original_optimized/csrc/ (for vec_t, math helpers, cp.async, ...).
# include_dirs is a compile flag (not a packaged source), so an absolute
# path is fine here. sources MUST stay relative — setuptools' egg_info /
# manifest step rejects absolute paths for packaged files.
FI_INC = str(ROOT / "original_optimized" / "csrc")


def cuda_ext(pkg: str, source_rel: str) -> CUDAExtension:
    """Build ``<pkg>._cuda_ops`` from ``<pkg>/csrc/<source_rel>``."""
    return CUDAExtension(
        name=f"{pkg}._cuda_ops",
        sources=[f"{pkg}/csrc/{source_rel}"],
        include_dirs=[FI_INC],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )


setup(
    ext_modules=[
        # FlashInfer-extracted paged decode (the dense baseline).
        cuda_ext("original_optimized", "paged_decode_fi.cu"),
        # Sparse paged decode — caller supplies sparse_idx/len/weights.
        cuda_ext("sparse_optimized", "sparse_decode_kernel.cu"),
        # Oracle top-K only.
        cuda_ext("sparse_oracle_topk_optimized", "oracle_topk_score_kernel.cu"),
        # Oracle top-K + sink + local (compact, optimized).
        cuda_ext(
            "sparse_oracle_topk_sink_local_optimized",
            "oracle_topk_score_kernel_compact.cu",
        ),
    ],
    # use_ninja=True is the default; we set it explicitly for clarity.
    # Passing verbose at the CLI (``-v``) surfaces every nvcc invocation.
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
