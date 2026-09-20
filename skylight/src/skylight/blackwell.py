"""B200 / sm_100 base prerequisites for vllm.

Applied at plugin load time. Dense and sparse runs alike benefit. Sparse
mode additionally relies on FLASHINFER_TOPK_ALGO=radix because flashinfer
0.6.x clusters_topk segfaults at L>=64K on sm_100.

Every adjustment is recorded in :data:`state` so operators can introspect
what the module did, via :func:`print_banner` or programmatic access.
User-set values are preserved; we only fill in defaults.
"""
from __future__ import annotations

import os
import sys
from typing import List, Tuple

# (name, value, source, reason). source ∈ {"default", "user-set", "applied", "FAILED"}.
state: List[Tuple[str, str, str, str]] = []

_APPLIED = False


def _set_env_default(name: str, value: str, reason: str) -> None:
    """Set ``os.environ[name] = value`` only if unset; record outcome in :data:`state`."""
    if name in os.environ:
        state.append((name, os.environ[name], "user-set", reason))
    else:
        os.environ[name] = value
        state.append((name, value, "default", reason))


def _ensure_in_path(*dirs: str) -> None:
    """Append each existing dir to PATH if not already present. Idempotent."""
    parts = [p for p in os.environ.get("PATH", "").split(":") if p]
    added = [d for d in dirs if d and d not in parts and os.path.isdir(d)]
    if not added:
        return
    os.environ["PATH"] = ":".join([*parts, *added])
    state.append((
        "PATH", f"+{len(added)}", "applied",
        f"appended for subprocess/JIT discovery: {added}",
    ))


def apply() -> None:
    """Apply Blackwell base prereqs. Idempotent — second call is a no-op."""
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _set_env_default(
        "FLASHINFER_TOPK_ALGO", "radix",
        "flashinfer 0.6.x clusters_topk segfaults on sm_100 at L>=64K",
    )
    _set_env_default(
        "VLLM_USE_DEEP_GEMM", "0",
        "BF16 warmup probes FP8 DeepGEMM and crashes when the pkg is missing",
    )

    # Spawn-launched vllm EngineCore workers can inherit a minimal PATH that
    # lacks /usr/bin (so even `which` doesn't resolve). flashinfer's JIT
    # compile path (sampling kernels) shells out to `which nvcc` and to
    # `ninja` directly, and fails hard without these. Defensive — caller-set
    # PATH wins via the "skip if already present" check.
    _ensure_in_path("/usr/bin", "/bin", "/usr/local/bin")
    # The venv's bin dir holds ninja, uv, and any console-script entry points
    # (including `skylight` itself). sys.executable is always .venv/bin/python
    # for an active venv, so its dirname is the right place to look.
    _ensure_in_path(os.path.dirname(sys.executable))
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        _ensure_in_path(os.path.join(cuda_home, "bin"))

    try:
        import torch
        torch.backends.cuda.enable_cudnn_sdp(False)
        state.append((
            "torch.backends.cuda.enable_cudnn_sdp", "False", "applied",
            "CUDA 13 / B200 cuDNN SDPA mismatch with torch+cu130",
        ))
    except Exception as exc:  # pragma: no cover — torch failure would block everything
        state.append((
            "torch.backends.cuda.enable_cudnn_sdp", "FAILED", "FAILED",
            f"{type(exc).__name__}: {exc}",
        ))


def print_banner() -> None:
    """Pretty-print the recorded state to stdout."""
    print("[skylight.blackwell] B200 base prereqs:")
    if not state:
        print("  (no actions recorded)")
        return
    width = max(len(name) for name, *_ in state)
    for name, value, source, reason in state:
        print(f"  {name:<{width}} = {value:<11} ({source} — {reason})")
