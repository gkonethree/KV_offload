"""End-to-end smoke: load a small model + generate with sparse backend.

Spawns a subprocess (`_smoke_helper.py`) so VLLM_ATTENTION_BACKEND +
SKYLIGHT_SPARSE_* env vars take effect at vllm import time. Validates the
full chain: plugin entry-point → register_backend → backend selected →
metadata builder constructs sparse wrapper → kernel runs → vllm produces
tokens.

Skip path:
  SKYLIGHT_SKIP_SMOKE=1 → skip (useful for fast iteration on unrelated changes)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
HELPER = HERE / "_smoke_helper.py"


@pytest.mark.skipif(
    os.environ.get("SKYLIGHT_SKIP_SMOKE") == "1",
    reason="SKYLIGHT_SKIP_SMOKE=1",
)
@pytest.mark.skipif(
    not os.environ.get("CUDA_HOME"),
    reason="CUDA_HOME not set (required for flashinfer JIT in the subprocess)",
)
def test_smoke_sparse_backend_end_to_end():
    """Subprocess: import vllm + generate via sparse backend; assert SMOKE_OK.

    The helper passes ``attention_backend="CUSTOM"`` to ``LLM(...)`` directly
    (vllm 0.21+ no longer reads ``VLLM_ATTENTION_BACKEND``). The kwarg
    triggers our registered backend via ``AttentionBackendEnum.CUSTOM``.
    """
    env = {
        **os.environ,
        # Lenient sparsity — just smoke-test the routing path. Real correctness
        # vs dense is covered by skylight_kernels' own pytest suite.
        "SKYLIGHT_SPARSE_TOPK": "0.5",
        "SKYLIGHT_SPARSE_SINK": "16",
        "SKYLIGHT_SPARSE_LOCAL": "32",
    }
    result = subprocess.run(
        [sys.executable, str(HELPER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min cold model load
    )
    assert result.returncode == 0, (
        f"smoke subprocess failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )
    assert "SMOKE_OK:" in result.stdout, (
        f"missing SMOKE_OK marker in stdout:\n{result.stdout!r}\n"
        f"stderr:\n{result.stderr!r}\n"
    )
