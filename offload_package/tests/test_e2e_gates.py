"""End-to-end correctness gates for the pq + offload pipeline.

Opt-in: set ``RUN_E2E_GATES=1``. Each gate shells out to the bench scripts
in a subprocess so engine env vars, GPU memory, and module state are fully
isolated per gate (a fresh process per gate is also how the results these
gates lock in were produced).

Known limitation documented by these gates (measured 2026-10-08):
    token-for-token equality between two LLM instances in one process is
    only valid within a noise horizon of ~100-128 decode tokens. The engine
    carries ~1e-5 per-instance nondeterminism (cuBLASLt algo selection is
    pointer-alignment dependent) that eventually flips near-tied greedy
    argmaxes: a pq-vs-pq control (identical configs, no offload) diverges at
    token 574 on a 4096-token decode. Long-decode correctness is therefore
    gated per-step by the fp64 probe (gate 4), not by token equality.

Gates:
    1. short-gate token match: pq == pq+off99 (262-token prompt, 64 tokens)
       -- locks in the partial-tail completeness fix and the F2/F6 fetch
       machinery.
    2. composition-change token match: two requests, one finishes mid-decode
       -- locks in current-step (runner-derived) request ids (F1-A).
    3. fault injection: SKYLIGHT_OFFLOAD_DISABLE_RUNNER_IDS=1 must fail
       LOUDLY (RuntimeError) instead of silently reading freed pages (F1-C).
    4. probe exactness: every decode step shows byte-exact staged reads and
       fp64-reference attention (offload is per-step exact).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("RUN_E2E_GATES"),
        reason="opt-in: set RUN_E2E_GATES=1 (boots LLM engines, ~2-4 min/gate)",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required",
    ),
]

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
GATE_TIMEOUT_S = int(os.environ.get("E2E_GATE_TIMEOUT", "2400"))


def _run(cmd: list[str], env_extra: dict[str, str] | None = None):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("E2E_GPU", "0")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=GATE_TIMEOUT_S,
    )


def test_short_gate_token_match():
    """pq vs pq+offload@0.99, 262-token prompt + 64 greedy tokens."""
    r = _run(
        [PY, "bench_1req_off99.py", "pq,pq+off99"],
        env_extra={"BENCH_NEW_TOKENS": "64", "BENCH_PROMPT_REPS": "10"},
    )
    assert r.returncode == 0, f"bench failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    assert "pq+off99: PASS" in r.stdout, (
        "short-gate token mismatch:\n"
        + "\n".join(r.stdout.splitlines()[-8:])
    )


def test_composition_gate_token_match():
    """Two requests, staggered decode budgets (96/40): per-request equality
    across the mid-decode finish (batch-composition change)."""
    r = _run([PY, "bench_2req_composition.py", "pq,pq+offload"])
    assert r.returncode == 0, f"bench failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    assert "overall: PASS" in r.stdout, (
        "composition-gate token mismatch:\n"
        + "\n".join(r.stdout.splitlines()[-8:])
    )


def test_runner_ids_fault_injection_raises():
    """Disabling the runner-derived id stash must make the build-side guard
    RAISE (refusing to read freed pages), not silently corrupt."""
    r = _run(
        [PY, "bench_2req_composition.py", "pq+offload"],
        env_extra={"SKYLIGHT_OFFLOAD_DISABLE_RUNNER_IDS": "1"},
    )
    out = r.stdout + r.stderr
    assert r.returncode != 0, "fault injection should have failed the run"
    assert "refusing to let the kernel read freed pages" in out, (
        f"expected the fail-loud guard, got:\n{out[-2000:]}"
    )


def test_probe_step_exactness():
    """Every decode step: staged reads byte-identical to source-of-truth and
    kernel attention equal to the fp64 reference over the selected set."""
    r = _run(
        [PY, "bench_1req_off99.py", "pq+off99"],
        env_extra={
            "BENCH_NEW_TOKENS": "32",
            "BENCH_PROMPT_REPS": "10",
            "SKYLIGHT_PQ_VERIFY": "1",
            "SKYLIGHT_PQ_VERIFY_STEPS": "40",
        },
    )
    assert r.returncode == 0, f"bench failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    steps = 0
    for line in r.stdout.splitlines():
        m = re.search(
            r"\[PQ VERIFY\] step=(\d+) layer=0:.*?"
            r"k_bytes_mismatch=(\d+) v_bytes_mismatch=(\d+) "
            r"page_wrong=(\d+) null_page_reads=(\d+).*?"
            r"ref_cos=([0-9.]+)", line)
        if not m:
            continue
        steps += 1
        kmis, vmis, pwrong, nulls = (int(m.group(i)) for i in range(2, 6))
        cos = float(m.group(6))
        assert kmis == 0 and vmis == 0, f"byte mismatch at step {m.group(1)}: {line}"
        assert pwrong == 0 and nulls == 0, f"bad page read at step {m.group(1)}: {line}"
        assert cos >= 0.9999, f"fp64 reference mismatch at step {m.group(1)}: {cos}"
    assert steps >= 8, f"expected >=8 probe steps, saw {steps}"
