"""End-to-end gauge test: ``skylight serve`` → ``/v1/completions`` → ``/metrics``.

Closes Track 1 across the process boundary: vllm's EngineCore runs in a
spawned subprocess and updates the Gauge in that worker's local
``prometheus_client`` REGISTRY. The aggregated value reaches ``/metrics``
via vllm's multi-process Prometheus collector. The previous offline smoke
in :mod:`test_smoke_sparse` can't see across that boundary; this test does.

Asserts:
  1. Server starts and ``/health`` returns 200.
  2. ``POST /v1/completions`` produces non-empty text (decode actually ran).
  3. ``GET /metrics`` exposes ``skylight_effective_sparsity_fraction``.
  4. The gauge value is in ``(0, 1)``:
       - ``0.0`` → wrapper.run() never updated the gauge (silent dense fallback,
         or multi-process aggregation not wired up).
       - ``1.0`` → ctx degenerate (<= sink+local), or kernel selected every
         position (genuine dense behaviour, not sparse).
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


# -------------------------------- helpers ------------------------------------


def _free_port() -> int:
    """Bind+release on port 0 to ask the OS for an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float, proc: subprocess.Popen) -> None:
    """Poll ``url`` until it returns 200 or the timeout elapses.

    Bails early if the server process has already exited (no point polling
    a dead URL for the full timeout).
    """
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            raise RuntimeError(
                f"server process exited (rc={proc.returncode}) before /health "
                f"became ready; output:\n{stdout[-4000:]}"
            )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_err = exc
        time.sleep(2)
    raise TimeoutError(
        f"{url} did not return 200 within {timeout}s; last error: {last_err!r}"
    )


def _http_post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_get_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def _parse_prometheus_metric(text: str, name: str) -> float | None:
    """Extract a single un-labeled metric from Prometheus text format.

    Prometheus text format: ``metric_name [labels] value [timestamp]``.
    Our gauge has no labels, so we match a line whose first token equals ``name``.
    """
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts and parts[0] == name:
            return float(parts[1])
    return None


def _terminate_server_group(proc: subprocess.Popen, label: str) -> str:
    """Send SIGTERM/SIGKILL to the whole process group and collect stdout."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        stdout, _ = proc.communicate()
        return stdout

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        stdout, _ = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = proc.communicate()
    return stdout


# -------------------------------- test ---------------------------------------


METRIC_NAME = "skylight_effective_sparsity_fraction"
SINK = 16
LOCAL = 32


@pytest.mark.skipif(
    os.environ.get("SKYLIGHT_SKIP_E2E") == "1",
    reason="SKYLIGHT_SKIP_E2E=1",
)
@pytest.mark.skipif(
    not os.environ.get("CUDA_HOME"),
    reason="CUDA_HOME not set (required for flashinfer JIT in the worker)",
)
def test_metrics_gauge_via_serve_endpoint():
    """End-to-end: gauge populates with fraction in (0, 1) after a real decode."""
    model = os.environ.get("SKYLIGHT_E2E_MODEL", "Qwen/Qwen3-0.6B")
    port = _free_port()
    base = f"http://localhost:{port}"

    # Prompt + max_tokens chosen so the decode-time n_keys exceeds SINK+LOCAL,
    # otherwise the formula sink+k_eff_mid+local degenerates to >= n_keys
    # and the gauge would read 1.0 (legitimately, but not what this test asserts).
    prompt = (
        "The history of artificial intelligence spans more than seventy years, "
        "beginning with early symbolic systems in the 1950s and progressing "
        "through expert systems, neural networks, and finally the transformer "
        "architecture introduced in 2017. List five milestones in order:"
    )

    server_cmd = [
        sys.executable, "-m", "skylight.cli", "serve",
        "--model", model,
        "--port", str(port),
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.30",
        "--enforce-eager",
    ]
    env = {
        **os.environ,
        "SKYLIGHT_SPARSE_TOPK": "0.5",
        "SKYLIGHT_SPARSE_SINK": str(SINK),
        "SKYLIGHT_SPARSE_LOCAL": str(LOCAL),
    }

    proc = subprocess.Popen(
        server_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # own process group so we can kill workers too
    )
    server_log = ""
    try:
        _wait_for_health(f"{base}/health", timeout=180, proc=proc)

        # Drive a decode that produces ctx > SINK+LOCAL (== 48).
        resp = _http_post_json(
            f"{base}/v1/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.0,
            },
            timeout=120,
        )
        text = resp["choices"][0]["text"]
        assert text, f"empty completion text in response: {resp!r}"

        # Now scrape /metrics. vllm exposes the prometheus_client default
        # registry — including our Gauge — through this endpoint with
        # multi-process aggregation (PROMETHEUS_MULTIPROC_DIR is set up
        # internally by vllm when the server starts).
        metrics_text = _http_get_text(f"{base}/metrics", timeout=10)
        fraction = _parse_prometheus_metric(metrics_text, METRIC_NAME)

        assert fraction is not None, (
            f"{METRIC_NAME} not exposed in /metrics. First 60 lines:\n"
            + "\n".join(metrics_text.splitlines()[:60])
        )
        assert 0.0 < fraction < 1.0, (
            f"expected {METRIC_NAME} in (0, 1) after a real sparse decode, "
            f"got {fraction}. "
            f"0.0 → wrapper.run() never updated the gauge (silent fallback, "
            f"or multi-process aggregation not wired up). "
            f"1.0 → ctx <= SINK+LOCAL={SINK + LOCAL} (degenerate), "
            f"or kernel selected every position (dense behaviour)."
        )
    finally:
        server_log = _terminate_server_group(proc, "skylight serve")
        if os.environ.get("SKYLIGHT_E2E_VERBOSE") == "1":
            print("\n--- server stdout (truncated) ---\n" + server_log[-4000:])
