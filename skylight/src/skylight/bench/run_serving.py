"""A/B-compare sparse vs dense backends via vllm's benchmark_serving.

We don't reinvent load generation, percentile collection, or dataset
loading — :mod:`vllm.benchmarks.benchmark_serving` already does all of
that. This module just:

  1. Spawns the right server (``skylight serve`` for sparse, vllm
     api_server with ``--attention-backend FLASHINFER`` for dense).
  2. Waits for ``/health`` to return 200.
  3. Invokes vLLM's public ``vllm bench serve`` client against the server.
  4. Tears the server down via its process group (SIGTERM, SIGKILL fallback).

Output is whatever ``vllm bench serve`` writes via ``--save-result`` —
typically a JSON with throughput, TTFT, TPOT, and latency percentiles.

Usage::

    # Sparse — uses skylight.cli (auto --attention-backend CUSTOM)
    python -m skylight.bench.run_serving --backend sparse \\
        --model Qwen/Qwen3-0.6B --num-prompts 100 \\
        --topk 0.10 --sink 64 --local 64

    # Dense baseline — uses vllm api_server with FLASHINFER
    python -m skylight.bench.run_serving --backend dense \\
        --model Qwen/Qwen3-0.6B --num-prompts 100

    # Compare
    diff <(jq . bench-results/dense-Qwen-Qwen3-0.6B.json) \\
         <(jq . bench-results/sparse-Qwen-Qwen3-0.6B.json)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from skylight.profiles import BMMProfile
from skylight.runtime import Platform, configure_runtime


# -------------------------------- helpers ------------------------------------


@dataclass(frozen=True)
class ArtifactPaths:
    """Stable per-condition artifact layout."""

    root: Path
    result: Path
    server_log: Path
    client_log: Path
    resolved: Path
    versions: Path
    telemetry: Path

    @classmethod
    def create(cls, root: Path) -> "ArtifactPaths":
        root = Path(root)
        telemetry = root / "telemetry"
        telemetry.mkdir(parents=True, exist_ok=True)
        server_log = root / "server.log"
        client_log = root / "client.log"
        server_log.touch()
        client_log.touch()
        return cls(
            root=root,
            result=root / "result.json",
            server_log=server_log,
            client_log=client_log,
            resolved=root / "resolved.json",
            versions=root / "versions.json",
            telemetry=telemetry,
        )


_RECORDED_ENV_PREFIXES = (
    "SKYLIGHT_",
    "TORCH_",
    "VLLM_",
    "FLASHINFER_",
    "CUDA_",
    "PYTORCH_",
    "NVIDIA_",
    "CUBLAS_",
    "NCCL_",
)
_RECORDED_ENV_NAMES = {
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "PYTHONPYCACHEPREFIX",
}
_SECRET_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "AUTH", "KEY")


def safe_environment_snapshot(env: dict[str, str]) -> dict[str, str]:
    """Return only reproducibility settings, never credentials or arbitrary env."""
    result = {}
    for name, value in env.items():
        if any(fragment in name.upper() for fragment in _SECRET_FRAGMENTS):
            continue
        if name in _RECORDED_ENV_NAMES or name.startswith(_RECORDED_ENV_PREFIXES):
            result[name] = value
    return dict(sorted(result.items()))


def _command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return f"unavailable (rc={result.returncode}): {result.stderr.strip()}"
    return result.stdout.strip()


def collect_gpu_identity(device: int) -> dict[str, object]:
    """Return fail-closed physical identity for one selected NVIDIA GPU."""
    output = _command_output(
        [
            "nvidia-smi",
            f"--id={device}",
            "--query-gpu=index,name,uuid,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    fields = [field.strip() for field in output.split(",")]
    try:
        index = int(fields[0])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"cannot establish selected GPU identity for device {device}: {output}"
        ) from exc
    valid = (
        len(fields) == 5
        and index == device
        and bool(fields[1])
        and fields[2].startswith("GPU-")
        and bool(fields[3])
        and re.fullmatch(r"\d+(?:\.\d+)+", fields[4]) is not None
    )
    if not valid:
        raise RuntimeError(
            f"cannot establish selected GPU identity for device {device}: {output}"
        )
    return {
        "index": index,
        "name": fields[1],
        "uuid": fields[2],
        "driver_version": fields[3],
        "compute_capability": fields[4],
    }


def _git_snapshot(path: Path) -> dict[str, object]:
    if not (path / ".git").exists():
        return {"revision": "unknown", "dirty": None}
    revision = _command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
    status = _command_output(["git", "-C", str(path), "status", "--porcelain"])
    return {"revision": revision, "dirty": bool(status)}


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _installed_package_versions() -> dict[str, list[str]]:
    """Return the complete installed Python distribution identity."""
    versions: dict[str, set[str]] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        versions.setdefault(canonical, set()).add(distribution.version)
    return {
        name: sorted(package_versions)
        for name, package_versions in sorted(versions.items())
    }


def collect_versions(platform: Platform) -> dict[str, object]:
    """Collect source, package, GPU, and driver identity for one condition."""
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    except ImportError:
        torch_version = "not-installed"
        torch_cuda = None

    repo_root = Path(__file__).resolve().parents[3]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": asdict(platform),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "python": sys.version,
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "vllm": _package_version("vllm"),
        "flashinfer_python": _package_version("flashinfer-python"),
        "python_packages": _installed_package_versions(),
        "skylight": _git_snapshot(repo_root),
        "skylight_kernels": _git_snapshot(repo_root / "skylight_kernels"),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def free_port() -> int:
    """Bind+release on port 0 to ask the OS for an unused TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def build_serve_cmd(
    backend: str,
    model: str,
    port: int,
    max_model_len: int,
    enforce_eager: bool,
    gpu_memory_utilization: float = 0.5,
    tool_call_parser: Optional[str] = None,
    enable_prefix_caching: bool = False,
    max_num_seqs: Optional[int] = None,
    gdn_prefill_backend: str = "triton",
    dtype: str = "bfloat16",
    block_size: int = 16,
    model_revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
) -> list[str]:
    """Build the command line for launching the model server.

    Sparse goes through ``python -m skylight.cli serve`` which
    auto-injects ``--attention-backend CUSTOM`` (the placeholder enum
    member our plugin registers SkylightSparseBackend under).

    Dense uses vllm's api_server directly with ``--attention-backend
    FLASHINFER`` — the production dense path on B200.

    ``gpu_memory_utilization`` defaults to 0.5 to match the handoff's
    tested operational setting and to coexist with other tenants on a
    shared B200 (vLLM's own default is 0.92, which crashes on a
    contended GPU).

    ``tool_call_parser`` is opt-in. When set (e.g. ``"qwen3_coder"``),
    both ``--enable-auto-tool-choice`` and ``--tool-call-parser
    <name>`` are appended — required for the agentic bench, which
    issues OpenAI-shaped ``tools=[...]`` requests that vLLM otherwise
    rejects with ``BadRequestError`` ("auto tool choice requires
    --enable-auto-tool-choice and --tool-call-parser to be set"). Left
    ``None`` for the perf bench, which sends raw prompts.
    """
    if backend == "sparse":
        cmd = [
            sys.executable, "-m", "skylight.cli", "serve",
            "--model", model,
            "--port", str(port),
            "--max-model-len", str(max_model_len),
        ]
    elif backend == "dense":
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--max-model-len", str(max_model_len),
            "--attention-backend", "FLASHINFER",
        ]
    else:
        raise ValueError(f"unknown --backend {backend!r}; expected 'sparse' or 'dense'")

    cmd.extend([
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--gdn-prefill-backend", gdn_prefill_backend,
        "--dtype", dtype,
        "--block-size", str(block_size),
    ])
    if model_revision is not None:
        cmd.extend(["--revision", model_revision])
    if tokenizer_revision is not None:
        cmd.extend(["--tokenizer-revision", tokenizer_revision])
    if os.environ.get("SKYLIGHT_HF_OVERRIDES"):
        cmd.extend(["--hf-overrides", os.environ["SKYLIGHT_HF_OVERRIDES"]])
    if tool_call_parser:
        cmd.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", tool_call_parser,
        ])
    if enforce_eager:
        cmd.append("--enforce-eager")
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    if os.environ.get("SKYLIGHT_BENCH_PROFILE") == "1":
        _pd = os.environ.get("VLLM_TORCH_PROFILER_DIR", "/tmp/torchprof")
        cmd.extend(["--profiler-config.profiler=torch", "--profiler-config.torch_profiler_dir=" + _pd])
    _tp = os.environ.get("SKYLIGHT_TP")
    if _tp:
        cmd.extend(["--tensor-parallel-size", _tp])
    _kd = os.environ.get("SKYLIGHT_KV_DTYPE")
    if _kd:
        cmd.extend(["--kv-cache-dtype", _kd])
    return cmd


def build_env(
    backend: str,
    topk: Optional[float],
    sink: Optional[int],
    local: Optional[int],
    channel_num: Optional[int] = None,
    *,
    method: str = "block_minmax",
    block_size: int = 16,
    metrics_dir: Optional[Path] = None,
) -> dict:
    """Construct the env passed to the server subprocess.

    Both paths first remove inherited sparse/BMM settings so a shell used for
    an earlier experiment cannot contaminate the next control. Sparse BMM
    runs then receive the complete qualified incremental profile; oracle runs
    receive only their explicit policy knobs.

    ``channel_num`` is the doubly-sparse axis: only the first N head
    channels participate in the score-kernel ``Q @ K^T`` (the full
    ``head_dim`` is still used for the actual attention compute). Set to
    ``-1`` to use the full ``head_dim`` (no channel sparsity); typical
    "double sparse" baseline is ``8`` or ``16``.
    """
    env = os.environ.copy()
    for name in tuple(env):
        if (
            name.startswith("SKYLIGHT_SPARSE_")
            or name.startswith("SKYLIGHT_INCR_")
            or name
            in {
                "SKYLIGHT_BLOCK_SIZE",
                "SKYLIGHT_FI_BSR",
                "SKYLIGHT_METRICS_LOG_DIR",
                "SKYLIGHT_METRICS_SAMPLING",
            }
        ):
            env.pop(name, None)

    if backend == "dense":
        return env
    if backend != "sparse":
        raise ValueError(f"unknown --backend {backend!r}; expected 'sparse' or 'dense'")

    if method == "block_minmax":
        profile = BMMProfile(
            target_density=topk if topk is not None else 0.10,
            sink=sink if sink is not None else 64,
            local=local if local is not None else 64,
            channel_num=channel_num if channel_num is not None else -1,
            block_size=block_size,
        )
        env.update(profile.to_env())
    elif method == "oracle":
        env["SKYLIGHT_SPARSE_METHOD"] = "oracle"
        if topk is not None:
            env["SKYLIGHT_SPARSE_TOPK"] = str(topk)
        if sink is not None:
            env["SKYLIGHT_SPARSE_SINK"] = str(sink)
        if local is not None:
            env["SKYLIGHT_SPARSE_LOCAL"] = str(local)
        if channel_num is not None:
            env["SKYLIGHT_SPARSE_CHANNEL_NUM"] = str(channel_num)
    else:
        raise ValueError(
            f"unknown sparse method {method!r}; expected 'block_minmax' or 'oracle'"
        )

    if metrics_dir is not None:
        env["SKYLIGHT_METRICS_LOG_DIR"] = str(metrics_dir)
        env["SKYLIGHT_METRICS_SAMPLING"] = "1.0"
    return env


def build_benchmark_cmd(
    model: str,
    port: int,
    num_prompts: int,
    dataset_name: str,
    request_rate: float,
    output_file: str,
    random_input_len: Optional[int] = None,
    random_output_len: Optional[int] = None,
    num_warmups: int = 2,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    tokenizer: Optional[str] = None,
) -> list[str]:
    """Build the command line for vllm's ``bench serve`` CLI subcommand.

    We use the ``vllm bench serve`` entry rather than ``python -m
    vllm.benchmarks.benchmark_serving`` because the CLI subcommand is the
    public, version-stable surface (the internal module path varied
    across vllm releases). The ``vllm`` console script is installed in the
    venv by ``uv sync``, so it resolves on PATH for any caller running
    inside an active venv.

    ``random_input_len`` / ``random_output_len`` are the dominant sweep
    knobs for sparse vs dense crossover analysis: sparse pays off as
    input length grows. Pass these for ``--dataset-name random``; vllm
    picks sensible defaults otherwise.
    """
    cmd = [
        str(Path(sys.executable).with_name("vllm")), "bench", "serve",
        "--backend", "openai",
        "--endpoint", "/v1/completions",
        "--host", "localhost",
        "--port", str(port),
        "--model", model,
        "--dataset-name", dataset_name,
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),
        "--num-warmups", str(num_warmups),
        "--temperature", str(temperature),
        "--save-result",
        "--result-filename", output_file,
    ]
    if random_input_len is not None:
        cmd += ["--random-input-len", str(random_input_len)]
    if random_output_len is not None:
        cmd += ["--random-output-len", str(random_output_len)]
    if tokenizer is not None:
        cmd += ["--tokenizer", tokenizer]
    if ignore_eos:
        cmd += ["--ignore-eos"]
    if os.environ.get("SKYLIGHT_BENCH_PROFILE") == "1":
        cmd += ["--profile"]
    if os.environ.get("SKYLIGHT_BENCH_DETAILED") == "1":
        cmd += ["--save-detailed"]
    return cmd


def wait_for_health(port: int, timeout: float, proc: subprocess.Popen) -> None:
    """Poll /health until it returns 200; bail if the server process exits."""
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited (rc={proc.returncode}) before /health became ready"
            )
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=2
            ) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_err = exc
        time.sleep(2)
    raise TimeoutError(
        f"/health did not respond 200 within {timeout}s; last error: {last_err!r}"
    )


def terminate_group(proc: subprocess.Popen) -> None:
    """SIGTERM the server's process group, escalating to SIGKILL after 15s."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


class ServingSession:
    """Own one reusable vLLM server process and its benchmark clients."""

    def __init__(
        self,
        serve_cmd: list[str],
        env: dict[str, str],
        port: int,
        timeout: float,
        server_log: Optional[Path] = None,
    ) -> None:
        self.serve_cmd = list(serve_cmd)
        self.env = env
        self.port = int(port)
        self.timeout = float(timeout)
        self.server_log = Path(server_log) if server_log is not None else None
        self._proc: Optional[subprocess.Popen] = None
        self._server_stream = None

    def __enter__(self) -> "ServingSession":
        if self._proc is not None:
            raise RuntimeError("serving session is already started")

        popen_kwargs = {"env": self.env, "start_new_session": True}
        if self.server_log is not None:
            self.server_log.parent.mkdir(parents=True, exist_ok=True)
            self._server_stream = self.server_log.open("a", encoding="utf-8")
            popen_kwargs.update(
                {
                    "stdout": self._server_stream,
                    "stderr": subprocess.STDOUT,
                }
            )
        try:
            self._proc = subprocess.Popen(self.serve_cmd, **popen_kwargs)
            wait_for_health(self.port, self.timeout, self._proc)
        except BaseException:
            self.close()
            raise
        return self

    def _require_running(self) -> subprocess.Popen:
        if self._proc is None:
            raise RuntimeError("serving session is not started")
        returncode = self._proc.poll()
        if returncode is not None:
            raise RuntimeError(f"server exited (rc={returncode})")
        return self._proc

    def run_benchmark(
        self,
        command: list[str],
        client_log: Optional[Path] = None,
        timeout: Optional[float] = None,
    ) -> subprocess.CompletedProcess:
        """Run one benchmark client against the live server."""
        self._require_running()
        run_kwargs = {}
        if timeout is not None:
            run_kwargs["timeout"] = timeout
        if client_log is None:
            return subprocess.run(command, **run_kwargs)

        client_log = Path(client_log)
        client_log.parent.mkdir(parents=True, exist_ok=True)
        with client_log.open("w", encoding="utf-8") as client_stream:
            return subprocess.run(
                command,
                stdout=client_stream,
                stderr=subprocess.STDOUT,
                **run_kwargs,
            )

    def is_healthy(self) -> bool:
        """Return whether the process is alive and its health endpoint responds."""
        try:
            self._require_running()
            with urllib.request.urlopen(
                f"http://localhost:{self.port}/health",
                timeout=2,
            ) as response:
                return response.status == 200
        except (
            RuntimeError,
            urllib.error.URLError,
            ConnectionError,
            TimeoutError,
        ):
            return False

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        try:
            if proc is not None:
                terminate_group(proc)
        finally:
            if self._server_stream is not None:
                self._server_stream.close()
                self._server_stream = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


# -------------------------------- entry point --------------------------------


def _default_output_path(backend: str, model: str) -> str:
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"bench-results/{backend}-{safe_model}.json"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend", choices=["sparse", "dense"], required=True)
    parser.add_argument("--model", required=True,
                        help="Model name as passed to vllm (e.g. Qwen/Qwen3-0.6B)")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--dataset-name", default="random",
                        help="vllm benchmark_serving --dataset-name (default: random)")
    parser.add_argument("--random-input-len", type=int, default=None,
                        help="(--dataset-name random) tokens per prompt input. "
                             "Sweep this to find sparse-vs-dense crossover.")
    parser.add_argument("--random-output-len", type=int, default=None,
                        help="(--dataset-name random) tokens to generate per prompt")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Requests/s; default inf = send all at once")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--gdn-prefill-backend", default="triton")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-warmups", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Disable CUDA graph capture (default ON; faster startup)")
    parser.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager")
    parser.add_argument(
        "--method",
        choices=["block_minmax", "oracle"],
        default="block_minmax",
        help="(sparse only) selector implementation",
    )
    parser.add_argument("--topk", type=float, default=0.10,
                        help="(sparse only) SKYLIGHT_SPARSE_TOPK fraction")
    parser.add_argument("--sink", type=int, default=64,
                        help="(sparse only) SKYLIGHT_SPARSE_SINK tokens")
    parser.add_argument("--local", type=int, default=64,
                        help="(sparse only) SKYLIGHT_SPARSE_LOCAL tokens")
    parser.add_argument("--channel-num", type=int, default=-1,
                        help="(sparse only) SKYLIGHT_SPARSE_CHANNEL_NUM: head channels "
                             "used for the score dot-product. -1 = full head_dim "
                             "(token-sparse only). Set to 8/16 for the doubly-sparse "
                             "baseline (channel-sparse + token-sparse).")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        default=None,
        help="Result JSON path; default bench-results/<backend>-<model>.json",
    )
    output_group.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Release artifact directory containing result, logs, settings, and versions",
    )
    parser.add_argument("--server-timeout", type=float, default=180.0,
                        help="/health wait timeout, seconds")

    args = parser.parse_args(argv)

    platform = configure_runtime()
    artifacts = (
        ArtifactPaths.create(args.artifacts_dir)
        if args.artifacts_dir is not None
        else None
    )
    port = free_port()
    output = str(
        artifacts.result
        if artifacts is not None
        else args.output or _default_output_path(args.backend, args.model)
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    serve_cmd = build_serve_cmd(
        args.backend, args.model, port, args.max_model_len, args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        gdn_prefill_backend=args.gdn_prefill_backend,
        dtype=args.dtype,
        block_size=args.block_size,
    )
    env = build_env(
        args.backend,
        args.topk,
        args.sink,
        args.local,
        args.channel_num,
        method=args.method,
        block_size=args.block_size,
        metrics_dir=artifacts.telemetry if artifacts and args.backend == "sparse" else None,
    )
    bench_cmd = build_benchmark_cmd(
        args.model,
        port,
        args.num_prompts,
        args.dataset_name,
        args.request_rate,
        output,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        num_warmups=args.num_warmups,
        temperature=args.temperature,
        ignore_eos=args.ignore_eos,
    )

    if artifacts is not None:
        bmm = None
        if args.backend == "sparse" and args.method == "block_minmax":
            bmm = asdict(
                BMMProfile(
                    target_density=args.topk,
                    sink=args.sink,
                    local=args.local,
                    channel_num=args.channel_num,
                    block_size=args.block_size,
                )
            )
        resolved = {
            "model": args.model,
            "backend": args.backend,
            "method": args.method if args.backend == "sparse" else None,
            "mode": "eager" if args.enforce_eager else "cudagraph",
            "workload": {
                "num_prompts": args.num_prompts,
                "random_input_len": args.random_input_len,
                "random_output_len": args.random_output_len,
                "request_rate": args.request_rate,
            },
            "server": {
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enable_prefix_caching": False,
                "gdn_prefill_backend": args.gdn_prefill_backend,
                "dtype": args.dtype,
                "block_size": args.block_size,
            },
            "generation": {
                "num_warmups": args.num_warmups,
                "temperature": args.temperature,
                "ignore_eos": args.ignore_eos,
            },
            "bmm": bmm,
            "serve_command": serve_cmd,
            "benchmark_command": bench_cmd,
            "environment": safe_environment_snapshot(env),
        }
        _write_json(artifacts.versions, collect_versions(platform))
        _write_json(artifacts.resolved, resolved)

    print(f"[bench] starting {args.backend} server on :{port}", flush=True)
    print(f"[bench] cmd: {' '.join(serve_cmd)}", flush=True)
    if artifacts is not None:
        artifacts.server_log.write_text("", encoding="utf-8")

    with ServingSession(
        serve_cmd,
        env,
        port,
        args.server_timeout,
        server_log=artifacts.server_log if artifacts is not None else None,
    ) as session:
        print(f"[bench] waiting for /health (timeout={args.server_timeout}s)...", flush=True)
        print(f"[bench] server ready, running benchmark", flush=True)

        print(f"[bench] cmd: {' '.join(bench_cmd)}", flush=True)
        result = session.run_benchmark(
            bench_cmd,
            client_log=artifacts.client_log if artifacts is not None else None,
        )
        print(f"[bench] benchmark done (rc={result.returncode}); result → {output}",
              flush=True)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
