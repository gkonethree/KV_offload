#!/usr/bin/env python3
"""Run a resumable dense/BMM long-context crossover experiment.

The runner deliberately delegates server startup and request generation to
``skylight.bench.run_serving``.  This file owns only the experiment matrix,
capacity probing, GPU telemetry, and pair summaries.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Iterable


MODEL = "Qwen/Qwen3.5-9B"
BATCH_CANDIDATES = (64, 32, 16, 8, 4, 2, 1)
DEFAULT_CONTEXTS = (65536, 131072, 262144, 524288, 786432, 1048576)
DEFAULT_GENERATION_LENGTHS = (64, 256, 512, 1024, 2048, 4096)


def parse_int_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated positive integer list."""
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def condition_key(
    mode: str,
    backend: str,
    context: int,
    batch: int,
    generated: int,
) -> str:
    """Return the stable artifact path for one server/benchmark condition."""
    return f"context-{context}/batch-{batch:02d}/{mode}/{backend}-gen{generated}"


def build_condition_command(
    *,
    backend: str,
    model: str,
    context: int,
    generated: int,
    batch: int,
    gpu: int,
    output_dir: Path,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    rope_overrides: str | None = None,
    kv_cache_dtype: str | None = None,
    python: str | None = None,
) -> list[str]:
    """Build the exact ``run_serving`` invocation for one condition."""
    command = ["env", f"CUDA_VISIBLE_DEVICES={gpu}"]
    if rope_overrides:
        command.append(f"SKYLIGHT_HF_OVERRIDES={rope_overrides}")
    if kv_cache_dtype:
        command.append(f"SKYLIGHT_KV_DTYPE={kv_cache_dtype}")
    command.extend([
        python or sys.executable,
        "-m",
        "skylight.bench.run_serving",
        "--backend",
        backend,
        "--model",
        model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(batch),
        "--random-input-len",
        str(context),
        "--random-output-len",
        str(generated),
        "--request-rate",
        "inf",
        "--max-model-len",
        str(context + generated + 512),
        "--max-num-seqs",
        str(batch),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--gdn-prefill-backend",
        "triton",
        "--dtype",
        "bfloat16",
        "--block-size",
        "16",
        "--num-warmups",
        "2",
        "--temperature",
        "0",
        "--ignore-eos",
        "--server-timeout",
        "1800",
        "--enforce-eager" if enforce_eager else "--no-enforce-eager",
        "--artifacts-dir",
        str(output_dir),
    ])
    if backend == "sparse":
        command.extend([
            "--method",
            "block_minmax",
            "--topk",
            "0.10",
            "--sink",
            "64",
            "--local",
            "64",
            "--channel-num",
            "-1",
        ])
    return command


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def condition_complete(output_dir: Path) -> bool:
    """Return true only for a result with a successful runner status."""
    result = _read_json(output_dir / "result.json")
    status = _read_json(output_dir / "status.json")
    return bool(
        result is not None
        and status is not None
        and status.get("success") is True
        and int(result.get("failed", 1)) == 0
    )


def summarize_pair(dense: dict[str, Any], bmm: dict[str, Any]) -> dict[str, Any]:
    """Classify one dense/BMM pair using both latency and throughput."""
    dense_tpot = float(dense["mean_tpot_ms"])
    bmm_tpot = float(bmm["mean_tpot_ms"])
    dense_output = float(dense["output_throughput"])
    bmm_output = float(bmm["output_throughput"])
    throughput_ratio = bmm_output / dense_output if dense_output else 0.0
    tpot_speedup = dense_tpot / bmm_tpot if bmm_tpot else 0.0
    if int(dense.get("failed", 0)) or int(bmm.get("failed", 0)):
        classification = "regression"
    elif tpot_speedup >= 1.05 and throughput_ratio >= 1.05:
        classification = "win"
    elif tpot_speedup < 1.0 or throughput_ratio < 1.0:
        classification = "regression"
    else:
        classification = "neutral"
    return {
        "classification": classification,
        "throughput_ratio": round(throughput_ratio, 4),
        "tpot_speedup": round(tpot_speedup, 4),
        "dense_output_throughput": dense_output,
        "bmm_output_throughput": bmm_output,
        "dense_mean_tpot_ms": dense_tpot,
        "bmm_mean_tpot_ms": bmm_tpot,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _start_gpu_monitor(path: Path) -> tuple[subprocess.Popen[str], Any]:
    stream = path.open("w", encoding="utf-8")
    stream.write("timestamp,utilization.gpu [%],utilization.memory [%],memory.used [MiB],power.draw [W]\n")
    stream.flush()
    process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-l",
            "1",
        ],
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, stream


def _stop_gpu_monitor(process: subprocess.Popen[str], stream: Any) -> None:
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()
    finally:
        stream.close()


def run_condition(
    command: list[str],
    output_dir: Path,
    *,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Run one condition and persist a durable success/failure status."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and condition_complete(output_dir):
        return {"success": True, "skipped": True, "output_dir": str(output_dir)}
    if dry_run:
        print("RUN " + shlex.join(command), flush=True)
        return {"success": True, "dry_run": True, "output_dir": str(output_dir)}

    started = datetime.now(timezone.utc).isoformat()
    log_path = output_dir / "runner.log"
    monitor_process = None
    monitor_stream = None
    returncode = 1
    try:
        monitor_process, monitor_stream = _start_gpu_monitor(output_dir / "gpu.csv")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            returncode = completed.returncode
    except OSError as exc:
        log_path.write_text(f"runner error: {exc}\n")
    finally:
        if monitor_process is not None and monitor_stream is not None:
            _stop_gpu_monitor(monitor_process, monitor_stream)

    result = _read_json(output_dir / "result.json")
    success = returncode == 0 and result is not None and int(result.get("failed", 1)) == 0
    status = {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "success": success,
        "result_present": result is not None,
        "failed_requests": None if result is None else result.get("failed"),
        "command": command,
    }
    _write_json(output_dir / "status.json", status)
    return status | {"output_dir": str(output_dir)}


def _run_pair(
    *,
    root: Path,
    mode: str,
    context: int,
    generated: int,
    candidates: Iterable[int],
    fixed_batch: int | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    enforce_eager = mode == "eager"
    batches = (fixed_batch,) if fixed_batch is not None else tuple(candidates)
    for batch in batches:
        dense_dir = root / condition_key(mode, "dense", context, batch, generated)
        sparse_dir = root / condition_key(mode, "sparse", context, batch, generated)
        dense_command = build_condition_command(
            backend="dense", model=args.model, context=context, generated=generated,
            batch=batch, gpu=args.gpu, output_dir=dense_dir,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=enforce_eager, rope_overrides=args.rope_overrides,
            kv_cache_dtype=args.kv_cache_dtype, python=args.python,
        )
        dense_status = run_condition(dense_command, dense_dir, resume=args.resume, dry_run=args.dry_run)
        if args.dry_run:
            sparse_command = build_condition_command(
                backend="sparse", model=args.model, context=context, generated=generated,
                batch=batch, gpu=args.gpu, output_dir=sparse_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=enforce_eager, rope_overrides=args.rope_overrides,
                kv_cache_dtype=args.kv_cache_dtype, python=args.python,
            )
            run_condition(sparse_command, sparse_dir, resume=args.resume, dry_run=True)
            continue
        if not dense_status.get("success"):
            continue
        sparse_command = build_condition_command(
            backend="sparse", model=args.model, context=context, generated=generated,
            batch=batch, gpu=args.gpu, output_dir=sparse_dir,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=enforce_eager, rope_overrides=args.rope_overrides,
            kv_cache_dtype=args.kv_cache_dtype, python=args.python,
        )
        sparse_status = run_condition(sparse_command, sparse_dir, resume=args.resume, dry_run=False)
        if not sparse_status.get("success"):
            continue
        dense = _read_json(dense_dir / "result.json")
        sparse = _read_json(sparse_dir / "result.json")
        assert dense is not None and sparse is not None
        pair = {
            "mode": mode,
            "context": context,
            "generated": generated,
            "batch": batch,
            "dense_dir": str(dense_dir),
            "bmm_dir": str(sparse_dir),
            **summarize_pair(dense, sparse),
        }
        pair_dir = root / "pairs"
        pair_dir.mkdir(parents=True, exist_ok=True)
        _write_json(pair_dir / f"{mode}-context{context}-gen{generated}.json", pair)
        return pair
    return None


def _write_summary(root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "gpu": args.gpu,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "rope_overrides": args.rope_overrides,
        "kv_cache_dtype": args.kv_cache_dtype,
        "rows": rows,
    }
    _write_json(root / "summary.json", payload)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _dry_run(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    for context in args.contexts:
        for mode in ("cudagraph", "eager"):
            if mode == "eager" and context not in args.eager_contexts:
                continue
            for batch in args.batch_candidates:
                for backend in ("dense", "sparse"):
                    output_dir = root / condition_key(mode, backend, context, batch, args.generated)
                    command = build_condition_command(
                        backend=backend, model=args.model, context=context,
                        generated=args.generated, batch=batch, gpu=args.gpu,
                        output_dir=output_dir,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        enforce_eager=mode == "eager", rope_overrides=args.rope_overrides,
                        kv_cache_dtype=args.kv_cache_dtype, python=args.python,
                    )
                    print("RUN " + shlex.join(command))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--contexts", type=parse_int_list, default=DEFAULT_CONTEXTS)
    parser.add_argument("--generated", type=int, default=128)
    parser.add_argument("--generation-context", type=int, default=32768)
    parser.add_argument("--generation-lengths", type=parse_int_list, default=DEFAULT_GENERATION_LENGTHS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.99)
    parser.add_argument("--batch-candidates", type=parse_int_list, default=BATCH_CANDIDATES)
    parser.add_argument("--eager-contexts", type=parse_int_list, default=(65536, 262144))
    parser.add_argument("--rope-overrides")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        return _dry_run(args)

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    selected_batches: dict[int, int] = {}
    for context in args.contexts:
        pair = _run_pair(
            root=root, mode="cudagraph", context=context, generated=args.generated,
            candidates=args.batch_candidates, fixed_batch=None, args=args,
        )
        if pair:
            rows.append(pair)
            selected_batches[context] = int(pair["batch"])
        print(json.dumps({"context": context, "mode": "cudagraph", "pair": pair}, sort_keys=True), flush=True)

    for context in args.eager_contexts:
        if context not in selected_batches:
            continue
        pair = _run_pair(
            root=root, mode="eager", context=context, generated=args.generated,
            candidates=args.batch_candidates, fixed_batch=selected_batches[context], args=args,
        )
        if pair:
            rows.append(pair)
        print(json.dumps({"context": context, "mode": "eager", "pair": pair}, sort_keys=True), flush=True)

    generation_batch = selected_batches.get(args.generation_context)
    if generation_batch is None and selected_batches:
        closest = min(selected_batches, key=lambda value: abs(value - args.generation_context))
        generation_batch = selected_batches[closest]
    if generation_batch is not None:
        for generated in args.generation_lengths:
            pair = _run_pair(
                root=root, mode="cudagraph", context=args.generation_context,
                generated=generated, candidates=args.batch_candidates,
                fixed_batch=generation_batch, args=args,
            )
            if pair:
                rows.append(pair)
            print(json.dumps({"context": args.generation_context, "generated": generated, "pair": pair}, sort_keys=True), flush=True)

    _write_summary(root, rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
