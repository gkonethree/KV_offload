"""Long-context benchmark runner: launch a (dense|sparse) skylight server, run a
vendored hub benchmark against it via ApiServerAdapter, write summary.json.

The long-context twin of agentic.py:run_one_config, minus the agent/preds/
instances machinery: long-context evals are request/response with per-task
scores (post_run_evaluate), so the flow is launch-server -> run_benchmark ->
summary. Sparsity lives in the server (SKYLIGHT_SPARSE_METHOD + knobs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from skylight.bench.run_serving import (
    build_env, build_serve_cmd, free_port, terminate_group, wait_for_health)
from skylight.bench.longctx import REGISTRY
from skylight.bench.longctx.api_server import ApiServerAdapter

# setup.sh installs the workspace package. This override remains available for
# kernel developers testing a separate checkout, but has no machine-specific default.
_KERNELS_PATH = os.environ.get("SKYLIGHT_KERNELS_PATH")


def run_one(bench_name: str, backend: str, model: str, subsets: Optional[List[str]],
            out_dir: str, *, method: str = "block_minmax", topk: float = 0.01,
            sink: int = 64, local: int = 64, channel_num: Optional[int] = None,
            max_ctx: int = 131072, max_new: int = 128, gpu: int = 0,
            enforce_eager: bool = False, gpu_mem: float = 0.5,
            server_timeout: float = 1800) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    cmd = build_serve_cmd(backend, model, port, max_ctx + max_new + 64, enforce_eager,
                          gpu_memory_utilization=gpu_mem)
    env = build_env(
        backend,
        topk,
        sink,
        local,
        channel_num,
        method=method,
    )
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if backend == "sparse":
        if _KERNELS_PATH:
            env["PYTHONPATH"] = _KERNELS_PATH + (
                ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    try:
        wait_for_health(port, server_timeout, proc)
        bench = REGISTRY[bench_name](subsets_to_run=subsets)
        adapter = ApiServerAdapter(model, base_url=f"http://127.0.0.1:{port}")
        metrics = bench.run_benchmark(
            adapter, result_dir=str(out),
            generation_kwargs={"max_new_tokens": max_new},
            request_kwargs={"max_context_length": max_ctx})
        summary = dict(benchmark=bench_name, backend=backend, method=method, model=model,
                       subsets=subsets, max_context_length=max_ctx, max_new_tokens=max_new,
                       topk=topk, sink=sink, local=local, channel_num=channel_num,
                       elapsed_s=round(time.time() - t0, 1), metrics=metrics)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        return summary
    finally:
        terminate_group(proc)
        log.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(REGISTRY))
    ap.add_argument("--backend", default="sparse", choices=["dense", "sparse"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--subsets", default=None, help="comma-separated; default = all for the benchmark")
    ap.add_argument("--method", default="block_minmax", choices=["oracle", "block_minmax"])
    ap.add_argument("--topk", type=float, default=0.01)
    ap.add_argument("--sink", type=int, default=64)
    ap.add_argument("--local", type=int, default=64)
    ap.add_argument("--channel-num", type=int, default=None)
    ap.add_argument("--max-context", type=int, default=131072)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    subsets = a.subsets.split(",") if a.subsets else None
    s = run_one(a.benchmark, a.backend, a.model, subsets, a.out, method=a.method,
                topk=a.topk, sink=a.sink, local=a.local, channel_num=a.channel_num,
                max_ctx=a.max_context, max_new=a.max_new_tokens, gpu=a.gpu, gpu_mem=a.gpu_mem)
    print("RESULT " + json.dumps(s.get("metrics", {}), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
