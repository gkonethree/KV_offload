"""Sweep multiple configs on the same benchmark and emit a pass@1 table.

Wraps :func:`skylight.bench.agentic.run_one_config` so a single command
runs dense + sparse-token + sparse-double against the same instance
subset and tabulates pass@1.

Usage::

    python -m skylight.bench.agentic_sweep \\
        --benchmark mini-swe-agent \\
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --instances bench-resources/swebench-subset-32.txt \\
        --configs dense,sparse-token,sparse-double

Outputs::

    bench-results/agentic/<sweep_id>/<config>/   per-config run dir
    bench-results/agentic/<sweep_id>/sweep-summary.json
    bench-results/agentic/<sweep_id>/sweep.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from skylight.bench.agentic import run_one_config
from skylight.bench.benchmarks import REGISTRY


# Named configurations: the delta from a dense baseline. Adding a new config
# here propagates automatically through CLI choices and the loop logic.
CONFIGS: dict[str, dict] = {
    "dense": {"backend": "dense"},
    "sparse-token": {
        "backend": "sparse",
        "topk": 0.10, "sink": 64, "local": 64, "channel_num": -1,
    },
    "sparse-double": {
        "backend": "sparse",
        "topk": 0.10, "sink": 64, "local": 64, "channel_num": 8,
    },
}


def build_run_one_kwargs(
    config_name: str,
    topk_override: Optional[float] = None,
    sink_override: Optional[int] = None,
    local_override: Optional[int] = None,
    channel_num_override: Optional[int] = None,
) -> dict:
    """Translate a config name + optional overrides into run_one_config kwargs.

    Sparse-axis overrides only apply to sparse configs; for dense they
    are silently ignored.
    """
    cfg = CONFIGS[config_name]
    kwargs: dict = {"backend": cfg["backend"]}
    if cfg["backend"] == "sparse":
        kwargs["topk"] = topk_override if topk_override is not None else cfg["topk"]
        kwargs["sink"] = sink_override if sink_override is not None else cfg["sink"]
        kwargs["local"] = local_override if local_override is not None else cfg["local"]
        kwargs["channel_num"] = (
            channel_num_override if channel_num_override is not None
            else cfg["channel_num"]
        )
    return kwargs


def format_accuracy_table(results: list[Optional[dict]]) -> str:
    """Render a one-row-per-config pass@1 comparison table."""
    headers = ["config", "n_solved", "n_total", "pass@1", "elapsed", "errors"]
    widths = [16, 10, 10, 10, 10, 20]
    lines = []
    sep = "  ".join("-" * w for w in widths)
    lines.append("  ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    lines.append(sep)
    for r in results:
        if r is None:
            continue
        errs = ",".join(r.get("errors", []))[: widths[5]] or "-"
        row = [
            f"{r['config']:<{widths[0]}}",
            f"{r.get('n_solved', 0):>{widths[1]}}",
            f"{r.get('n_total', 0):>{widths[2]}}",
            f"{r.get('pass_at_1', 0.0):>{widths[3]}.3f}",
            f"{int(r.get('elapsed_s', 0.0)):>{widths[4]}}s",
            f"{errs:<{widths[5]}}",
        ]
        lines.append("  ".join(row))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--benchmark", default="mini-swe-agent",
                        choices=list(REGISTRY))
    parser.add_argument("--model", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--configs", default=",".join(CONFIGS.keys()),
                        help=f"comma-separated from: {','.join(CONFIGS)}")
    parser.add_argument("--output-dir", default=None,
                        help="default: bench-results/agentic/sweep_<timestamp>")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--server-timeout", type=float, default=180.0)
    parser.add_argument("--topk", type=float, default=None,
                        help="override --topk for every sparse config")
    parser.add_argument("--sink", type=int, default=None)
    parser.add_argument("--local", type=int, default=None)
    parser.add_argument("--channel-num", type=int, default=None)
    args = parser.parse_args(argv)

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    invalid = [c for c in config_names if c not in CONFIGS]
    if invalid:
        print(f"[agentic-sweep] unknown configs {invalid!r}; "
              f"valid: {list(CONFIGS)}", file=sys.stderr)
        return 2

    benchmark = REGISTRY[args.benchmark]
    ts = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.output_dir or f"bench-results/agentic/sweep_{ts}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    sweep_log = sweep_dir / "sweep.log"

    def slog(msg: str) -> None:
        line = f"[agentic-sweep] {msg}"
        print(line, flush=True)
        with sweep_log.open("a") as f:
            f.write(line + "\n")

    slog(f"sweep_dir={sweep_dir} benchmark={args.benchmark} configs={config_names}")
    slog(f"instances={args.instances}")
    if any([args.topk, args.sink, args.local, args.channel_num is not None]):
        slog(
            f"overrides: topk={args.topk} sink={args.sink} "
            f"local={args.local} channel_num={args.channel_num}"
        )

    results: list[Optional[dict]] = []
    t_start = time.time()
    for name in config_names:
        kwargs = build_run_one_kwargs(
            name,
            topk_override=args.topk, sink_override=args.sink,
            local_override=args.local, channel_num_override=args.channel_num,
        )
        slog(f"▶ {name}")
        try:
            summary = run_one_config(
                benchmark=benchmark,
                config_name=name,
                model=args.model,
                instances=Path(args.instances),
                output_dir=sweep_dir / name,
                max_model_len=args.max_model_len,
                server_timeout_s=args.server_timeout,
                **kwargs,
            )
            results.append(summary)
            slog(
                f"✓ {name}: n_solved={summary.get('n_solved')} "
                f"n_total={summary.get('n_total')} "
                f"pass@1={summary.get('pass_at_1', 0.0):.3f} "
                f"elapsed={int(summary.get('elapsed_s', 0))}s "
                f"errors={summary.get('errors', [])}"
            )
        except Exception as exc:
            slog(f"✗ {name}: orchestrator raised {exc!r}")
            results.append(None)

    elapsed = time.time() - t_start
    slog(f"done — {len(config_names)} configs in {elapsed:.0f}s")
    print("")
    print(format_accuracy_table(results))

    summary_path = sweep_dir / "sweep-summary.json"
    summary_path.write_text(json.dumps({
        "benchmark": args.benchmark,
        "model": args.model,
        "instances": args.instances,
        "configs": config_names,
        "results": [r for r in results if r is not None],
        "elapsed_s": elapsed,
    }, indent=2) + "\n")
    slog(f"summary → {summary_path}")

    failures = sum(1 for r in results if r is None or r.get("errors"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
