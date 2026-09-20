"""Sweep multiple bench configurations and emit a clean comparison table.

Wraps :mod:`skylight.bench.run_serving` so a single command runs the full
A/B/C: dense baseline vs. token-sparse vs. doubly-sparse (channel + token)
across multiple input lengths. Per-run server logs are suppressed by
default (saved to per-run log files); the orchestrator only emits one
structured status line per run plus a summary table at the end.

Usage::

    python -m skylight.bench.sweep \\
        --model Qwen/Qwen3-0.6B \\
        --num-prompts 50 \\
        --input-lens 1024,4096,16384 \\
        --configs dense,sparse-token,sparse-double

Output:
    bench-results/sweep/<config>-L<input_len>.json    # per-run vllm bench result
    bench-results/sweep/<config>-L<input_len>.log     # captured stdout/stderr
    bench-results/sweep/sweep-summary.json            # aggregated headline numbers
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# Named configurations. Each entry is the **delta** from the dense baseline;
# the orchestrator translates these into ``skylight.bench.run_serving`` flags.
# Add more here without touching the loop logic.
CONFIGS: dict[str, dict] = {
    "dense": {
        "backend": "dense",
    },
    "sparse-token": {
        "backend": "sparse",
        "topk": 0.10,
        "sink": 64,
        "local": 64,
        "channel_num": -1,  # full head_dim → token-sparse only
    },
    "sparse-double": {
        "backend": "sparse",
        "topk": 0.10,
        "sink": 64,
        "local": 64,
        "channel_num": 8,  # 8/head_dim of the score = doubly-sparse
    },
}


# --------------------------------- helpers -----------------------------------


def build_run_serving_cmd(
    model: str,
    config_name: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    max_model_len: int,
    output_json: str,
    topk_override: Optional[float] = None,
    sink_override: Optional[int] = None,
    local_override: Optional[int] = None,
    channel_num_override: Optional[int] = None,
) -> list[str]:
    """Translate a (model, config_name, input_len, ...) tuple into a
    ``skylight.bench.run_serving`` argv list.

    Per-axis overrides win over the named config's baked-in values. This
    lets ``sweep --configs sparse-token,sparse-double --topk 0.02`` rerun
    every sparse config at the more aggressive 2% sparsity without
    editing the CONFIGS table. Overrides only apply to sparse configs;
    they're silently ignored for dense.
    """
    cfg = CONFIGS[config_name]
    cmd = [
        sys.executable, "-m", "skylight.bench.run_serving",
        "--backend", cfg["backend"],
        "--model", model,
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--max-model-len", str(max_model_len),
        "--output", output_json,
    ]
    if cfg["backend"] == "sparse":
        topk = topk_override if topk_override is not None else cfg["topk"]
        sink = sink_override if sink_override is not None else cfg["sink"]
        local = local_override if local_override is not None else cfg["local"]
        channel_num = (
            channel_num_override
            if channel_num_override is not None
            else cfg["channel_num"]
        )
        cmd += [
            "--topk", str(topk),
            "--sink", str(sink),
            "--local", str(local),
            "--channel-num", str(channel_num),
        ]
    return cmd


def extract_summary(result_json: str, config_name: str, input_len: int,
                    elapsed_s: float) -> Optional[dict]:
    """Parse the headline numbers from a vllm bench serve result JSON."""
    try:
        with open(result_json) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    try:
        return {
            "config": config_name,
            "input_len": input_len,
            "req_per_sec": float(data["request_throughput"]),
            "out_tok_per_sec": float(data["output_throughput"]),
            "ttft_p50_ms": float(data["median_ttft_ms"]),
            "ttft_p99_ms": float(data["p99_ttft_ms"]),
            "tpot_p50_ms": float(data["median_tpot_ms"]),
            "tpot_p99_ms": float(data["p99_tpot_ms"]),
            "elapsed_s": elapsed_s,
        }
    except KeyError as exc:
        print(f"[sweep] WARNING: missing key {exc} in {result_json}", file=sys.stderr)
        return None


def format_table(results: list[Optional[dict]]) -> str:
    """Pretty-print the per-run headline numbers in a one-row-per-run table."""
    headers = ["config", "L", "req/s", "out_tok/s", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99"]
    widths = [18, 8, 10, 12, 10, 10, 10, 10]
    lines = []
    sep = "  ".join("-" * w for w in widths)
    lines.append("  ".join(f"{h:<{w}}" for h, w in zip(headers, widths)))
    lines.append(sep)
    for r in results:
        if r is None:
            continue
        row = [
            f"{r['config']:<{widths[0]}}",
            f"{r['input_len']:>{widths[1]}}",
            f"{r['req_per_sec']:>{widths[2]}.2f}",
            f"{r['out_tok_per_sec']:>{widths[3]}.1f}",
            f"{r['ttft_p50_ms']:>{widths[4]}.1f}",
            f"{r['ttft_p99_ms']:>{widths[5]}.1f}",
            f"{r['tpot_p50_ms']:>{widths[6]}.2f}",
            f"{r['tpot_p99_ms']:>{widths[7]}.2f}",
        ]
        lines.append("  ".join(row))
    return "\n".join(lines)


# --------------------------------- run-one -----------------------------------


def run_one(
    model: str,
    config_name: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    max_model_len: int,
    output_dir: Path,
    timeout_s: float,
    topk_override: Optional[float] = None,
    sink_override: Optional[int] = None,
    local_override: Optional[int] = None,
    channel_num_override: Optional[int] = None,
) -> Optional[dict]:
    """Run a single (config, input_len) bench. Captures stdout/stderr to a log
    file so the orchestrator's terminal stays clean.

    Per-axis overrides (topk/sink/local/channel_num) are passed through to
    :func:`build_run_serving_cmd` and influence the result file's name so
    runs at different override values land in distinct JSON files.
    """
    # Suffix the output paths with any overrides so multiple sweep
    # invocations at different aggressiveness levels don't collide.
    suffix_parts = []
    if topk_override is not None:
        suffix_parts.append(f"topk{topk_override}")
    if sink_override is not None:
        suffix_parts.append(f"sink{sink_override}")
    if local_override is not None:
        suffix_parts.append(f"local{local_override}")
    if channel_num_override is not None:
        suffix_parts.append(f"ch{channel_num_override}")
    suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""

    result_json = str(output_dir / f"{config_name}-L{input_len}{suffix}.json")
    log_file = output_dir / f"{config_name}-L{input_len}{suffix}.log"

    cmd = build_run_serving_cmd(
        model, config_name, input_len, output_len, num_prompts,
        max_model_len, result_json,
        topk_override=topk_override,
        sink_override=sink_override,
        local_override=local_override,
        channel_num_override=channel_num_override,
    )

    print(f"[sweep] ▶ {config_name:<14} L={input_len:<6}  starting...", flush=True)
    t0 = time.time()
    with log_file.open("w") as logf:
        result = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout_s,
        )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[sweep] ✗ {config_name:<14} L={input_len:<6}  FAILED rc={result.returncode} "
              f"after {elapsed:.0f}s — see {log_file}", flush=True)
        return None

    summary = extract_summary(result_json, config_name, input_len, elapsed)
    if summary is None:
        print(f"[sweep] ✗ {config_name:<14} L={input_len:<6}  result parse failed — "
              f"see {log_file}", flush=True)
        return None

    print(
        f"[sweep] ✓ {summary['config']:<14} L={summary['input_len']:<6}  "
        f"req/s={summary['req_per_sec']:>6.2f}  "
        f"out_tok/s={summary['out_tok_per_sec']:>8.1f}  "
        f"ttft_p50={summary['ttft_p50_ms']:>6.1f}ms  "
        f"tpot_p50={summary['tpot_p50_ms']:>6.2f}ms  "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    return summary


# --------------------------------- entry point -------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--output-len", type=int, default=64,
                        help="random-output-len per prompt (default: 64)")
    parser.add_argument("--input-lens", default="1024,4096,16384",
                        help="comma-separated input lengths (default: 1024,4096,16384)")
    parser.add_argument("--configs", default=",".join(CONFIGS.keys()),
                        help=f"comma-separated config names from: {','.join(CONFIGS)}")
    parser.add_argument("--max-model-len-headroom", type=int, default=512,
                        help="bytes of headroom over (input_len + output_len) "
                             "for max-model-len (default: 512)")
    parser.add_argument("--output-dir", default="bench-results/sweep")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-run timeout in seconds (default: 900s = 15min)")

    # Per-axis overrides over the named config's baked-in values. Useful
    # for sweeping aggressiveness without editing the CONFIGS table:
    #   sweep --configs sparse-token,sparse-double --topk 0.02
    parser.add_argument("--topk", type=float, default=None,
                        help="override --topk for every sparse config (default: per-config)")
    parser.add_argument("--sink", type=int, default=None,
                        help="override --sink for every sparse config (default: per-config)")
    parser.add_argument("--local", type=int, default=None,
                        help="override --local for every sparse config (default: per-config)")
    parser.add_argument("--channel-num", type=int, default=None,
                        help="override --channel-num for every sparse config (default: per-config)")

    args = parser.parse_args(argv)

    input_lens = [int(x.strip()) for x in args.input_lens.split(",") if x.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    invalid = [c for c in config_names if c not in CONFIGS]
    if invalid:
        print(f"[sweep] error: unknown configs {invalid!r}; "
              f"valid: {list(CONFIGS)}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(input_lens) * len(config_names)
    print(f"[sweep] {n_runs} runs  model={args.model}  input_lens={input_lens}  "
          f"configs={config_names}  output={output_dir}/", flush=True)

    overrides_msg = []
    if args.topk is not None:
        overrides_msg.append(f"topk={args.topk}")
    if args.sink is not None:
        overrides_msg.append(f"sink={args.sink}")
    if args.local is not None:
        overrides_msg.append(f"local={args.local}")
    if args.channel_num is not None:
        overrides_msg.append(f"channel_num={args.channel_num}")
    if overrides_msg:
        print(f"[sweep] overrides: {', '.join(overrides_msg)}", flush=True)

    results: list[Optional[dict]] = []
    t_start = time.time()
    for L in input_lens:
        max_model_len = L + args.output_len + args.max_model_len_headroom
        for c in config_names:
            results.append(run_one(
                args.model, c, L, args.output_len, args.num_prompts,
                max_model_len, output_dir, args.timeout,
                topk_override=args.topk,
                sink_override=args.sink,
                local_override=args.local,
                channel_num_override=args.channel_num,
            ))

    elapsed = time.time() - t_start
    print(f"\n[sweep] done — {n_runs} runs in {elapsed:.0f}s\n", flush=True)
    print(format_table(results), flush=True)

    sweep_json = output_dir / "sweep-summary.json"
    with sweep_json.open("w") as f:
        json.dump({
            "model": args.model,
            "num_prompts": args.num_prompts,
            "output_len": args.output_len,
            "input_lens": input_lens,
            "configs": config_names,
            "results": [r for r in results if r is not None],
            "elapsed_s": elapsed,
        }, f, indent=2)
    print(f"\n[sweep] summary → {sweep_json}", flush=True)

    failures = sum(1 for r in results if r is None)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
