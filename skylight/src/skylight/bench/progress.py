"""Metrics scraping + progress reporting for agentic bench runs.

Lightweight observability for one-config runs:

  * MetricsScraper — daemon thread that polls /metrics every 5s, writes
    a CSV timeseries, keeps the latest sample in memory.
  * format_progress_line — pure function that renders one status line.
  * ProgressReporter — daemon thread that ticks every 30s (configurable
    via SKYLIGHT_BENCH_PROGRESS_INTERVAL_S), writes a progress.jsonl row
    and prints the status line on stdout.

Runtime overhead is negligible (<0.1% wall clock on 5-15 GPU-min
instances). All threads are daemons; the main thread never blocks on
metric I/O.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# Metrics we extract from /metrics. Everything else in the payload is ignored.
# ``skylight_effective_sparsity_fraction`` is what backend.py exports; the
# legacy ``skylight_observed_sparsity_fraction`` name is aliased in the parser.
_TRACKED_METRICS = (
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "skylight_effective_sparsity_fraction",
)

_SPARSITY_METRIC = "skylight_effective_sparsity_fraction"
_SPARSITY_LEGACY = "skylight_observed_sparsity_fraction"


def _parse_prometheus_text(text: str) -> dict:
    """Extract tracked metrics from a Prometheus text-format payload.

    Returns {metric_name: float}. Labels are stripped — we report a
    single value per metric (the last one seen if multiple labels
    exist for the same name). Missing tracked metrics simply aren't
    present in the dict. Malformed lines are skipped, not raised.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        left, val_str = line.rsplit(" ", 1)
        name = left.split("{", 1)[0]
        if name == _SPARSITY_LEGACY and _SPARSITY_METRIC not in out:
            name = _SPARSITY_METRIC
        elif name == _SPARSITY_LEGACY:
            continue
        if name not in _TRACKED_METRICS:
            continue
        try:
            out[name] = float(val_str)
        except ValueError:
            continue
    return out


class MetricsScraper(threading.Thread):
    """Daemon thread: GETs /metrics every interval_s; snapshots + CSV row each scrape."""

    def __init__(
        self,
        port: int,
        csv_path: Path,
        interval_s: float = 5.0,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.csv_path = csv_path
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._latest: dict = {}
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        """Return a copy of the most-recent /metrics sample."""
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w") as f:
            f.write("timestamp," + ",".join(_TRACKED_METRICS) + "\n")
            f.flush()
            while not self._stop_event.is_set():
                try:
                    with urllib.request.urlopen(
                        f"http://localhost:{self.port}/metrics", timeout=2.0,
                    ) as r:
                        text = r.read().decode("utf-8", errors="replace")
                    sample = _parse_prometheus_text(text)
                    if sample:
                        with self._lock:
                            self._latest = sample
                        row = [f"{time.time():.2f}"] + [
                            str(sample.get(m, "")) for m in _TRACKED_METRICS
                        ]
                        f.write(",".join(row) + "\n")
                        f.flush()
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    # Scrape failures are silent; the progress reporter shows N/A.
                    pass
                self._stop_event.wait(self.interval_s)


def format_progress_line(
    config: str,
    instances_done: int,
    instances_total: int,
    snapshot: dict,
    elapsed_s: float,
    prev_tok_count: Optional[float] = None,
    prev_tok_time: Optional[float] = None,
) -> str:
    """Render a single status line. Pure function (no I/O)."""
    pct = (instances_done / max(1, instances_total)) * 100
    bar_width = 12
    filled = int(bar_width * instances_done / max(1, instances_total))
    bar = "█" * filled + "░" * (bar_width - filled)

    tok_total = snapshot.get("vllm:generation_tokens_total")
    if (
        tok_total is not None
        and prev_tok_count is not None
        and prev_tok_time is not None
    ):
        dt = max(1.0, time.time() - prev_tok_time)
        tok_per_s = max(0.0, (tok_total - prev_tok_count) / dt)
        tok_str = f"{tok_per_s:>5.0f} tok/s"
    else:
        tok_str = "  N/A tok/s"

    spars = snapshot.get(_SPARSITY_METRIC) or snapshot.get(_SPARSITY_LEGACY)
    spars_str = f"spars={spars:.3f}" if spars is not None else "spars=N/A  "

    if instances_done > 0 and instances_done < instances_total:
        per_inst = elapsed_s / instances_done
        eta_s = per_inst * (instances_total - instances_done)
        m, s = divmod(int(eta_s), 60)
        eta_str = f"ETA {m}m{s:02d}s"
    else:
        eta_str = "ETA --:--"

    m, s = divmod(int(elapsed_s), 60)
    elapsed_str = f"{m}m{s:02d}s elapsed"

    return (
        f"[agentic] {config:<14} {instances_done:>3}/{instances_total} "
        f"[{bar}] {pct:>3.0f}%  {tok_str}  {spars_str}  {elapsed_str}  {eta_str}"
    )


class ProgressReporter(threading.Thread):
    """Daemon thread: every interval_s, read scraper + count patches, emit one line."""

    def __init__(
        self,
        config_name: str,
        scraper,
        patches_path: Path,
        instances_total: int,
        progress_jsonl: Path,
        interval_s: Optional[float] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.config_name = config_name
        self.scraper = scraper
        self.patches_path = patches_path
        self.instances_total = instances_total
        self.progress_jsonl = progress_jsonl

        if interval_s is None:
            env_interval = os.environ.get("SKYLIGHT_BENCH_PROGRESS_INTERVAL_S")
            interval_s = float(env_interval) if env_interval else 30.0
        self.interval_s = interval_s

        self._stop_event = threading.Event()
        self._t_start = time.time()
        self._prev_tok_count: Optional[float] = None
        self._prev_tok_time: Optional[float] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self.progress_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_jsonl.open("a") as f:
            while not self._stop_event.is_set():
                snapshot = self.scraper.snapshot()
                instances_done = self._count_patches()
                elapsed = time.time() - self._t_start
                line = format_progress_line(
                    config=self.config_name,
                    instances_done=instances_done,
                    instances_total=self.instances_total,
                    snapshot=snapshot,
                    elapsed_s=elapsed,
                    prev_tok_count=self._prev_tok_count,
                    prev_tok_time=self._prev_tok_time,
                )
                print(line, flush=True)
                record = {
                    "t": time.time(),
                    "config": self.config_name,
                    "elapsed_s": elapsed,
                    "instances_done": instances_done,
                    "instances_total": self.instances_total,
                    **snapshot,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                tok_now = snapshot.get("vllm:generation_tokens_total")
                if tok_now is not None:
                    self._prev_tok_count = tok_now
                    self._prev_tok_time = time.time()
                self._stop_event.wait(self.interval_s)

    def _count_patches(self) -> int:
        if not self.patches_path.exists():
            return 0
        with self.patches_path.open() as f:
            return sum(1 for line in f if line.strip())
