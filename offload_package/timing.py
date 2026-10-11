"""Env-gated stage timing for the offload pipeline.

Enable with SKYLIGHT_OFFLOAD_TIMING=1 (must be set before the process starts;
the flag is read once at import). All entry points are no-ops when disabled,
so instrumented call sites cost one function call and a boolean check.

Stages (accumulated across the process, reported every 32 engine steps and
once at exit):

  sched.*        scheduler-side, per engine step:
                   score        re-scoring pass of offload candidates
                   offload      eviction pick + GPU->CPU copy (D2H + sync)
                   surgery      store-completion processing (true freeing)
  build.*        worker-side, per step / per layer:
                   patch        index remediation + staging re-point
                   pattern_d2h  sparse-pattern D2H cache (oracle family only)
  pq.*           PQ codebook machinery (once per layer per process):
                   fit          k-means++ + Lloyd fit (one-time init cost)
  wrap.*         decode wrapper, per layer:
                   discover     cold-ref discovery (incl. indptr host sync)
                   fetch        staging fetch + index patch
                   patch        pos/val build + index_copy_ inside fetch
                   decode       sparse decode kernel passes
  lru.*           LRU scorer signal:
                   note         per-step top-k selection -> recency ticks
  pool.*         ensure_resident:
                   alloc        phase-1 slot allocation + eviction scoring
                   (phase-2 sub-stages below, per layer during fetch):
                   cpu_gather   pinned-cache slot gather on CPU
                   transform    permute/cat layout transform on CPU
                   h2d          H2D copy launch (pageable today -> sync here)
                   h2d_sync   deferred ring-buffer event sync
"""
from __future__ import annotations

import atexit
import os
import time
from collections import defaultdict

ENABLED = os.environ.get("SKYLIGHT_OFFLOAD_TIMING", "0") == "1"

_total: defaultdict[str, float] = defaultdict(float)
_count: defaultdict[str, int] = defaultdict(int)
_max: defaultdict[str, float] = defaultdict(float)
_t0: dict[str, float] = {}
_step_counter = 0


def start(stage: str) -> None:
    if ENABLED:
        _t0[stage] = time.perf_counter()


def end(stage: str) -> None:
    if not ENABLED:
        return
    t0 = _t0.pop(stage, None)
    if t0 is None:
        return
    dt = (time.perf_counter() - t0) * 1e3
    _total[stage] += dt
    _count[stage] += 1
    if dt > _max[stage]:
        _max[stage] = dt


def step() -> None:
    """Call once per engine step; prints a table every 32 steps."""
    global _step_counter
    if not ENABLED:
        return
    _step_counter += 1
    if _step_counter % 32 == 0:
        report()


def report(final: bool = False) -> None:
    if not ENABLED or not _total:
        return
    tag = "FINAL" if final else f"step {_step_counter}"
    print(f"\n[OFFLOAD TIMING] {tag}  "
          f"(calls/step shown as calls; mean/max per call)", flush=True)
    print(f"{'stage':<18} {'total_ms':>10} {'calls':>8} "
          f"{'mean_ms':>9} {'max_ms':>9}", flush=True)
    for stage in sorted(_total, key=lambda s: -_total[s]):
        n = _count[stage]
        print(f"{stage:<18} {_total[stage]:>10.1f} {n:>8} "
              f"{_total[stage] / max(n, 1):>9.3f} {_max[stage]:>9.3f}",
              flush=True)


@atexit.register
def _flush() -> None:
    report(final=True)
