#!/usr/bin/env python3
"""Simulate offload_package's real staging design on a selection trace.

The real KVStagingPool stages a BLOCK (16 tokens) for ALL layers at once into
one shared pool of C slots (LRU eviction, blocks needed by the current layer
protected). Within a decode step, layers run 0..NL-1; each layer's CPU-resident,
unstaged blocks are fetched on demand (the compute stream waits for them).

  none   on-demand only
  prev   at the start of step t, fetch (one batch) every block any layer used
         at step t-1 that is no longer staged; then on-demand as usual

Residency model is off99: every block whose tokens are all middle tokens lives
on CPU unless staged; sink/local blocks are GPU-resident.

Reports per step: on-demand blocks, number of on-demand fetch calls (layers that
had to fetch), prefetched blocks, and prefetched blocks that went unused.

Usage: python analysis/simulate_shared_pool.py TRACE_DIR OUT.json
"""
import json
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from simulate_prefetch import load  # noqa: E402


def run(need, offl, cap, policy):
    S, NL, P = need.shape
    lru = OrderedDict()                      # block -> None, oldest first
    od_blocks = od_calls = pf_blocks = pf_unused = 0
    pf_pending = set()

    def insert(blocks, protect):
        nonlocal pf_unused
        for b in blocks:
            if b in lru:
                lru.move_to_end(b)
                continue
            if len(lru) >= cap:
                victim = next((v for v in lru if v not in protect), None)
                if victim is None:
                    continue                 # pool full of protected blocks: skipped
                del lru[victim]
                if victim in pf_pending:
                    pf_pending.discard(victim)
                    pf_unused += 1
            lru[b] = None

    for t in range(S):
        if policy == 'prev' and t > 0:
            want = set(np.where((need[t - 1] & offl[t][None, :]).any(0))[0].tolist())
            fetch = [b for b in want if b not in lru]
            pf_blocks += len(fetch)
            pf_pending |= set(fetch)
            insert(fetch, protect=set())
        for l in range(NL):
            n = np.where(need[t, l] & offl[t])[0].tolist()
            ns = set(n)
            miss = [b for b in n if b not in lru]
            for b in n:
                if b in lru:
                    lru.move_to_end(b)
                    pf_pending.discard(b)
            if miss:
                od_blocks += len(miss)
                od_calls += 1
                insert(miss, protect=ns)
    pf_unused += len(pf_pending)
    return dict(od_blocks=od_blocks / S, od_calls=od_calls / S,
                pf_blocks=pf_blocks / S, pf_unused=pf_unused / S)


def main():
    trace, out_path = sys.argv[1], sys.argv[2]
    meta, need, offl, S, NL, P = load(trace)
    union = (need & offl[:, None, :]).any(1).sum(1)
    per_layer = need.sum(2).mean()
    offl_blocks = offl.sum(1)
    U = float(union.mean())
    print(f"S={S} NL={NL} offloadable blocks/step={offl_blocks.mean():.0f} "
          f"per-layer need={per_layer:.1f} union over layers={U:.1f}", flush=True)
    fracs = (0.25, 0.5, 0.75, 0.9, 1.0, 1.25)
    rows = []
    for f in fracs:
        cap = max(4, int(round(U * f)))
        r = {p: run(need, offl, cap, p) for p in ('none', 'prev')}
        rows.append(dict(frac=f, cap=cap, **{f"{p}_{k}": v for p, d in r.items() for k, v in d.items()}))
        a, b = r['none'], r['prev']
        print(f"C={cap:4d} ({f:.0%} of union) | none: on-demand {a['od_blocks']:6.1f} blk in "
              f"{a['od_calls']:4.1f} calls | prev: on-demand {b['od_blocks']:6.1f} blk in "
              f"{b['od_calls']:4.1f} calls, prefetched {b['pf_blocks']:6.1f} (unused {b['pf_unused']:5.1f})",
              flush=True)
    with open(out_path, 'w') as f:
        json.dump(dict(union=U, per_layer=float(per_layer), rows=rows), f)


if __name__ == '__main__':
    main()
