#!/usr/bin/env python3
"""Replay a trace_selection.py trace against a GPU staging cache plus a prefetcher.

Model (off99-style): every middle page lives on CPU unless cached; sink and
local-window pages are always GPU-resident. Each layer has its own LRU cache of
C pages holding pages it actually used. Per decode step t, per layer l:

  predict  the policy names pages it expects layer l to need, using only what is
           known before layer l runs: earlier steps of any layer, and layer l-1
           at step t. Predicted, uncached pages are fetched into a separate
           in-flight buffer, overlapped with earlier compute (they never evict
           the cache).
  demand   layer l's real selection. Each needed, uncached page is either
             hidden   -- it was predicted, so it arrived in the background
             exposed  -- not predicted, fetched on the critical path (a stall)
           Predicted pages that were not needed are wasted transfer.
  update   used pages enter the cache (LRU).

The volume streamed per step (hidden + exposed) is set by capacity alone; a
prefetcher only decides how much of it is off the critical path.

Policies
  none           no prediction: every uncached page is exposed
  prev           what this layer used last step
  prev+spatial1  that, plus +-1 neighbouring pages
  history16      anything this layer used in the last 16 steps
  crosslayer     what layer l-1 used this step (layer 0: last step)
  cross+prev     both
  perfect        exactly the needed set (bound: nothing exposed, nothing wasted)

Also reports predictor recall on NEW pages (needed at t, not at t-1).

Usage: python analysis/simulate_prefetch.py TRACE_DIR OUT.json [--workers N]
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

POLICIES = ['none', 'prev', 'prev+spatial1', 'history16', 'crosslayer', 'cross+prev', 'perfect']
NEED = OFFL = None


def load(trace):
    meta = json.load(open(f"{trace}/meta.json"))
    sel = np.load(f"{trace}/sel.npy", mmap_mode='r')
    lengths = np.load(f"{trace}/lengths.npy").astype(np.int64)
    S = min(len(lengths), meta['steps_recorded'], sel.shape[0])
    NL, Lt = sel.shape[1], sel.shape[2]
    PAGE = meta['page_size']
    SINK = int(meta['sparse_env']['SKYLIGHT_SPARSE_SINK'])
    LOCAL = int(meta['sparse_env']['SKYLIGHT_SPARSE_LOCAL'])
    P = (Lt + PAGE - 1) // PAGE
    lengths = lengths[:S]
    pos = np.arange(P * PAGE)
    mid = (pos[None, :] >= SINK) & (pos[None, :] < (lengths[:, None] - LOCAL))
    pg = np.arange(P)
    # A page is offloadable only if every token in it is a middle token.
    offl = (pg[None, :] * PAGE >= SINK) & (pg[None, :] * PAGE + PAGE - 1 < (lengths[:, None] - LOCAL))
    need = np.zeros((S, NL, P), dtype=bool)
    for l in range(NL):
        X = np.zeros((S, P * PAGE), dtype=np.uint8)
        X[:, :Lt] = sel[:S, l, :]
        need[:, l, :] = ((X > 0) & mid).reshape(S, P, PAGE).any(2) & offl
    return meta, need, offl, S, NL, P


def dilate(mask, r, axis=-1):
    out = mask.copy()
    m = np.moveaxis(mask, axis, 0)
    o = np.moveaxis(out, axis, 0)
    for d in range(1, r + 1):
        o[d:] |= m[:-d]
        o[:-d] |= m[d:]
    return out


def predict(policy, t, l):
    need = NEED
    if policy == 'perfect':
        return need[t, l]
    if policy == 'none' or t == 0:
        return None
    prev = need[t - 1, l]
    cross = need[t, l - 1] if l > 0 else prev
    if policy == 'prev':
        return prev
    if policy == 'prev+spatial1':
        return dilate(prev, 1)
    if policy == 'history16':
        return need[max(0, t - 16):t, l].any(0)
    if policy == 'crosslayer':
        return cross
    if policy == 'cross+prev':
        return cross | prev
    raise ValueError(policy)


def simulate(args):
    policy, cap, l = args
    S, NL, P = NEED.shape
    last = np.full(P, -1, dtype=np.int64)
    cached = np.zeros(P, dtype=bool)
    hidden = exposed = wasted = 0
    for t in range(S):
        n = NEED[t, l] & OFFL[t]
        miss = n & ~cached
        pf = predict(policy, t, l)
        if pf is None:
            exposed += int(miss.sum())
        else:
            pf = pf & OFFL[t] & ~cached
            hidden += int((miss & pf).sum())
            exposed += int((miss & ~pf).sum())
            wasted += int((pf & ~n).sum())
        # used pages enter the cache, LRU
        new = n & ~cached
        k = int(new.sum()) - (cap - int(cached.sum()))
        if k > 0:
            cand = np.where(cached & ~n)[0]
            if len(cand):
                v = cand[np.argpartition(last[cand], min(k, len(cand)) - 1)[:min(k, len(cand))]]
                cached[v] = False
        room = cap - int(cached.sum())
        add = np.where(new)[0][:max(room, 0)]
        cached[add] = True
        last[n & cached] = t
    return policy, cap, l, hidden / S, exposed / S, wasted / S


def predictor_recall(need, offl):
    S, NL, P = need.shape
    rows = []
    for l in range(NL):
        n = need[1:, l] & offl[1:]
        prev = need[:-1, l]
        new = n & ~prev
        hist = np.zeros_like(prev)
        for d in range(1, min(17, S)):
            hist[d - 1:] |= need[:S - d, l]
        cross = need[1:, l - 1] if l > 0 else prev
        preds = {'spatial1': dilate(prev, 1), 'spatial2': dilate(prev, 2),
                 'history16': hist, 'crosslayer': cross, 'cross+spatial1': cross | dilate(prev, 1)}
        tot = max(int(new.sum()), 1)
        row = dict(layer=l, need=float(n.sum() / (S - 1)), new=float(new.sum() / (S - 1)))
        for k, p in preds.items():
            row[k] = float((new & p).sum() / tot)
        rows.append(row)
    return rows


def main():
    global NEED, OFFL
    trace, out_path = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 32
    meta, NEED, OFFL, S, NL, P = load(trace)
    hot = float(NEED.sum(2).mean())
    fracs = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    caps = [max(4, int(round(hot * f))) for f in fracs]
    print(f"S={S} NL={NL} P={P} mean needed pages/step/layer={hot:.1f} caps={caps}", flush=True)
    jobs = [(pol, c, l) for pol in POLICIES for c in caps for l in range(NL)]
    with Pool(workers) as pool:
        res = pool.map(simulate, jobs, chunksize=4)
    table = {pol: {str(c): dict(hidden=[0.0] * NL, exposed=[0.0] * NL, wasted=[0.0] * NL)
                   for c in caps} for pol in POLICIES}
    for pol, c, l, h, e, w in res:
        t = table[pol][str(c)]
        t['hidden'][l], t['exposed'][l], t['wasted'][l] = h, e, w
    for pol in POLICIES:
        print(pol.ljust(14), '  '.join(
            f"C={c}: exp {np.mean(table[pol][str(c)]['exposed']):5.1f} "
            f"waste {np.mean(table[pol][str(c)]['wasted']):5.1f}" for c in caps[1:5]))
    out = dict(steps=S, layers=NL, pages=P, hot_pages=hot, cap_fracs=fracs, caps=caps,
               policies=POLICIES, table=table, recall=predictor_recall(NEED, OFFL),
               page_bytes=meta['page_size'] * meta['n_kv_heads'] * 64 * 2 * 2)
    with open(out_path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print("wrote", out_path)


if __name__ == '__main__':
    main()
