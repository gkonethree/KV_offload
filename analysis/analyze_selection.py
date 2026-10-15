#!/usr/bin/env python3
"""Turn a trace_selection.py trace into compact per-layer metrics (JSON).

Granularity model (one decode step, one layer, middle region only -- sink and
local are always resident so they never need fetching):
  page        fetch every 16-token page any kv-head touched, all heads   (today)
  page_head   fetch a page only for the heads that touched it (HND split)
  token       fetch each selected token, all heads (union over heads)
  token_head  fetch exactly the (token, head) pairs that were selected
Units are (token x head) slots, so ratios are byte ratios.

Usage: python analysis/analyze_selection.py TRACE_DIR OUT.json
"""
import base64
import json
import sys

import numpy as np

trace, out_path = sys.argv[1], sys.argv[2]
meta = json.load(open(f"{trace}/meta.json"))
sel = np.load(f"{trace}/sel.npy", mmap_mode='r')          # [S, NL, Lt] uint8 bits
lengths = np.load(f"{trace}/lengths.npy").astype(np.int64)
S = min(len(lengths), meta['steps_recorded'], sel.shape[0])
NL, Lt = sel.shape[1], sel.shape[2]
H = meta['n_kv_heads']
PAGE = meta['page_size']
SINK = int(meta['sparse_env']['SKYLIGHT_SPARSE_SINK'])
LOCAL = int(meta['sparse_env']['SKYLIGHT_SPARSE_LOCAL'])
lengths = lengths[:S]

P = (Lt + PAGE - 1) // PAGE
Lp = P * PAGE
pos = np.arange(Lp)
valid = pos[None, :] < lengths[:, None]                                   # [S, Lp]
mid = (pos[None, :] >= SINK) & (pos[None, :] < (lengths[:, None] - LOCAL))

STEP_BIN = max(1, S // 256)
SB = (S + STEP_BIN - 1) // STEP_BIN
SAMPLE_EVERY = max(1, S // 16)
sample_steps = list(range(0, S, SAMPLE_EVERY))[:16]
RECENCY_EDGES = [0, 64, 128, 256, 512, 1024, 2048, 4096, 1 << 30]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


layers = []
heat = np.zeros((NL, SB, P), dtype=np.uint8)
samples = np.zeros((NL, len(sample_steps), Lp), dtype=np.uint8)

for l in range(NL):
    X = np.zeros((S, Lp), dtype=np.uint8)
    X[:, :Lt] = sel[:S, l, :]
    bits = ((X[:, :, None] >> np.arange(H, dtype=np.uint8)) & 1).astype(bool)   # [S, Lp, H]
    bits &= valid[:, :, None]
    mbits = bits & mid[:, :, None]

    # ---- heatmap: mean fraction of heads selecting, binned to (step-bin, page)
    frac = bits.mean(axis=2)                                                 # [S, Lp]
    pg = frac.reshape(S, P, PAGE).sum(2)
    pv = valid.reshape(S, P, PAGE).sum(2)
    pad = SB * STEP_BIN - S
    pg_b = np.pad(pg, ((0, pad), (0, 0))).reshape(SB, STEP_BIN, P).sum(1)
    pv_b = np.pad(pv, ((0, pad), (0, 0))).reshape(SB, STEP_BIN, P).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.where(pv_b > 0, pg_b / np.maximum(pv_b, 1), np.nan)
    heat[l] = np.where(np.isnan(v), 255, np.clip(np.round(v * 250), 0, 250)).astype(np.uint8)
    samples[l] = X[sample_steps]

    # ---- per-step selection sizes
    k_h = mbits.sum(1)                                                       # [S, H]
    union = mbits.any(2)                                                     # [S, Lp]
    inter = mbits.all(2) & mid
    n_mid = np.maximum(mid.sum(1), 1)

    # ---- page utilisation
    ch = mbits.reshape(S, P, PAGE, H).sum(2)                                 # [S, P, H]
    cu = union.reshape(S, P, PAGE).sum(2)                                    # [S, P]
    hist_head = np.bincount(ch[ch > 0].ravel(), minlength=PAGE + 1)[1:]
    hist_union = np.bincount(cu[cu > 0].ravel(), minlength=PAGE + 1)[1:]
    pages_union = (cu > 0).sum(1)
    pages_head = (ch > 0).sum((1, 2))
    vol = dict(
        page=pages_union * PAGE * H,
        page_head=pages_head * PAGE,
        token=union.sum(1) * H,
        token_head=k_h.sum(1),
    )

    # ---- temporal reuse: what a 1-step and a 16-step staging cache would miss
    def new_vs_window(w):
        new_tok, new_pg = np.zeros(S), np.zeros(S)
        seen = np.zeros_like(mbits)
        seen_pg = np.zeros((S, P), dtype=bool)
        for d in range(1, w + 1):
            seen[d:] |= mbits[:-d]
            seen_pg[d:] |= (cu[:-d] > 0)
        new_tok = (mbits & ~seen).sum((1, 2))
        new_pg = ((cu > 0) & ~seen_pg).sum(1)
        return new_tok[1:], new_pg[1:]

    nt1, np1 = new_vs_window(1)
    nt16, np16 = new_vs_window(16)
    inter_t = (mbits[1:] & mbits[:-1]).sum((1, 2))
    union_t = (mbits[1:] | mbits[:-1]).sum((1, 2))
    jac = inter_t / np.maximum(union_t, 1)

    # ---- head agreement
    agree = inter.sum(1) / np.maximum(union.sum(1), 1)
    pair = []
    for a in range(H):
        for b in range(a + 1, H):
            i = (mbits[:, :, a] & mbits[:, :, b]).sum(1)
            u = (mbits[:, :, a] | mbits[:, :, b]).sum(1)
            pair.append(i / np.maximum(u, 1))
    pair = np.mean(pair, 0)

    # ---- popularity over prompt+generated positions eligible >= 64 steps
    elig = mid.sum(0)                                                        # [Lp]
    chosen = mbits.sum(0)                                                    # [Lp, H]
    ok = elig >= 64
    freq = (chosen[ok] / elig[ok, None]).ravel()
    pop_hist = np.histogram(freq, bins=20, range=(0, 1))[0]
    srt = np.sort(chosen[ok].ravel())[::-1].astype(np.float64)
    cum = np.cumsum(srt) / max(srt.sum(), 1)
    lorenz = [float(cum[min(len(cum) - 1, int(q * len(cum)))]) if len(cum) else 0.0
              for q in np.linspace(0.01, 1, 50)]

    # ---- recency: distance from the current token of each middle selection
    dist = (lengths[:, None] - 1 - pos[None, :])
    dsel = np.repeat(dist[:, :, None], H, 2)[mbits]
    rec_hist = np.histogram(dsel, bins=RECENCY_EDGES)[0]

    def binned(x):
        x = np.asarray(x, dtype=np.float64)
        n = (len(x) + STEP_BIN - 1) // STEP_BIN
        x = np.pad(x, (0, n * STEP_BIN - len(x)), constant_values=np.nan)
        return [None if np.isnan(v) else round(float(v), 4)
                for v in np.nanmean(x.reshape(n, STEP_BIN), 1)]

    tot = {k: float(v.sum()) for k, v in vol.items()}
    layers.append(dict(
        layer=l,
        sparsity=float((k_h.sum(1) / (n_mid * H)).mean()),
        union_frac=float((union.sum(1) / n_mid).mean()),
        k_per_head=float(k_h.mean()),
        util_head=float(ch[ch > 0].mean()) if (ch > 0).any() else 0.0,
        util_union=float(cu[cu > 0].mean()) if (cu > 0).any() else 0.0,
        hist_head=hist_head.tolist(), hist_union=hist_union.tolist(),
        volume=tot,
        new_tok_1=float(nt1.mean()), new_tok_16=float(nt16.mean()),
        new_pg_1=float(np1.mean()), new_pg_16=float(np16.mean()),
        sel_tok=float(k_h.sum(1).mean()), sel_pg=float(pages_union.mean()),
        jaccard=float(jac.mean()),
        head_all_agree=float(agree.mean()), head_pair_jaccard=float(pair.mean()),
        never_frac=float((freq == 0).mean()) if len(freq) else 0.0,
        pop_hist=pop_hist.tolist(), lorenz=lorenz,
        recency=rec_hist.tolist(),
        series=dict(
            sparsity=binned(k_h.sum(1) / (n_mid * H)),
            util_union=binned(np.where(pages_union > 0, union.sum(1) / np.maximum(pages_union, 1), np.nan)),
            jaccard=binned(np.concatenate([[np.nan], jac])),
            waste=binned(vol['page'] / np.maximum(vol['token_head'], 1)),
        ),
    ))
    print(f"layer {l:2d}: sparsity={layers[-1]['sparsity']:.3f} "
          f"util_union={layers[-1]['util_union']:.2f}/16 "
          f"page/token_head={tot['page'] / max(tot['token_head'], 1):.2f}x "
          f"jaccard={layers[-1]['jaccard']:.3f}", flush=True)

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta['model'])
    ids = meta['prompt_token_ids'] + meta['output_token_ids']
    pieces = [tok.convert_ids_to_tokens(i) for i in ids]
except Exception as e:
    print("tokenizer unavailable:", e)
    pieces = []

out = dict(
    meta={k: meta[k] for k in ('model', 'n_layers', 'n_kv_heads', 'n_q_heads', 'page_size',
                                'prompt_mode', 'prompt_len', 'new_tokens', 'max_model_len',
                                'sparse_env', 'output_text')},
    steps=S, Lp=Lp, P=P, step_bin=STEP_BIN, sb=SB, sink=SINK, local=LOCAL,
    lengths=lengths.tolist(), sample_steps=sample_steps,
    recency_edges=RECENCY_EDGES[:-1],
    heat=b64(heat), samples=b64(samples), tokens=pieces,
    layers=layers,
)
with open(out_path, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print("wrote", out_path)
