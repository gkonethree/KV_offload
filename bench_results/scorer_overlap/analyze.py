"""Compare scorers: offload-set overlap, fetch timeline, selection hit rate."""
import itertools, json, os
D = os.path.dirname(os.path.abspath(__file__))
names = [n for n in ("paged_eviction", "lru", "random") if os.path.exists(f"{D}/{n}.json")]
R = {n: json.load(open(f"{D}/{n}.json")) for n in names}

def series(r, key):
    d = {int(k): v for k, v in r[key].items()}
    return [d.get(s, 0) for s in range(1, r["step"] + 1)]

print("== offload triggers (decode step: #blocks)")
for n in names:
    print(f"  {n:<15}", ", ".join(f"{o['step']}:{len(o['blocks'])}" for o in R[n]["offloads"]))

print("\n== overlap of offloaded block sets (Jaccard)")
pre = {n: set(R[n]["offloads"][0]["blocks"]) if R[n]["offloads"] else set() for n in names}
tot = {n: set(b for o in R[n]["offloads"] for b in o["blocks"]) for n in names}
for a, b in itertools.combinations(names, 2):
    j = lambda x, y: len(x & y) / max(len(x | y), 1)
    print(f"  {a} vs {b}: prefill {j(pre[a], pre[b]):.2f} "
          f"(shared {len(pre[a] & pre[b])}/{len(pre[a])}), all offloads {j(tot[a], tot[b]):.2f}")

print("\n== fetched tokens (sum over layers)")
for n in names:
    f = series(R[n], "fetched"); total = sum(f); c, s50, s90 = 0, None, None
    for i, x in enumerate(f, 1):
        c += x
        if s50 is None and c >= 0.5 * total: s50 = i
        if s90 is None and c >= 0.9 * total: s90 = i
    w = lambda lo, hi: sum(f[lo - 1:hi])
    print(f"  {n:<15} total {total:>9,}  first 16 steps {w(1,16):>8,}  17-256 {w(17,256):>8,}  "
          f"257+ {w(257,len(f)):>8,}  50% by step {s50}  90% by step {s90}")

print("\n== where selected tokens live, before each layer's fetch (share of distinct selected tokens)")
for lo, hi in ((1, 16), (17, 256), (257, 2048)):
    print(f"  steps {lo}-{hi}:")
    for n in names:
        sel = sum(series(R[n], "sel")[lo - 1:hi]) or 1
        g = sum(series(R[n], "on_gpu")[lo - 1:hi]) / sel
        st = sum(series(R[n], "staged")[lo - 1:hi]) / sel
        cp = sum(series(R[n], "on_cpu")[lo - 1:hi]) / sel
        nd = sum(series(R[n], "need")[lo - 1:hi]) / sel
        print(f"    {n:<15} GPU {g:6.1%}  staged {st:6.1%}  CPU {cp:6.1%}  -> fetched/filled {nd:6.1%}")
