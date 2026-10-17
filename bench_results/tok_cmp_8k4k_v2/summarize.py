"""pq vs pq+token eviction (FastKVzip) vs pq+token eviction (random)."""
import json, os, re, statistics
D = os.path.dirname(os.path.abspath(__file__))
CFGS = [("pq", "pq"), ("pq_tok70", "tok-fastkvzip"), ("pq_tokrand70", "tok-random")]

def parse(tag):
    p = os.path.join(D, tag + ".log")
    if not os.path.exists(p):
        return None
    log = open(p).read()
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, log)) else None)
    r = {"tpot": g(r"decode-only TPOT: ([\d.]+)"), "ttft": g(r"TTFT: ([\d.]+)"),
         "err": "Traceback" in log, "fallback": "[PQ ADC] FALLBACK" in log}
    m = re.search(r"KV blocks in use: peak (\d+), mean (\d+), final (\d+) \(\+(\d+) staging", log)
    r["kv"] = tuple(map(int, m.groups())) if m else None
    fin = log[log.rfind("[OFFLOAD TIMING] FINAL"):] if "[OFFLOAD TIMING] FINAL" in log else ""
    m = re.search(r"\[bytes moved, MB\] (.*)", fin)
    r["mb"] = sum(float(v) for _, v in re.findall(r"(\S+)=([\d.]+)", m.group(1))) if m else 0.0
    m = re.search(r"^wrap\.tok_chunked\s+[\d.]+\s+(\d+)", fin, re.M)
    r["chunked"] = int(m.group(1)) if m else 0
    tp = os.path.join(D, tag + ".tokens.json")
    r["tokens"] = json.load(open(tp)) if os.path.exists(tp) else None
    meta = os.path.join(D, tag + ".meta")
    r["md5"] = [x.split()[1] for x in open(meta).read().split("\n") if x.startswith("code_md5")] if os.path.exists(meta) else []
    return r

def agree(a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    return first, sum(x == y for x, y in zip(a, b)) / max(n, 1)

print("Qwen3-4B-Instruct-2507 | LongBench 8k prompt + 4096 decode | 70% token offload, W=256 | 230 staging pages | batched gate")
print(f"{'config':<15}{'rnd':>4}{'TPOT ms':>9}{'TTFT ms':>9}{'KV peak':>8}{'KV mean':>8}{'KV final':>9}{'+stage':>7}{'fetched MB':>11}{'1st diff vs pq':>15}{'pos-agree':>10}")
res, md5s, means = {}, set(), {}
for stem, name in CFGS:
    for rnd in ("r1", "r2"):
        res[(stem, rnd)] = parse(f"{stem}_{rnd}")
for stem, name in CFGS:
    vals = []
    for rnd in ("r1", "r2"):
        r = res[(stem, rnd)]
        if r is None:
            print(f"{name:<15}{rnd:>4}  (not run yet)"); continue
        if r["err"] or r["tpot"] is None:
            print(f"{name:<15}{rnd:>4}  {'ERROR' if r['err'] else '(running)'}"); continue
        vals.append(r["tpot"]); md5s.update(r["md5"])
        base = res.get(("pq", rnd)) or res.get(("pq", "r1"))
        fd, ag = (agree(base["tokens"], r["tokens"]) if base and base.get("tokens") and r["tokens"] else (None, float("nan")))
        kv = r["kv"] or (0, 0, 0, 0)
        print(f"{name:<15}{rnd:>4}{r['tpot']:>9.1f}{r['ttft'] or 0:>9.1f}{kv[0]:>8}{kv[1]:>8}{kv[2]:>9}{kv[3]:>7}{r['mb']:>11.0f}"
              f"{str(fd) if stem != 'pq' else '-':>15}{ag if stem != 'pq' else 1.0:>10.3f}"
              + ("  ADC-FALLBACK" if r["fallback"] else "") + (f"  chunked={r['chunked']}" if r["chunked"] else ""))
    if vals:
        means[name] = statistics.mean(vals)
print("\nmean TPOT:", "  ".join(f"{k} {v:.1f} ms" for k, v in means.items()))
if "pq" in means:
    print("vs pq:     ", "  ".join(f"{k} {100 * (v / means['pq'] - 1):+.1f}%" for k, v in means.items() if k != "pq"))
p1, p2 = res.get(("pq", "r1")), res.get(("pq", "r2"))
if p1 and p2 and p1["tokens"] and p2["tokens"]:
    print("pq r1 vs r2 (run-to-run baseline):", "first diff %s, pos-agree %.3f" % agree(p1["tokens"], p2["tokens"]))
print("code checksums seen:", sorted(md5s), "(must be a single value)")
