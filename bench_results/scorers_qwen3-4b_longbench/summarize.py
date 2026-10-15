"""Summarize the LongBench scorer sweep: TPOT, offload time breakdown per
decoded token, fetch traffic, and token agreement with pq."""
import json, os, re
D = os.path.dirname(os.path.abspath(__file__))
STAGES = ["wrap.discover", "wrap.fetch", "pool.alloc", "pool.cpu_gather",
          "pool.transform", "pool.h2d", "pool.h2d_sync", "build.patch",
          "sched.offload", "sched.score", "sched.surgery", "lru.note"]

def parse(tag):
    p = os.path.join(D, tag + ".log")
    if not os.path.exists(p):
        return None
    log = open(p).read()
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, log)) else None)
    r = {"tpot": g(r"decode-only TPOT: ([\d.]+)"), "ttft": g(r"TTFT: ([\d.]+)"),
         "time": g(r"total time: ([\d.]+)s"),
         "new": g(r"new tokens: (\d+)"), "err": bool(re.search(r"Traceback", log))}
    fin = log.rfind("[OFFLOAD TIMING] FINAL")
    r["stages"], r["bytes"] = {}, {}
    if fin >= 0:
        blk = log[fin:]
        for st in STAGES:
            m = re.search(rf"^{re.escape(st)}\s+([\d.]+)\s+(\d+)", blk, re.M)
            if m:
                r["stages"][st] = float(m.group(1))
        m = re.search(r"\[bytes moved, MB\] (.*)", blk)
        if m:
            r["bytes"] = {k: float(v) for k, v in re.findall(r"(\S+)=([\d.]+)", m.group(1))}
    tp = os.path.join(D, f"tokens_{tag}.json")
    try:
        r["tokens"] = json.load(open(tp))
    except (OSError, ValueError):
        r["tokens"] = None
    return r

base = parse("pq_stauto")
runs = [("pq", "auto")] + [(c, s) for s in ("auto", "709")
                          for c in ("pq+off50", "pq+off50-lru", "pq+off50-rand")]
print("TPOT = decode-only (first->last token, after a warm-up request that pays the PQ fit).")
print(f"{'config':<15}{'staging':>8}{'TPOT ms':>9}{'Δ vs pq':>9}{'TTFT s':>8}  "
      f"{'discover':>9}{'fetch':>7}{'alloc':>7}{'sched':>7} (ms/token)  "
      f"{'tokMB':>8}{'blkMB':>8}  tokens vs pq")
for cfg, st in runs:
    r = parse(f"{cfg.replace('+', '_')}_st{st}")
    if r is None:
        print(f"{cfg:<15}{st:>8}  (not run yet)"); continue
    if r["err"]:
        print(f"{cfg:<15}{st:>8}  ERROR -- see log"); continue
    if r["tpot"] is None:
        print(f"{cfg:<15}{st:>8}  (running)"); continue
    n = r["new"] or 1
    per = lambda k: r["stages"].get(k, 0.0) / n
    sched = per("sched.offload") + per("sched.score") + per("sched.surgery")
    d = (f"{(r['tpot'] / base['tpot'] - 1) * 100:+.1f}%"
         if base and base["tpot"] and cfg != "pq" else "")
    agree = ""
    if cfg != "pq" and base and base["tokens"] and r["tokens"]:
        i = next((i for i, (a, b) in enumerate(zip(base["tokens"], r["tokens"])) if a != b), None)
        agree = "identical" if i is None else f"diff @{i}"
    ttft = f"{r['ttft'] / 1000:.1f}" if r.get("ttft") else "-"
    print(f"{cfg:<15}{st:>8}{r['tpot']:>9.1f}{d:>9}{ttft:>8}  "
          f"{per('wrap.discover'):>9.2f}{per('wrap.fetch'):>7.2f}{per('pool.alloc'):>7.2f}{sched:>7.2f}"
          f"{'':12}{r['bytes'].get('pool.d2h_token', 0):>8.0f}{r['bytes'].get('pool.d2h_block', 0):>8.0f}  {agree}")
print("\nblkMB > 0 means the chunked whole-block fallback fired (per-layer cold set > staging).")
print("Token differences vs pq are expected noise (pq itself is not run-to-run deterministic).")
