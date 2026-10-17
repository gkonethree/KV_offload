"""Paged eviction vs LRU at 70% offload: TPOT, offload cost breakdown, and
fetch traffic (re-fetches show up as token traffic above the one-pass
ceiling once staging is smaller than the offloaded set)."""
import math, os, re
D = os.path.dirname(os.path.abspath(__file__))
PAGE, LAYERS, TOK_BYTES = 16, 36, 8 * 128 * 2 * 2      # Qwen3-4B, bf16 K+V
STAGES = ("wrap.discover", "wrap.fetch", "pool.alloc", "sched.offload", "build.patch")
CALLS = ("wrap.discover", "wrap.fetch")

def auto_slots(max_len, f=0.7, topk=0.1, g=8, sink=64, local=64):
    mid_pages = -(-(max_len - sink - local) // PAGE)
    union = 1 - (1 - topk) ** g
    return max(64, int(min(math.ceil(f * union * mid_pages), math.ceil(f * mid_pages)) * 1.10))

def fetch_ratio(r):
    """wrap.fetch calls per layer-step that fetched anything is not logged
    directly; report fetch calls / discovery calls (>1 = chunked fallback)."""
    d, f = r.get("calls", {}).get("wrap.discover"), r.get("calls", {}).get("wrap.fetch")
    return f"{f / d:.2f}" if d and f else "-"

def parse(tag):
    p = os.path.join(D, tag + ".log")
    if not os.path.exists(p):
        return None
    log = open(p).read()
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, log)) else None)
    r = {"tpot": g(r"decode-only TPOT: ([\d.]+)"), "ttft": g(r"TTFT: ([\d.]+)"),
         "prompt": g(r"prompt tokens: (\d+)"), "new": g(r"new tokens: (\d+)"),
         "err": "Traceback" in log, "stages": {}, "bytes": {}}
    fin = log.rfind("[OFFLOAD TIMING] FINAL")
    if fin >= 0:
        blk = log[fin:]
        r["calls"] = {}
        for st in STAGES:
            m = re.search(rf"^{re.escape(st)}\s+([\d.]+)\s+(\d+)", blk, re.M)
            if m:
                r["stages"][st] = float(m.group(1))
                r["calls"][st] = int(m.group(2))
        m = re.search(r"\[bytes moved, MB\] (.*)", blk)
        if m:
            r["bytes"] = {k: float(v) for k, v in re.findall(r"(\S+)=([\d.]+)", m.group(1))}
    return r

print("TPOT = decode-only, after a warm-up request (PQ codebook fit excluded).")
print("ceiling = one-pass traffic if every token of every block offloaded by the end were fetched once.\n")
print(f"{'run':<13}{'TPOT ms':>9}{'TTFT s':>8}{'discover':>10}{'fetch':>7}{'sched':>7} (ms/tok)"
      f"{'staging':>9}{'offl.blk':>9}{'tokMB':>8}{'ceilMB':>8}{'blkMB':>7}{'chunks/fetch':>13}")
STAGING = {"8k4k": 400, "16k8k": 355}   # fixed staging slots per run
for ctx, max_len, prompt_new in (("8k4k", 18000, (8192, 4096)), ("16k8k", 25600, (16384, 8192))):
    for sc in ("paged", "lru"):
        tag = f"{ctx}_{sc}"; r = parse(tag)
        if r is None:
            print(f"{tag:<13} (not run yet)"); continue
        if r["err"]:
            print(f"{tag:<13} ERROR -- see {tag}.log"); continue
        if r["tpot"] is None:
            print(f"{tag:<13} (running)"); continue
        n = r["new"] or 1
        per = lambda k: r["stages"].get(k, 0.0) / n
        total_blocks = (prompt_new[0] + prompt_new[1]) // PAGE
        pinned = 64 // PAGE + -(-64 // PAGE) + 1          # sink + local + tail
        offl = math.ceil(0.7 * (total_blocks - pinned))
        ceil_mb = offl * PAGE * LAYERS * TOK_BYTES / 1e6
        print(f"{tag:<13}{r['tpot']:>9.1f}{(r['ttft'] or 0)/1000:>8.2f}{per('wrap.discover'):>10.2f}"
              f"{per('wrap.fetch'):>7.2f}{per('sched.offload'):>7.2f}{'':9}"
              f"{STAGING[ctx]:>9}{offl:>9}{r['bytes'].get('pool.d2h_token', 0):>8.0f}"
              f"{ceil_mb:>8.0f}{r['bytes'].get('pool.d2h_block', 0):>7.0f}"
              f"{fetch_ratio(r):>13}")
print("\ntokMB above ceilMB = re-fetches after staging evictions (scorer quality shows up here).")
print("chunks/fetch > 1 = per-token chunked fallback fired (a layer's cold set exceeded staging).")
