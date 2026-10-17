"""Clean comparison summary: per config and round, plus the mean."""
import os, re, statistics
D = os.path.dirname(os.path.abspath(__file__))
CFGS = [("pq", "pq"), ("pq_off70", "paged_eviction"), ("pq_off70-lru", "lru"), ("pq_off70-rand", "random")]
CEIL_MB = 1255      # one pass over every token of the ~532 blocks offloaded by the end

def parse(tag):
    p = os.path.join(D, tag + ".log")
    if not os.path.exists(p):
        return None
    log = open(p).read()
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, log)) else None)
    r = {"tpot": g(r"decode-only TPOT: ([\d.]+)"), "ttft": g(r"TTFT: ([\d.]+)"),
         "new": g(r"new tokens: (\d+)"), "err": "Traceback" in log}
    fin = log[log.rfind("[OFFLOAD TIMING] FINAL"):] if "[OFFLOAD TIMING] FINAL" in log else ""
    st = {k: (float(v), int(c)) for k, v, c in re.findall(r"^(\S+)\s+([\d.]+)\s+(\d+)", fin, re.M)}
    r["disc"] = st.get("wrap.discover", (0, 0))[0] / (r["new"] or 1)
    r["fetch"] = st.get("wrap.fetch", (0, 0))[0] / (r["new"] or 1)
    m = re.search(r"\[bytes moved, MB\] (.*)", fin)
    r["mb"] = dict((k, float(v)) for k, v in re.findall(r"(\S+)=([\d.]+)", m.group(1))) if m else {}
    r["chunked"] = log.count("token store -> chunked")
    meta = os.path.join(D, tag + ".meta")
    r["meta"] = open(meta).read().split("\n") if os.path.exists(meta) else []
    return r

print("Qwen3-4B | LongBench 8k prompt + 4096 decode | 70% offload, W=256 | token staging store, 400 pages")
print("TPOT = decode-only (after warm-up). Fetched: one-pass ceiling ~%d MB.\n" % CEIL_MB)
print(f"{'config':<16}{'round':>6}{'TPOT ms':>9}{'TTFT ms':>9}{'disc ms/tok':>12}{'fetch ms/tok':>13}{'fetched MB':>12}  gpu_at_start / code")
means = {}
md5s = set()
for stem, name in CFGS:
    vals = []
    for rnd in ("r1", "r2"):
        r = parse(f"{stem}_{rnd}")
        if r is None:
            print(f"{name:<16}{rnd:>6}  (not run yet)"); continue
        if r["err"] or r["tpot"] is None:
            print(f"{name:<16}{rnd:>6}  {'ERROR' if r['err'] else '(running)'}"); continue
        vals.append(r["tpot"])
        meta = " ".join(x.split(" ", 1)[1] for x in r["meta"] if x)
        md5s.update(x.split()[1] for x in r["meta"] if x.startswith("code_md5"))
        print(f"{name:<16}{rnd:>6}{r['tpot']:>9.1f}{r['ttft'] or 0:>9.1f}{r['disc']:>12.2f}{r['fetch']:>13.2f}"
              f"{r['mb'].get('pool.d2h_token', 0):>12.0f}  {meta}" + (f"  chunked={r['chunked']}" if r['chunked'] else ""))
    if vals:
        means[name] = statistics.mean(vals)
print("\nmean TPOT:", "  ".join(f"{k} {v:.1f} ms" for k, v in means.items()))
if "pq" in means:
    print("vs pq:     ", "  ".join(f"{k} {100 * (v / means['pq'] - 1):+.1f}%" for k, v in means.items() if k != "pq"))
print("code checksums seen:", sorted(md5s), "(must be a single value for a clean comparison)")
