"""Forced decoding: FastKVzip vs random at 70% offload, per top-k."""
import glob, os, re, statistics
D = os.path.dirname(os.path.abspath(__file__))
NAMES = [("pq", "pq"), ("pq_tok70", "tok-fastkvzip"), ("pq_tokrand70", "tok-random")]
L = 36

def parse(p):
    log = open(p).read()
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, log)) else None)
    r = {"tpot": g(r"decode-only TPOT: ([\d.]+)"), "ttft": g(r"TTFT: ([\d.]+)"),
         "err": "Traceback" in log, "fallback": "FALLBACK" in log}
    m = re.search(r"KV blocks in use: peak (\d+), mean (\d+), final (\d+) \(\+(\d+) staging", log)
    r["kv"] = tuple(map(int, m.groups())) if m else (0, 0, 0, 0)
    fin = log[log.rfind("[OFFLOAD TIMING] FINAL"):] if "[OFFLOAD TIMING] FINAL" in log else ""
    m = re.search(r"\[bytes moved, MB\] .*?pool\.d2h_token=([\d.]+)", fin)
    r["gb"] = float(m.group(1)) / 1e3 if m else 0.0
    m = re.search(r"^wrap\.tok_chunked\s+[\d.]+\s+(\d+)", fin, re.M)
    r["chunked"] = int(m.group(1)) if m else 0
    m = re.search(r"^wrap\.discover\s+[\d.]+\s+(\d+)", fin, re.M)
    calls = int(m.group(1)) if m else 0
    c = dict(re.findall(r"(\S+)=(\d+)", fin[fin.find("[counts]"):].split("\n")[0])) if "[counts]" in fin else {}
    sel, cold, need = (int(c.get(k, 0)) for k in ("tok.sel_entries", "tok.sel_cold", "tok.need_keys"))
    r["cold_rate"] = cold / sel if sel else 0.0
    r["miss_per_ls"] = need / calls if calls else 0.0
    m = re.search(r"forced: steps (\d+), agree ([\d.]+)%, ref NLL ([\d.]+), first disagreement at step (-?\d+)", log)
    r["agree"], r["nll"], r["first"] = (float(m.group(2)), float(m.group(3)), int(m.group(4))) if m else (None, None, None)
    return r

def fmt(vals, f="{:.1f}"):
    if not vals:
        return "-"
    return f.format(statistics.mean(vals)) + (f" ±{(max(vals) - min(vals)) / 2:.1f}" if len(vals) > 1 else "")

for tk in sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(D, "k*_r?.log"))}):
    print(f"\n=== top-k {tk[1:]} | Qwen3-4B-Instruct-2507 | 8k + 4096 | 70% token offload, W=256 ===")
    print(f"{'config':<15}{'TPOT ms':>15}{'TTFT ms':>14}{'cold-sel %':>13}{'miss/layer-step':>16}"
          f"{'fetched GB':>13}{'chunked':>9}{'agree %':>9}{'ref NLL':>9}{'1st diff':>9}{'KV mean/final (+stage)':>24}  runs")
    for stem, name in NAMES:
        rs = [parse(p) for p in sorted(glob.glob(os.path.join(D, f"{tk}_{stem}_r*.log")))]
        rs = [r for r in rs if r["tpot"] is not None and not r["err"]]
        if not rs:
            print(f"{name:<15}  (not done)"); continue
        kv = rs[0]["kv"]
        tok = stem != "pq"
        print(f"{name:<15}{fmt([r['tpot'] for r in rs]):>15}{fmt([r['ttft'] for r in rs]):>14}"
              f"{(fmt([100 * r['cold_rate'] for r in rs]) if tok else '-'):>13}"
              f"{(fmt([r['miss_per_ls'] for r in rs]) if tok else '-'):>16}"
              f"{(fmt([r['gb'] for r in rs]) if tok else '-'):>13}"
              f"{(fmt([r['chunked'] for r in rs], '{:.0f}') if tok else '-'):>9}"
              f"{fmt([r['agree'] for r in rs if r['agree'] is not None], '{:.2f}'):>9}"
              f"{fmt([r['nll'] for r in rs if r['nll'] is not None], '{:.4f}'):>9}"
              f"{'/'.join(str(r['first']) for r in rs):>9}"
              f"{f'{kv[1]}/{kv[2]} (+{kv[3]})':>24}  {len(rs)}"
              + ("  FALLBACK!" if any(r["fallback"] for r in rs) else ""))
md5 = {open(p).read().split()[1] for p in glob.glob(os.path.join(D, "*.meta"))}
print("\ncode checksums:", sorted(md5))
