import json, os, re
D = os.path.dirname(os.path.abspath(__file__))
rows = []
base = None
for cfg in ["pq", "pq+off50", "pq+off50-lru", "pq+off50-rand"]:
    stem = cfg.replace("+", "_")
    log = open(os.path.join(D, stem + ".log")).read() if os.path.exists(os.path.join(D, stem + ".log")) else ""
    g = lambda pat: (m.group(1) if (m := re.search(pat, log)) else "-")
    toks = None
    p = os.path.join(D, f"tokens_{stem}.json")
    if os.path.exists(p):
        toks = json.load(open(p))
    if cfg == "pq":
        base = toks
    agree = "-"
    if toks is not None and base is not None and cfg != "pq":
        diff = next((i for i, (a, b) in enumerate(zip(base, toks)) if a != b), None)
        agree = "identical" if diff is None else f"first diff @{diff}"
    rows.append((cfg, g(r"prompt tokens: (\d+)"), g(r"new tokens: (\d+)"), g(r"total time: ([\d.]+)s"),
                 g(r"decode TPOT: ([\d.]+)"), g(r"throughput: ([\d.]+)"), g(r"peak GPU allocated: ([\d.]+)"), agree))
print(f"{'config':<16}{'prompt':>8}{'new':>6}{'time(s)':>9}{'TPOT(ms)':>10}{'tok/s':>8}{'peakGB':>8}  tokens vs pq")
for r in rows:
    print(f"{r[0]:<16}{r[1]:>8}{r[2]:>6}{r[3]:>9}{r[4]:>10}{r[5]:>8}{r[6]:>8}  {r[7]}")
