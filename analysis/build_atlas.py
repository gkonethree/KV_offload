#!/usr/bin/env python3
"""Embed analyze_selection.py outputs into the Selection Atlas page.

Usage: python analysis/build_atlas.py OUT.html name=analysis.json[+sim.json] [...]
"""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
out_path, pairs = sys.argv[1], sys.argv[2:]
data = {}
for p in pairs:
    name, paths = p.split('=', 1)
    an_path, _, sim_path = paths.partition('+')
    with open(an_path) as f:
        data[name] = json.load(f)
    if sim_path:
        with open(sim_path) as f:
            data[name]['sim'] = json.load(f)
blob = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
with open(os.path.join(here, 'selection_atlas.template.html')) as f:
    page = f.read().replace('__DATA__', blob)
with open(out_path, 'w') as f:
    f.write(page)
print(f"wrote {out_path} ({len(page) / 1e6:.1f} MB)")
