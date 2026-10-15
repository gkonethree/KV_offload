"""Record, per scorer, what gets offloaded and how selection hits residency.

One engine per scorer (pq+off50 / -lru / -rand), LongBench prompt, 2k decode,
W=512, auto staging. Hooks (monkeypatches, no package changes):
  * scheduler _trigger_request_offload -> offloaded block idx per trigger
  * staging pool ensure_resident_tokens -> tokens fetched per decode step
  * wrapper _discover_vec -> per layer: distinct selected tokens, split by
    residency (GPU / staged / CPU) before this layer's fetch
Writes <scorer>.json next to this script.
"""
import collections, json, os, sys
sys.path.insert(0, '/home/gaurav_kumar/KV_offload')
import runpy
mod = runpy.run_path('/home/gaurav_kumar/KV_offload/bench_1req_off99.py', run_name='bench')
import torch
from offload_package.integration.scheduler_connector import OffloadPackageScheduler as S
from offload_package.staging.pool import KVStagingPool as P
from offload_package.staging.residency import DEV_GPU, DEV_STAGING, DEV_CPU
import pq_optimized.batch_decode_with_paged_kv_cache_wrapper as W
OUT = os.path.dirname(os.path.abspath(__file__))
rec = {}

def reset():
    rec.clear()
    rec.update(step=0, offloads=[], fetched=collections.Counter(),
               sel=collections.Counter(), on_gpu=collections.Counter(),
               staged=collections.Counter(), on_cpu=collections.Counter(),
               need=collections.Counter())

_bcm = S.build_connector_meta
def bcm(self, so):
    rec["step"] += 1
    return _bcm(self, so)
S.build_connector_meta = bcm

_tro = S._trigger_request_offload
def tro(self, req_id):
    n0 = len(self._step_store_acc)
    _tro(self, req_id)
    blocks = sorted(r.block_idx for e in self._step_store_acc[n0:] for r in e[0])
    if blocks:
        rec["offloads"].append({"step": rec["step"], "req": req_id, "blocks": blocks})
S._trigger_request_offload = tro

_ert = P.ensure_resident_tokens
def ert(self, layer, requests, protect=None, device_protect=True):
    rec["fetched"][rec["step"]] += sum(len(o) for _, o in requests)
    return _ert(self, layer, requests, protect=protect, device_protect=device_protect)
P.ensure_resident_tokens = ert

_dv = W.BatchDecodeWithPagedKVCacheWrapper._discover_vec
def dv(self, sparse_idx, page_size, request_ids, layer, res, indptr_l, slots_t):
    out = _dv(self, sparse_idx, page_size, request_ids, layer, res, indptr_l, slots_t)
    st_ = rec["step"]
    for b in range(sparse_idx.shape[0]):
        u = torch.unique(sparse_idx[b].reshape(-1))
        s = res.dev_state[slots_t[b], torch.div(u, int(page_size), rounding_mode="floor")]
        rec["sel"][st_] += int(u.numel())
        rec["on_gpu"][st_] += int((s == DEV_GPU).sum())
        rec["staged"][st_] += int((s == DEV_STAGING).sum())
        rec["on_cpu"][st_] += int((s == DEV_CPU).sum())
    if out is not None and layer is not None:
        rec["need"][st_] += sum(len(o) for _, o in out[0]) + sum(len(o) for _, o in out[1])
    return out
W.BatchDecodeWithPagedKVCacheWrapper._discover_vec = dv

for cfg in os.environ.get("SCORERS", "pq+off50,pq+off50-lru,pq+off50-rand").split(","):
    reset()
    mod['run'](cfg, **mod['CONFIGS'][cfg])
    name = {"pq+off50": "paged_eviction", "pq+off50-lru": "lru", "pq+off50-rand": "random"}[cfg]
    json.dump({k: (dict(v) if isinstance(v, collections.Counter) else v)
               for k, v in rec.items()}, open(f"{OUT}/{name}.json", "w"))
    print(f"RECORDED {name}: steps={rec['step']} triggers={len(rec['offloads'])} "
          f"fetched_tokens={sum(rec['fetched'].values())}", flush=True)
