#!/usr/bin/env python3
"""Compare block-granular vs per-layer staging on one request (PQ sparse decode).

  --config pq      no offloading (reference output)
  --config block   offload, KVStagingPool  (stage a block for all layers)
  --config layer   offload, PerLayerStagingPool (stage only the selecting layer)

A 2-token warm-up request fits the PQ codebooks first (one-time, ~minutes on a
shared GPU); the timed request is a fresh one, so the measurement is decode
plus prefill only. Writes a JSON with tokens, timing and fetch counters.

Usage: PYTHONPATH=... VLLM_PLUGINS= python analysis/bench_staging.py \
         --config layer --slots 64 --new-tokens 512 --out /tmp/x.json
"""
import argparse
import json
import os
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ.setdefault('SKYLIGHT_OFFLOAD_TIMING', '1')   # per-stage breakdown

ap = argparse.ArgumentParser()
ap.add_argument('--config', choices=['pq', 'block', 'layer'], required=True)
ap.add_argument('--slots', type=int, default=64)
ap.add_argument('--new-tokens', type=int, default=512)
ap.add_argument('--prompt-reps', type=int, default=40)
ap.add_argument('--max-model-len', type=int, default=4096)
ap.add_argument('--fraction', type=float, default=0.99)
ap.add_argument('--interval', type=int, default=128)
ap.add_argument('--repeats', type=int, default=1,
                help='timed requests after one warm-up (median is reported)')
ap.add_argument('--out', required=True)
args = ap.parse_args()

os.environ.update({
    'SKYLIGHT_SPARSE_METHOD': 'pq',
    'SKYLIGHT_SPARSE_TOPK': '0.1',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
    'SKYLIGHT_PQ_GQA_SHARE': '1',
})

import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'skylight.backend.SkylightSparseBackend')
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

_orig_get = getattr(cuda_module, '_get_attn_backend_class', None)


def _patched_get(backend, *a, **k):
    if backend is AttentionBackendEnum.CUSTOM or (
        backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES
    ):
        backend = AttentionBackendEnum.CUSTOM
    return _orig_get(backend, *a, **k) if _orig_get else backend


if _orig_get:
    cuda_module._get_attn_backend_class = _patched_get

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

import offload_package
from offload_package import timing as _timing
from offload_package.staging.residency import Residency, ResidencyTable
import skylight
import pq_optimized
from offload_package.staging.pool import KVStagingPool
from pq_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as PQWrapper,
)

print(f"[BENCH] offload_package from {offload_package.__file__}", flush=True)
print(f"[BENCH] skylight from {skylight.__file__}", flush=True)
print(f"[BENCH] pq_optimized from {pq_optimized.__file__}", flush=True)

# ---- counters (wrapping, so the block-mode code itself is untouched) --------
C = dict(block_fetch_calls=0, block_blocks=0, chunk_merges=0)
_orig_acq = KVStagingPool._acquire_slab


def _acq(self, n):
    C['block_fetch_calls'] += 1
    C['block_blocks'] += int(n)
    return _orig_acq(self, n)


KVStagingPool._acquire_slab = _acq
_orig_merge = PQWrapper.__dict__['_lse2_merge']
_merge_fn = _orig_merge.__func__ if isinstance(_orig_merge, staticmethod) else _orig_merge


def _merge(*a, **k):
    C['chunk_merges'] += 1
    return _merge_fn(*a, **k)


PQWrapper._lse2_merge = staticmethod(_merge) if isinstance(_orig_merge, staticmethod) else _merge

from offload_package.staging.per_layer_pool import PerLayerStagingPool  # noqa: E402
POOLS = []
_orig_pl_init = PerLayerStagingPool.__init__


def _pl_init(self, *a, **k):
    _orig_pl_init(self, *a, **k)
    POOLS.append(self)


PerLayerStagingPool.__init__ = _pl_init

# ---- KV memory sampling: block counts by residency, once per engine step ------
TABLES = []
_orig_rt_init = ResidencyTable.__init__


def _rt_init(self, *a, **k):
    _orig_rt_init(self, *a, **k)
    TABLES.append(self)


ResidencyTable.__init__ = _rt_init
MEM = dict(peak_total=0, peak_gpu=0, peak_cpu=0, steps=0, sum_gpu=0, sum_cpu=0)
_orig_step = _timing.step


def _step():
    _orig_step()
    for t in TABLES:
        g = c = 0
        for loc in t._blocks.values():
            if loc.residency == Residency.GPU:
                g += 1
            elif loc.residency == Residency.CPU:
                c += 1          # STAGING copies count as neither: they live in the pool
        if g + c == 0:
            continue
        MEM['steps'] += 1
        MEM['sum_gpu'] += g
        MEM['sum_cpu'] += c
        MEM['peak_gpu'] = max(MEM['peak_gpu'], g)
        MEM['peak_cpu'] = max(MEM['peak_cpu'], c)
        MEM['peak_total'] = max(MEM['peak_total'], len(t._blocks))
        break


_timing.step = _step

PROMPT = ('The history of artificial intelligence is a story of ambition, '
          'setbacks, and breakthroughs across decades of research. ') * args.prompt_reps


def main():
    kv = None
    if args.config != 'pq':
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True, 'offload_fraction': args.fraction, 'num_cpu_slots': 4096,
                'staging_policy': 'fixed', 'num_staging_slots': args.slots,
                'interval_tokens': args.interval, 'true_free': True,
                'scorer_type': 'paged_eviction',
                'staging_granularity': 'layer' if args.config == 'layer' else 'block',
            },
            kv_role='kv_both',
        )
    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16',
              max_model_len=args.max_model_len, gpu_memory_utilization=0.20,
              enforce_eager=True, max_num_seqs=1, disable_log_stats=True,
              attention_config={'backend': 'CUSTOM'}, async_scheduling=False,
              **({'kv_transfer_config': kv} if kv is not None else {}))

    t0 = time.time()
    llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=2, ignore_eos=True))
    warm_s = time.time() - t0
    print(f"[BENCH] warm-up (codebook fit) {warm_s:.1f}s", flush=True)

    pools = POOLS
    print(f"[BENCH] per-layer pools tracked: {len(pools)}", flush=True)
    runs = []
    out = None
    for rep in range(args.repeats):
        for k in C:
            C[k] = 0
        for p in pools:
            for k in p.stats:
                p.stats[k] = 0
        for k in MEM:
            MEM[k] = 0
        _timing._total.clear(); _timing._count.clear(); _timing._max.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        out = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=args.new_tokens,
                                                    ignore_eos=True))[0]
        torch.cuda.synchronize()
        el = time.time() - t0
        n = len(out.outputs[0].token_ids)
        r = dict(time_s=el, tpot_ms=el / max(n, 1) * 1e3, new_tokens=n,
                 counters=dict(C), mem=dict(MEM),
                 timing={k: dict(total_ms=_timing._total[k], calls=_timing._count[k],
                                 max_ms=_timing._max[k]) for k in _timing._total},
                 tokens=list(out.outputs[0].token_ids))
        if pools:
            r['layer_pool'] = dict(pools[0].stats)
        runs.append(r)
        print(f"[BENCH] rep {rep}: {args.config} slots={args.slots} frac={args.fraction}: "
              f"{n} tokens in {el:.1f}s -> {r['tpot_ms']:.2f} ms/token  "
              f"counters {r['counters']} pool {r.get('layer_pool')} mem {MEM}", flush=True)

    tp = sorted(r['tpot_ms'] for r in runs)
    res = dict(config=args.config, slots=args.slots, fraction=args.fraction,
               max_model_len=args.max_model_len, prompt_reps=args.prompt_reps,
               new_tokens=runs[-1]['new_tokens'], prompt_tokens=len(out.prompt_token_ids),
               warmup_s=warm_s, tpot_ms=tp[len(tp) // 2], tpot_all=[r['tpot_ms'] for r in runs],
               time_s=runs[-1]['time_s'], tokens=runs[-1]['tokens'],
               tokens_stable=all(r['tokens'] == runs[0]['tokens'] for r in runs),
               text=out.outputs[0].text, counters=runs[-1]['counters'], runs=runs)
    if pools:
        res['layer_pool'] = runs[-1].get('layer_pool')
        p = pools[0]
        res['kv'] = dict(layers=p.num_layers, page_bytes_per_layer=int(
            p.gpu_k_caches[0][0].numel() * p.gpu_k_caches[0].element_size()))
    print(f"[BENCH] {args.config} slots={args.slots} frac={args.fraction}: median "
          f"{res['tpot_ms']:.2f} ms/token over {args.repeats} (all {res['tpot_all']})", flush=True)
    print(f"[BENCH] out: {out.outputs[0].text[:120]!r}", flush=True)
    with open(args.out, 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    main()
