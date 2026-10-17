#!/usr/bin/env python3
"""Two-request composition-change bench: pq vs pq+offload.

One request finishes mid-decode while the other continues, which changes the
batch composition between steps. This is the scenario where the one-step-stale
request-ids used to make both fetch paths bail for a step (reading freed GPU
pages with weight 1) before the F1 fix. Correctness: greedy decode; the two
configs must match token-for-token PER REQUEST.

Usage: .venv/bin/python bench_2req_composition.py [config,config,...]
"""
import os
import gc
import sys
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'skylight.backend.SkylightSparseBackend')
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

_orig = getattr(cuda_module, '_get_attn_backend_class', None)


def _patched_get(backend, *a, **k):
    if backend is AttentionBackendEnum.CUSTOM or (
        backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES
    ):
        backend = AttentionBackendEnum.CUSTOM
    if _orig:
        return _orig(backend, *a, **k)
    return backend


if _orig:
    cuda_module._get_attn_backend_class = _patched_get

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
MAX_MODEL_LEN = 8300
OFFLOAD_FRACTION = 0.99
INTERVAL_TOKENS = 32

PROMPT_A = ('The history of artificial intelligence is a story of ambition, '
            'setbacks, and breakthroughs across decades of research. ') * 10
PROMPT_B = ('Quantum computing promises to reshape cryptography, optimization, '
            'and molecular simulation once fault-tolerant hardware arrives. ') * 10

# Staggered decode budgets: B finishes at step 40, A keeps decoding to 96.
MAX_TOKENS_A = 96
MAX_TOKENS_B = 40

SPARSE_ENV = {
    'SKYLIGHT_SPARSE_TOPK': '0.3',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
    'SKYLIGHT_PQ_GQA_SHARE': '1',   # operating mode: selection per kv-head
}

_TOK = dict(offload_granularity='token', token_scorer='random', offload_fraction=0.7)
CONFIGS = {
    'pq': dict(method='pq', offload=False),
    'pq+offload': dict(method='pq', offload=True),
    # Token-level offloading under batching (prefix caching off for all three).
    'pq-nopc': dict(method='pq', offload=False, nopc=True),
    'pq+tok': dict(method='pq', offload=True, nopc=True, extra=dict(_TOK)),
    'pq+tokh': dict(method='pq', offload=True, nopc=True,
                    extra=dict(_TOK, token_budget_scope='layer')),
}


def run(name, method, offload, nopc=False, extra=None):
    print(f"\n{'=' * 70}\nCONFIG: {name}\n{'=' * 70}", flush=True)
    for k, v in SPARSE_ENV.items():
        os.environ[k] = v
    os.environ['SKYLIGHT_SPARSE_METHOD'] = method
    if method == 'pq':
        # Skylight upstream (ak/pq): 'pq' is the pure-torch reference; the
        # production path is 'pq_optimized' with its slot-code maintainer,
        # driven on the FULL-cg slot path (eager here) with top-k sized from
        # the live context.
        os.environ['SKYLIGHT_SPARSE_METHOD'] = 'pq_optimized'
        os.environ['SKYLIGHT_INCR_FULLCG'] = '1'
        os.environ['SKYLIGHT_INCR_SLOT'] = '1'
        os.environ['SKYLIGHT_SPARSE_DYNAMIC_TOPK'] = '1'

    kv = None
    if offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True,
                'offload_fraction': OFFLOAD_FRACTION,
                'num_cpu_slots': 4096,
                'staging_policy': 'auto',
                'interval_tokens': INTERVAL_TOKENS,
                'true_free': True,
                **(extra or {}),
            },
            kv_role='kv_both',
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    kwargs = dict(
        model=MODEL,
        dtype='float16',
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.20,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=2,
        disable_log_stats=True,
        attention_config={'backend': 'CUSTOM'},
        async_scheduling=False,
        enable_prefix_caching=not nopc,
    )
    if kv is not None:
        kwargs['kv_transfer_config'] = kv

    t_init0 = time.time()
    llm = LLM(**kwargs)
    t_init = time.time() - t_init0

    sp = [
        SamplingParams(temperature=0, top_p=1.0, max_tokens=MAX_TOKENS_A, ignore_eos=True),
        SamplingParams(temperature=0, top_p=1.0, max_tokens=MAX_TOKENS_B, ignore_eos=True),
    ]
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate([PROMPT_A, PROMPT_B], sp)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  init: {t_init:.1f}s")
    print(f"  prompt tokens: {[len(o.prompt_token_ids) for o in outputs]}")
    print(f"  new tokens: {[len(o.outputs[0].token_ids) for o in outputs]}")
    print(f"  total time: {elapsed:.1f}s ({total_out} out tokens)")
    print(f"  out[0]: {outputs[0].outputs[0].text[:60]!r}")
    print(f"  out[1]: {outputs[1].outputs[0].text[:60]!r}")

    result = {
        'name': name, 'time_s': elapsed,
        'texts': [o.outputs[0].text for o in outputs],
        'tokens': [list(o.outputs[0].token_ids) for o in outputs],
    }
    # vLLM gc.freeze()s the heap at engine start: `del llm` alone frees
    # nothing; EngineCore.shutdown() unfreezes and shuts the connector down.
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:
        print(f"  engine shutdown failed ({e}); falling back to gc.unfreeze", flush=True)
    gc.unfreeze()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    return result


if __name__ == '__main__':
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else list(CONFIGS)
    results = {}
    for name in names:
        results[name] = run(name, **CONFIGS[name])

    print(f"\n{'=' * 92}")
    base_name = 'pq' if 'pq' in results else 'pq-nopc'
    print(f"CORRECTNESS (per-request greedy token-for-token vs {base_name!r}):")
    base = results.get(base_name)
    ok = True
    if base is None:
        print("  pq baseline not run; skipping comparison")
    else:
        for name, r in results.items():
            if name == base_name:
                continue
            for i, (bt, ot) in enumerate(zip(base['tokens'], r['tokens'])):
                # Compare over the common prefix length (B finishes earlier).
                n = min(len(bt), len(ot))
                match = bt[:n] == ot[:n] and len(ot) >= n
                tag = f"{name}[{i}]"
                if not match:
                    for j, (x, y) in enumerate(zip(bt[:n], ot[:n])):
                        if x != y:
                            print(f"  {tag}: diverges at token {j}: "
                                  f"{base['texts'][i][:60]!r} vs {r['texts'][i][:60]!r}")
                            break
                    else:
                        print(f"  {tag}: length mismatch {len(bt)} vs {len(ot)}")
                print(f"  {tag}: {'PASS' if match else 'FAIL'}")
                ok = ok and match
    print(f"  overall: {'PASS' if ok else 'FAIL'}")
