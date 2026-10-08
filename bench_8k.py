#!/usr/bin/env python3
"""8k-context bench: {oracle, pq} x {+offload, -offload}.

Offload at 50% of GPU blocks, staging pool auto-sized (staging_policy='auto').
Correctness: greedy decode (temperature=0); we require
  oracle == oracle+offload and pq == pq+offload  (token-for-token),
cross-family (oracle vs pq) divergence is expected (different selections).

Note on the PQ path: the sparse-attention scorer uses PQ codes computed on
GPU-resident codes — attention scores over KV blocks are never computed for
selection; the offload orchestrator fetches purely by residency lookup.

Usage: .venv/bin/python bench_8k.py [config,config,...]
"""
import os
import gc
import sys
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
# Pick a free GPU; override externally if needed.
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
MAX_MODEL_LEN = 17000            # ~8.1k input + 8k out headroom
MAX_NEW_TOKENS = 1024
OFFLOAD_FRACTION = 0.5
INTERVAL_TOKENS = 32

# ~26 tokens per repetition -> 310 reps ~ 8.1k tokens per prompt.
PROMPT_A = ('The history of artificial intelligence is a story of ambition, '
            'setbacks, and breakthroughs across decades of research. ') 


SPARSE_ENV = {
    'SKYLIGHT_SPARSE_TOPK': '0.5',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
}

CONFIGS = {
    'oracle': dict(method='oracle', offload=False),
    'oracle+offload': dict(method='oracle', offload=True),
    'pq': dict(method='pq', offload=False),
    'pq+offload': dict(method='pq', offload=True),
}


def run(name, method, offload):
    print(f"\n{'=' * 70}\nCONFIG: {name}\n{'=' * 70}", flush=True)
    for k, v in SPARSE_ENV.items():
        os.environ[k] = v
    os.environ['SKYLIGHT_SPARSE_METHOD'] = method

    kv = None
    if offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True,
                'offload_fraction': OFFLOAD_FRACTION,
                'num_cpu_slots': 4096,
                'staging_policy': 'auto',   # auto-sized to hot selected set
                'interval_tokens': INTERVAL_TOKENS,
                'true_free': True,
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
        enforce_eager=os.environ.get("BENCH_GRAPHS", "0") != "1",
        tensor_parallel_size=1,
        max_num_seqs=8,
        disable_log_stats=True,
        attention_config={'backend': 'CUSTOM'},
        async_scheduling=False,
    )
    if kv is not None:
        kwargs['kv_transfer_config'] = kv

    t_init0 = time.time()
    llm = LLM(**kwargs)
    t_init = time.time() - t_init0

    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate(PROMPT_A, sp)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    in_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tpot_ms = elapsed / max(out_tokens, 1) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"  init: {t_init:.1f}s")
    print(f"  prompt tokens: {in_tokens} ({in_tokens // 2}/prompt)")
    print(f"  new tokens: {out_tokens}")
    print(f"  total time: {elapsed:.1f}s")
    print(f"  decode TPOT: {tpot_ms:.2f} ms/token")
    print(f"  throughput: {out_tokens / max(elapsed, 1e-9):.1f} tok/s")
    print(f"  peak GPU allocated: {peak_gb:.2f} GB")
    for i, o in enumerate(outputs):
        print(f"  out[{i}]: {o.outputs[0].text[:70]!r}")

    result = {
        'name': name, 'prompt_tokens': in_tokens, 'new_tokens': out_tokens,
        'time_s': elapsed, 'tpot_ms': tpot_ms, 'tps': out_tokens / max(elapsed, 1e-9),
        'peak_gb': peak_gb, 'init_s': t_init,
        'texts': [o.outputs[0].text for o in outputs],
    }
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

    rs = list(results.values())
    print(f"\n{'=' * 92}")
    print(f"SUMMARY 8k-context (input={rs[0]['prompt_tokens']}, +{MAX_NEW_TOKENS} out, 2 prompts)")
    print(f"{'=' * 92}")
    print(f"{'config':<18} {'time(s)':>8} {'TPOT(ms)':>9} {'tok/s':>8} {'peak GPU(GB)':>13} {'init(s)':>8}")
    for r in rs:
        print(f"{r['name']:<18} {r['time_s']:>8.1f} {r['tpot_ms']:>9.2f} {r['tps']:>8.1f} "
              f"{r['peak_gb']:>13.2f} {r['init_s']:>8.1f}")

    # Offload must be observationally identical to its non-offload counterpart.
    print("\nCORRECTNESS (greedy token-for-token):")
    ok = True
    for fam in ('oracle', 'pq'):
        if fam in results and f'{fam}+offload' in results:
            match = results[fam]['texts'] == results[f'{fam}+offload']['texts']
            for i, (a, b) in enumerate(zip(results[fam]['texts'], results[f'{fam}+offload']['texts'])):
                if a != b:
                    for j, (ca, cb) in enumerate(zip(a, b)):
                        if ca != cb:
                            print(f"  {fam}: prompt[{i}] diverges at char {j}: {a[j-20:j]!r} | {b[j-20:j]!r}")
                            break
            print(f"  {fam}: {'PASS' if match else 'FAIL'}")
            ok = ok and match
    print(f"  overall: {'PASS' if ok else 'FAIL'}")
