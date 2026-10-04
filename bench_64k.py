#!/usr/bin/env python3
"""64k-context benchmark: sparse selection x offloading matrix.

Configs (in-process, sequentially):
  oracle+offload   oracle topk selection, offload 50% / W=128 decode tokens + post-prefill
  pq+offload       PQ-code selection, same offload schedule
  oracle           oracle selection, no offloading (no connector)
  pq               PQ selection, no offloading

Metrics: wall time, decode TPOT (tok/s), peak allocated GPU memory, prompt length,
freed-block evidence (from [SCHED DEBUG] lines when offload is on).

Usage: .venv/bin/python bench_64k.py [config,config,...]
"""
import os
import gc
import sys
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
# Pick a free GPU (0 is typically busy on this box); override by setting it externally.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

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
MAX_MODEL_LEN = 66000          # >64k input + headroom
TARGET_INPUT_TOKENS = 64500    # ~64k input
MAX_NEW_TOKENS = 256
OFFLOAD_FRACTION = 0.5
INTERVAL_TOKENS = 128

# Two distinct long prompts (distinct content keeps prefix sharing from
# trivializing the offload path).
PROMPT_A = ('The history of artificial intelligence is a story of ambition, '
            'setbacks, and breakthroughs across decades of research. ') * 2300
PROMPT_B = ('Quantum computing promises to reshape cryptography, optimization, '
            'and molecular simulation once fault-tolerant hardware arrives. ') * 2300

SPARSE_ENV = {
    'SKYLIGHT_SPARSE_TOPK': '0.1',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
}


def run(name, method, offload):
    print(f"\n{'=' * 70}\nCONFIG: {name}\n{'=' * 70}", flush=True)
    for k, v in SPARSE_ENV.items():
        os.environ[k] = v
    os.environ['SKYLIGHT_SPARSE_METHOD'] = method
    # Offload evidence (True-freed / ORCH PROMOTE lines) lands in the log.
    os.environ['OFFLOAD_PACKAGE_DEBUG'] = '1' if offload else '0'

    kv = None
    if offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True,
                'offload_fraction': OFFLOAD_FRACTION,
                'num_cpu_slots': 8192,
                'num_staging_slots': 64,
                'interval_tokens': INTERVAL_TOKENS,
                'scorer_decay': 1.0,
                'scorer_head_reduction': 'mean',
                'true_free': True,
                'promote_per_step': 4,
            },
            kv_role='kv_both',
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free0, total = torch.cuda.mem_get_info()
    gpu_used_before = (total - free0) / 1e9

    kwargs = dict(
        model=MODEL,
        dtype='float16',
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.80,
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
    outputs = llm.generate([PROMPT_A, PROMPT_B], sp)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    in_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tpot_ms = (elapsed - 0) / max(out_tokens, 1) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    free1, _ = torch.cuda.mem_get_info()
    gpu_used_after = (total - free1) / 1e9

    print(f"  init: {t_init:.1f}s")
    print(f"  prompt tokens: {in_tokens} ({in_tokens // 2}/prompt)")
    print(f"  new tokens: {out_tokens}")
    print(f"  total time: {elapsed:.1f}s")
    print(f"  decode TPOT: {tpot_ms:.1f} ms/token")
    print(f"  throughput: {out_tokens / max(elapsed, 1e-9):.1f} tok/s")
    print(f"  peak GPU allocated: {peak_gb:.2f} GB")
    print(f"  sample out: {outputs[0].outputs[0].text[:60]!r}")

    result = {
        'name': name, 'prompt_tokens': in_tokens, 'new_tokens': out_tokens,
        'time_s': elapsed, 'tpot_ms': tpot_ms, 'tps': out_tokens / max(elapsed, 1e-9),
        'peak_gb': peak_gb, 'init_s': t_init,
    }
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    return result


CONFIGS = {
    'oracle+offload': dict(method='oracle', offload=True),
    'pq+offload': dict(method='pq', offload=True),
    'pqis+offload': dict(method='pqis', offload=True),
    'oracle': dict(method='oracle', offload=False),
    'pqis': dict(method='pqis', offload=False),
    'pq': dict(method='pq', offload=False),
}

if __name__ == '__main__':
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else list(CONFIGS)
    results = []
    for name in names:
        results.append(run(name, **CONFIGS[name]))

    print(f"\n{'=' * 92}")
    print(f"SUMMARY 64k-context (input={results[0]['prompt_tokens']}, +{MAX_NEW_TOKENS} out, 2 prompts)")
    print(f"{'=' * 92}")
    print(f"{'config':<18} {'time(s)':>8} {'TPOT(ms)':>9} {'tok/s':>8} {'peak GPU(GB)':>13} {'init(s)':>8}")
    for r in results:
        print(f"{r['name']:<18} {r['time_s']:>8.1f} {r['tpot_ms']:>9.1f} {r['tps']:>8.1f} {r['peak_gb']:>13.2f} {r['init_s']:>8.1f}")
