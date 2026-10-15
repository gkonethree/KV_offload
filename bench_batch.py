#!/usr/bin/env python3
"""Batch/memory test: N concurrent long-context requests, no-offload vs offload.

Demonstrates the point of KV offloading: blocks true-freed by the offload
trigger return to the GPU BlockPool's free queue and are consumed by the
OTHER requests (new admissions / decode growth). Evidence per run:

  1. "GPU KV cache usage" trajectory -- with offload the curve DROPS at each
     trigger instead of climbing monotonically to ~100% and stalling there.
  2. Running/Waiting request counts -- with offload all N run concurrently;
     without offload requests queue once the pool saturates.
  3. Preemptions counter (pool exhaustion during decode growth).
  4. "offloaded N blocks (batched this step)" connector lines.
  5. Wall clock + aggregate throughput + peak GPU allocated.

Same workload both modes; identical gpu_memory_utilization. Each mode runs
in a fresh subprocess (clean engine + pools). The staging pool is part of the
offload cost and is reported explicitly (BATCH_STAGING_SLOTS x ~360KB).

Usage:
  .venv/bin/python bench_batch.py pq,offload
Env: BATCH_REQS (32), BATCH_REPS (420 -> ~10.9k-token prompts),
     BATCH_DECODE (48), BATCH_GPU_UTIL (0.08), BATCH_STAGING_SLOTS (1024),
     BENCH_TOPK (0.05)
"""
import os
import re
import sys
import json
import time
import subprocess

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('SKYLIGHT_OFFLOAD_TIMING', '1')
os.environ.setdefault('VLLM_LOG_STATS_INTERVAL', '2')

import torch

from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum, register_backend, _ATTN_OVERRIDES,
)
register_backend(AttentionBackendEnum.CUSTOM, 'skylight.backend.SkylightSparseBackend')
import vllm.platforms.cuda as cuda_module

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
MAX_MODEL_LEN = 17000
N_REQS = int(os.environ.get('BATCH_REQS', '32'))
PROMPT_REPS = int(os.environ.get('BATCH_REPS', '40'))
DECODE = int(os.environ.get('BATCH_DECODE', '32'))
GPU_UTIL = float(os.environ.get('BATCH_GPU_UTIL', '0.08'))
STAGING_SLOTS = int(os.environ.get('BATCH_STAGING_SLOTS', '1024'))
CPU_SLOTS = int(os.environ.get('BATCH_CPU_SLOTS', '16384'))
FRACTION = float(os.environ.get('BATCH_FRACTION', '0.5'))
INTERVAL_TOKENS = int(os.environ.get('BENCH_INTERVAL', '128'))

# Distinct topic word per request so prompt prefixes diverge immediately ->
# no prefix-cache sharing across requests (which would shrink real KV demand).
TOPICS = ['neural networks', 'quantum computing', 'climate models', 'protein folding',
          'autonomous vehicles', 'language models', 'robotic control', 'graph theory',
          'supply chains', 'medical imaging', 'audio synthesis', 'urban planning',
          'fusion energy', 'soil chemistry', 'orbital mechanics', 'market design',
          'crop breeding', 'ocean currents', 'battery chemistry', 'optical networks',
          'seismic imaging', 'drug discovery', 'speech recognition', 'image retrieval',
          'recommender systems', 'formal proofs', 'compiler design', 'cache coherence',
          'air traffic', 'power grids', 'water systems', 'proteomics']
BASE = ('The history of artificial intelligence is a story of ambition, '
        'setbacks, and breakthroughs across decades of research. ')
PROMPTS = [
    (f'All about {TOPICS[i % len(TOPICS)]}: ' + BASE) * PROMPT_REPS
    for i in range(N_REQS)
]

SPARSE_ENV = {
    'SKYLIGHT_SPARSE_TOPK': os.environ.get('BENCH_TOPK', '0.05'),
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
    'SKYLIGHT_PQ_GQA_SHARE': '1',
}

RESULT_DIR = '/tmp/opencode'


def run(mode):
    print(f"\n{'=' * 70}\nCONFIG: {mode} | reqs={N_REQS} x ~{26 * PROMPT_REPS} tok"
          f" + {DECODE} decode | gpu_util={GPU_UTIL}\n{'=' * 70}", flush=True)
    for k, v in SPARSE_ENV.items():
        os.environ[k] = v
    os.environ['SKYLIGHT_SPARSE_METHOD'] = 'pq'

    kv = None
    if mode != 'pq':
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True,
                'offload_fraction': FRACTION,
                'num_cpu_slots': CPU_SLOTS,
                'staging_policy': 'fixed',
                'num_staging_slots': STAGING_SLOTS,
                'interval_tokens': INTERVAL_TOKENS,
                'true_free': True,
                'scorer_type': 'paged_eviction',
            },
            kv_role='kv_both',
        )

    llm = LLM(
        model=MODEL,
        dtype='float16',
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_UTIL,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=N_REQS,
        disable_log_stats=False,   # keep periodic KV-usage / Running / Waiting lines
        attention_config={'backend': 'CUSTOM'},
        async_scheduling=False,
        **({} if kv is None else {'kv_transfer_config': kv}),
    )

    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=DECODE, ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate(PROMPTS, sp)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    in_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  requests finished: {len(outputs)}/{N_REQS}")
    print(f"  prompt tokens: {in_tokens} ({in_tokens // N_REQS}/request)")
    print(f"  new tokens: {out_tokens}")
    print(f"  total time: {elapsed:.1f}s")
    print(f"  aggregate gen: {out_tokens / max(elapsed, 1e-9):.1f} tok/s")
    print(f"  peak GPU allocated: {peak_gb:.2f} GB")

    with open(f'{RESULT_DIR}/batch_result_{mode}.json', 'w') as f:
        json.dump({'mode': mode, 'reqs': len(outputs), 'in': in_tokens,
                   'out': out_tokens, 'time_s': elapsed,
                   'tps': out_tokens / max(elapsed, 1e-9), 'peak_gb': peak_gb}, f)


def parse_log(logfile):
    usage, running, waiting = [], [], []
    preempts, pool_blocks, offload_events, fit_s = 0, None, 0, 0.0
    offload_ms, fetch_batches, fetch_ms = 0.0, 0, 0.0
    for line in open(logfile, errors='replace'):
        m = re.search(r'GPU KV cache usage: ([0-9.]+)%', line)
        if m:
            usage.append(float(m.group(1)))
        m = re.search(r'Running: (\d+) reqs, Waiting: (\d+) reqs', line)
        if m:
            running.append(int(m.group(1)))
            waiting.append(int(m.group(2)))
        m = re.search(r'Preemptions: (\d+)', line)
        if m:
            preempts = max(preempts, int(m.group(1)))
        m = re.search(r'Bound GPU BlockPool \((\d+) blocks\)', line)
        if m and pool_blocks is None:
            pool_blocks = int(m.group(1))
        # timing-table rows: last occurrence (the FINAL report) wins
        m = re.match(r'^pq\.fit\s+([0-9.]+)\s+(\d+)', line)
        if m:
            fit_s = float(m.group(1)) / 1e3
        m = re.match(r'^sched\.offload\s+([0-9.]+)\s+(\d+)', line)
        if m:
            offload_ms, offload_events = float(m.group(1)), int(m.group(2))
        m = re.match(r'^wrap\.fetch\s+([0-9.]+)\s+(\d+)', line)
        if m:
            fetch_ms, fetch_batches = float(m.group(1)), int(m.group(2))
    return {
        'kv_usage': usage, 'running': running, 'waiting': waiting,
        'preempts': preempts, 'pool_blocks': pool_blocks,
        'offload_events': offload_events, 'offload_ms': offload_ms,
        'fetch_batches': fetch_batches, 'fetch_ms': fetch_ms,
        'fit_s': fit_s,
    }


def summarize(mode, p):
    try:
        with open(f'{RESULT_DIR}/batch_result_{mode}.json') as f:
            r = json.load(f)
    except FileNotFoundError:
        print(f"{mode}: no result JSON")
        return
    u = p['kv_usage']
    print(f"\n--- {mode} ---")
    pool_txt = f"{p['pool_blocks']} blocks" if p['pool_blocks'] else 'n/a (no connector)'
    print(f"  pool: {pool_txt} | finished: {r['reqs']}/{N_REQS} reqs"
          f" | gen {r['out']} tok in {r['time_s']:.0f}s ({r['tps']:.1f} tok/s)"
          f" | peak {r['peak_gb']:.2f} GB")
    if u:
        ramp = min(max(1, len(u) // 10), max(0, len(u) - 1))
        print(f"  KV usage: peak {max(u):.0f}% | post-ramp high {max(u[ramp:]):.0f}%"
              f" | min {min(u):.0f}% | final {u[-1]:.0f}% | samples {len(u)}")
    if p['running']:
        print(f"  Running: max {max(p['running'])} | Waiting: max {max(p['waiting'])}"
              f" | samples {len(p['running'])}")
    print(f"  preemptions: {p['preempts']} | offload events: {p['offload_events']}"
          f" ({p['offload_ms'] / 1e3:.1f}s) | fetch batches: {p['fetch_batches']}"
          f" ({p['fetch_ms'] / 1e3:.1f}s)"
          + (f" | codebook fit: {p['fit_s']:.0f}s" if p['fit_s'] else ""))
    if u:
        step = max(1, len(u) // 16)
        traj = ' '.join(f'{x:.0f}' for x in u[::step])
        print(f"  KV%: {traj}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'pq,offload'
    if arg.startswith('single:'):
        run(arg[len('single:'):])
        return
    for mode in arg.split(','):
        logfile = f'{RESULT_DIR}/batch_{mode}.log'
        print(f"\n==== batch mode '{mode}' -> {logfile}", flush=True)
        with open(logfile, 'w') as f:
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), 'single:' + mode],
                stdout=f, stderr=subprocess.STDOUT, check=True,
            )
        summarize(mode, parse_log(logfile))


if __name__ == '__main__':
    main()
