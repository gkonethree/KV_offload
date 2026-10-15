#!/usr/bin/env python3
"""Single-request check: pq vs pq + 99% offload.

Hypothesis under test: with offload_fraction=0.99 almost every KV block is
moved to CPU each offload interval, and the PQ sparse decode must still be
token-for-token identical to plain pq, because selected CPU-resident pages
are fetched into the GPU staging pool (chunked exact decode when the cold
set exceeds staging capacity). pq+offload is expected to be SLOWER: the
fetch (and possibly multi-pass chunked decode) is on the critical path of
every decode step for a single request.

Configs:
  pq            baseline, no offloading
  pq+off99      99% offload, staging_policy='auto' (auto-sized to hot set)
  pq+off99-tiny 99% offload, fixed tiny staging (64 slots) -> forced chunked
                fallback (cold set ~160 blocks > 64 slots)

Usage: .venv/bin/python bench_1req_off99.py [config,config,...]
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

MODEL = os.environ.get('BENCH_MODEL', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
MAX_MODEL_LEN = int(os.environ.get('BENCH_MAX_MODEL_LEN', '18000'))  # prompt + decode + headroom
MAX_NEW_TOKENS = int(os.environ.get('BENCH_NEW_TOKENS', '8192'))
INTERVAL_TOKENS = int(os.environ.get('BENCH_INTERVAL', '128'))

# Single request: ~26 tokens per repetition. BENCH_PROMPT_REPS sets the
# length (40 -> ~1k tokens).
PROMPT = ('The history of artificial intelligence is a story of ambition, '
          'setbacks, and breakthroughs across decades of research. ') * \
    int(os.environ.get('BENCH_PROMPT_REPS', '1'))

# BENCH_PROMPT_FILE replaces the repeated-sentence prompt with a real text
# (e.g. a LongBench sample already truncated to the target length).
if os.environ.get('BENCH_PROMPT_FILE'):
    with open(os.environ['BENCH_PROMPT_FILE']) as _f:
        PROMPT = _f.read()

# Second request's prompt for the long-lived-process reuse test.
PROMPT_B = ('Quantum computing promises to reshape cryptography, optimization, '
            'and molecular simulation once fault-tolerant hardware arrives. ') * \
    int(os.environ.get('BENCH_PROMPT_REPS', '1'))

# if os.environ.get('BENCH_PROMPT') == 'varied':
#     # Non-degenerate prompt: varied content per block, so greedy logits are
#     # not a tie-fest; ~15 tokens/sentence -> 540 sentences ~ 8.1k tokens.
#     PROMPT = ''.join(
#         f"Observation {i}: the calibrated reading was {(i * 37) % 997}, "
#         f"which the field team called noteworthy, and the reviewers of "
#         f"section {(i * 11) % 23} marked it pending further review. "
#         for i in range(170))

SPARSE_ENV = {
    'SKYLIGHT_SPARSE_TOPK': os.environ.get('BENCH_TOPK', '0.1'),
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
    'SKYLIGHT_PQ_GQA_SHARE': '1',   # operating mode: selection per kv-head
}

CONFIGS = {
    'pq': dict(method='pq', offload=False),

    'pq+off50': dict(method='pq', offload=True, fraction=0.5,
                     staging_policy='auto', num_staging_slots=None,
                     scorer_type='paged_eviction'),
    'pq+off50-rand': dict(method='pq', offload=True, fraction=0.5,
                          staging_policy='auto', num_staging_slots=None,
                          scorer_type='random'),
    'pq+off50-lru': dict(method='pq', offload=True, fraction=0.5,
                          staging_policy='auto', num_staging_slots=None,
                          scorer_type='lru'),
    'pq+off50-tok': dict(method='pq', offload=True, fraction=0.5,
                         staging_policy='auto', num_staging_slots=None,
                         scorer_type='paged_eviction',
                         fetch_granularity='per_layer_tokens'),
    'pq+off50-lru-tok': dict(method='pq', offload=True, fraction=0.5,
                             staging_policy='auto', num_staging_slots=None,
                             scorer_type='lru',
                             fetch_granularity='per_layer_tokens'),
    # 70% offload, per-layer token recall (scorer comparison under staging
    # pressure: the auto pool is smaller than the offloaded set late in a run).
    'pq+off70': dict(method='pq', offload=True, fraction=0.7,
                     staging_policy='auto', num_staging_slots=None,
                     scorer_type='paged_eviction',
                     fetch_granularity='per_layer_tokens'),
    'pq+off70-lru': dict(method='pq', offload=True, fraction=0.7,
                         staging_policy='auto', num_staging_slots=None,
                         scorer_type='lru',
                         fetch_granularity='per_layer_tokens'),
    # Explicit block-granularity baselines for the token-vs-block matrix
    # (independent of the fetch_granularity default).
    'pq+off50-blk': dict(method='pq', offload=True, fraction=0.5,
                         staging_policy='auto', num_staging_slots=None,
                         scorer_type='paged_eviction',
                         fetch_granularity='block'),
    'pq+off50-lru-blk': dict(method='pq', offload=True, fraction=0.5,
                             staging_policy='auto', num_staging_slots=None,
                             scorer_type='lru',
                             fetch_granularity='block'),

    # Gate configs (e2e gate suite). off99 forces nearly-everything-fetched;
    # t12 forces the chunked-exact fallback under 12-slot staging pressure.
    'pq+off99': dict(method='pq', offload=True, fraction=0.99,
                     staging_policy='auto', num_staging_slots=None,
                     scorer_type='paged_eviction'),
    'pq+off99-t12': dict(method='pq', offload=True, fraction=0.99,
                         staging_policy='fixed', num_staging_slots=12,
                         scorer_type='paged_eviction'),
    'pq+off99-tok': dict(method='pq', offload=True, fraction=0.99,
                         staging_policy='auto', num_staging_slots=None,
                         scorer_type='paged_eviction',
                         fetch_granularity='per_layer_tokens'),
    'pq+off99-t12-tok': dict(method='pq', offload=True, fraction=0.99,
                             staging_policy='fixed', num_staging_slots=12,
                             scorer_type='paged_eviction',
                             fetch_granularity='per_layer_tokens'),

}



def shutdown_llm(llm):
    """Release an in-process engine's GPU memory.

    vLLM gc.freeze()s the heap at engine startup, so `del llm` alone frees
    nothing; EngineCore.shutdown() unfreezes it and runs the worker/connector
    shutdown (which drops the offload package's engine references).
    """
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:  # attribute path changed / already shut down
        print(f"  engine shutdown failed ({e}); falling back to gc.unfreeze", flush=True)
    gc.unfreeze()

def run(name, method, offload, fraction=0.5, staging_policy='auto',
        num_staging_slots=None, scorer_type='paged_eviction',
        fetch_granularity='per_layer_tokens'):
    print(f"\n{'=' * 70}\nCONFIG: {name}\n{'=' * 70}", flush=True)
    for k, v in SPARSE_ENV.items():
        os.environ[k] = v
    os.environ['SKYLIGHT_SPARSE_METHOD'] = method

    kv = None
    if offload and os.environ.get('BENCH_STAGING_SLOTS'):
        # Fixed staging size for every offload config (staging-pressure sweeps).
        staging_policy = 'fixed'
        num_staging_slots = int(os.environ['BENCH_STAGING_SLOTS'])
    if offload:
        extra = {
            'enabled': True,
            'offload_fraction': fraction,
            'num_cpu_slots': 4096,
            'staging_policy': staging_policy,
            'interval_tokens': INTERVAL_TOKENS,
            'true_free': True,
            'scorer_type': scorer_type,
            'fetch_granularity': fetch_granularity,
        }
        if num_staging_slots is not None:
            extra['num_staging_slots'] = num_staging_slots
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config=extra,
            kv_role='kv_both',
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    kwargs = dict(
        model=MODEL,
        dtype=os.environ.get('BENCH_DTYPE', 'float16'),
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.20,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        disable_log_stats=True,
        attention_config={'backend': 'CUSTOM'},
        async_scheduling=False,
    )
    if kv is not None:
        kwargs['kv_transfer_config'] = kv
    # BENCH_WARMUP=1: measure steady-state decode only. A short warm-up
    # generation on the same prompt pays the one-time per-process costs (PQ
    # codebook fit at the first decode step, JIT); prefix caching is off so the
    # timed request re-prefills identically in every config (offloaded blocks
    # lose their cache hashes, which would otherwise make prefill
    # config-dependent); stats are on so per-request token timestamps exist.
    warmup = os.environ.get('BENCH_WARMUP') == '1'
    if warmup:
        kwargs['disable_log_stats'] = False
        kwargs['enable_prefix_caching'] = False

    t_init0 = time.time()
    llm = LLM(**kwargs)
    t_init = time.time() - t_init0

    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True)
    if warmup:
        llm.generate([PROMPT], SamplingParams(
            temperature=0, top_p=1.0, max_tokens=8, ignore_eos=True))
        from offload_package import timing as _timing
        _timing.reset()  # breakdown covers the timed request only
    torch.cuda.synchronize()
    t0 = time.time()
    if os.environ.get('BENCH_TRACE_MULTINOMIAL') == '1':
        import traceback as _tb
        _orig_mn = torch.multinomial
        _mn_count = [0]
        def _traced_multinomial(*a, **k):
            if _mn_count[0] < 3:
                _mn_count[0] += 1
                print(f"\n=== torch.multinomial call #{_mn_count[0]} stack ===", flush=True)
                _tb.print_stack()
            return _orig_mn(*a, **k)
        torch.multinomial = _traced_multinomial

    if os.environ.get('BENCH_PROFILE') == '1':
        from torch.profiler import profile, ProfilerActivity
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
        ) as prof:
            outputs = llm.generate([PROMPT], sp)
        print(prof.key_averages(group_by_stack_n=8).table(
            sort_by='self_cpu_time_total', row_limit=35))
    else:
        outputs = llm.generate([PROMPT], sp)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    reuse = None
    if os.environ.get('BENCH_REUSE') == '1':
        torch.cuda.synchronize()
        t_r0 = time.time()
        outputs_b = llm.generate([PROMPT_B], sp)
        torch.cuda.synchronize()
        reuse = time.time() - t_r0

    in_tokens = len(outputs[0].prompt_token_ids)
    out_tokens = len(outputs[0].outputs[0].token_ids)
    tpot_ms = elapsed / max(out_tokens, 1) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    print(f"  init: {t_init:.1f}s")
    print(f"  prompt tokens: {in_tokens}")
    print(f"  new tokens: {out_tokens}")
    print(f"  total time: {elapsed:.1f}s")
    print(f"  decode TPOT: {tpot_ms:.2f} ms/token")
    m = getattr(outputs[0], 'metrics', None)
    if m is not None and m.first_token_ts and m.last_token_ts and out_tokens > 1:
        print(f"  TTFT: {m.first_token_latency * 1000:.1f} ms")
        print(f"  decode-only TPOT: "
              f"{(m.last_token_ts - m.first_token_ts) / (out_tokens - 1) * 1000:.2f} ms/token")
    print(f"  throughput: {out_tokens / max(elapsed, 1e-9):.1f} tok/s")
    print(f"  peak GPU allocated: {peak_gb:.2f} GB")
    print(f"  out: {outputs[0].outputs[0].text[:110]!r}")
    print(f"  token_ids[:32]: {list(outputs[0].outputs[0].token_ids)[:32]}")
    if reuse is not None:
        rb = len(outputs_b[0].outputs[0].token_ids)
        print(f"  REUSE req#2: total {reuse:.1f}s, TPOT {reuse / max(rb,1) * 1000:.2f} ms/token "
              f"({rb} tokens; codebook already fitted -> no pq.fit)")

    result = {
        'name': name, 'prompt_tokens': in_tokens, 'new_tokens': out_tokens,
        'time_s': elapsed, 'tpot_ms': tpot_ms, 'tps': out_tokens / max(elapsed, 1e-9),
        'peak_gb': peak_gb, 'init_s': t_init,
        'text': outputs[0].outputs[0].text,
        'tokens': list(outputs[0].outputs[0].token_ids),
    }
    import json
    with open(f"/tmp/opencode/tokens_{name.replace('+', '_')}.json", "w") as f:
        json.dump(result['tokens'], f)
    shutdown_llm(llm)
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
    print(f"SUMMARY single-request (input={rs[0]['prompt_tokens']}, +{MAX_NEW_TOKENS} out)")
    print(f"{'=' * 92}")
    print(f"{'config':<16} {'time(s)':>8} {'TPOT(ms)':>9} {'tok/s':>8} {'peak GPU(GB)':>13} {'init(s)':>8}")
    for r in rs:
        print(f"{r['name']:<16} {r['time_s']:>8.1f} {r['tpot_ms']:>9.2f} {r['tps']:>8.1f} "
              f"{r['peak_gb']:>13.2f} {r['init_s']:>8.1f}")

    # Correctness: every offload config must be token-for-token identical to pq.
    print("\nCORRECTNESS (greedy token-for-token vs 'pq' baseline):")
    base = results.get('pq')
    if base is None:
        print("  pq baseline not run; skipping comparison")
    else:
        for name, r in results.items():
            if name == 'pq':
                continue
            match = r['tokens'] == base['tokens']
            if not match:
                for j, (ta, tb) in enumerate(zip(base['tokens'], r['tokens'])):
                    if ta != tb:
                        print(f"  {name}: diverges at token {j}: {base['text'][:200]!r}")
                        print(f"  {' ' * len(name)}  vs off: {r['text'][:200]!r}")
                        break
            print(f"  {name}: {'PASS' if match else 'FAIL'}")
