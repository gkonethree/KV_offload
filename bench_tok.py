#!/usr/bin/env python3
"""Small-context correctness bench for TOKEN-granularity offloading.

Single request, varied prompt, greedy decode.  Token mode keeps the top
`token_budget` tokens GPU-resident (TriAttention scoring, compaction +
page reclaim every `divide_length` decode tokens) and per-step the PQ top-k
sees ALL logical tokens; CPU-resident selections are fetched into packed
staging (chunked exact decode when the straggler set exceeds staging).

Configs (default run):
  pq             baseline, no offloading (greedy reference)
  pq+tok20       keep ~80% of the prompt as token budget (~20% offloaded)
  pq+tok50       keep ~50% (~50% offloaded)
  pq+tok50-tiny  keep ~50%, staging 2 pages -> forces the chunked path

Correctness: every token config must be token-for-token identical to `pq`
(selection sees the full logical history; staging + LSE2 merge are exact).

Usage: .venv/bin/python bench_tok.py [config,config,...]
"""
import os
import gc
import json
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
STATS = os.path.abspath('stats/tinyllama_tri_stats.pt')
MAX_MODEL_LEN = 2048  # TinyLlama max_position_embeddings
MAX_NEW_TOKENS = int(os.environ.get('BENCH_NEW_TOKENS', '128'))
ROUND_INTERVAL = 128

# Varied prompt (~26 tokens/sentence), sized for a small context.
N_SENT = int(os.environ.get('BENCH_SENTENCES', '40'))
PROMPT = ''.join(
    f"Observation {i}: the calibrated reading was {(i * 37) % 997}, "
    f"which the field team called noteworthy, and the reviewers of "
    f"section {(i * 11) % 23} marked it pending further review. "
    for i in range(N_SENT))

SPARSE_ENV = {
    'SKYLIGHT_SPARSE_TOPK': os.environ.get('BENCH_TOPK', '0.1'),
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
    'SKYLIGHT_PQ_GQA_SHARE': '1',
}

CONFIGS = {
    'pq': dict(method='pq', token=False),
    'pq2': dict(method='pq', token=False),  # determinism control
    'pq+tok20': dict(method='pq', token=True, keep_frac=0.80,
                     staging_pages=16),
    'pq+tok50': dict(method='pq', token=True, keep_frac=0.50,
                     staging_pages=16),
    'pq+tok50-tiny': dict(method='pq', token=True, keep_frac=0.50,
                          staging_pages=2),
    'pq+tok20-noround': dict(method='pq', token=True, keep_frac=0.80,
                             staging_pages=16, divide_length=100000),
}


def _prompt_len():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    return len(tok(PROMPT).input_ids)


def _get_token_engine(llm):
    """Best-effort walk to the in-process token engine for metrics."""
    try:
        core = llm.llm_engine.engine_core
        core = getattr(core, 'engine_core', core)
        conn = core.scheduler.connector
        orch = conn._scheduler_manager._orchestrator
        if orch is None:
            return None, None
        return orch.token_engine, conn._scheduler_manager
    except AttributeError:
        return None, None


def _engine_metrics(engine):
    if engine is None:
        return {}
    staged = getattr(engine.straggler, 'staging', None)
    # Per-request state is cleaned up when the request finishes; the
    # cumulative counters survive and describe the whole run.
    metrics = dict(engine.stats)
    metrics.update(getattr(engine.straggler, 'stats', {}))
    metrics['staged_now'] = staged.num_staged if staged else 0
    metrics['pool_used'] = engine.token_pool.num_slots - engine.token_pool.num_free
    return metrics



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

def run(name, method, token=False, keep_frac=0.5, staging_pages=16,
        prompt_len=None, divide_length=None):
    print(f"\n{'=' * 70}\nCONFIG: {name}\n{'=' * 70}", flush=True)
    for k, v in SPARSE_ENV.items():
        os.environ[k] = v
    os.environ['SKYLIGHT_SPARSE_METHOD'] = method

    kv = None
    if token:
        budget = max(1, -(-int(prompt_len * keep_frac) // 1))  # ceil
        extra = {
            'enabled': True,
            'granularity': 'token',
            'scorer_type': 'triattention',
            'triattention_stats_path': STATS,
            'token_budget': budget,
            'divide_length': divide_length or ROUND_INTERVAL,
            'window_size': 64,
            'sink_tokens': 64,          # must match SKYLIGHT_SPARSE_SINK
            'protect_prefill': False,  # mid-prompt tokens must compete
            'num_cpu_slots': 512,
            'num_staging_slots': staging_pages,
            'interval_tokens': ROUND_INTERVAL,
            'true_free': True,
        }
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config=extra,
            kv_role='kv_both',
        )
        print(f"  token budget: {budget} tokens "
              f"(keep {keep_frac:.0%} of {prompt_len})")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    kwargs = dict(
        model=MODEL,
        dtype='float16',
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.20,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        disable_log_stats=True,
        enable_prefix_caching=False,   # required by token-granularity mode
        attention_config={'backend': 'CUSTOM'},
        async_scheduling=False,
    )
    if kv is not None:
        kwargs['kv_transfer_config'] = kv

    t_init0 = time.time()
    llm = LLM(**kwargs)
    t_init = time.time() - t_init0

    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=MAX_NEW_TOKENS,
                        ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate([PROMPT], sp)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    in_tokens = len(outputs[0].prompt_token_ids)
    out_tokens = len(outputs[0].outputs[0].token_ids)
    tpot_ms = elapsed / max(out_tokens, 1) * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    engine, sched_mgr = _get_token_engine(llm)
    metrics = _engine_metrics(engine)
    if metrics:
        print(f"  token-engine: rounds={metrics['rounds']} "
              f"evicted={metrics['evicted']} promoted={metrics['promoted']} "
              f"freed_pages={metrics['freed_pages']} "
              f"fetch_batches={metrics['fetch_batches']} "
              f"fetched={metrics['fetched_tokens']} "
              f"chunked_steps={metrics['chunked_steps']} "
              f"chunk_groups={metrics['chunk_groups']} "
              f"staged_now={metrics['staged_now']} "
              f"cpu_pool_used={metrics['pool_used']}")
    free_blocks = None
    if sched_mgr is not None and sched_mgr._gpu_block_pool is not None:
        free_blocks = sched_mgr._gpu_block_pool.get_num_free_blocks()

    print(f"  init: {t_init:.1f}s")
    print(f"  prompt tokens: {in_tokens}")
    print(f"  new tokens: {out_tokens}")
    print(f"  total time: {elapsed:.1f}s")
    print(f"  decode TPOT: {tpot_ms:.2f} ms/token")
    print(f"  throughput: {out_tokens / max(elapsed, 1e-9):.1f} tok/s")
    print(f"  peak GPU allocated: {peak_gb:.2f} GB")
    if free_blocks is not None:
        print(f"  GPU pool free blocks at end: {free_blocks}")
    print(f"  out: {outputs[0].outputs[0].text[:110]!r}")
    print(f"  token_ids[:32]: {list(outputs[0].outputs[0].token_ids)[:32]}")

    result = {
        'name': name, 'prompt_tokens': in_tokens, 'new_tokens': out_tokens,
        'time_s': elapsed, 'tpot_ms': tpot_ms,
        'tps': out_tokens / max(elapsed, 1e-9),
        'peak_gb': peak_gb, 'init_s': t_init,
        'metrics': metrics,
        'text': outputs[0].outputs[0].text,
        'tokens': list(outputs[0].outputs[0].token_ids),
    }
    with open(f"/tmp/opencode/tok_tokens_{name.replace('+', '_')}.json", "w") as f:
        json.dump(result['tokens'], f)
    shutdown_llm(llm)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    return result


if __name__ == '__main__':
    if os.environ.get('BENCH_WATCHDOG_S'):
        import faulthandler
        faulthandler.dump_traceback_later(
            int(os.environ['BENCH_WATCHDOG_S']), exit=True
        )
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else \
        ['pq', 'pq+tok20', 'pq+tok50']
    prompt_len = _prompt_len()
    print(f"prompt: {N_SENT} sentences -> ~{prompt_len} tokens; "
          f"decode {MAX_NEW_TOKENS}")

    results = {}
    for name in names:
        results[name] = run(name, prompt_len=prompt_len, **CONFIGS[name])

    rs = list(results.values())
    print(f"\n{'=' * 92}")
    print(f"SUMMARY token-mode (input={prompt_len}, +{MAX_NEW_TOKENS} out)")
    print(f"{'=' * 92}")
    print(f"{'config':<16} {'time(s)':>8} {'TPOT(ms)':>9} {'tok/s':>8} "
          f"{'peak GPU(GB)':>13} {'evicted':>8} {'promoted':>9} "
          f"{'pages reclaimed':>16} {'fetch':>7} {'chunked':>8}")
    for r in rs:
        m = r['metrics']
        if m:
            ex = m.get('evicted', '-')
            pr = m.get('promoted', '-')
            pg = m.get('freed_pages', '-')
            fe = m.get('fetched_tokens', '-')
            ch = m.get('chunked_steps', '-')
        else:
            ex = pr = pg = fe = ch = '-'
        print(f"{r['name']:<16} {r['time_s']:>8.1f} {r['tpot_ms']:>9.2f} "
              f"{r['tps']:>8.1f} {r['peak_gb']:>13.2f} {ex:>8} {pr:>9} "
              f"{pg:>16} {fe:>7} {ch:>8}")

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
                        print(f"  {name}: diverges at token {j}:")
                        print(f"      pq  : {base['text'][:200]!r}")
                        print(f"      {name}: {r['text'][:200]!r}")
                        break
            print(f"  {name}: {'PASS' if match else 'FAIL'}")
