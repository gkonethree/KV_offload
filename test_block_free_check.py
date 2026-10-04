#!/usr/bin/env python3
"""Empirical check: do offloaded GPU blocks actually return to the free pool?

Runs the sparse sink-local config WITH the offload connector on a few prompts
and instruments:
  - BlockPool free-block count before/after each _process_store_completion
  - ref_cnt of the blocks the connector tries to free
  - per-step free/running/waiting timeline

Debug print spam is redirected to a log file; the BLOCKPOOL free_blocks debug
lines there are post-processed to tabulate ref_cnt after decrement.
"""
import os
import sys
import time
import gc
from collections import Counter, defaultdict

import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ['VLLM_PLUGINS'] = ''

N_PROMPTS = int(os.environ.get('N_PROMPTS', '8'))
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '128'))
PROMPT_REPS = int(os.environ.get('PROMPT_REPS', '192'))
LOG_PATH = os.environ.get('BLOCK_FREE_LOG', '/tmp/opencode/block_free_check.log')

# 1. Register Skylight backend (same as test_offload_integrated.py)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'skylight.backend.SkylightSparseBackend')

# 2. Monkey patch for CUSTOM backend
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

_original_get = getattr(cuda_module, '_get_attn_backend_class', None)

def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    if _original_get:
        return _original_get(backend, *args, **kwargs)
    return backend

if _original_get:
    cuda_module._get_attn_backend_class = patched_get

# 3. Sparse sink-local env (oracle topk sink+local), offload variant
for k, v in {
    'SKYLIGHT_SPARSE_METHOD': 'oracle',
    'SKYLIGHT_SPARSE_TOPK': '0.1',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
    'SKYLIGHT_SYNC_STAGING': '1',
    'SKYLIGHT_SPARSE_BUDGET': 'frac:0.5',
}.items():
    os.environ[k] = v

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


class FdRedirect:
    """Redirect fd-level stdout (print spam + vLLM logs) to a file; stderr stays."""
    def __enter__(self):
        sys.stdout.flush()
        self.saved = os.dup(1)
        self.f = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(self.f, 1)
        os.close(self.f)
        return self
    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self.saved, 1)
        os.close(self.saved)
        return False


def main():
    kv = KVTransferConfig(
        kv_connector='OffloadPackageConnector',
        kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
        kv_connector_extra_config={
            'enabled': True, 'offload_fraction': 0.5, 'num_cpu_slots': 2048,
            'num_staging_slots': 64, 'interval_tokens': 64,
            'scorer_decay': 1.0, 'scorer_head_reduction': 'mean',
        },
        kv_role='kv_both',
    )
    llm = LLM(
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
        gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
        kv_transfer_config=kv, disable_log_stats=True,
        attention_config={'backend': 'CUSTOM'},
        enable_prefix_caching=False, max_num_seqs=N_PROMPTS,
    )

    # ---- probes (in-process: VLLM_ENABLE_V1_MULTIPROCESSING=0) ----
    engine = llm.llm_engine.engine_core.engine_core
    scheduler = engine.scheduler
    bp = scheduler.kv_cache_manager.block_pool
    print(f"KV pool: {bp.num_gpu_blocks} GPU blocks", flush=True)

    store_events = []
    sm = getattr(scheduler.connector, '_scheduler_manager', None)
    if sm is not None and getattr(sm, '_gpu_block_pool', None) is not None:
        orig_psc = sm._process_store_completion

        def probe_psc(event_idx):
            entry = sm._pending_store_events.get(event_idx)
            gpu_ids = list(entry[0]) if entry else []
            refs_before = {bid: bp.blocks[bid].ref_cnt for bid in gpu_ids[:12]}
            free_before = sm._gpu_block_pool.get_num_free_blocks()
            orig_psc(event_idx)
            free_after = sm._gpu_block_pool.get_num_free_blocks()
            refs_after = {bid: bp.blocks[bid].ref_cnt for bid in refs_before}
            store_events.append({
                'event': event_idx, 'n_blocks': len(gpu_ids),
                'free_before': free_before, 'free_after': free_after,
                'free_delta': free_after - free_before,
                'ref_before_sample': refs_before, 'ref_after_sample': refs_after,
            })
        sm._process_store_completion = probe_psc
        print("attached _process_store_completion probe", flush=True)
    else:
        print("WARNING: could not reach scheduler manager / gpu block pool", flush=True)

    timeline = []
    orig_sched = scheduler.schedule

    def probe_sched(*a, **k):
        out = orig_sched(*a, **k)
        timeline.append({
            'running': len(scheduler.running), 'waiting': len(scheduler.waiting),
            'free': bp.get_num_free_blocks(),
            'sched': len(out.num_scheduled_tokens),
            'preempted': len(out.preempted_req_ids or ()),
        })
        return out
    scheduler.schedule = probe_sched
    print("attached schedule probe", flush=True)

    # ---- per-request offload probe (wrap offload_fraction: the single
    # offload selection+copy point driven by _trigger_global_offload) ----
    offloaded_by_req = {}
    orch = getattr(scheduler.connector, '_worker_handler', None)
    orch = getattr(orch, '_orchestrator', None) if orch is not None else None
    if orch is not None:
        offloaded_by_req = defaultdict(set)
        orig_frac = orch.offload_manager.offload_fraction

        def probe_frac(candidates, fraction, scored=None):
            result = orig_frac(candidates, fraction, scored)
            for ref in result:
                offloaded_by_req[ref.request_id].add(ref.block_idx)
            return result
        orch.offload_manager.offload_fraction = probe_frac
        print("attached offload_fraction probe", flush=True)
    else:
        print("WARNING: could not reach orchestrator for offload probe", flush=True)

    prompts = ['The future of AI is ' * PROMPT_REPS] * N_PROMPTS
    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=MAX_TOKENS, ignore_eos=True)

    t0 = time.time()
    with FdRedirect():
        outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0

    # ---- report ----
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\ngenerated {total_tokens} tokens in {elapsed:.1f}s over {len(timeline)} steps")
    print(f"scheduler timeline (step: running/waiting/free):")
    for i, row in enumerate(timeline):
        if i < 10 or i % max(1, len(timeline) // 30) == 0 or i == len(timeline) - 1:
            print(f"  step {i:>4}: running={row['running']:>3} waiting={row['waiting']:>3} "
                  f"free={row['free']:>5} sched={row['sched']:>3} preempted={row['preempted']}")

    if store_events:
        print(f"\nstore completions observed: {len(store_events)}")
        tot_delta = sum(e['free_delta'] for e in store_events)
        print(f"total free-block delta across events: {tot_delta} "
              f"(blocks the pool actually regained: {sum(1 for e in store_events if e['free_delta'] > 0)} events)")
        for e in store_events:
            print(f"  event {e['event']}: tried to free {e['n_blocks']} blocks | "
                  f"free {e['free_before']} -> {e['free_after']} (delta {e['free_delta']}) | "
                  f"ref_cnt after: {list(e['ref_after_sample'].values())[:6]}")
    else:
        print("\nNO store completions observed — offloaded blocks were never even "
              "passed to free_blocks(). Check worker store-event plumbing.")

    if offloaded_by_req:
        print(f"\nper-request offload coverage ({len(offloaded_by_req)} of {N_PROMPTS} requests offloaded):")
        for req_id in sorted(offloaded_by_req, key=lambda r: int(r.split('-')[0])):
            print(f"  req {req_id}: {len(offloaded_by_req[req_id])} distinct blocks offloaded "
                  f"(block_idx range {min(offloaded_by_req[req_id])}..{max(offloaded_by_req[req_id])})")
    else:
        print("\nper-request offload coverage: NO blocks ever offloaded")

    print(f"\nfinal free blocks: {bp.get_num_free_blocks()} / {bp.num_gpu_blocks}")


if __name__ == '__main__':
    main()
