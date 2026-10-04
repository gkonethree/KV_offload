#!/usr/bin/env python3
"""Final benchmark: Compare working configurations."""
import os, time, gc, torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

# Register backends BEFORE importing vllm
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# FlashInfer baseline
register_backend(AttentionBackendEnum.FLASHINFER, 'vllm.v1.attention.backends.flashinfer.FlashInferBackend')

# Skylight Oracle (sparse_oracle_topk_sink_local_optimized)
register_backend(AttentionBackendEnum.CUSTOM, 'skylight.backend.SkylightSparseBackend')

import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module._get_attn_backend_class
def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)
cuda_module._get_attn_backend_class = patched_get

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import torch, time

def run_bench(name, env_dict, use_offload, max_model_len=4096, input_len=3072, output_len=512, num_prompts=4):
    for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK', 
              'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM', 
              'SKYLIGHT_SYNC_STAGING', 'SKYLIGHT_SPARSE_BUDGET',
              'VLLM_ATTENTION_BACKEND']:
        if k in env_dict:
            os.environ[k] = env_dict[k]
        elif k in os.environ:
            del os.environ[k]
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    if use_offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 
                'num_cpu_slots': 4096, 'num_staging_slots': 64, 
                'interval_tokens': 16, 'scorer_decay': 1.0, 
                'scorer_head_reduction': 'mean'},
            kv_role='kv_both',
        )
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=max_model_len,
            gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
            kv_transfer_config=kv, disable_log_stats=True,
            attention_config={'backend': env_dict.get('VLLM_ATTENTION_BACKEND', 'FLASHINFER')},
        )
    else:
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=max_model_len,
            gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
            disable_log_stats=True,
            attention_config={'backend': env_dict.get('VLLM_ATTENTION_BACKEND', 'FLASHINFER')},
        )
    
    prompts = ['The future of AI is ' * (input_len // 16)] * num_prompts
    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=output_len, ignore_eos=True)
    
    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tpot = (elapsed / total_tokens) * 1000
    throughput = total_tokens / elapsed
    
    free, total = torch.cuda.mem_get_info()
    mem_gb = (total - free) / 1e9
    
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  TPOT: {tpot:.2f} ms/token")
    print(f"  GPU Mem: {mem_gb:.2f} GB")
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    
    return {'name': name, 'time': elapsed, 'throughput': throughput, 'tpot': tpot, 'mem': mem_gb}

if __name__ == '__main__':
    configs = [
        # Dense baselines
        ('FlashInfer (Dense)', {'VLLM_ATTENTION_BACKEND': 'FLASHINFER'}, False),
        # Skylight Sparse - Oracle (slow but accurate)
        ('Skylight Oracle', {
            'VLLM_ATTENTION_BACKEND': 'CUSTOM',
            'SKYLIGHT_SPARSE_METHOD': 'oracle',
            'SKYLIGHT_SPARSE_TOPK': '0.1',
            'SKYLIGHT_SPARSE_SINK': '64',
            'SKYLIGHT_SPARSE_LOCAL': '64',
            'SKYLIGHT_SPARSE_CHANNEL_NUM': '-1',
            'SKYLIGHT_SYNC_STAGING': '0',
            'SKYLIGHT_SPARSE_BUDGET': 'frac:0.1',
        }, False),
        # Skylight Sparse + Offload (Oracle)
        ('Skylight Oracle + Offload', {
            'VLLM_ATTENTION_BACKEND': 'CUSTOM',
            'SKYLIGHT_SPARSE_METHOD': 'oracle',
            'SKYLIGHT_SPARSE_TOPK': '0.1',
            'SKYLIGHT_SPARSE_SINK': '64',
            'SKYLIGHT_SPARSE_LOCAL': '64',
            'SKYLIGHT_SPARSE_CHANNEL_NUM': '-1',
            'SKYLIGHT_SYNC_STAGING': '1',
            'SKYLIGHT_SPARSE_BUDGET': 'frac:0.1',
        }, True),
    ]
    
    results = []
    for name, env, offload in configs:
        r = run_bench(name, env, offload, max_model_len=4096, input_len=3072, output_len=512, num_prompts=4)
        r['case'] = '4k context'
        results.append(r)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)
    
    # Summary
    print(f"\n{'='*90}")
    print("FINAL BENCHMARK RESULTS: 4k Context (TinyLlama, 3k in + 512 out, 4 prompts)")
    print(f"{'='*90}")
    print(f"{'Configuration':<30} {'Time(s)':>8} {'Throughput':>12} {'TPOT(ms)':>10} {'GPU Mem':>8}")
    print(f"{'-'*90}")
    for r in results:
        if 'error' not in r:
            print(f"{r['name']:<30} {r['time']:>8.2f} {r['throughput']:>12.1f} {r['tpot']:>10.2f} {r['mem']:>8.2f}")
        else:
            print(f"{r['name']:<30} {'ERROR':>8} {'':>12} {'':>10} {'':>8}")
    print(f"{'='*90}")
