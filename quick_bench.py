#!/usr/bin/env python3
"""Quick benchmark: FlashInfer vs Skylight Oracle vs Skylight Oracle+Offload"""
import os, time, gc, torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.FLASHINFER, 'vllm.v1.attention.backends.flashinfer.FlashInferBackend')
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
import torch, time, gc

def run_one(name, env_dict, use_offload):
    for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK', 
              'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM', 
              'SKYLIGHT_SYNC_STAGING', 'SKYLIGHT_SPARSE_BUDGET',
              'VLLM_ATTENTION_BACKEND']:
        if k in env_dict: os.environ[k] = env_dict[k]
        elif k in os.environ: del os.environ[k]
    
    print(f"\n--- {name} ---")
    
    if use_offload:
        kv = KVTransferConfig(kv_connector='OffloadPackageConnector', kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 16, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'}, kv_role='kv_both')
        llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=4096, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv, disable_log_stats=True, attention_config={'backend': 'CUSTOM'})
    else:
        llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=4096, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, disable_log_stats=True, attention_config={'backend': 'FLASHINFER' if not 'CUSTOM' in str(env_dict.get('VLLM_ATTENTION_BACKEND', '')) else 'CUSTOM'})
    
    prompts = ['The future of AI is ' * 192] * 4  # 3072 tokens
    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=512, ignore_eos=True)
    
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
    
    print(f"  Time: {elapsed:.2f}s, Throughput: {throughput:.1f} tok/s, TPOT: {tpot:.2f} ms, Mem: {mem_gb:.2f} GB")
    
    del llm; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(2)
    return {'name': name, 'time': elapsed, 'throughput': throughput, 'tpot': tpot, 'mem': mem_gb}

if __name__ == '__main__':
    configs = [
        ('FlashInfer Dense', {'VLLM_ATTENTION_BACKEND': 'FLASHINFER'}, False),
        ('Skylight Oracle', {'VLLM_ATTENTION_BACKEND': 'CUSTOM', 'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '-1'}, False),
        ('Skylight Oracle + Offload', {'VLLM_ATTENTION_BACKEND': 'CUSTOM', 'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '-1', 'SKYLIGHT_SYNC_STAGING': '1'}, True),
    ]
    
    results = []
    for name, env, offload in configs:
        r = run_one(name, env, False if 'Offload' not in name else True)
        r['name'] = name
        results.append(r)
    
    print(f"\n{'='*80}")
    print(f"{'Config':<25} {'Time(s)':>8} {'Throughput':>12} {'TPOT(ms)':>10} {'Mem(GB)':>8}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['name']:<25} {r['time']:>8.2f} {r['throughput']:>12.1f} {r['tpot']:>10.2f} {r['mem']:>8.2f}")
