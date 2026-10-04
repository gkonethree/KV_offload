#!/usr/bin/env python3
"""Simple test for Skylight backend with vLLM 0.29.0"""
import os
import time
import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '8'
os.environ['SKYLIGHT_SYNC_STAGING'] = '1'
os.environ['SKYLIGHT_SPARSE_BUDGET'] = 'frac:0.1'

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import torch, time, gc

def run_bench(name, use_offload=False):
    if use_offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True, 'offload_fraction': 0.75, 
                'num_cpu_slots': 4096, 'num_staging_slots': 64, 
                'interval_tokens': 16, 'scorer_decay': 1.0, 
                'scorer_head_reduction': 'mean'
            },
            kv_role='kv_both',
        )
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
            dtype='float16', max_model_len=2048, 
            gpu_memory_utilization=0.05, enforce_eager=True, 
            tensor_parallel_size=1, kv_transfer_config=kv, 
            disable_log_stats=True, 
            attention_config={'backend': 'CUSTOM'},
        )
    else:
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
            dtype='float16', max_model_len=2048, 
            gpu_memory_utilization=0.05, enforce_eager=True, 
            tensor_parallel_size=1, disable_log_stats=True,
            attention_config={'backend': 'CUSTOM'},
        )
    
    prompts = ['The future of AI is'] * 4
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
    # Register backend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')
    
    # Monkey patch for CUSTOM backend
    import vllm.platforms.cuda as cuda_module
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
    _original_get = cuda_module._get_attn_backend_class
    def patched_get(backend, *args, **kwargs):
        if backend is AttentionBackendEnum.CUSTOM or (backend is None and 'CUSTOM' in str(AttentionBackendEnum.CUSTOM)):
            backend = AttentionBackendEnum.CUSTOM
        return _original_get(backend, *args, **kwargs)
    cuda_module._get_attn_backend_class = patched_get
    
    # Test configurations
    configs = [
        ('FlashInfer Dense', {'VLLM_ATTENTION_BACKEND': 'FLASHINFER'}, False),
        ('Skylight Oracle', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '8', 'SKYLIGHT_SYNC_STAGING': '0'}, False),
        ('Skylight Oracle + Offload', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SYNC_STAGING': '1'}, True),
    ]
    
    results = []
    for name, env, offload in configs:
        for k, v in {'VLLM_ATTENTION_BACKEND': 'FLASHINFER', 'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '8', 'SKYLIGHT_SYNC_STAGING': '0'}.items():
            if k in os.environ:
                del os.environ[k]
        for k, v in {'VLLM_ATTENTION_BACKEND': 'FLASHINFER'}.items():
            os.environ[k] = v
        for k, v in {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '8', 'SKYLIGHT_SYNC_STAGING': '0'}.items():
            os.environ[k] = v
        
        r = run_bench(name, use_offload=offload)
        results.append(r)
        time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: 4k Context (TinyLlama, 3k in + 512 out, 4 prompts)")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'Time(s)':>8} {'Throughput':>12} {'TPOT(ms)':>10} {'GPU Mem':>8}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['name']:<30} {r['time']:>8.2f} {r['throughput']:>12.1f} {r['tpot']:>10.2f} {r['mem']:>8.2f}")

if __name__ == '__main__':
    import os, time, gc, torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    import torch, time, gc
    
    def run_bench(name, use_offload):
        if 'CUSTOM' in os.environ.get('VLLM_ATTENTION_BACKEND', ''):
            # Use CUSTOM backend
            pass
        
        if 'Skylight' in name and 'Offload' in name:
            kv = KVTransferConfig(
                kv_connector='OffloadPackageConnector',
                kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
                kv_connector_extra_config={
                    'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 
                    'num_staging_slots': 64, 'interval_tokens': 16, 'scorer_decay': 1.0, 
                    'scorer_head_reduction': 'mean'
                },
                kv_role='kv_both',
            )
            llm = LLM(
                model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, 
                gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
                kv_transfer_config=kv, disable_log_stats=True, 
                attention_config={'backend': 'CUSTOM'},
            )
        else:
            llm = LLM(
                model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, 
                gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, 
                disable_log_stats=True,
                attention_config={'backend': 'CUSTOM' if 'Skylight' in name else 'FLASHINFER'},
            )
        
        prompts = ['The future of AI is ' * 192] * 4
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
    
    # Run tests
    configs = [
        ('FlashInfer Dense', {}, False),
        ('Skylight Oracle', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '8', 'SKYLIGHT_SYNC_STAGING': '0'}, False),
        ('Skylight Oracle + Offload', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SYNC_STAGING': '1'}, True),
    ]
    
    results = []
    for name, env, offload in configs:
        for k, v in env.items():
            os.environ[k] = v
        for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK', 'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM', 'SKYLIGHT_SYNC_STAGING']:
            if k not in env and k in os.environ:
                del os.environ[k]
        
        r = run_bench(name, use_offload=offload)
        results.append(r)
        time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: 4k Context (TinyLlama, 3k in + 512 out, 4 prompts)")
    print(f"{'='*80}")
    print(f"{'Configuration':<30} {'Time(s)':>8} {'Throughput':>12} {'TPOT(ms)':>10} {'GPU Mem':>8}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['name']:<30} {r['time']:>8.2f} {r['throughput']:>12.1f} {r['tpot']:>10.2f} {r['mem']:>8.2f}")

if __name__ == '__main__':
    main()
