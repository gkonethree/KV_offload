#!/usr/bin/env python3
"""Test just Skylight sparse_optimized backend without offload."""
import os, time
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'sparse_optimized'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '8'
os.environ['SKYLIGHT_SYNC_STAGING'] = '0'
os.environ['SKYLIGHT_SPARSE_BUDGET'] = 'frac:0.1'

# Register backend BEFORE importing vllm
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.sparse_optimized_backend.SparseOptimizedBackend')

# Monkey patch for CUSTOM backend
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module._get_attn_backend_class
def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)
cuda_module._get_attn_backend_class = patched_get

from vllm import LLM, SamplingParams
import torch, time, gc

def run_bench(name, model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', 
              max_model_len=4096, input_len=3072, output_len=512, num_prompts=4):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    llm = LLM(
        model=model, dtype='float16', max_model_len=max_model_len,
        gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
        disable_log_stats=True, attention_config={'backend': 'CUSTOM'},
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
    # Test with 4k context
    print("Testing Skylight sparse_optimized (fast CUDA kernel)")
    r = run_bench('Skylight sparse_optimized', max_model_len=4096, input_len=3072, output_len=512, num_prompts=4)
    print(f"\nResult: {r['time']:.2f}s, {r['throughput']:.1f} tok/s, TPOT: {r['tpot']:.2f}ms")
