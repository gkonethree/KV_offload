import os, time
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_PLUGINS'] = ''
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'sparse_optimized'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '8'
os.environ['SKYLIGHT_SYNC_STAGING'] = '1'

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

# Register backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

# Monkey patch cuda
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module._get_attn_backend_class
def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)
cuda_module._get_attn_backend_class = patched_get

# Register backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

def run_test(name, use_offload):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    
    if use_offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 8, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'},
            kv_role='kv_both',
        )
    else:
        kv = None
    
    from vllm import LLM, SamplingParams
    import torch
    
    def m():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9
    
    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv, disable_log_stats=True, attention_config={'backend': 'CUSTOM'})
    
    def m():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9
    
    print(f'After load: {m():.2f} GB')
    _ = llm.generate(['test'], SamplingParams(temperature=0, max_tokens=1))
    print(f'After warmup: {m():.2f} GB')
    
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, seed=42, stop_token_ids=[], ignore_eos=True)
    print(f'Before generate: {m():.2f} GB')
    start = time.time()
    outputs = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=sp)
    elapsed = time.time() - start
    print(f'After generate: {m():.2f} GB (took {elapsed:.1f}s)')
    print(f'Output length: {len(outputs[0].outputs[0].text)} chars')
    del llm
    torch.cuda.empty_cache()
    time.sleep(2)
    return m()

# Test 1: With offload
kv = KVTransferConfig(
    kv_connector='OffloadPackageConnector',
    kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
    kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 8, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'},
    kv_role='kv_both',
)

# Test 2: Without offload (no kv_transfer_config)
# We'll run sequentially

print("Testing WITH offload (75%)...")
mem_with = run_test("WITH 75% offload", True)

print("\n\nTesting WITHOUT offload...")
mem_without = run_test("WITHOUT offload", False)

print(f"\n{'='*60}")
print(f"SUMMARY:")
print(f"  With 75% offload:    {mem_with:.2f} GB")
print(f"  Without offload:     {mem_without:.2f} GB")
print(f"  Memory saved:        {mem_without - mem_with:.2f} GB ({(mem_without - mem_with)/mem_without*100:.1f}%)")
