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

import torch, time
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

# Register backend BEFORE importing vllm
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

# Monkey patch the underlying function in CudaPlatformBase - save original FIRST
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum

# Save original FIRST
_original_get = cuda_module._get_attn_backend_class

def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)

# Patch the function
cuda_module._get_attn_backend_class = patched_get

# Register backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

kv = KVTransferConfig(
    kv_connector='OffloadPackageConnector',
    kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
    kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 16, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'},
    kv_role='kv_both',
)

from vllm import LLM, SamplingParams
import torch, time

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

def get_free_blocks(llm):
    """Get free block count from block pool"""
    return llm.llm_engine.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv, disable_log_stats=True, attention_config={'backend': 'CUSTOM'})

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

print("=== BLOCK POOL REUSE TEST ===")
print(f"Initial free blocks: {get_free_blocks(llm)}")
print(f"After load: {m():.2f} GB")

_ = llm.generate(['test'], SamplingParams(temperature=0, max_tokens=1))
print(f"After warmup: {m():.2f} GB")
print(f"Free blocks after warmup: {get_free_blocks(llm)}")

sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256, seed=42, stop_token_ids=[], ignore_eos=True)

# First generation - should allocate blocks
print("\n=== FIRST GENERATION ===")
print(f"Before generate: free={get_free_blocks(llm)}, mem={m():.2f} GB")
outputs1 = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=sp)
print(f"After gen1: free={get_free_blocks(llm)}, mem={m():.2f} GB")

# Second generation - should reuse blocks
print("\n=== SECOND GENERATION (should reuse blocks) ===")
print(f"Before gen2: free={get_free_blocks(llm)}, mem={m():.2f} GB")
outputs2 = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=sp)
print(f"After gen2: free={get_free_blocks(llm)}, mem={m():.2f} GB")

# Third generation - should still reuse
print("\n=== THIRD GENERATION (should still reuse) ===")
print(f"Before gen3: free={get_free_blocks(llm)}, mem={m():.2f} GB")
outputs3 = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=sp)
print(f"After gen3: free={get_free_blocks(llm)}, mem={m():.2f} GB")

print("\n=== FINAL STATE ===")
print(f"Final free blocks: {get_free_blocks(llm)}")
print(f"Final memory: {m():.2f} GB")

print("\n=== VERIFICATION ===")
print("If blocks are reused: free count should stay ~same across generations")
print("If blocks are NOT reused: free count would decrease each generation")
