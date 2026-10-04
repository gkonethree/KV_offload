import os, time
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_PLUGINS'] = ''
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def get_mem():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

def run_config(name, use_sparse=False, use_offload=False):
    # Reset env
    for k in ['SKYLIGHT_SPARSE_METHOD','SKYLIGHT_SPARSE_TOPK','SKYLIGHT_SPARSE_SINK','SKYLIGHT_SPARSE_LOCAL','SKYLIGHT_SPARSE_CHANNEL_NUM','SKYLIGHT_SYNC_STAGING']:
        os.environ.pop(k, None)
    os.environ['VLLM_PLUGINS'] = ''
    os.environ['VLLM_ATTENTION_BACKEND'] = 'CUSTOM' if use_sparse else 'FLASHINFER'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    
    if use_sparse:
        os.environ['SKYLIGHT_SPARSE_METHOD'] = 'sparse_optimized'
        os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
        os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
        os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
        os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '8'
    
    # Register backend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    if use_sparse:
        register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')
    else:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        AttentionBackendEnum.CUSTOM.clear_override()
    
    # Monkey patch
    import vllm.platforms.cuda as cuda_module
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
    original = cuda_module.CudaPlatform.get_attn_backend_cls
    def patched(cls, selected_backend, attn_selector_config, num_heads=None):
        if selected_backend is AttentionBackendEnum.CUSTOM or (selected_backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
            selected_backend = AttentionBackendEnum.CUSTOM
        return original(selected_backend, attn_selector_config, num_heads)
    cuda_module.CudaPlatform.get_attn_backend_cls = classmethod(patched)
    
    if use_offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 32, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'},
            kv_role='kv_both',
        )
    else:
        kv = None
    
    from vllm import LLM, SamplingParams
    llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv, disable_log_stats=True)
    
    def m():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e9
    
    print(f'  {name}: After load: {m():.2f} GB', flush=True)
    _ = llm.generate(['test'], SamplingParams(temperature=0, max_tokens=1))
    print(f'  {name}: After warmup: {m():.2f} GB', flush=True)
    
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256, seed=42, stop_token_ids=[], ignore_eos=True)
    for i in range(5):
        _ = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=sp)
        print(f'  {name}: Step {i+1}: {m():.2f} GB used', flush=True)
    
    return m()

# Run all 4 configs sequentially
print("="*60)
print("MEMORY PROFILING - 2048 context, 256 decode tokens")
print("="*60)

print("\n[1/4] FlashInfer baseline...")
_ = run_config("FlashInfer", use_sparse=False, use_offload=False)

print("\n[2/4] sparse_optimized (no offload)...")
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'sparse_optimized'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '8'
_ = run_config("sparse_optimized", use_sparse=True, use_offload=False)

print("\n[3/4] sparse_optimized + 75% offload...")
os.environ['SKYLIGHT_SYNC_STAGING'] = '1'
_ = run_config("sparse_optimized+offload", use_sparse=True, use_offload=True)

print("\n[4/4] FlashInfer + 75% offload...")
os.environ.pop('SKYLIGHT_SPARSE_METHOD', None)
os.environ.pop('SKYLIGHT_SYNC_STAGING', None)
_ = run_config("FlashInfer+offload", use_sparse=False, use_offload=True)

print("\n" + "="*60)
print("SUMMARY")
