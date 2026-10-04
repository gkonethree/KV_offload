import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_PLUGINS'] = ''
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import vllm.distributed.kv_transfer.kv_transfer_state as kv_state
original_ensure = kv_state.ensure_kv_transfer_initialized

def debug_ensure(vllm_config, kv_cache_config):
    print(f'[DEBUG] ensure_kv_transfer_initialized called', flush=True)
    print(f'  kv_transfer_config: {vllm_config.kv_transfer_config}', flush=True)
    result = original_ensure(vllm_config, kv_cache_config)
    print(f'  _KV_CONNECTOR_AGENT type: {type(kv_state._KV_CONNECTOR_AGENT)}', flush=True)
    print(f'  _KV_CONNECTOR_AGENT: {kv_state._KV_CONNECTOR_AGENT}', flush=True)
    return result

kv_state.ensure_kv_transfer_initialized = debug_ensure

# Now run the test
import os
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

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module.CudaPlatformBase.get_attn_backend_cls.__func__
def patched_get(cls, selected_backend, attn_selector_config, num_heads=None):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, _ATTN_OVERRIDES
    if selected_backend is AttentionBackendEnum.CUSTOM or (selected_backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        selected_backend = AttentionBackendEnum.CUSTOM
    return _original_get(cls, selected_backend, attn_selector_config, num_heads)
cuda_module.CudaPlatformBase.get_attn_backend_cls.__func__ = patched_get

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

kv = KVTransferConfig(
    kv_connector='OffloadPackageConnector',
    kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
    kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 32, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'},
    kv_role='kv_both',
)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import torch

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv, disable_log_stats=True, attention_config={'backend': 'CUSTOM'})

print('Starting generate...', flush=True)
outputs = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256, seed=42, stop_token_ids=[], ignore_eos=True))
print('Done', flush=True)
