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

# Register backend BEFORE importing vllm
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

# Monkey patch cuda.py PROPERLY
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
original_get = cuda_module.CudaPlatform.get_attn_backend_cls.__func__

def patched_get(cls, selected_backend, attn_selector_config, num_heads=None):
    if selected_backend is AttentionBackendEnum.CUSTOM or (selected_backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        selected_backend = AttentionBackendEnum.CUSTOM
    return original_get(cls, selected_backend, attn_selector_config, num_heads)
cuda_module.CudaPlatform.get_attn_backend_cls = classmethod(patched_get)

# Patch connector to debug
import offload_package.integration.connector as conn_mod
original_conn_init = conn_mod.OffloadPackageConnector.__init__
def debug_init(self, *args, **kwargs):
    print(f'[DEBUG] Connector __init__', flush=True)
    return original_conn_init(self, *args, **kwargs)
conn_mod.OffloadPackageConnector.__init__ = debug_init

original_init_config = conn_mod.OffloadPackageConnector.initialize_from_config
def debug_init_config(self, kv_cache_config):
    print(f'[DEBUG] Connector initialize_from_config: kv_cache_config={kv_cache_config}', flush=True)
    return original_init_config(self, kv_cache_config)
conn_mod.OffloadPackageConnector.initialize_from_config = debug_init_config

# Patch scheduler
import offload_package.integration.scheduler_connector as sc
original_build = sc.OffloadPackageScheduler.build_connector_meta
def debug_build(self, scheduler_output):
    print(f'[DEBUG] build_connector_meta: orch={self._orchestrator is not None}, pool={self._gpu_block_pool is not None}, new={len(scheduler_output.scheduled_new_reqs)}, cached={len(scheduler_output.scheduled_cached_reqs.req_ids)}', flush=True)
    return original_build(self, scheduler_output)
sc.OffloadPackageScheduler.build_connector_meta = debug_build

# Patch worker connector
import offload_package.integration.worker_connector as wc
original_worker_init = wc.OffloadPackageWorker.__init__
def debug_worker_init(self, *args, **kwargs):
    print(f'[DEBUG] Worker __init__', flush=True)
    return original_worker_init(self, *args, **kwargs)
wc.OffloadPackageWorker.__init__ = debug_worker_init

original_worker_init_config = wc.OffloadPackageWorker.initialize_from_config
def debug_worker_init_config(self, kv_cache_config):
    print(f'[DEBUG] Worker initialize_from_config: kv_cache_config={kv_cache_config}', flush=True)
    return original_worker_init_config(self, kv_cache_config)
wc.OffloadPackageWorker.initialize_from_config = debug_worker_init_config

kv = KVTransferConfig(
    kv_connector='OffloadPackageConnector',
    kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
    kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 8, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'},
    kv_role='kv_both',
)

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
original_get = cuda_module.CudaPlatform.get_attn_backend_cls.__func__
def patched_get(cls, selected_backend, attn_selector_config, num_heads=None):
    if selected_backend is AttentionBackendEnum.CUSTOM or (selected_backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        selected_backend = AttentionBackendEnum.CUSTOM
    return original_get(cls, selected_backend, attn_selector_config, num_heads)
cuda_module.CudaPlatform.get_attn_backend_cls = classmethod(patched_get)

from vllm import LLM, SamplingParams
import torch
def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv, disable_log_stats=True)

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

print(f'After load: {m():.2f} GB', flush=True)
_ = llm.generate(['test'], SamplingParams(temperature=0, max_tokens=1))
print(f'After warmup: {m():.2f} GB', flush=True)

sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200, seed=42, stop_token_ids=[], ignore_eos=True)

print(f'Before generate: {m():.2f} GB', flush=True)
outputs = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=sp)
print(f'After generate: {m():.2f} GB', flush=True)
