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

# Register backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

# Patch cuda - save original FIRST
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module._get_attn_backend_class
def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)
cuda_module._get_attn_backend_class = patched_get
_original_get = cuda_module._get_attn_backend_class

# Register backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

# Patch scheduler to trace offload flow
import offload_package.integration.scheduler_connector as sc
original_trigger = sc.OffloadPackageScheduler._trigger_global_offload
def debug_trigger(self, metadata):
    print(f'[SCHED] _trigger_global_offload called, orch={self._orchestrator is not None}, scored={len(self._orchestrator._scored) if self._orchestrator else 0}', flush=True)
    return original_trigger(self, metadata)
sc.OffloadPackageScheduler._trigger_global_offload = debug_trigger

original_process = sc.OffloadPackageScheduler._process_store_completion
def debug_process(self, event_idx):
    print(f'[SCHED] _process_store_completion event={event_idx}', flush=True)
    return original_process(self, event_idx)
sc.OffloadPackageScheduler._process_store_completion = debug_process

original_update = sc.OffloadPackageScheduler.update_connector_output
def debug_update(self, connector_output):
    print(f'[SCHED] update_connector_output: finished_recving={connector_output.finished_recving}', flush=True)
    if connector_output.kv_connector_worker_meta:
        print(f'  worker_meta.completed_store_events={connector_output.kv_connector_worker_meta.completed_store_events}', flush=True)
    return original_update(self, connector_output)
sc.OffloadPackageScheduler.update_connector_output = debug_update

original_trigger = sc.OffloadPackageScheduler._trigger_global_offload
def debug_trigger(self, metadata):
    print(f'[SCHED] _trigger_global_offload called, orch={self._orchestrator is not None}, scored={len(self._orchestrator._scored) if self._orchestrator else 0}', flush=True)
    return original_trigger(self, metadata)
sc.OffloadPackageScheduler._trigger_global_offload = debug_trigger

# Register backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

# Patch cuda
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module._get_attn_backend_class
def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)
cuda_module._get_attn_backend_class = patched_get
_original_get = cuda_module._get_attn_backend_class

# Register backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'offload_package.integration.original_optimized_backend.OriginalOptimizedBackend')

# Patch cuda
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module._get_attn_backend_class
def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)
cuda_module._get_attn_backend_class = patched_get
_original_get = cuda_module._get_attn_backend_class

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import torch, time

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

kv = KVTransferConfig(
    kv_connector='OffloadPackageConnector',
    kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
    kv_connector_extra_config={'enabled': True, 'offload_fraction': 0.75, 'num_cpu_slots': 4096, 'num_staging_slots': 64, 'interval_tokens': 8, 'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'},
    kv_role='kv_both',
)

from vllm import LLM, SamplingParams
llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv, disable_log_stats=True, attention_config={'backend': 'CUSTOM'})

def m():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

print(f'After load: {m():.2f} GB', flush=True)
_ = llm.generate(['test'], SamplingParams(temperature=0, max_tokens=1))
print(f'After warmup: {m():.2f} GB', flush=True)

sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, seed=42, stop_token_ids=[], ignore_eos=True)

print(f'Before generate: {m():.2f} GB', flush=True)
start = time.time()
outputs = llm.generate(prompts=['The future of AI is'] * 4, sampling_params=sp)
elapsed = time.time() - start
print(f'After generate: {m():.2f} GB (took {elapsed:.1f}s)', flush=True)
print(f'Output length: {len(outputs[0].outputs[0].text)} chars', flush=True)
