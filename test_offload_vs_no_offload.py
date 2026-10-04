#!/usr/bin/env python3
"""Test Skylight Oracle + Offload vs Skylight Oracle (no offload)"""
import os
import torch
import gc

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

# Set env vars for sparse oracle FIRST (before importing vllm)
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '4'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '4'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '64'

# Register Skylight backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')

# Monkey patch for CUSTOM backend
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

_original_get = getattr(cuda_module, '_get_attn_backend_class', None)

def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend_enum = AttentionBackendEnum.CUSTOM
        return backend_enum.get_class()
    if _original_get:
        return _original_get(backend, *args, **kwargs)
    return backend

if _original_get:
    cuda_module._get_attn_backend_class = patched_get

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Small prompt: 10 tokens, generate 8
prompts = ['Hello world ' * 2]

sp = SamplingParams(
    temperature=0, 
    top_p=1.0, 
    max_tokens=8, 
    ignore_eos=True,
    logprobs=5
)

# Test 1: Skylight Oracle (no offload)
print("=" * 60)
print("Test 1: Skylight Oracle (no offload)")
print("=" * 60)

llm_sparse = LLM(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
    gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
    disable_log_stats=True,
    attention_config={'backend': 'CUSTOM'},
)

outputs_sparse = llm_sparse.generate(prompts, sp)

for o in outputs_sparse:
    print(f"Generated tokens: {o.outputs[0].token_ids}")
    print(f"Generated text: {o.outputs[0].text}")

sparse_tokens = outputs_sparse[0].outputs[0].token_ids

del llm_sparse
gc.collect()
torch.cuda.empty_cache()

# Test 2: Skylight Oracle + Offload
print("\n" + "=" * 60)
print("Test 2: Skylight Oracle + Offload (interval=16)")
print("=" * 60)

kv = KVTransferConfig(
    kv_connector='OffloadPackageConnector',
    kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
    kv_connector_extra_config={
        'enabled': True, 'offload_fraction': 0.5, 'num_cpu_slots': 2048, 
        'num_staging_slots': 64, 'interval_tokens': 16, 
        'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'
    },
    kv_role='kv_both',
)

llm_offload = LLM(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
    gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
    kv_transfer_config=kv, disable_log_stats=True,
    attention_config={'backend': 'CUSTOM'},
)

outputs_offload = llm_offload.generate(prompts, sp)

for o in outputs_offload:
    print(f"Generated tokens: {o.outputs[0].token_ids}")
    print(f"Generated text: {o.outputs[0].text}")

offload_tokens = outputs_offload[0].outputs[0].token_ids

print("\n" + "=" * 60)
print("COMPARISON: Skylight Oracle vs Skylight Oracle + Offload")
print("=" * 60)
print(f"No Offload:  {sparse_tokens}")
print(f"With Offload: {offload_tokens}")
print(f"Match: {sparse_tokens == offload_tokens}")

if sparse_tokens != offload_tokens:
    for i, (s, o) in enumerate(zip(sparse_tokens, offload_tokens)):
        if s != o:
            print(f"  First mismatch at step {i}: no_offload={s}, offload={o}")
            break
else:
    print("SUCCESS: Offloading does not add additional divergence!")