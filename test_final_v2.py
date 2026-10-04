#!/usr/bin/env python3
"""Test Skylight backend with vLLM 0.29.0"""
import os
import time
import gc
import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '64'

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')

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

llm = LLM(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=1024,
    gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
    disable_log_stats=True, attention_config={'backend': 'CUSTOM'})

prompts = ['test'] * 2
sp = SamplingParams(temperature=0, max_tokens=16)

torch.cuda.synchronize()
start = time.time()
outputs = llm.generate(prompts, sp)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f'Success! Time: {elapsed:.2f}s')

del llm
gc.collect()
torch.cuda.empty_cache()
