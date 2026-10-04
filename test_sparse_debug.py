#!/usr/bin/env python3
"""Debug sparse attention pattern for small prompt"""
import os
import torch

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

# Small prompt: 10 tokens, generate 8
prompts = ['Hello world ' * 2]  # ~20 chars * 2 = 40 chars ≈ 10 tokens

sp = SamplingParams(
    temperature=0, 
    top_p=1.0, 
    max_tokens=8, 
    ignore_eos=True,
    logprobs=5
)

llm = LLM(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
    gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
    disable_log_stats=True,
    attention_config={'backend': 'CUSTOM'},
)

outputs = llm.generate(prompts, sp)

for o in outputs:
    print(f"Prompt: {o.prompt}")
    print(f"Generated tokens: {o.outputs[0].token_ids}")
    print(f"Generated text: {o.outputs[0].text}")
    if o.outputs[0].logprobs:
        for i, step in enumerate(o.outputs[0].logprobs):
            top_token = max(step.items(), key=lambda x: x[1].logprob)
            print(f"  Step {i}: top_token={top_token[0]}, logprob={top_token[1].logprob:.4f}")