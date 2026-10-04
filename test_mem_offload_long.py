#!/usr/bin/env python
"""Test if offload eventually frees memory with longer run."""
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.1"
os.environ["SKYLIGHT_SPARSE_SINK"] = "64"
os.environ["SKYLIGHT_SPARSE_LOCAL"] = "64"
os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "8"
os.environ["SKYLIGHT_SYNC_STAGING"] = "1"

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def get_gpu_mem():
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9

def setup_backend():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, "offload_package.integration.original_optimized_backend.OriginalOptimizedBackend")
    import vllm.platforms.cuda as cuda_module
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES
    _orig_cuda_get_attn_backend_cls = cuda_module.CudaPlatform.get_attn_backend_cls
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    @classmethod
    def patched_cuda_get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
        if selected_backend is AttentionBackendEnum.CUSTOM or (selected_backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
            selected_backend = AttentionBackendEnum.CUSTOM
        return _orig_cuda_get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)
    cuda_module.CudaPlatform.get_attn_backend_cls = patched_cuda_get_attn_backend_cls

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadPackageConnector",
    kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
    kv_connector_extra_config={
        "enabled": True,
        "offload_fraction": 0.75,
        "num_cpu_slots": 4096,
        "num_staging_slots": 64,
        "interval_tokens": 64,  # More frequent offload
        "scorer_decay": 1.0,
        "scorer_head_reduction": "mean",
    },
    kv_role="kv_both",
)

setup_backend()

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.05,
    enforce_eager=True,
    tensor_parallel_size=1,
    kv_transfer_config=kv_transfer_config,
)

def get_mem():
    free, total = get_gpu_mem()
    return total - free

# Initial
print(f"After model load: {get_mem():.2f} GB used")

# Warmup
outputs = llm.generate(prompts=["test"], sampling_params=SamplingParams(temperature=0, max_tokens=1))
print(f"After warmup: {get_mem():.2f} GB used")

# Run many decode steps
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048, seed=42, stop_token_ids=[], ignore_eos=True)

for i in range(10):
    outputs = llm.generate(prompts=["The future of AI is"] * 4, sampling_params=sampling_params)
    mem = get_mem()
    print(f"After step {i+1}: {get_mem():.2f} GB used")

print(f"Final: {get_mem():.2f} GB used")
