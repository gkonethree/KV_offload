#!/usr/bin/env python
"""Test original_optimized kernel with offloading."""
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_PLUGINS"] = ""  # Disable all general plugins to avoid skylight plugin

# Now import vllm
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import time

# Register our backend AFTER vllm imports (plugins disabled)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, "offload_package.integration.original_optimized_backend.OriginalOptimizedBackend")

# Monkey-patch cuda.py get_attn_backend_cls to handle CUSTOM properly
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES
_orig_cuda_get_attn_backend_cls = cuda_module.CudaPlatform.get_attn_backend_cls
@classmethod
def patched_cuda_get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
    if selected_backend is AttentionBackendEnum.CUSTOM or (selected_backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        selected_backend = AttentionBackendEnum.CUSTOM
    return _orig_cuda_get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)
cuda_module.CudaPlatform.get_attn_backend_cls = patched_cuda_get_attn_backend_cls

# Test 0% offload with original_optimized backend (baseline)
frac = 0.0
label = "0% OFFLOAD + original_optimized"

print(f"\n{'='*60}")
print(f"TEST: {label}")
print("="*60)

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadPackageConnector",
    kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
    kv_connector_extra_config={
        "enabled": True,
        "offload_fraction": frac,
        "num_cpu_slots": 4096,
        "num_staging_slots": 64,
        "interval_tokens": 16,
        "scorer_decay": 1.0,
        "scorer_head_reduction": "mean",
    },
    kv_role="kv_both",
)

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    max_model_len=1024,
    gpu_memory_utilization=0.05,
    enforce_eager=True,
    tensor_parallel_size=1,
    kv_transfer_config=kv_transfer_config,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    seed=42,
)

print(f"[DEBUG] Starting generation (512 tokens)...")
start = time.time()

outputs = llm.generate(
    prompts=["The future of AI is"],
    sampling_params=sampling_params
)

elapsed = time.time() - start
tok_per_sec = 512 / elapsed

output_text = outputs[0].outputs[0].text
print(f"\n[RESULT] {label}: {elapsed:.2f}s, {tok_per_sec:.1f} tok/s")
print(f"Output: {output_text[:100]}...")
