#!/usr/bin/env python
"""Test 12k tokens with 75% offload, rescore after prefill + every 128 tokens."""
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"  # Allow longer context
os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.1"
os.environ["SKYLIGHT_SPARSE_SINK"] = "64"
os.environ["SKYLIGHT_SPARSE_LOCAL"] = "64"
os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "8"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import time

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, "offload_package.integration.original_optimized_backend.OriginalOptimizedBackend")

import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES
_orig_cuda_get_attn_backend_cls = cuda_module.CudaPlatform.get_attn_backend_cls
@classmethod
def patched_cuda_get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
    if selected_backend is AttentionBackendEnum.CUSTOM or (selected_backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        selected_backend = AttentionBackendEnum.CUSTOM
    return _orig_cuda_get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)
cuda_module.CudaPlatform.get_attn_backend_cls = patched_cuda_get_attn_backend_cls

# Test: 75% offload, interval 128 tokens, 12k tokens generation
frac = 0.75
label = "75% OFFLOAD + sparse_optimized (12k tokens, interval=128)"

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
        "interval_tokens": 128,  # Rescore every 128 tokens
        "scorer_decay": 1.0,
        "scorer_head_reduction": "mean",
    },
    kv_role="kv_both",
)

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    max_model_len=16384,  # Need longer context for 12k tokens
    gpu_memory_utilization=0.05,
    enforce_eager=True,
    tensor_parallel_size=1,
    kv_transfer_config=kv_transfer_config,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=12288,  # 12k tokens
    seed=42,
)

print(f"[DEBUG] Starting generation (12288 tokens)...")
start = time.time()

outputs = llm.generate(
    prompts=["The future of AI is"],
    sampling_params=sampling_params
)

elapsed = time.time() - start
tok_per_sec = 12288 / elapsed

output_text = outputs[0].outputs[0].text
print(f"\n[RESULT] {label}: {elapsed:.2f}s, {tok_per_sec:.1f} tok/s")
print(f"Output length: {len(output_text)} chars")
print(f"First 200 chars: {output_text[:200]}...")
print(f"Last 200 chars: {output_text[-200:] if len(output_text) > 200 else output_text}...")
