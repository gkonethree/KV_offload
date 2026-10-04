#!/usr/bin/env python
"""Memory profiling for sparse and sparse+offload at decode start."""
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def get_gpu_mem():
    """Get GPU memory in GB."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9

def run_test(label, use_offload=False, use_sparse=False, max_model_len=2048, max_tokens=256):
    """Run test and print memory at decode start."""
    if use_sparse:
        os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
        os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.1"
        os.environ["SKYLIGHT_SPARSE_SINK"] = "64"
        os.environ["SKYLIGHT_SPARSE_LOCAL"] = "64"
        os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "8"
    else:
        for k in ["SKYLIGHT_SPARSE_METHOD", "SKYLIGHT_SPARSE_TOPK", "SKYLIGHT_SPARSE_SINK", "SKYLIGHT_SPARSE_LOCAL", "SKYLIGHT_SPARSE_CHANNEL_NUM"]:
            os.environ.pop(k, None)
    
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM" if use_sparse else "FLASHINFER"
    
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    if use_sparse:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        register_backend(AttentionBackendEnum.CUSTOM, "offload_package.integration.original_optimized_backend.OriginalOptimizedBackend")
    else:
        AttentionBackendEnum.CUSTOM.clear_override()
    
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
    
    if use_offload:
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadPackageConnector",
            kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
            kv_connector_extra_config={
                "enabled": True,
                "offload_fraction": 0.75,
                "num_cpu_slots": 4096,
                "num_staging_slots": 64,
                "interval_tokens": 128,
                "scorer_decay": 1.0,
                "scorer_head_reduction": "mean",
            },
            kv_role="kv_both",
        )
    else:
        kv_transfer_config = None
    
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        tensor_parallel_size=1,
        kv_transfer_config=kv_transfer_config,
    )
    
    # Check memory after model loaded, before generation
    free, total = get_gpu_mem()
    used = total - free
    print(f"[{label}] After model load: {used:.2f}/{total:.2f} GB used ({free:.2f} GB free)")
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
        seed=42,
        stop_token_ids=[],
        ignore_eos=True,
    )
    
    # Force prefill to complete by running with a prompt
    # The first generate call will trigger prefill
    outputs = llm.generate(
        prompts=["The future of AI is"] * 4,  # Batch of 4
        sampling_params=sampling_params
    )
    
    # After first generate (prefill done, decode started)
    free, total = get_gpu_mem()
    used = total - free
    print(f"[{label}] After 1st generate (decode started): {used:.2f}/{total:.2f} GB used ({free:.2f} GB free)")
    
    # Run a few more decode steps
    for i in range(3):
        outputs = llm.generate(
            prompts=["The future of AI is"] * 4,
            sampling_params=sampling_params
        )
        free, total = get_gpu_mem()
        used = total - free
        print(f"[{label}] After generate {i+2}: {used:.2f}/{total:.2f} GB used ({free:.2f} GB free)")
    
    return outputs

# Test configurations
print("="*80)
print("MEMORY PROFILING AT DECODE START")
print("="*80)

# 1) FlashInfer baseline
print("\n[1] FlashInfer baseline...")
free, total = get_gpu_mem()
print(f"Initial: {total:.2f} GB total, {free:.2f} GB free")
run_test("FlashInfer", use_offload=False, use_sparse=False, max_model_len=2048, max_tokens=256)

# 2) sparse_optimized no offload
print("\n[2] sparse_optimized (no offload)...")
os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.1"
os.environ["SKYLIGHT_SPARSE_SINK"] = "64"
os.environ["SKYLIGHT_SPARSE_LOCAL"] = "64"
os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "8"
run_test("sparse_optimized", use_offload=False, use_sparse=True, max_model_len=2048, max_tokens=256)

# 3) sparse_optimized + 75% offload
print("\n[3] sparse_optimized + 75% offload...")
os.environ["SKYLIGHT_SYNC_STAGING"] = "1"
run_test("sparse_optimized+offload", use_offload=True, use_sparse=True, max_model_len=2048, max_tokens=256)

# 4) FlashInfer + 75% offload
print("\n[4] FlashInfer + 75% offload...")
os.environ.pop("SKYLIGHT_SPARSE_METHOD", None)
os.environ.pop("SKYLIGHT_SYNC_STAGING", None)
run_test("FlashInfer+offload", use_offload=True, use_sparse=False, max_model_len=2048, max_tokens=256)

print("\n" + "="*80)
print("DONE")
