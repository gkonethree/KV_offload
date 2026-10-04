#!/usr/bin/env python
"""Clean memory profiling for 4 configs."""
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
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9

def setup_env(use_sparse):
    for k in ["SKYLIGHT_SPARSE_METHOD", "SKYLIGHT_SPARSE_TOPK", "SKYLIGHT_SPARSE_SINK", "SKYLIGHT_SPARSE_LOCAL", "SKYLIGHT_SPARSE_CHANNEL_NUM", "SKYLIGHT_SYNC_STAGING"]:
        os.environ.pop(k, None)
    if use_sparse:
        os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
        os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.1"
        os.environ["SKYLIGHT_SPARSE_SINK"] = "64"
        os.environ["SKYLIGHT_SPARSE_LOCAL"] = "64"
        os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "8"

def setup_backend(use_sparse):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    if use_sparse:
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

def run_mem_test(label, use_offload=False, use_sparse=False):
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM" if use_sparse else "FLASHINFER"
    setup_env(use_sparse)
    setup_backend(use_sparse)
    
    if use_offload:
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadPackageConnector",
            kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
            kv_connector_extra_config={"enabled": True, "offload_fraction": 0.75, "num_cpu_slots": 4096, "num_staging_slots": 64, "interval_tokens": 128, "scorer_decay": 1.0, "scorer_head_reduction": "mean"},
            kv_role="kv_both",
        )
    else:
        kv_transfer_config = None
    
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype="float16", max_model_len=2048, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, kv_transfer_config=kv_transfer_config, disable_log_stats=True)
    
    free, total = get_gpu_mem()
    used_after_load = total - free
    print(f"  {label}: After model load: {used_after_load:.2f} GB used")
    
    # Warm up with a small generate
    outputs = llm.generate(prompts=["test"], sampling_params=SamplingParams(temperature=0, max_tokens=1))
    
    free, total = get_gpu_mem()
    used_after_prefill = total - free
    print(f"  {label}: After prefill (decode ready): {used_after_prefill:.2f} GB used")
    
    # Run a few decode steps
    outputs = llm.generate(prompts=["The future of AI is"] * 4, sampling_params=SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128, seed=42, stop_token_ids=[], ignore_eos=True))
    
    free, total = get_gpu_mem()
    used_after_decode = total - free
    print(f"  {label}: After decode steps: {used_after_decode:.2f} GB used")
    
    return used_after_load, used_after_prefill, used_after_decode

print("="*60)
print("GPU MEMORY PROFILING (2048 context, 256 decode tokens)")
print("="*60)

print("\n[1/4] FlashInfer baseline...")
b1, p1, d1 = run_mem_test("FlashInfer", use_offload=False, use_sparse=False)

print("\n[2/4] sparse_optimized (no offload)...")
b2, p2, d2 = run_mem_test("sparse_optimized", use_offload=False, use_sparse=True)

print("\n[3/4] sparse_optimized + 75% offload...")
b3, p3, d3 = run_mem_test("sparse_optimized+offload", use_offload=True, use_sparse=True)

print("\n[4/4] FlashInfer + 75% offload...")
b4, p4, d4 = run_mem_test("FlashInfer+offload", use_offload=True, use_sparse=False)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Config':<25} | {'After Load':>10} | {'After Prefill':>12} | {'After Decode':>11}")
print("-"*60)
for label, (b, p, d) in [
    ("FlashInfer", (b1, p1, d1)),
    ("sparse_optimized", (b2, p2, d2)),
    ("sparse+offload", (b3, p3, d3)),
    ("FlashInfer+offload", (b4, p4, d4)),
]:
    print(f"{label:<25} | {b:>9.2f} GB | {p:>11.2f} GB | {d:>10.2f} GB")

print("\nMemory freed by offload (load->decode delta):")
print(f"  FlashInfer:     {b1 - d1:.2f} GB")
print(f"  sparse_opt:     {b2 - d2:.2f} GB")
print(f"  sparse+offload: {b3 - d3:.2f} GB")
print(f"  FI+offload:     {b4 - d4:.2f} GB")
