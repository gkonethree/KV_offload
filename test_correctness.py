#!/usr/bin/env python
"""Test output correctness: FlashInfer vs sparse_optimized vs sparse_optimized+offload."""
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import time

def run_test(label, use_offload=False, use_sparse_optimized=False):
    """Run a single test and return output text."""
    if use_sparse_optimized:
        os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
        os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.1"
        os.environ["SKYLIGHT_SPARSE_SINK"] = "64"
        os.environ["SKYLIGHT_SPARSE_LOCAL"] = "64"
        os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "8"
    else:
        os.environ.pop("SKYLIGHT_SPARSE_METHOD", None)
        os.environ.pop("SKYLIGHT_SPARSE_TOPK", None)
        os.environ.pop("SKYLIGHT_SPARSE_SINK", None)
        os.environ.pop("SKYLIGHT_SPARSE_LOCAL", None)
        os.environ.pop("SKYLIGHT_SPARSE_CHANNEL_NUM", None)
    
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM" if use_sparse_optimized else "FLASHINFER"
    
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    if use_sparse_optimized:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        register_backend(AttentionBackendEnum.CUSTOM, "offload_package.integration.original_optimized_backend.OriginalOptimizedBackend")
    else:
        # Clear any existing override
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
                "offload_fraction": 0.5,
                "num_cpu_slots": 4096,
                "num_staging_slots": 64,
                "interval_tokens": 16,
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
    
    outputs = llm.generate(
        prompts=["The future of AI is"],
        sampling_params=sampling_params
    )
    
    return outputs[0].outputs[0].text

# Run tests
print("Testing FlashInfer baseline...")
flashinfer_out = run_test("FlashInfer", use_offload=False, use_sparse_optimized=False)
print(f"FlashInfer output: {flashinfer_out[:200]}...")

print("\nTesting sparse_optimized (no offload)...")
sparse_out = run_test("sparse_optimized", use_offload=False, use_sparse_optimized=True)
print(f"sparse_optimized output: {sparse_out[:200]}...")

print("\nTesting sparse_optimized + 50% offload...")
sparse_offload_out = run_test("sparse_optimized+offload", use_offload=True, use_sparse_optimized=True)
print(f"sparse_optimized+offload output: {sparse_offload_out[:200]}...")

# Compare outputs
print("\n" + "="*60)
print("CORRECTNESS CHECK:")
print("="*60)
print(f"FlashInfer == sparse_optimized: {flashinfer_out == sparse_out}")
print(f"FlashInfer == sparse_optimized+offload: {flashinfer_out == sparse_offload_out}")
print(f"sparse_optimized == sparse_optimized+offload: {sparse_out == sparse_offload_out}")

if flashinfer_out == sparse_out == sparse_offload_out:
    print("\n✓ ALL OUTPUTS MATCH!")
else:
    print("\n✗ OUTPUTS DIFFER!")
    # Show first difference
    for i, (a, b, c) in enumerate(zip(flashinfer_out, sparse_out, sparse_offload_out)):
        if a != b or b != c:
            print(f"First difference at char {i}: FlashInfer='{a}', sparse='{b}', offload='{c}'")
            break
