#!/usr/bin/env python
"""Run all 5 configurations at 32k tokens."""
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

def run_test(label, use_offload=False, use_sparse=False, use_sync=False):
    """Run a single test and return output text and timing."""
    if use_sparse:
        os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
        os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.1"
        os.environ["SKYLIGHT_SPARSE_SINK"] = "64"
        os.environ["SKYLIGHT_SPARSE_LOCAL"] = "64"
        os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "8"
    else:
        for k in ["SKYLIGHT_SPARSE_METHOD", "SKYLIGHT_SPARSE_TOPK", "SKYLIGHT_SPARSE_SINK", "SKYLIGHT_SPARSE_LOCAL", "SKYLIGHT_SPARSE_CHANNEL_NUM"]:
            os.environ.pop(k, None)
    
    if use_sync:
        os.environ["SKYLIGHT_SYNC_STAGING"] = "1"
    else:
        os.environ.pop("SKYLIGHT_SYNC_STAGING", None)
    
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM" if use_sparse else "FLASHINFER"
    
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
    
    if use_offload:
        from vllm.config import KVTransferConfig
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
    
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        max_model_len=32768,
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        tensor_parallel_size=1,
        kv_transfer_config=kv_transfer_config,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=32768,
        seed=42,
        stop_token_ids=[],
        ignore_eos=True,
    )
    
    start = time.time()
    outputs = llm.generate(
        prompts=["The future of AI is"],
        sampling_params=sampling_params
    )
    elapsed = time.time() - start
    tok_per_sec = 32768 / elapsed
    output_text = outputs[0].outputs[0].text
    print(f"[RESULT] {label}: {elapsed:.2f}s, {tok_per_sec:.1f} tok/s, {len(output_text)} chars")
    del llm
    return elapsed, tok_per_sec, output_text

# Run all 5 configurations
print("="*80)
print("32K TOKEN BENCHMARK")
print("="*80)

# 1) FlashInfer baseline
print("\n[1/5] FlashInfer baseline...")
r1 = run_test("FlashInfer", use_offload=False, use_sparse=False, use_sync=False)

# 2) sparse_optimized no offload
print("\n[2/5] sparse_optimized (no offload)...")
r2 = run_test("sparse_optimized (no offload)", use_offload=False, use_sparse=True, use_sync=False)

# 3) sparse_optimized + 75% offload (0 lag)
print("\n[3/5] sparse_optimized + 75% offload (0 lag)...")
r3 = run_test("sparse_optimized + 75% offload (0 lag)", use_offload=True, use_sparse=True, use_sync=True)

# 4) FlashInfer + 75% offload (0 lag)
print("\n[4/5] FlashInfer + 75% offload (0 lag)...")
r4 = run_test("FlashInfer + 75% offload (0 lag)", use_offload=True, use_sparse=False, use_sync=True)

# 5) sparse_optimized + 75% offload (1 lag)
print("\n[5/5] sparse_optimized + 75% offload (1 lag)...")
r5 = run_test("sparse_optimized + 75% offload (1 lag)", use_offload=True, use_sparse=True, use_sync=False)

# Summary
print("\n" + "="*80)
print("SUMMARY: 32K TOKENS")
print("="*80)
print(f"| {'Config':<50} | {'Time (s)':>10} | {'tok/s':>10} | {'Speedup':>8} |")
print(f"|{'-'*50}|{'-'*10}|{'-'*10}|{'-'*8}|")
for label, (elapsed, tok_per_sec, out) in [
    ("FlashInfer", r1),
    ("sparse_optimized (no offload)", r2),
    ("sparse_optimized + 75% offload (0 lag)", r3),
    ("FlashInfer + 75% offload (0 lag)", r4),
    ("sparse_optimized + 75% offload (1 lag)", r5),
]:
    speedup = r1[1] / tok_per_sec if tok_per_sec > 0 else 0
    print(f"| {label:<50} | {elapsed:>10.2f} | {tok_per_sec:>10.1f} | {speedup:>7.2f}x |")

# Verify correctness
print("\n" + "="*80)
print("CORRECTNESS:")
outputs = [r[2] for r in [r1, r2, r3, r4, r5]]
labels = ["FlashInfer", "sparse_optimized", "sparse+offload(0lag)", "FI+offload(0lag)", "sparse+offload(1lag)"]
for i in range(len(outputs)):
    for j in range(i+1, len(outputs)):
        match = outputs[i] == outputs[j]
        print(f"  {labels[i]} == {labels[j]}: {'✓' if match else '✗'}")
