#!/usr/bin/env python
"""Compare FlashAttention with different offloading fractions."""

import os
import sys
import time

# Disable multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ.pop("VLLM_ATTENTION_BACKEND", None)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import time

# Offload fractions to test
FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]  # 0%, 25%, 50%, 75%, 100%
# Note: 1.0 might not be valid, so we'll use 0.99 instead

def run_test(offload_fraction, label):
    """Run a single test with given offload fraction."""
    print(f"\n{'='*60}")
    print(f"TEST: {label} (offload_fraction={offload_fraction})")
    print("="*60)
    
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadPackageConnector",
        kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
        kv_connector_extra_config={
            "enabled": True,
            "offload_fraction": offload_fraction,
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
    
    prompts = ["The future of AI is"] * 1  # Single prompt
    
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
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            seed=42,
        )
    )
    
    elapsed = time.time() - start
    tok_per_sec = 512 / elapsed
    
    output_text = outputs[0].outputs[0].text
    return elapsed, outputs[0].outputs[0].text

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    import time
    
    # Test fractions: 0%, 25%, 50%, 75%, 99% (100% not allowed)
    fractions = [0.0, 0.25, 0.50, 0.75, 0.99]
    labels = ["0% OFFLOAD", "25% OFFLOAD", "50% OFFLOAD", "75% OFFLOAD", "99% OFFLOAD"]
    
    results = []
    
    for frac, label in zip(fractions, labels):
        try:
            elapsed, text = run_test(offload_fraction=frac, label=label)
            results.append((label, frac, elapsed, 512/elapsed))
            print(f"\n[RESULT] {label}: {elapsed:.2f}s, {512/elapsed:.1f} tok/s")
        except Exception as e:
            print(f"[ERROR] {label} failed: {e}")
            results.append((label, frac, None, None))
        
        # Small delay between runs
        import time
        time.sleep(2)
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<20} | {'Time (s)':>10} | {'Throughput (tok/s)':>15} | {'Speedup':>10}")
    print("-"*80)
    
    baseline_time = None
    for label, frac, elapsed, tok_per_sec in results:
        if elapsed is not None:
            if baseline_time is None:
                baseline_time = elapsed
                speedup = 1.0
            else:
                speedup = baseline_time / elapsed
            print(f"{label:<20} | {elapsed:>10.2f} | {512/elapsed:>15.1f} | {speedup:>9.2f}x")
        else:
            print(f"{label:<20} | {'FAILED':>10} | {'N/A':>15} | {'N/A':>10}")
    
    print("="*80)
