#!/usr/bin/env python
"""FlashAttention WITH offloading"""

import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ.pop("VLLM_ATTENTION_BACKEND", None)
os.environ.pop("SKYLIGHT_SPARSE_TOPK", None)
os.environ.pop("SKYLIGHT_SPARSE_SINK", None)
os.environ.pop("SKYLIGHT_SPARSE_LOCAL", None)
os.environ.pop("SKYLIGHT_SPARSE_CHANNEL_NUM", None)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ.pop("VLLM_ATTENTION_BACKEND", None)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import time

# KV Transfer config with offloading enabled
# EASY TOGGLE: Set ENABLE_OFFLOADING = True to disable offloading
ENABLE_OFFLOADING = True

# Offload fraction (0 = no offloading, 0.5 = 50% offloading)
OFFLOAD_FRACTION = 0.5 if ENABLE_OFFLOADING else 0.0

kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadPackageConnector",
    kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
    kv_connector_extra_config={
        "enabled": ENABLE_OFFLOADING,
        "offload_fraction": OFFLOAD_FRACTION,
        "num_cpu_slots": 4096,
        "num_staging_slots": 64,
        "interval_tokens": 16,
        "scorer_decay": 1.0,
        "scorer_head_reduction": "mean",
    },
    kv_role="kv_both",
)

def test_flashattn_with_offload():
    print(f"[TEST] FLASHATTN + 50% OFFLOAD")
    start = time.time()
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        tensor_parallel_size=1,
        kv_transfer_config=kv_transfer_config,
    )
    init_time = time.time() - start
    print(f"[DEBUG] LLM created in {init_time:.2f}s")
    
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
    print(f"[PERF] FLASHATTN + {"ON" if ENABLE_OFFLOADING else "OFF"} OFFLOAD: Time: {elapsed:.2f}s, Tokens: 512, Tok/s: {512/elapsed:.1f}")
    
    output_text = outputs[0].outputs[0].text
    return elapsed, outputs[0].outputs[0].text

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    import time
    
    time_taken, text = test_flashattn_with_offload()
    print(f"\n=== FLASHATTN + 50% OFFLOAD RESULT ===")
    print(f"Time: {time_taken:.2f}s")
    print(f"Text: {text[:200]}...")
