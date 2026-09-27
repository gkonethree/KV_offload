#!/usr/bin/env python
"""Test 99% offload in fresh process."""
import os
import sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ.pop("VLLM_ATTENTION_BACKEND", None)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import time

frac = 0.99
label = "99% OFFLOAD"

print(f"\n{'='*60}")
print(f"TEST: {label} (offload_fraction={frac})")
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
