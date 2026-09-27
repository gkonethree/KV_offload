#!/usr/bin/env python
"""Test 12k tokens FlashInfer baseline."""
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

from vllm import LLM, SamplingParams
import time

# FlashInfer baseline
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    max_model_len=16384,
    gpu_memory_utilization=0.05,
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=12288,
    seed=42,
    stop_token_ids=[],
    ignore_eos=True,
)

print(f"[DEBUG] Starting FlashInfer 12288 tokens...")
start = time.time()

outputs = llm.generate(
    prompts=["The future of AI is"],
    sampling_params=sampling_params
)

elapsed = time.time() - start
tok_per_sec = 12288 / elapsed

output_text = outputs[0].outputs[0].text
print(f"\n[RESULT] FlashInfer 12k tokens: {elapsed:.2f}s, {tok_per_sec:.1f} tok/s")
print(f"Output length: {len(output_text)} chars")
