#!/usr/bin/env python
"""1) BASELINE: No sparse kernel, no offloading - standard vLLM FlashAttention"""

import os
import sys
import time

# Disable multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

# Standard vLLM - no custom attention backend
os.environ.pop("VLLM_ATTENTION_BACKEND", None)
os.environ.pop("SKYLIGHT_SPARSE_TOPK", None)
os.environ.pop("SKYLIGHT_SPARSE_SINK", None)
os.environ.pop("SKYLIGHT_SPARSE_LOCAL", None)
os.environ.pop("SKYLIGHT_SPARSE_CHANNEL_NUM", None)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ.pop("VLLM_ATTENTION_BACKEND", None)

from vllm import LLM, SamplingParams
import time

def test_baseline():
    print(f"[TEST] BASELINE: No sparse, no offloading")
    start = time.time()
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        tensor_parallel_size=1,
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
    print(f"[PERF] BASELINE: Time: {elapsed:.2f}s, Tokens: 512, Tok/s: {512/elapsed:.1f}")
    
    output_text = outputs[0].outputs[0].text
    return elapsed, outputs[0].outputs[0].text

if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    import time
    time_taken, text = test_baseline()
    print(f"\n=== BASELINE RESULT ===")
    print(f"Time: {time_taken:.2f}s")
    print(f"Text: {text[:200]}...")
