#!/usr/bin/env python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["VLLM_PLUGINS"] = ""
os.environ["SKYLIGHT_SPARSE_METHOD"] = "sparse_optimized"
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

def run_test(topk_frac, label):
    os.environ["SKYLIGHT_SPARSE_TOPK"] = str(topk_frac)
    
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        seed=42,
    )
    
    start = time.time()
    outputs = llm.generate(
        prompts=["The future of AI is"],
        sampling_params=sampling_params
    )
    elapsed = time.time() - start
    tok_per_sec = 512 / elapsed
    print(f"{label} (topk={topk_frac}): {elapsed:.2f}s, {tok_per_sec:.1f} tok/s")
    del llm

# FlashInfer baseline
os.environ.pop("SKYLIGHT_SPARSE_METHOD", None)
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    max_model_len=1024,
    gpu_memory_utilization=0.05,
    enforce_eager=True,
    tensor_parallel_size=1,
)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, seed=42)
start = time.time()
_ = llm.generate(prompts=["The future of AI is"], sampling_params=sampling_params)
elapsed = time.time() - start
print(f"FlashInfer: {elapsed:.2f}s, {512/elapsed:.1f} tok/s")
del llm

# sparse_optimized at different sparsities
for topk_frac, label in [(0.1, "10%"), (0.05, "5%"), (0.02, "2%"), (0.01, "1%")]:
    run_test(topk_frac, label)
