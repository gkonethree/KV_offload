#!/usr/bin/env python3
"""Compare Skylight Oracle vs Skylight Oracle + Offload in same process"""
import os
import time
import gc
import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

# Set env vars for sparse oracle FIRST (before importing vllm)
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '64'

# Register Skylight backend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')

# Monkey patch for CUSTOM backend
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

_original_get = getattr(cuda_module, '_get_attn_backend_class', None)

def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend_enum = AttentionBackendEnum.CUSTOM
        return backend_enum.get_class()
    if _original_get:
        return _original_get(backend, *args, **kwargs)
    return backend

if _original_get:
    cuda_module._get_attn_backend_class = patched_get

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# 4 prompts × ~300 tokens input, generate 512 tokens each
prompts = ['The future of AI is ' * 192] * 4
sp = SamplingParams(
    temperature=0, 
    top_p=1.0, 
    max_tokens=512, 
    ignore_eos=True,
    logprobs=5
)

def run_with_config(name, use_offload):
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    if use_offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True, 'offload_fraction': 0.5, 'num_cpu_slots': 2048, 
                'num_staging_slots': 64, 'interval_tokens': 128, 
                'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'
            },
            kv_role='kv_both',
        )
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
            gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
            kv_transfer_config=kv, disable_log_stats=True,
            attention_config={'backend': 'CUSTOM'},
        )
    else:
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
            gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
            disable_log_stats=True,
            attention_config={'backend': 'CUSTOM'},
        )
    
    torch.manual_seed(0)
    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    generated_token_ids = []
    generated_top_logprobs = []
    
    for o in outputs:
        toks = list(o.outputs[0].token_ids)
        generated_token_ids.append(toks)
        
        step_lps = []
        if o.outputs[0].logprobs:
            for step in o.outputs[0].logprobs:
                top_val = max(lp.logprob for lp in step.values())
                step_lps.append(top_val)
        generated_top_logprobs.append(torch.tensor(step_lps, dtype=torch.float32))
    
    total_tokens = sum(len(toks) for toks in generated_token_ids)
    tpot = (elapsed / total_tokens) * 1000 if total_tokens > 0 else 0
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    
    free, total = torch.cuda.mem_get_info()
    mem_gb = (total - free) / 1e9
    
    print(f"  Generated Tokens: {total_tokens}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  TPOT: {tpot:.2f} ms/token")
    print(f"  GPU Mem: {mem_gb:.2f} GB")
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    
    return {
        'name': name, 
        'token_ids': generated_token_ids,
        'top_logprobs': generated_top_logprobs
    }

# Run both configs in SAME process
print("Running both configs in SAME process for fair comparison")
no_offload_res = run_with_config("Skylight Oracle (no offload)", False)
offload_res = run_with_config("Skylight Oracle + Offload", True)

# Compare
print(f"\n{'='*90}")
print(f"LOGIT & OUTPUT MATCH ANALYSIS ({no_offload_res['name']} vs {offload_res['name']})")
print(f"{'='*90}")

all_matched = True
for prompt_idx in range(len(no_offload_res['token_ids'])):
    toks_no_off = no_offload_res['token_ids'][prompt_idx]
    toks_off = offload_res['token_ids'][prompt_idx]
    
    lp_no_off = no_offload_res['top_logprobs'][prompt_idx]
    lp_off = offload_res['top_logprobs'][prompt_idx]
    
    min_len = min(len(lp_no_off), len(lp_off))
    if min_len > 0:
        diff = torch.abs(lp_no_off[:min_len] - lp_off[:min_len])
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
    else:
        max_diff, mean_diff = 0.0, 0.0

    is_token_match = (toks_no_off == toks_off)
    if not is_token_match:
        all_matched = False

    status_str = "MATCH" if is_token_match else "MISMATCH"
    print(f"  Prompt {prompt_idx + 1}: [{status_str}]")
    print(f"    - Max Logit/Logprob Diff:  {max_diff:.6e}")
    print(f"    - Mean Logit/Logprob Diff: {mean_diff:.6e}")
    
    if not is_token_match:
        first_mismatch_step = next((i for i, (t1, t2) in enumerate(zip(toks_no_off, toks_off)) if t1 != t2), None)
        print(f"    - First token mismatch at step: {first_mismatch_step}")
        print(f"    - No-offload tokens: {toks_no_off[:10]}...")
        print(f"    - Offload tokens: {toks_off[:10]}...")
        if max_diff < 1e-2:
            print("    - Evaluation: Mismatch is caused by FP16 floating-point noise on borderline logits.")
        else:
            print("    - Evaluation: High logit divergence (>1e-2) indicates KV cache staging/data corruption.")
    print()

if all_matched:
    print("SUCCESS: Exact token matches achieved across all prompts between non-offloaded and offloaded runs.")
else:
    print("NOTICE: Output diverged between non-offloaded and offloaded runs. Check logit difference metrics above.")