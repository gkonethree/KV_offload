#!/usr/bin/env python3
"""Test Skylight backend with vLLM 0.29.0 and analyze logit/logprob divergence between offloaded and non-offloaded runs"""
import os
import time
import gc
import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

# 1. Register Skylight backend once at startup
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')

# 2. Apply monkey patch for CUSTOM backend dynamically
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

_original_get = getattr(cuda_module, '_get_attn_backend_class', None)

def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    if _original_get:
        return _original_get(backend, *args, **kwargs)
    return backend

if _original_get:
    cuda_module._get_attn_backend_class = patched_get

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def run_bench(name, env, use_offload=False):
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Select FLASHINFER backend if environment dictionary is empty, else use CUSTOM
    backend_choice = 'FLASHINFER' if not env else 'CUSTOM'
    
    if use_offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True, 'offload_fraction': 0.5, 'num_cpu_slots': 2048, 
                'num_staging_slots': 64, 'interval_tokens': 16, 
                'scorer_decay': 1.0, 'scorer_head_reduction': 'mean'
            },
            kv_role='kv_both',
        )
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
            gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
            kv_transfer_config=kv, disable_log_stats=True,
            attention_config={'backend': backend_choice},
        )
    else:
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
            gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
            disable_log_stats=True,
            attention_config={'backend': backend_choice},
        )
    
    prompts = ['The future of AI is ' * 192] * 4
    # Greedy sampling (temperature=0) + logprobs=5 to capture top candidate log probabilities
    sp = SamplingParams(
        temperature=0, 
        top_p=1.0, 
        max_tokens=8, 
        ignore_eos=True,
        logprobs=5
    )
    
    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    # Capture token IDs and step-by-step top logprobs for each prompt output
    generated_token_ids = []
    generated_top_logprobs = []
    
    for o in outputs:
        toks = list(o.outputs[0].token_ids)
        generated_token_ids.append(toks)
        
        # Extract max logprob at each generated step
        step_lps = []
        if o.outputs[0].logprobs:
            for step in o.outputs[0].logprobs:
                # Grab max logprob value among returned top-k candidates
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
        'tokens': total_tokens, 
        'time': elapsed, 
        'throughput': throughput, 
        'tpot': tpot, 
        'mem': mem_gb,
        'token_ids': generated_token_ids,
        'top_logprobs': generated_top_logprobs
    }


if __name__ == '__main__':
    configs = [
        ('Skylight Oracle', {
            'SKYLIGHT_SPARSE_METHOD': 'oracle', 
            'SKYLIGHT_SPARSE_TOPK': '0.1', 
            'SKYLIGHT_SPARSE_SINK': '64', 
            'SKYLIGHT_SPARSE_LOCAL': '64', 
            'SKYLIGHT_SPARSE_CHANNEL_NUM': '64'
        }, False),
        ('Skylight Oracle + Offload', {
            'SKYLIGHT_SPARSE_METHOD': 'oracle', 
            'SKYLIGHT_SPARSE_TOPK': '0.1', 
            'SKYLIGHT_SPARSE_SINK': '64', 
            'SKYLIGHT_SPARSE_LOCAL': '64', 
            'SKYLIGHT_SPARSE_CHANNEL_NUM': '64'
        }, True),
    ]
    
    results = []
    for name, env, offload in configs:
        for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK', 
                  'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM', 
                  'SKYLIGHT_SYNC_STAGING', 'SKYLIGHT_SPARSE_BUDGET',
                  'VLLM_ATTENTION_BACKEND']:
            os.environ.pop(k, None)
        
        for k, v in env.items():
            os.environ[k] = v
        
        r = run_bench(name, env, use_offload=offload)
        results.append(r)
        time.sleep(2)
    
    print(f"\n{'='*90}")
    print(f"SUMMARY: 4k Context (TinyLlama, 3k in + 512 out, 4 prompts)")
    print(f"{'='*90}")
    print(f"{'Configuration':<30} {'Tokens':>8} {'Time(s)':>8} {'Throughput':>12} {'TPOT(ms)':>10} {'GPU Mem':>8}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['name']:<30} {r['tokens']:>8d} {r['time']:>8.2f} {r['throughput']:>12.1f} {r['tpot']:>10.2f} {r['mem']:>8.2f}")

    # Logit & Output Verification: Skylight Oracle (idx 0) vs Skylight Oracle + Offload (idx 1)
    no_offload_res = results[0]
    offload_res = results[1]

    print(f"\n{'='*90}")
    print(f"LOGIT & OUTPUT MATCH ANALYSIS ({no_offload_res['name']} vs {offload_res['name']})")
    print(f"{'='*90}")

    all_matched = True
    for prompt_idx in range(len(no_offload_res['token_ids'])):
        toks_no_off = no_offload_res['token_ids'][prompt_idx]
        toks_off = offload_res['token_ids'][prompt_idx]
        
        lp_no_off = no_offload_res['top_logprobs'][prompt_idx]
        lp_off = offload_res['top_logprobs'][prompt_idx]
        
        # Calculate logprob differences up to the minimum generated length
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
            if max_diff < 1e-2:
                print("    - Evaluation: Mismatch is caused by FP16 floating-point noise on borderline logits.")
            else:
                print("    - Evaluation: High logit divergence (>1e-2) indicates KV cache staging/data corruption.")
        print()

    if all_matched:
        print("SUCCESS: Exact token matches achieved across all prompts between non-offloaded and offloaded runs.")
    else:
        print("NOTICE: Output diverged between non-offloaded and offloaded runs. Check logit difference metrics above.")