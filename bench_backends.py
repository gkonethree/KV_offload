#!/usr/bin/env python3
"""Benchmark different attention backends (dense vs sparse) without offloading."""
import os, time, torch
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_PLUGINS'] = ''
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

# ===== BACKEND CONFIGS TO TEST =====
# Dense FlashInfer baseline
DENSE_ENV = {
    'VLLM_ATTENTION_BACKEND': 'FLASHINFER',
}

# OriginalOptimized (dense, from skylight_kernels)
ORIGINAL_OPTIMIZED_ENV = {
    'VLLM_ATTENTION_BACKEND': 'CUSTOM',
    'SKYLIGHT_SPARSE_METHOD': 'dense',  # Not a real option, will fail
}

# sparse_optimized (fast CUDA kernel, needs external pattern)
SPARSE_OPTIMIZED_ENV = {
    'VLLM_ATTENTION_BACKEND': 'CUSTOM',
    'SKYLIGHT_SPARSE_METHOD': 'sparse_optimized',
    'SKYLIGHT_SPARSE_TOPK': '0.1',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '8',
    'SKYLIGHT_SYNC_STAGING': '0',
}

# Oracle sparse (internal scoring, slow)
ORACLE_ENV = {
    'VLLM_ATTENTION_BACKEND': 'CUSTOM',
    'SKYLIGHT_SPARSE_METHOD': 'oracle',
    'SKYLIGHT_SPARSE_TOPK': '0.1',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '-1',
    'SKYLIGHT_SYNC_STAGING': '0',
}

# Block minmax (incremental)
BLOCK_MINMAX_ENV = {
    'VLLM_ATTENTION_BACKEND': 'CUSTOM',
    'SKYLIGHT_SPARSE_METHOD': 'block_minmax',
    'SKYLIGHT_SPARSE_TOPK': '0.1',
    'SKYLIGHT_SPARSE_SINK': '64',
    'SKYLIGHT_SPARSE_LOCAL': '64',
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '-1',
    'SKYLIGHT_SPARSE_BUDGET': 'frac:0.1',
}

from vllm import LLM, SamplingParams

def run_bench(name, env_dict, model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', max_model_len=2048, input_len=512, output_len=128, num_prompts=4):
    """Run benchmark with given env config."""
    # Apply env
    for k, v in env_dict.items():
        os.environ[k] = v
    
    # Clear conflicting env vars
    for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK', 'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM', 'SKYLIGHT_SYNC_STAGING', 'SKYLIGHT_SPARSE_BUDGET']:
        if k not in env_dict and k in os.environ:
            del os.environ[k]
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Env: {env_dict}")
    print(f"{'='*60}")
    
    llm = None
    try:
        llm = LLM(
            model=model,
            dtype='float16',
            max_model_len=max_model_len,
            gpu_memory_utilization=0.05,  # Lower to avoid OOM between runs
            enforce_eager=True,
            tensor_parallel_size=1,
            disable_log_stats=True,
        )
        
        prompts = ['The future of AI is ' * (input_len // 16)] * num_prompts
        sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=output_len, ignore_eos=True)
        
        torch.cuda.synchronize()
        start = time.time()
        outputs = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tpot = (elapsed / total_tokens) * 1000  # ms per token
        throughput = total_tokens / elapsed
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Throughput: {throughput:.1f} tok/s")
        print(f"  TPOT: {tpot:.2f} ms/token")
        print(f"  Sample output: {outputs[0].outputs[0].text[:80]}")
        
        return {
            'name': name,
            'time': elapsed,
            'tokens': total_tokens,
            'throughput': throughput,
            'tpot': tpot,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'name': name, 'error': str(e)}
    finally:
        import gc
        if llm is not None:
            del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(3)

if __name__ == '__main__':
    import gc
    
    configs = [
        ('FlashInfer (Dense)', DENSE_ENV),
        ('sparse_optimized (fast CUDA)', SPARSE_OPTIMIZED_ENV),
        # ('Oracle (sparse_oracle_topk_sink_local)', ORACLE_ENV),  # Too slow, skip
        ('Block Minmax (incremental)', BLOCK_MINMAX_ENV),
    ]
    
    # Test with longer context (TinyLlama max is 2048)
    test_cases = [
        ('short (512 in, 128 out)', 512, 128, 4),
        ('medium (1024 in, 256 out)', 1024, 256, 4),
        # ('long (2048 in, 128 out)', 2048, 128, 2),
    ]
    
    results = []
    for case_name, in_len, out_len, n_prompts in test_cases:
        print(f"\n\n{'#'*60}")
        print(f"Test case: {case_name}")
        print(f"{'#'*60}")
        
        for name, env in configs:
            # Skip OriginalOptimized - it's dense with different kernel, not a backend name
            if name == 'OriginalOptimized (Dense)':
                # Test this with CUSTOM + dense method
                continue
            
            r = run_bench(name, env, max_model_len=8192, input_len=in_len, output_len=out_len, num_prompts=n_prompts)
            r['case'] = case_name
            results.append(r)
            
            # Clear memory between runs
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Case':<25} {'Backend':<35} {'Time(s)':>8} {'Throughput':>12} {'TPOT(ms)':>10}")
    print(f"{'-'*80}")
    for r in results:
        if 'error' not in r:
            print(f"{r['case']:<25} {r['name']:<35} {r['time']:>8.2f} {r['throughput']:>12.1f} {r['tpot']:>10.2f}")
        else:
            print(f"{r['case']:<25} {r['name']:<35} {'ERROR':>8}")
