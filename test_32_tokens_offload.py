#!/usr/bin/env python3
"""Test 32 tokens: Skylight Oracle + Offload vs Skylight Oracle (no offload)"""
import os
import json
import subprocess
import sys
import time

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

def run_worker(name, env_vars, use_offload):
    import os
    import time
    import gc
    import torch

    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

    for k, v in env_vars.items():
        os.environ[k] = v

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    is_custom = bool(env_vars)

    if is_custom:
        from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
        register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')

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

        attn_config = {'backend': 'CUSTOM'}
    else:
        attn_config = {'backend': 'FLASHINFER'}

    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

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
            attention_config=attn_config,
        )
    else:
        llm = LLM(
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
            gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
            disable_log_stats=True,
            attention_config=attn_config,
        )

    # 10 tokens input, generate 32 tokens
    prompts = ['Hello world ' * 2]

    sp = SamplingParams(
        temperature=0, 
        top_p=1.0, 
        max_tokens=32, 
        ignore_eos=True,
        logprobs=5
    )

    import torch
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

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} tok/s")
    print(f"  TPOT: {tpot:.2f} ms/token")
    print(f"  GPU Mem: {mem_gb:.2f} GB")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)

    result = {
        'name': name, 
        'time': elapsed, 
        'throughput': throughput, 
        'tpot': tpot, 
        'mem': mem_gb,
        'token_ids': generated_token_ids,
        'top_logprobs': [lp.tolist() for lp in generated_top_logprobs]
    }
    print(f"__RESULT_JSON_START__{json.dumps(result)}__RESULT_JSON_END__")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        import torch
        config_data = json.loads(sys.argv[2])
        run_worker(config_data['name'], config_data['env'], config_data['offload'])
        sys.exit(0)

    configs = [
        {
            'name': 'Skylight Oracle (no offload)',
            'env': {
                'SKYLIGHT_SPARSE_METHOD': 'oracle',
                'SKYLIGHT_SPARSE_TOPK': '0.1',
                'SKYLIGHT_SPARSE_SINK': '64',
                'SKYLIGHT_SPARSE_LOCAL': '64',
                'SKYLIGHT_SPARSE_CHANNEL_NUM': '64'
            },
            'offload': False
        },
        {
            'name': 'Skylight Oracle + Offload',
            'env': {
                'SKYLIGHT_SPARSE_METHOD': 'oracle',
                'SKYLIGHT_SPARSE_TOPK': '0.1',
                'SKYLIGHT_SPARSE_SINK': '64',
                'SKYLIGHT_SPARSE_LOCAL': '64',
                'SKYLIGHT_SPARSE_CHANNEL_NUM': '64'
            },
            'offload': True
        },
    ]

    results = []
    for cfg in configs:
        cmd = [sys.executable, __file__, '--worker', json.dumps(cfg)]
        env = os.environ.copy()

        for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK',
                  'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM',
                  'SKYLIGHT_SYNC_STAGING', 'SKYLIGHT_SPARSE_BUDGET', 'VLLM_ATTENTION_BACKEND']:
            env.pop(k, None)

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        output = ""
        for line in proc.stdout:
            print(line, end="")
            output += line
        proc.wait()

        print(f"[DEBUG] Return code: {proc.returncode}")
        if "__RESULT_JSON_START__" in output:
            json_str = output.split("__RESULT_JSON_START__")[1].split("__RESULT_JSON_END__")[0]
            results.append(json.loads(json_str))
        else:
            print("[ERROR] No JSON result found in output")

    print(f"\n{'='*90}")
    print(f"COMPARISON: Skylight Oracle vs Skylight Oracle + Offload (32 tokens)")
    print(f"{'='*90}")
    print(f"Number of results: {len(results)}")
    for i, r in enumerate(results):
        print(f"  Result {i}: {r['name']}")

    if len(results) < 2:
        print("ERROR: Not enough results to compare!")
        sys.exit(1)

    no_offload_res = results[0]
    offload_res = results[1]

    import torch
    all_matched = True
    for prompt_idx in range(len(no_offload_res['token_ids'])):
        toks_no_off = no_offload_res['token_ids'][prompt_idx]
        toks_off = offload_res['token_ids'][prompt_idx]

        lp_no_off = torch.tensor(no_offload_res['top_logprobs'][prompt_idx])
        lp_off = torch.tensor(offload_res['top_logprobs'][prompt_idx])

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
            print(f"    - No-offload tokens: {toks_no_off}")
            print(f"    - Offload tokens:    {toks_off}")
        print()

    if all_matched:
        print("SUCCESS: Exact token matches achieved!")
    else:
        print("NOTICE: Output diverged between non-offloaded and offloaded runs.")