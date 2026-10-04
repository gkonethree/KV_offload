#!/usr/bin/env python3
"""Compare throughput: Skylight Oracle vs Skylight Oracle + Offload with longer sequences"""
import os
import json
import subprocess
import sys

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
                'enabled': True, 'offload_fraction': 0.5, 'num_cpu_slots': 4096,
                'num_staging_slots': 64, 'interval_tokens': 256,
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

    # 4 prompts × 300 tokens input, generate 1536 tokens each (longer for offload to kick in)
    prompts = ['The future of AI is ' * 192] * 4

    sp = SamplingParams(
        temperature=0, 
        top_p=1.0, 
        max_tokens=1536, 
        ignore_eos=True,
    )

    import torch
    torch.manual_seed(0)
    torch.cuda.synchronize()
    start = time.time()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
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

    result = {
        'name': name, 
        'time': elapsed, 
        'throughput': throughput, 
        'tpot': tpot, 
        'mem': mem_gb,
        'total_tokens': total_tokens
    }
    print(f"__RESULT_JSON_START__{json.dumps(result)}__RESULT_JSON_END__")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
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
    print(f"THROUGHPUT COMPARISON: 4 requests × 3k input + 1536 output tokens each")
    print(f"{'='*90}")
    print(f"{'Configuration':<35} {'Tokens':>8} {'Time(s)':>8} {'Throughput(tok/s)':>15} {'TPOT(ms)':>10} {'GPU Mem(GB)':>10}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['name']:<35} {r['total_tokens']:>8d} {r['time']:>8.2f} {r['throughput']:>15.1f} {r['tpot']:>10.2f} {r['mem']:>10.2f}")

    if len(results) == 2:
        no_off = results[0]
        off = results[1]
        speedup = off['throughput'] / no_off['throughput'] if no_off['throughput'] > 0 else 0
        mem_savings = no_off['mem'] - off['mem']
        print(f"\n{'='*90}")
        print(f"SPEEDUP: {speedup:.2f}x")
        print(f"GPU MEM SAVINGS: {mem_savings:.2f} GB")
        print(f"{'='*90}")