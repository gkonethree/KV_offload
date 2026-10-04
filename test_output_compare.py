#!/usr/bin/env python3
"""Compare outputs between FlashInfer Dense, Skylight Oracle, and Skylight Oracle + Offload"""
import os
import time
import gc
import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '128'
os.environ['SKYLIGHT_SYNC_STAGING'] = '1'
os.environ['SKYLIGHT_SPARSE_BUDGET'] = 'frac:0.5'

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')

# Monkey patch for CUSTOM backend
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = None
try:
    _original_get = cuda_module._get_attn_backend_class
except AttributeError:
    pass
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
import torch, time, gc
import os

def run_and_get_outputs(use_offload=False):
    """Run inference and return generated token IDs"""
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
    
    prompts = ['The future of AI is ' * 192] * 4
    sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=512, ignore_eos=True)
    
    torch.cuda.synchronize()
    outputs = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    
    # Get all token IDs
    all_token_ids = [o.outputs[0].token_ids for o in outputs]
    
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    
    return all_token_ids

if __name__ == '__main__':
    import os, time, gc, torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    import torch, time, gc
    import os
    
    # Register backend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')
    
    # Monkey patch for CUSTOM backend
    import vllm.platforms.cuda as cuda_module
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
    _original_get = None
    try:
        _original_get = cuda_module._get_attn_backend_class
    except AttributeError:
        pass
    def patched_get(backend, *args, **kwargs):
        if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
            backend = AttentionBackendEnum.CUSTOM
        if _original_get:
            return _original_get(backend, *args, **kwargs)
        return backend
    if _original_get:
        cuda_module._get_attn_backend_class = patched_get
    
    # Test configs
    configs = [
        ('FlashInfer Dense', {}, False),
        ('Skylight Oracle (no offload)', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '64', 'VLLM_ATTENTION_BACKEND': 'CUSTOM'}, False),
        ('Skylight Oracle + Offload', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SYNC_STAGING': '1', 'SKYLIGHT_SPARSE_BUDGET': 'frac:0.5', 'VLLM_ATTENTION_BACKEND': 'CUSTOM'}, True),
    ]
    
    results = {}
    for name, env, offload in configs:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        
        for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK', 
                  'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM', 
                  'SKYLIGHT_SYNC_STAGING', 'SKYLIGHT_SPARSE_BUDGET',
                  'VLLM_ATTENTION_BACKEND']:
            os.environ.pop(k, None)
        
        for k, v in env.items():
            os.environ[k] = v
        
        # Run and get outputs
        token_ids = run_and_get_outputs(use_offload)
        results[name] = token_ids
        
        # Print first few tokens for each prompt
        for i, tokens in enumerate(token_ids):
            print(f"  Prompt {i}: first 20 tokens: {tokens[:20]}")
            print(f"  Prompt {i}: total tokens: {len(tokens)}")
        
        time.sleep(2)
    
    # Compare outputs
    print(f"\n{'='*80}")
    print(f"OUTPUT COMPARISON")
    print(f"{'='*80}")
    
    # Compare token by token
    configs = ['FlashInfer Dense', 'Skylight Oracle', 'Skylight Oracle + Offload']
    for i in range(4):  # 4 prompts
        print(f"\nPrompt {i}:")
        tokens_dense = results['FlashInfer Dense'][i]
        tokens_oracle = results['Skylight Oracle'][i]
        tokens_offload = results['Skylight Oracle + Offload'][i]
        
        min_len = min(len(tokens_dense), len(tokens_oracle), len(tokens_offload))
        match_dense_oracle = sum(1 for a, b in zip(tokens_dense[:min_len], tokens_oracle[:min_len]) if a == b)
        match_dense_offload = sum(1 for a, b in zip(tokens_dense[:min_len], tokens_offload[:min_len]) if a == b)
        match_oracle_offload = sum(1 for a, b in zip(tokens_oracle[:min_len], tokens_offload[:min_len]) if a == b)
        
        print(f"  Dense vs Oracle:     {match_dense_oracle}/{min_len} ({100*match_dense_oracle/min_len:.1f}%)")
        print(f"  Dense vs Offload:    {match_dense_offload}/{min_len} ({100*match_dense_offload/min_len:.1f}%)")
        print(f"  Oracle vs Offload:   {match_oracle_offload}/{min_len} ({100*match_oracle_offload/min_len:.1f}%)")
        
        # Show first divergence
        for j in range(min_len):
            if tokens_dense[j] != tokens_oracle[j]:
                print(f"  First diff Dense vs Oracle at pos {j}: dense={tokens_dense[j]}, oracle={tokens_oracle[j]}")
                break
        for j in range(min_len):
            if tokens_dense[j] != tokens_offload[j]:
                print(f"  First diff Dense vs Offload at pos {j}: dense={tokens_dense[j]}, offload={tokens_offload[j]}")
                break

if __name__ == '__main__':
    import os, time, gc, torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    import torch, time, gc
    import os
    
    # Register backend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')
    
    # Monkey patch for CUSTOM backend
    import vllm.platforms.cuda as cuda_module
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
    _original_get = None
    try:
        _original_get = cuda_module._get_attn_backend_class
    except AttributeError:
        pass
    def patched_get(backend, *args, **kwargs):
        if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
            backend = AttentionBackendEnum.CUSTOM
        if _original_get:
            return _original_get(backend, *args, **kwargs)
        return backend
    if _original_get:
        cuda_module._get_attn_backend_class = patched_get
    
    # Set env
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
    os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
    os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
    os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
    os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
    os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '128'
    os.environ['SKYLIGHT_SYNC_STAGING'] = '1'
    os.environ['SKYLIGHT_SPARSE_BUDGET'] = 'frac:0.5'
    
    # Register backend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')
    
    # Monkey patch for CUSTOM backend
    import vllm.platforms.cuda as cuda_module
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
    _original_get = None
    try:
        _original_get = cuda_module._get_attn_backend_class
    except AttributeError:
        pass
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
    import torch, time, gc
    import os
    
    def run_and_get_outputs(use_offload=False):
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
        
        prompts = ['The future of AI is ' * 192] * 4
        sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=512, ignore_eos=True)
        
        torch.cuda.synchronize()
        outputs = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        
        all_token_ids = [o.outputs[0].token_ids for o in outputs]
        
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)
        
        return all_token_ids
    
    # Set up environment
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
    os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
    os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
    os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
    os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
    os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '128'
    os.environ['SKYLIGHT_SYNC_STAGING'] = '1'
    os.environ['SKYLIGHT_SPARSE_BUDGET'] = 'frac:0.5'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
    
    # Register backend
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend('CUSTOM', 'skylight.backend.SkylightSparseBackend')
    
    # Monkey patch for CUSTOM backend
    import vllm.platforms.cuda as cuda_module
    from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
    _original_get = None
    try:
        _original_get = cuda_module._get_attn_backend_class
    except AttributeError:
        pass
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
    import torch, time, gc
    import os
    
    def run_and_get_outputs(use_offload=False):
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
        
        prompts = ['The future of AI is ' * 192] * 4
        sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=512, ignore_eos=True)
        
        torch.cuda.synchronize()
        outputs = llm.generate(prompts, sp)
        torch.cuda.synchronize()
        
        all_token_ids = [o.outputs[0].token_ids for o in outputs]
        
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)
        
        return all_token_ids

    results = {}
    for name, env, offload in [
        ('FlashInfer Dense', {}, False),
        ('Skylight Oracle', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SPARSE_CHANNEL_NUM': '64', 'VLLM_ATTENTION_BACKEND': 'CUSTOM'}, False),
        ('Skylight Oracle + Offload', {'SKYLIGHT_SPARSE_METHOD': 'oracle', 'SKYLIGHT_SPARSE_TOPK': '0.1', 'SKYLIGHT_SPARSE_SINK': '64', 'SKYLIGHT_SPARSE_LOCAL': '64', 'SKYLIGHT_SYNC_STAGING': '1', 'SKYLIGHT_SPARSE_BUDGET': 'frac:0.5', 'VLLM_ATTENTION_BACKEND': 'CUSTOM'}, True),
    ]:
        for k in ['SKYLIGHT_SPARSE_METHOD', 'SKYLIGHT_SPARSE_TOPK', 'SKYLIGHT_SPARSE_SINK', 
                  'SKYLIGHT_SPARSE_LOCAL', 'SKYLIGHT_SPARSE_CHANNEL_NUM', 
                  'SKYLIGHT_SYNC_STAGING', 'SKYLIGHT_SPARSE_BUDGET',
                  'VLLM_ATTENTION_BACKEND']:
            if k in os.environ:
                del os.environ[k]
        
        for k, v in env.items():
            os.environ[k] = v
        
        token_ids = run_and_get_outputs(use_offload)
        results[name] = token_ids
        
        time.sleep(2)
    
    # Compare outputs
    print(f"\n{'='*80}")
    print(f"OUTPUT COMPARISON")
    print(f"{'='*80}")
    
    for i in range(4):
        print(f"\nPrompt {i}:")
        tokens_dense = results['FlashInfer Dense'][i]
        tokens_oracle = results['Skylight Oracle'][i]
        tokens_offload = results['Skylight Oracle + Offload'][i]
        
        min_len = min(len(tokens_dense), len(tokens_oracle), len(tokens_offload))
        match_dense_oracle = sum(1 for a, b in zip(tokens_dense[:min_len], tokens_oracle[:min_len]) if a == b)
        match_dense_offload = sum(1 for a, b in zip(tokens_dense[:min_len], tokens_offload[:min_len]) if a == b)
        match_oracle_offload = sum(1 for a, b in zip(tokens_oracle[:min_len], tokens_offload[:min_len]) if a == b)
        
        print(f"  Dense vs Oracle:     {match_dense_oracle}/{min_len} ({100*match_dense_oracle/min_len:.1f}%)")
        print(f"  Dense vs Offload:    {match_dense_offload}/{min_len} ({100*match_dense_offload/min_len:.1f}%)")
        print(f"  Oracle vs Offload:   {match_oracle_offload}/{min_len} ({100*match_oracle_offload/min_len:.1f}%)")
        
        for j in range(min_len):
            if tokens_dense[j] != tokens_oracle[j]:
                print(f"  First diff Dense vs Oracle at pos {j}: dense={tokens_dense[j]}, oracle={tokens_oracle[j]}")
                break
        for j in range(min_len):
            if tokens_dense[j] != tokens_offload[j]:
                print(f"  First diff Dense vs Offload at pos {j}: dense={tokens_dense[j]}, offload={tokens_offload[j]}")
                break

if __name__ == '__main__':
    import os, time, gc, torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    import torch, time, gc
    import os
    
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
    os.environ['SKYLIGHT_SPARSE_METHOD'] = 'oracle'
    os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
    os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
    os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
    os.environ['SKYLIGHT_SPARSE_CHANNEL_NUM'] = '128'
    os.environ['SKYLIGHT_SYNC_STAGING'] = '1'
    os.environ['SKYLIGHT_SPARSE_BUDGET'] = 'frac:0.5'
    
    main()
