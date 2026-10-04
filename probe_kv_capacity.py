#!/usr/bin/env python3
"""Probe: KV cache capacity (num_gpu_blocks) at a given gpu_memory_utilization."""
import os
import sys
import torch

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

util = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05

from vllm import LLM

free_before, total = torch.cuda.mem_get_info()
if True:
    llm = LLM(
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
        gpu_memory_utilization=util, enforce_eager=True, tensor_parallel_size=1,
        disable_log_stats=True,
    )

sched = llm.llm_engine.engine_core.engine_core.scheduler
bp = sched.kv_cache_manager.block_pool
n_blocks = bp.num_gpu_blocks
page = bp.block_size
kv_tokens = n_blocks * page

# KV bytes: TinyLlama fp16: 22 layers * 2 * 4 kv_heads * 64 head_dim * 2 bytes per token
kv_bytes = kv_tokens * 22 * 2 * 4 * 64 * 2
free_after, _ = torch.cuda.mem_get_info()
used = (total - free_after) / 1e9

print(f"gpu_memory_utilization : {util}")
print(f"num_gpu_blocks         : {n_blocks} (page_size={page})")
print(f"KV token capacity      : {kv_tokens}")
print(f"KV bytes              : {kv_bytes/1e9:.3f} GB")
print(f"total GPU used         : {used:.3f} GB (budget was {util*total/1e9:.3f} GB)")
print(f"overhead (budget-KV)   : {(util*total - kv_bytes)/1e9:.3f} GB")
