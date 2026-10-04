import os, time, gc, torch
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.FLASHINFER, 'vllm.v1.attention.backends.flashinfer.FlashInferBackend')

import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES, AttentionBackendEnum
_original_get = cuda_module._get_attn_backend_class
def patched_get(backend, *args, **kwargs):
    if backend is AttentionBackendEnum.CUSTOM or (backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES):
        backend = AttentionBackendEnum.CUSTOM
    return _original_get(backend, *args, **kwargs)
cuda_module._get_attn_backend_class = patched_get

from vllm import LLM, SamplingParams
import torch, time, gc

llm = LLM(model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=4096, gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1, disable_log_stats=True)

prompts = ['The future of AI is ' * 192] * 4
sp = SamplingParams(temperature=0, top_p=1.0, max_tokens=512, ignore_eos=True)

torch.cuda.synchronize()
start = time.time()
outputs = llm.generate(prompts, sp)
torch.cuda.synchronize()
elapsed = time.time() - start

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
tpot = (elapsed / total_tokens) * 1000
throughput = total_tokens / elapsed
free, total = torch.cuda.mem_get_info()
mem_gb = (total - free) / 1e9

print(f"FlashInfer Dense: Time: {elapsed:.2f}s, Throughput: {throughput:.1f} tok/s, TPOT: {tpot:.2f} ms, Mem: {mem_gb:.2f} GB")

del llm; gc.collect(); torch.cuda.empty_cache()
