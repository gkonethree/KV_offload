#!/usr/bin/env python3
"""B14.3: selection-quality measurement for the PQ family (recall harness).

Runs the skylight sparse backend with offload enabled across PQ variants and
reports the in-run attention-error summary (true softmax mass selected,
recall vs exact top-k, rel_l2 vs dense). SKYLIGHT_PQ_ERR_DBG=1 makes the pq
wrapper recompute exact attention per sampled step; SKYLIGHT_PQ_ERR_EVERY must
stay coprime with the layer count (default 31).

Usage: .venv/bin/python test_pq_recall.py
"""
import os

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ['SKYLIGHT_SPARSE_TOPK'] = '0.1'
os.environ['SKYLIGHT_SPARSE_SINK'] = '64'
os.environ['SKYLIGHT_SPARSE_LOCAL'] = '64'
os.environ['SKYLIGHT_PQ_ERR_DBG'] = '1'      # enable the attn_error harness

import gc
import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
register_backend(AttentionBackendEnum.CUSTOM, 'skylight.backend.SkylightSparseBackend')
import vllm.platforms.cuda as cuda_module
from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

_orig_get = getattr(cuda_module, '_get_attn_backend_class', None)


def _patched_get(backend, *a, **k):
    if backend is AttentionBackendEnum.CUSTOM or (
        backend is None and AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES
    ):
        backend = AttentionBackendEnum.CUSTOM
    if _orig_get:
        return _orig_get(backend, *a, **k)
    return backend


if _orig_get:
    cuda_module._get_attn_backend_class = _patched_get

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def run(method: str, use_offload: bool) -> None:
    print(f"\n{'=' * 60}\nMETHOD={method}  OFFLOAD={use_offload}\n{'=' * 60}")

    # Method -> env knobs. oracle_shadow = exact-selection upper bound
    # (offload copies but never frees); oracle_free = true freeing without
    # promotion (shows the damage); oracle_free_promo = R2 (promotion watch).
    if method == "oracle_shadow":
        os.environ['OFFLOAD_PACKAGE_TRUE_FREE'] = '0'
    elif method == "oracle_free":
        os.environ['OFFLOAD_PACKAGE_TRUE_FREE'] = '1'
        os.environ['OFFLOAD_PACKAGE_PROMOTE_PER_STEP'] = '0'
    elif method == "oracle_free_promo":
        os.environ['OFFLOAD_PACKAGE_TRUE_FREE'] = '1'
        os.environ['OFFLOAD_PACKAGE_PROMOTE_PER_STEP'] = '4'
    else:
        os.environ.pop('OFFLOAD_PACKAGE_TRUE_FREE', None)
        os.environ.pop('OFFLOAD_PACKAGE_PROMOTE_PER_STEP', None)

    kv = None
    if use_offload:
        kv = KVTransferConfig(
            kv_connector='OffloadPackageConnector',
            kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
            kv_connector_extra_config={
                'enabled': True, 'offload_fraction': 0.5, 'num_cpu_slots': 2048,
                'num_staging_slots': 64, 'interval_tokens': 32,
            },
            kv_role='kv_both',
        )
    kwargs = dict(
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', dtype='float16', max_model_len=2048,
        gpu_memory_utilization=0.05, enforce_eager=True, tensor_parallel_size=1,
        disable_log_stats=True, attention_config={'backend': 'CUSTOM'},
        async_scheduling=False,
    )
    if kv is not None:
        kwargs['kv_transfer_config'] = kv
    llm = LLM(**kwargs)
    outs = llm.generate(
        ['The future of AI is ' * 64],
        SamplingParams(temperature=0, top_p=1.0, max_tokens=64, ignore_eos=True),
    )
    text = outs[0].outputs[0].text[:60]
    print('OUT:', repr(text))
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == '__main__':
    import sys

    # Row label -> sparse backend method (the label also selects offload knobs
    # inside run()).
    method_of = {
        'pq': 'pq',
        'pqrs': 'pqrs',
        'oracle_shadow': 'oracle',
        'oracle_free': 'oracle',
        'oracle_free_promo': 'oracle',
    }
    methods = sys.argv[1].split(',') if len(sys.argv) > 1 else list(method_of)
    for row in methods:
        os.environ['SKYLIGHT_SPARSE_METHOD'] = method_of[row]
        if method_of[row] == 'pqrs':
            os.environ.setdefault('SKYLIGHT_PQRS_HEAVY_FRAC', '0.5')
        run(row, use_offload=True)
