#!/usr/bin/env python3
"""Record which tokens PQ sparse attention selects, per decode step / layer / kv-head.

Output (in --out DIR):
  sel.npy     memmap uint8 [steps, layers, L_total]; bit h set => kv-head h selected
              that token position at that step (sink + top-k middle + local).
  lengths.npy int32 [steps]   context length seen by the decode step
  meta.json   config, token ids, layer/head counts

Usage:
  PYTHONPATH=<repo> VLLM_PLUGINS= python analysis/trace_selection.py \
      --out /tmp/x --new-tokens 2048 --prompt repeat --max-model-len 4096
"""
import argparse
import json
import os

os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--new-tokens', type=int, default=2048)
ap.add_argument('--prompt', choices=['repeat', 'varied'], default='repeat')
ap.add_argument('--prompt-reps', type=int, default=40)
ap.add_argument('--varied-n', type=int, default=40)
ap.add_argument('--max-model-len', type=int, default=4096)
ap.add_argument('--topk', default='0.1')
ap.add_argument('--sink', default='64')
ap.add_argument('--local', default='64')
args = ap.parse_args()

SPARSE_ENV = {
    'SKYLIGHT_SPARSE_METHOD': 'pq',
    'SKYLIGHT_SPARSE_TOPK': args.topk,
    'SKYLIGHT_SPARSE_SINK': args.sink,
    'SKYLIGHT_SPARSE_LOCAL': args.local,
    'SKYLIGHT_SPARSE_CHANNEL_NUM': '64',
    'SKYLIGHT_PQ_GQA_SHARE': '1',
}
os.environ.update(SPARSE_ENV)

import numpy as np
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
    return _orig_get(backend, *a, **k) if _orig_get else backend


if _orig_get:
    cuda_module._get_attn_backend_class = _patched_get

from vllm import LLM, SamplingParams
from pq_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as PQWrapper,
)

if args.prompt == 'repeat':
    PROMPT = ('The history of artificial intelligence is a story of ambition, '
              'setbacks, and breakthroughs across decades of research. ') * args.prompt_reps
else:
    PROMPT = ''.join(
        f"Observation {i}: the calibrated reading was {(i * 37) % 997}, "
        f"which the field team called noteworthy, and the reviewers of "
        f"section {(i * 11) % 23} marked it pending further review. "
        for i in range(args.varied_n))


class Recorder:
    def __init__(self):
        self.on = False
        self.layer_of_ptr = {}
        self.layer = -1
        self.step = -1
        self.buf = None
        self.lengths = []
        self.n_layers = None
        self.group_checked = False

    def open(self, n_steps, n_layers, l_total):
        self.n_layers = n_layers
        self.l_total = l_total
        self.buf = np.lib.format.open_memmap(
            os.path.join(args.out, 'sel.npy'), mode='w+', dtype=np.uint8,
            shape=(n_steps, n_layers, l_total))
        self.n_steps = n_steps

    def enter_layer(self, paged_kv_cache):
        t = paged_kv_cache[0] if isinstance(paged_kv_cache, (tuple, list)) else paged_kv_cache
        ptr = t.data_ptr()
        if ptr not in self.layer_of_ptr:
            self.layer_of_ptr[ptr] = len(self.layer_of_ptr)
        self.layer = self.layer_of_ptr[ptr]
        if self.on and self.layer == 0:
            self.step += 1

    def record(self, sparse_idx, sparse_w, lengths):
        if not self.on or torch.cuda.is_current_stream_capturing():
            return
        if self.step < 0 or self.step >= self.n_steps:
            return
        B, Hq, K = sparse_idx.shape
        n_kv = int(os.environ.get('_TRACE_NKV', '4'))
        group = Hq // n_kv
        idx = sparse_idx[0, ::group]            # [Hkv, K] (rows shared within a group)
        w = sparse_w[0, ::group] > 0
        if not self.group_checked:
            self.group_checked = True
            same = bool((sparse_idx[0].view(n_kv, group, K) == idx[:, None]).all())
            print(f"[TRACE] Hq={Hq} Hkv={n_kv} group={group} K={K} rows-shared-in-group={same}",
                  flush=True)
        L = int(lengths[0])
        bits = torch.zeros(L, dtype=torch.int32, device=idx.device)
        for h in range(n_kv):
            sel = idx[h][w[h] & (idx[h] < L)]
            m = torch.zeros(L, dtype=torch.bool, device=idx.device)
            m[sel] = True
            bits |= m.to(torch.int32) << h
        L_w = min(L, self.l_total)
        self.buf[self.step, self.layer, :L_w] = bits[:L_w].to(torch.uint8).cpu().numpy()
        if self.layer == 0:
            self.lengths.append(L)


REC = Recorder()

_orig_run = PQWrapper.run
_orig_bg = PQWrapper._build_gather
_orig_bgr = PQWrapper._build_gather_ranked


def _run(self, q, paged_kv_cache, *a, **k):
    REC.enter_layer(paged_kv_cache)
    return _orig_run(self, q, paged_kv_cache, *a, **k)


def _bg(self, s_hat, lengths, sink, local, k_mid):
    out = _orig_bg(self, s_hat, lengths, sink, local, k_mid)
    REC.record(out[0], out[1], lengths)
    return out


def _bgr(self, s_hat, lengths, *a, **k):
    out = _orig_bgr(self, s_hat, lengths, *a, **k)
    REC.record(out[0], out[1], lengths)
    return out


PQWrapper.run = _run
PQWrapper._build_gather = _bg
PQWrapper._build_gather_ranked = _bgr


def main():
    os.makedirs(args.out, exist_ok=True)
    llm = LLM(
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        dtype='float16',
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.20,
        enforce_eager=True,
        max_num_seqs=1,
        disable_log_stats=True,
        attention_config={'backend': 'CUSTOM'},
        async_scheduling=False,
    )
    hf = llm.llm_engine.model_config.hf_config
    n_layers = hf.num_hidden_layers
    os.environ['_TRACE_NKV'] = str(hf.num_key_value_heads)
    prompt_ids = llm.get_tokenizer()(PROMPT)['input_ids']
    l_total = len(prompt_ids) + args.new_tokens + 1
    if l_total > args.max_model_len:
        raise SystemExit(f"prompt {len(prompt_ids)} + {args.new_tokens} > max_model_len")
    REC.open(args.new_tokens, n_layers, l_total)

    sp = SamplingParams(temperature=0, max_tokens=args.new_tokens, ignore_eos=True)
    REC.on = True
    out = llm.generate([PROMPT], sp)[0]
    REC.on = False
    REC.buf.flush()

    meta = dict(
        model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        n_layers=n_layers, n_kv_heads=hf.num_key_value_heads,
        n_q_heads=hf.num_attention_heads, page_size=16,
        prompt_mode=args.prompt, prompt_len=len(out.prompt_token_ids),
        new_tokens=len(out.outputs[0].token_ids),
        steps_recorded=REC.step + 1, l_total=l_total,
        max_model_len=args.max_model_len, sparse_env=SPARSE_ENV,
        prompt_token_ids=list(out.prompt_token_ids),
        output_token_ids=list(out.outputs[0].token_ids),
        output_text=out.outputs[0].text,
    )
    np.save(os.path.join(args.out, 'lengths.npy'), np.array(REC.lengths, dtype=np.int32))
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f"[TRACE] done: steps={REC.step + 1} layers={n_layers} -> {args.out}", flush=True)
    print(f"[TRACE] out: {out.outputs[0].text[:200]!r}", flush=True)


if __name__ == '__main__':
    main()
