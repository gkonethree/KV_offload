"""Subprocess target for tests/integration/test_smoke_sparse.py.

Runs in its own process so VLLM_ATTENTION_BACKEND + SKYLIGHT_SPARSE_* env
vars take effect at vllm import time. Loads Qwen/Qwen3-0.6B (GQA ratio 2,
head_dim 64 — both supported by the kernel) and generates 16 tokens.

Exits 0 on success and prints a 'SMOKE_OK: <text>' line. Exits non-zero on
any failure with a diagnostic message on stderr.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    model = os.environ.get("SKYLIGHT_SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    max_model_len = int(os.environ.get("SKYLIGHT_SMOKE_MAX_MODEL_LEN", "2048"))
    gpu_mem_util = float(os.environ.get("SKYLIGHT_SMOKE_GPU_MEM_UTIL", "0.30"))

    from vllm import LLM, SamplingParams

    # vllm 0.21 dropped the VLLM_ATTENTION_BACKEND env var; selection is now
    # via the LLM kwarg (or `--attention-backend` CLI). "CUSTOM" resolves to
    # our SkylightSparseBackend via the plugin's register_backend() call.
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        enforce_eager=True,  # smoke: skip CUDA graph capture
        gpu_memory_utilization=gpu_mem_util,
        attention_backend="CUSTOM",
    )
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    outputs = llm.generate(["The capital of France is"], sp)

    text = outputs[0].outputs[0].text
    print(f"GENERATED: {text!r}", flush=True)

    if not text:
        print("FAIL: empty output", file=sys.stderr)
        return 1

    # Sanity: at least one finite token id.
    token_ids = outputs[0].outputs[0].token_ids
    if not token_ids:
        print("FAIL: no token ids generated", file=sys.stderr)
        return 1

    print(f"SMOKE_OK: model={model} n_tokens={len(token_ids)} text={text[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
