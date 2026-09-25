#!/usr/bin/env python
"""Test tensor parallelism with packed KV cache layout."""
import os

# Set BEFORE any imports
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["OFFLOAD_PACKAGE_ENABLED"] = "1"
os.environ["OFFLOAD_PACKAGE_OFFLOAD_FRACTION"] = "0.20"
os.environ["OFFLOAD_PACKAGE_CPU_SLOTS"] = "4096"
os.environ["OFFLOAD_PACKAGE_STAGING_SLOTS"] = "16"
os.environ["OFFLOAD_PACKAGE_INTERVAL"] = "32"
os.environ["OFFLOAD_PACKAGE_SCORER_DECAY"] = "1.0"
os.environ["OFFLOAD_PACKAGE_SCORER_HEAD_REDUCTION"] = "mean"


def main():
    # Add paths
    import sys
    sys.path.insert(0, '/home/gaurav_kumar/KV_offload/vllm-0.29.0/vllm-integration')
    sys.path.insert(0, '/home/gaurav_kumar/KV_offload/skylight_kernels')

    from offload_package_hooks import install_all
    install_all()
    print("✓ Offload package hooks installed")

    from sparse_oracle_topk_optimized import BatchDecodeWithPagedKVCacheWrapper
    print("✓ Sparse kernel available")

    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model='Qwen/Qwen2.5-7B-Instruct',
        dtype='float16',
        max_model_len=1024,
        gpu_memory_utilization=0.2,
        tensor_parallel_size=2,
        enforce_eager=True,
    )
    print("✓ LLM created successfully")

    outputs = llm.generate(
        ["Lets start small. are large models necessarily better than samll ones?"],
        SamplingParams(max_tokens=32, temperature=0.7)
    )
    print(f"Generated: {outputs[0].outputs[0].text}")

    # Check offload stats
    def _check_orchestrator(worker):
        if hasattr(worker, 'model_runner') and worker.model_runner is not None:
            orchestrator = getattr(worker.model_runner, '_offload_orchestrator', None)
            if orchestrator is not None:
                return {'enabled': True, 'residency': str(orchestrator.residency)}
            return {'enabled': False, 'reason': 'orchestrator is None'}
        return {'enabled': False, 'reason': 'no model_runner'}

    results = llm.llm_engine.engine_core.collective_rpc(_check_orchestrator, args=())
    print(f"Orchestrator check: {results}")


if __name__ == '__main__':
    main()
