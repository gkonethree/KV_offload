#!/bin/bash
set -e

echo "Running FlashAttention offload comparison (each test in fresh process)"
echo "================================================================"

for frac in 0.0 0.25 0.50 0.75 0.99; do
    label=$(awk "BEGIN {printf \"%d%%\", $frac*100}")
    echo ""
    echo "Testing $label (offload_fraction=$frac)..."
    python3 -c "
import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ.pop('VLLM_ATTENTION_BACKEND', None)

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import time

kv_transfer_config = KVTransferConfig(
    kv_connector='OffloadPackageConnector',
    kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector',
    kv_connector_extra_config={
        'enabled': True,
        'offload_fraction': $frac,
        'num_cpu_slots': 4096,
        'num_staging_slots': 64,
        'interval_tokens': 16,
        'scorer_decay': 1.0,
        'scorer_head_reduction': 'mean',
    },
    kv_role='kv_both',
)

llm = LLM(
    model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    dtype='float16',
    max_model_len=1024,
    gpu_memory_utilization=0.05,
    enforce_eager=True,
    tensor_parallel_size=1,
    kv_transfer_config=kv_transfer_config,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    seed=42,
)

start = time.time()
outputs = llm.generate(
    prompts=['The future of AI is'],
    sampling_params=sampling_params
)
elapsed = time.time() - start
tok_per_sec = 512 / elapsed
print(f'RESULT: offload_fraction={$frac}, time={elapsed:.2f}s, tok/s={tok_per_sec:.1f}')
" 2>&1 | grep -E "(RESULT:|Free memory|Desired GPU)"
done

echo ""
echo "================================================================"
echo "Done"
