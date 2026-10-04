import os
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
os.environ['VLLM_PLUGINS'] = ''
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import torch
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Test worker directly
from offload_package.integration.worker_connector import OffloadPackageWorker
from vllm.config import VllmConfig, KVCacheConfig
from offload_package.config import OffloadConfig
from vllm.v1.attention.backends.registry import KVConnectorRole

# Create minimal configs
vllm_config = VllmConfig()
kv_cache_config = KVCacheConfig(num_blocks=100, kv_cache_groups=[])
offload_config = OffloadConfig(enabled=True, offload_fraction=0.75, num_cpu_slots=4096, num_staging_slots=64, interval_tokens=8)

print('Creating worker...')
worker = OffloadPackageWorker(
    vllm_config=None,  # Simplified
    role=None,
    kv_cache_config=None,
    offload_config=None
)
print('Worker created')
print(f'_staging_slots: {getattr(worker, "_staging_slots", "NOT SET")}')
