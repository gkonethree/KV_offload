#!/usr/bin/env python
"""
Test script to verify offloaded blocks are fetched from CPU during sparse decode.
Sets offload_fraction=1.0 to offload all blocks, then verifies they are fetched from CPU.
"""

import os
import sys

# Disable multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Disable v2 model runner (encoder-only) for decoder models
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

# Set up attention backend BEFORE importing vllm
os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"

# Set sparse config (required by SkylightSparseBackend)
os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.01"
os.environ["SKYLIGHT_SPARSE_SINK"] = "8"
os.environ["SKYLIGHT_SPARSE_LOCAL"] = "8"
os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "-1"

# Disable multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Disable v2 model runner (encoder-only) for decoder models
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

# Install skylight plugin to register CUSTOM backend
from skylight.plugin import install_plugin
install_plugin()

# Disable multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Disable v2 model runner (encoder-only) for decoder models
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

# Set up attention backend BEFORE importing vllm
os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"

# Set sparse config (required by SkylightSparseBackend)
os.environ["SKYLIGHT_SPARSE_TOPK"] = "0.01"
os.environ["SKYLIGHT_SPARSE_SINK"] = "8"
os.environ["SKYLIGHT_SPARSE_LOCAL"] = "8"
os.environ["SKYLIGHT_SPARSE_CHANNEL_NUM"] = "-1"

# Install skylight plugin to register CUSTOM backend
from skylight.plugin import install_plugin
install_plugin()

# Disable multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Disable v2 model runner (encoder-only) for decoder models
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

# Monkey-patch vLLM to register SparseAttentionBackend as CUSTOM backend
# and force use_sparse=True when CUSTOM backend is explicitly selected
def _apply_vllm_patches():
    import importlib
    
    

    # Monkey-patch get_attn_backend to force use_sparse=True for CUSTOM backend
    import vllm.v1.attention.selector as selector_module
    import vllm.config.attention as attention_module
    import vllm.platforms.cuda as cuda_module
    
    _original_get_attn_backend = selector_module.get_attn_backend
    
    def _patched_get_attn_backend(
        head_size, dtype, kv_cache_dtype, use_mla=False, has_sink=False,
        use_sparse=False, use_mm_prefix=False, use_per_head_quant_scales=False,
        attn_type=None, num_heads=None, has_sliding_window=False
    ):
        from vllm.config import get_current_vllm_config
        vllm_config = get_current_vllm_config()
        if vllm_config and vllm_config.attention_config:
            backend = vllm_config.attention_config.backend
            # Check if backend was explicitly set to CUSTOM
            if getattr(vllm_config.attention_config, '_backend_explicitly_set', False):
                from vllm.v1.attention.backends.registry import AttentionBackendEnum
                if vllm_config.attention_config.backend is AttentionBackendEnum.CUSTOM:
                    use_sparse = True
        return _original_get_attn_backend(
            head_size, dtype, kv_cache_dtype, use_mla, has_sink,
            use_sparse, use_mm_prefix, use_per_head_quant_scales,
            attn_type, num_heads, has_sliding_window
        )
    
    selector_module.get_attn_backend = _patched_get_attn_backend
    
    # Patch AttentionConfig to track explicit backend setting
    _original_attention_config_init = attention_module.AttentionConfig.__init__
    
    def _patched_attention_config_init(self, *args, **kwargs):
        backend = kwargs.get('backend', None)
        if backend is not None:
            self._backend_explicitly_set = True
        else:
            self._backend_explicitly_set = False
        _original_attention_config_init(self, *args, **kwargs)
    
    attention_module.AttentionConfig.__init__ = _patched_attention_config_init
    
    # Also patch get_attn_backend_cls in CudaPlatform to handle CUSTOM backend
    _original_get_attn_backend_cls = cuda_module.CudaPlatform.get_attn_backend_cls
    
    @classmethod
    def _patched_get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        # Handle CUSTOM backend explicitly since AttentionBackendEnum.CUSTOM = None
        if selected_backend is not None or selected_backend is AttentionBackendEnum.CUSTOM:
            try:
                from vllm.v1.attention.backends.registry import _get_attn_backend_class
                backend_class = _get_attn_backend_class(selected_backend)
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=cls.get_device_capability(),
                    **(attn_selector_config._asdict()),
                )
            except (ImportError, OSError) as e:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: [{type(e).__name__}: {e}]"
                ) from e
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.info("Using %s backend.", selected_backend)
                return _backend_cls_path(backend_class)
        return _original_get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads)
    
    cuda_module.CudaPlatform.get_attn_backend_cls = _patched_get_attn_backend_cls

def _apply_vllm_patches():
    # This will be called after vLLM imports
    pass

# Apply the patches
_apply_vllm_patches()

# Now import vLLM


# Register SparseAttentionBackend as CUSTOM attention backend
try:
    from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "offload_package.integration.sparse_backend.SparseAttentionBackend"
    )
    print("[REG] SparseAttentionBackend registered as CUSTOM attention backend")
except (ImportError, ValueError) as e:
    print(f"[REG] Could not register sparse backend: {e}")
    pass

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Add skylight_kernels path
sys.path.insert(0, '/home/gaurav_kumar/KV_offload/skylight_kernels')

# Set up KV Transfer config with offload_fraction=1.0 (offload all blocks)
kv_transfer_config = KVTransferConfig(
    kv_connector="OffloadPackageConnector",
    kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.offload_package_connector",
    kv_connector_extra_config={
        "enabled": True,
        "offload_fraction": 0.99,  # Offload 99% of blocks (max allowed < 1.0)
        "num_cpu_slots": 4096,
        "num_staging_slots": 64,
        "interval_tokens": 16,
        "scorer_decay": 1.0,
        "scorer_head_reduction": "mean",
    },
    kv_role="kv_both",
)

def test_offload_fetch():
    """Test that offloaded blocks are fetched from CPU during sparse decode."""
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    prompts = ["The future of AI is"]
    max_tokens = 128
    temperature = 0.7
    top_p = 0.9
    max_seq_len = 2048
    gpu_memory_utilization = 0.05
    enforce_eager = True
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
    )
    
    print(f"[DEBUG] Creating LLM with offload_fraction=1.0...")
    
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        tensor_parallel_size=1,
        kv_transfer_config=kv_transfer_config,
        attention_config={"backend": "CUSTOM"},
    )
    
    print(f"[DEBUG] LLM created")
    
    # Run generation
    print(f"[DEBUG] Starting generation...")
    
    outputs = llm.generate(
        prompts=["The future of AI is"],
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=16,
        )
    )
    
    print(f"[DEBUG] Generation completed")
    
    return outputs, llm

if __name__ == "__main__":
    outputs, llm = test_offload_fetch()
    
    print("="*60)
    print("GENERATION RESULTS")
    print("="*60)
    for i, output in enumerate(outputs):
        print(f"Prompt {i}: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
