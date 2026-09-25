# SPDX-License-Identifier: Apache-2.0
"""Registration of SparseAttentionBackend with vLLM attention backend registry."""

def register_sparse_attention_backend():
    """Register SparseAttentionBackend with vLLM's attention backend registry.
    
    This function should be called early, before vLLM initializes attention backends.
    It registers SparseAttentionBackend as the CUSTOM backend implementation.
    """
    try:
        from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
        from offload_package.integration.sparse_backend import SparseAttentionBackend
        
        # Register SparseAttentionBackend as the CUSTOM backend
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "offload_package.integration.sparse_backend.SparseAttentionBackend"
        )
        return True
    except ImportError as e:
        print(f"[SPARSE REG] ImportError during registration: {e}")
        return False
    except ValueError as e:
        if "already registered" in str(e):
            return True
        raise

# Also register the sparse backend's metadata builder if needed
def register_sparse_backend_early():
    """Register the sparse backend early, before vLLM initializes.
    
    This should be called before any vLLM attention operations.
    """
    try:
        from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
        
        # Register SparseAttentionBackend as the CUSTOM backend
        # Using the class path directly
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "offload_package.integration.sparse_backend.SparseAttentionBackend"
        )
        print("[SPARSE REG] SparseAttentionBackend registered as CUSTOM backend")
        return True
    except ImportError as e:
        print(f"[SPARSE REG] ImportError during registration: {e}")
        return False
    except ValueError as e:
        if "already registered" in str(e):
            print("[SPARSE REG] Already registered")
            return True
        raise
    except Exception as e:
        print(f"[SPARSE REG] Error during registration: {e}")
        import traceback
        traceback.print_exc()
        return False

# Auto-register when this module is imported
register_sparse_backend_early()
