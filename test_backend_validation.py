#!/usr/bin/env python3
"""Test script to check SkylightSparseBackend validation."""

import sys
sys.path.insert(0, '/home/gaurav_kumar/KV_offload/skylight/src')

from skylight.plugin import install_plugin
install_plugin()

from skylight.backend import SkylightSparseBackend
from vllm.v1.attention.backend import AttentionType

# Test the validation methods
print("supports_head_size(64):", SkylightSparseBackend.supports_head_size(64))
print("is_sparse():", SkylightSparseBackend.is_sparse())
print("supports_attn_type(AttentionType.DECODER):", end=" ")
from vllm.v1.attention.backend import AttentionType
try:
    result = SkylightSparseBackend.supports_attn_type(AttentionType.DECODER)
    print(result)
except Exception as e:
    print(f"Error: {e}")

# Now test with monkey-patching is_sparse
print("\n--- Testing with monkey-patched is_sparse ---")

# Monkey-patch is_sparse to return True
@classmethod
def is_sparse(cls):
    return True

from skylight.backend import SkylightSparseBackend
SkylightSparseBackend.is_sparse = classmethod(is_sparse)

print(f"After patching is_sparse(): {SkylightSparseBackend.is_sparse()}")

# Check the validation methods
from vllm.v1.attention.backend import AttentionType

print("supports_head_size(64):", SkylightSparseBackend.supports_head_size(64))
print("is_sparse():", SkylightSparseBackend.is_sparse())
print("supports_attn_type(AttentionType.DECODER):", end=" ")
from vllm.v1.attention.backend import AttentionType
try:
    result = SkylightSparseBackend.supports_attn_type("DECODER")
    print(result)
except Exception as e:
    print(f"Error: {e}")

# Now test the backend class selection
from vllm.platforms import current_platform
import torch

try:
    cls = current_platform.get_attn_backend_cls(
        "CUSTOM",
        None,
        32,
    )
    print(f"Backend class: {cls}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")