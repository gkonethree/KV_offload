#!/usr/bin/env python3
"""Test block_minmax_incr_optimized kernel"""
import sys
sys.path.insert(0, '/home/gaurav_kumar/KV_offload/skylight_kernels')

import torch
import math

from block_minmax_incr_optimized import BatchDecodeWithPagedKVCacheWrapper as BlockMinmaxWrapper
from block_minmax_incr_optimized.summary_maintain import PagedSummaryMaintainer

# Setup
device = torch.device("cuda")
dtype = torch.float16
B = 4
ctx = 131072
page = 16
H_q = 32
H_kv = 4
D = 128

pages_per_req = math.ceil(ctx / 16)
total_pages = B * math.ceil(131072 / 16)
last_page_len_value = 131072 % 16 or 16

kv_indptr = torch.arange(0, 4 * math.ceil(131072/16) + 1, math.ceil(131072/16), dtype=torch.int32, device="cuda")
kv_indices = torch.arange(4 * math.ceil(131072/16), dtype=torch.int32, device="cuda")
kv_last_page_len = torch.full((4,), 16, dtype=torch.int32, device="cuda")
kv_cache = torch.randn(4 * math.ceil(131072/16), 2, 16, 4, 128, dtype=torch.float16, device="cuda")
q = torch.randn(4, 32, 128, dtype=torch.float16, device="cuda")

# Dense FlashInfer reference
import flashinfer
fi_cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
if fi_cls is None:
    fi_cls = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper

ws_d = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
dense = fi_cls(ws_d, "NHD")
dense.plan(
    kv_indptr, kv_indices, kv_last_page_len,
    32, 4, 128, 16,
    pos_encoding_mode="NONE",
    q_data_type=torch.float16, kv_data_type=torch.float16,
)

def bench(fn, warmup=5, iters=20):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 20

dense_ms = bench(lambda: dense.run(q, kv_cache))
print(f"FlashInfer dense: {dense_ms:.4f} ms")

# BlockMinmax
ws_s = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
block_minmax = BlockMinmaxWrapper(
    ws_s, "NHD",
    max_seq_len=131072,
    sink_size=64,
    local_size=64,
    sub_page=8,
)
block_minmax.plan(
    kv_indptr, kv_indices, kv_last_page_len,
    32, 4, 128, 16,
    q_data_type=torch.float16, kv_data_type=torch.float16,
)

# First run to build summary
print("Building summary...")
block_minmax.run(q, kv_cache, top=0.1, channel_num=8)
torch.cuda.synchronize()

def bench(fn, iters=20):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 20

print("Running block_minmax...")
ms = bench(lambda: block_minmax.run(q, kv_cache, top=0.1, channel_num=8))
print(f"BlockMinmax: {ms:.4f} ms")
