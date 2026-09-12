# sparse_optimized

A standalone, FlashInfer-free CUDA implementation of a paged batch-decode
attention kernel with **per-(batch, q_head) sparse token selection** and
optional per-token weights. Built as a drop-in faster replacement for the
reference `sparse/` wrapper.

---

## What it does

For each `(batch_idx b, q_head qh)`, given:

- `Q[b, qh, :]` — query vector
- `paged_kv_cache` — paged K/V tensor + block table (`kv_indptr`, `kv_indices`, `kv_last_page_len`)
- `sparse_len[b, qh, 0]` — number of valid sparse indices for this row
- `sparse_idx[b, qh, :S]` — token positions (in the contiguous logical sequence) to attend over
- `sparse_weights[b, qh, :S]` — per-token weights (added in log space before softmax)

it computes weighted sparse attention:

```
qk_raw[s] = Q[b, qh] · K[t]              where t = sparse_idx[b, qh, s]
qk[s]     = sm_scale * qk_raw[s]         (or soft_cap*tanh(qk_raw*sm_scale/soft_cap) when enabled)
weighted  = qk + log(sparse_weights)
attn      = softmax(weighted)
O[b, qh]  = sum_s attn[s] * V[t]
LSE[b,qh] = logsumexp(weighted) / log(2)   # FlashInfer log2 convention
```

`K[t]` and `V[t]` are read **directly** from the paged cache via the block
table — there is no intermediate gather/`repeat_interleave` of K/V. The kernel
only ever materialises one K and one V vector per thread per token at a time.

---

## Performance summary

NVIDIA H100 80GB HBM3, fp16, `H_kv=8, H_q=32, D=128, page=16`, NHD layout.
`speedup` is `original_optimized` (full dense decode) latency ÷
`sparse_optimized` latency on the same shape.

### ctx = 32K

| B | 50% | 25% | 10% | 5% | 2% | 1% | 0.5% | 0.2% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.40x | 0.73x | **1.50x** | **2.13x** | **2.59x** | **3.09x** | **3.50x** | **3.50x** |
| 4  | 0.36x | 0.70x | **1.61x** | **3.01x** | **6.61x** | **10.82x** | **10.95x** | **11.08x** |
| 8  | 0.40x | 0.78x | **1.88x** | **3.55x** | **7.67x** | **14.47x** | **20.75x** | **21.15x** |
| 16 | 0.44x | 0.88x | **2.13x** | **4.12x** | **9.37x** | **16.34x** | **30.88x** | **42.22x** |

### ctx = 64K

| B | 50% | 25% | 10% | 5% | 2% | 1% | 0.5% | 0.2% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.36x | 0.67x | **1.45x** | **2.49x** | **3.39x** | **3.77x** | **4.17x** | **4.40x** |
| 4  | 0.34x | 0.68x | **1.63x** | **3.06x** | **6.81x** | **12.77x** | **21.48x** | **21.65x** |
| 8  | 0.39x | 0.77x | **1.88x** | **3.68x** | **8.42x** | **15.08x** | **27.77x** | **42.32x** |
| 16 | 0.44x | 0.89x | **2.17x** | **4.28x** | **10.06x** | **18.75x** | **32.74x** | **81.82x** |

### ctx = 128K

| B | 50% | 25% | 10% | 5% | 2% | 1% | 0.5% | 0.2% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.32x | 0.63x | **1.45x** | **2.58x** | **5.57x** | **10.25x** | **11.05x** | **11.14x** |
| 4  | 0.33x | 0.66x | **1.64x** | **3.18x** | **7.45x** | **13.36x** | **24.25x** | **42.04x** |
| 8  | 0.38x | 0.77x | **1.90x** | **3.75x** | **8.88x** | **16.82x** | **29.64x** | **76.14x** |
| 16 | 0.45x | 0.89x | **2.21x** | **4.35x** | **10.54x** | **20.09x** | **37.32x** | **76.77x** |

**Crossover** (sparse begins beating dense): consistently around **10% sparsity**
across all batch sizes and context lengths.

Headline numbers at typical sparse-attention operating points:

- **5% sparsity**: 2.1–4.4x faster than dense.
- **2% sparsity**: 2.6–10.5x faster.
- **1% sparsity**: 3.1–20.1x faster.
- **0.2% sparsity**: 3.5–82x faster.

---

## Public API

```python
from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper

wrapper = BatchDecodeWithPagedKVCacheWrapper(
    float_workspace_buffer,           # uint8 CUDA tensor (>=128 MB)
    kv_layout="NHD",                  # or "HND"
)
wrapper.plan(
    paged_kv_indptr,                  # int32 [B+1]
    paged_kv_indices,                 # int32 [total_pages]
    paged_kv_last_page_len,           # int32 [B]
    num_qo_heads, num_kv_heads, head_dim, page_size,
    q_data_type=torch.float16,
    kv_data_type=torch.float16,
    pos_encoding_mode="NONE",
    logits_soft_cap=None,             # or > 0 for soft-capping
)

out, lse = wrapper.run(
    q,                                # [B, num_qo_heads, head_dim]
    paged_kv_cache,                   # rank-5 [P, 2, page_size, H_kv, D] or HND equivalent
    sparse_len,                       # int32 [B, num_qo_heads, 1]
    sparse_idx,                       # int64 [B, num_qo_heads, max_S]
    sparse_weights,                   # float32 [B, num_qo_heads, max_S]
    return_lse=True,
)
```

The wrapper inherits from `original_optimized.BatchDecodeWithPagedKVCacheWrapper`,
so all of its plan-time validation (head sizing, layouts, dtypes) carries over.

---

## How it works

### 1. Direct paged access — no K/V gather

Inside the kernel, for each selected token index `t = sparse_idx[b, qh, s]`:

```cpp
page_local = t / page_size
in_page    = t % page_size
page_id    = kv_indices[kv_indptr[b] + page_local]
kv_off     = page_id * stride_page + kv_head * stride_h + in_page * stride_n
k_vec.load(k_data + kv_off + feat_base)   // 16-byte vector load
v_vec.load(v_data + kv_off + feat_base)
```

That's it. `sparse_idx`, `sparse_weights`, and the page table are read straight
from GMEM; `K[t]` and `V[t]` are fetched only when actually needed, then
consumed immediately and discarded. No `[B, H_q, S, D]` intermediate tensor is
ever allocated.

### 2. Per-(b, qh) thread block layout

Grid = `(B, H_q, K_SPLIT)` — one CTA per `(batch, q_head, k_split)`. Block =
`(BDX, BDY)` where:

| HEAD_DIM | dtype | VEC_SIZE | BDX | BDY | tokens / outer iter |
|---:|:---:|---:|---:|---:|---:|
| 64  | fp16/bf16 | 8 | 8  | 16 | 16 |
| 128 | fp16/bf16 | 8 | 16 | 8  | 8  |
| 256 | fp16/bf16 | 8 | 32 | 4  | 4  |

Each `ty` thread row processes one token per outer iteration; `BDX` threads
within a row cooperate to load and dot-product one `D`-vector. Cross-`tx`
reduction uses **warp shuffles** (no smem, no syncthreads inside the loop).

### 3. Online softmax in log2 domain

Per-`ty` running state is `(m, d, o_acc)` in registers. After the loop, one
shared-memory pass merges the `BDY` rows into a single `(m_g, d_g, o)` per CTA.
LSE is written in log2 units to match FlashInfer convention.

### 4. Split-K parallelism

For low-batch decoding, `B * H_q` is often well below the SM count (e.g.
B=1, H_q=32 → 32 CTAs vs 132 SMs). The kernel adds a third grid dim
`K_SPLIT`: each block processes a chunk of `ceil(S / K_SPLIT)` sparse tokens
and writes its partial `(m, d, o)` to a tmp workspace. A small
`sparse_decode_merge_kernel` then combines the K_SPLIT partials per `(b, qh)`
into the final output and LSE using the standard online-softmax merge.

`K_SPLIT` is chosen at run() time by `pick_k_split(B, H_q, max_S, target_blocks_per_sm=4)`:

```cpp
want = sm_count * target_blocks_per_sm           // ~4 CTAs/SM after split
K_SPLIT = clamp(ceil(want / (B*H_q)), [1, max_S/16, 32])
```

This lifted B=1 ctx=128K @ 5% sparsity from `0.91 ms` to `0.073 ms` (12x)
without touching the inner loop.

When `K_SPLIT == 1` the kernel writes directly to the output and the merge
kernel is skipped — zero overhead added for already-saturated cases.

### Things that didn't help

- **`cp.async` double-buffered K/V loads** — broke even or was slightly slower
  after split-K, because each CTA now has too few iterations (typically
  10–100) for the smem write+read overhead to be amortised by latency hiding.
  The hardware load buffer + simultaneous K+V issue already give enough MLP.
  The implementation is left in the git history but not in the current kernel.

- **Larger `VEC_SIZE` (32-byte loads)** — no measurable improvement in our
  configs; the compiler already emits `cp.async`-style overlapping loads from
  the standard `vec_t<half, 8>::load`.

---

## Repository layout

```
sparse_optimized/
├── __init__.py
├── batch_decode_with_paged_kv_cache_wrapper.py   # Python wrapper (extends original_optimized)
├── cuda_ops.py                                   # JIT compiles the .cu file
├── csrc/
│   └── sparse_decode_kernel.cu                   # All kernels + Torch bindings
└── README.md                                     # this file
```

The kernel reuses extracted FlashInfer headers from
`original_optimized/csrc/flashinfer/` (`vec_dtypes.cuh`, `math.cuh`) for
vectorised 16-byte loads and stable log2-domain math. There is no runtime
dependency on the `flashinfer` package.

---

## Build & run

PyTorch with CUDA 11.0+ is the only requirement. The extension JIT-compiles
on first use:

```bash
python -c "from sparse_optimized.cuda_ops import get_sparse_decode_ops; get_sparse_decode_ops()"
```

(First call takes ~50 s to compile; subsequent runs cache the build under
`~/.cache/torch_extensions/`.)

### Correctness tests

Compares against the pure-PyTorch `sparse/` reference across NHD/HND, fp16/bf16,
multiple `(B, ctx, H_kv, H_q, D, page, max_S)` configs, soft-cap, and a
zero-sparse-len edge case.

```bash
pytest test_sparse_optimized.py -v
```

All 13 cases pass.

### Benchmarks

Single-shape sparse profiling:

```bash
python profile_sparse.py --impl-folder sparse_optimized
```

Sparse-vs-dense sweep across batches and sparsity ratios:

```bash
python profile_sparse_vs_dense.py
```

Long-context (32K / 64K / 128K) sweep that produced the tables above:

```bash
python profile_sparse_long_ctx.py
```

---

## Limitations / not implemented

- `q_len_per_req > 1` (single-token decode only).
- Sliding window (`window_left >= 0`).
- Sinks, KV scaling factor (`kv_cache_sf`).
- FP8 / FP4 KV cache.
- CUDA Graph capture path (the wrapper's `use_cuda_graph=True` mode is
  inherited but the run-time `K_SPLIT` selection isn't graph-safe yet).

These would all be straightforward additions; nothing in the kernel structure
prevents them.
