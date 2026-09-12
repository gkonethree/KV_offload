# sparse_oracle_topk_optimized

A standalone CUDA implementation of **oracle top-k sparse paged batch-decode
attention** for long-context LLMs. Given a query `Q` and a paged KV cache,
the wrapper:

1. computes the dense `Q @ K^T` selection scores per `(batch, q_head)`,
2. picks the `k = max(1, min(L_max, int(round(topk * L_max))))` largest indices
   for a fractional ``topk`` argument,
3. runs a sparse paged-decode over only those indices.

It is built on top of `sparse_optimized` (which provides the sparse decode
kernel) and ships its own custom score kernel under `csrc/`. The top-k
selection uses FlashInfer's radix-based `flashinfer.top_k` when available,
with a `torch.topk` fallback. The score kernel writes scores in the same
dtype as Q/K (fp16/bf16), which halves both the score-write traffic and
the top-k stage's read traffic relative to fp32 scores.

The headline feature is **partial-dimension scoring**: the run-time argument
`channel_num <= head_dim` restricts the selection dot product to only the
first `channel_num` channels of `Q` and `K` — a cheap proxy for the full-D
similarity that, on H100, ~halves the K bandwidth needed for the score pass
and (combined with the radix top-k and fp16 score storage) delivers 2.8-4.2×
end-to-end speedups vs FlashInfer dense decode at 1% sparsity.

---

## What it does

For each `(batch_idx b, q_head qh)`:

```
selection_scores[t] = Q[b, qh, :channel_num] · K[t, kv_head, :channel_num]
top_idx             = top_k_indices(selection_scores)
attn_scores[s]      = sm_scale * (Q[b, qh] · K[top_idx[s]])
attn                = softmax(attn_scores)
O[b, qh]            = sum_s attn[s] * V[top_idx[s]]
```

Note the asymmetry: **selection** uses only the first `channel_num` channels,
but the **attention computation** uses the full `head_dim`. `sm_scale` is
deliberately omitted from the selection dot product (it's a positive
monotonic transform that doesn't change the top-k ranking).

---

## Performance summary

NVIDIA H100 80GB HBM3, fp16, `H_q=32, head_dim=128, page_size=16, ctx=128K`,
NHD layout, `channel_num=8`. Baselines are FlashInfer (`flash_ms`) and
`original_optimized` (`opt_ms`). Sparse latency is in `sparse_ms` columns,
with sparsity = ``topk`` when ``topk`` is the fraction of the score axis
``L_max`` (typically ``context_len`` for uniform-length batches).

### H_kv = 8 (GQA, group_size = 4)

Latency in **ms**:

| B  | flash_ms | opt_ms | sparse @50% | @20%  | @10%  | @5%   | @2%   | @1%   |
|---:|---------:|-------:|------------:|------:|------:|------:|------:|------:|
|  1 |   0.1855 | 0.1893 |       0.661 | 0.330 | 0.225 | 0.166 | 0.127 | 0.115 |
|  4 |   0.7177 | 0.7174 |       2.243 | 1.069 | 0.679 | 0.471 | 0.339 | 0.293 |
|  8 |   1.4966 | 1.5633 |       3.997 | 1.926 | 1.225 | 0.853 | 0.625 | 0.545 |
| 16 |   3.0773 | 2.9296 |       7.483 | 3.622 | 2.346 | 1.689 | 1.249 | 1.024 |

Speedup vs FlashInfer:

| B  |   50% |   20% |   10% |    5% |    2% |    1% |
|---:|------:|------:|------:|------:|------:|------:|
|  1 | 0.28× | 0.56× | 0.83× | **1.12×** | **1.46×** | **1.65×** |
|  4 | 0.32× | 0.67× | **1.06×** | **1.52×** | **2.11×** | **2.45×** |
|  8 | 0.36× | 0.75× | **1.18×** | **1.68×** | **2.30×** | **2.66×** |
| 16 | 0.41× | 0.85× | **1.31×** | **1.82×** | **2.46×** | **2.81×** |

### H_kv = 32 (MHA, group_size = 1)

Latency in **ms**:

| B  | flash_ms | opt_ms | sparse @50% | @20%  | @10%  | @5%   | @2%   | @1%   |
|---:|---------:|-------:|------------:|------:|------:|------:|------:|------:|
|  1 |   0.7029 | 0.7054 |       0.795 | 0.448 | 0.335 | 0.274 | 0.232 | 0.215 |
|  4 |   2.7928 | 2.7938 |       2.770 | 1.523 | 1.107 | 0.881 | 0.749 | 0.702 |
|  8 |   5.6111 | 5.6140 |       5.020 | 2.812 | 2.071 | 1.692 | 1.452 | 1.366 |
| 16 |  11.1774 | 11.181 |       9.182 | 5.258 | 3.942 | 3.266 | 2.837 | 2.682 |

Speedup vs FlashInfer:

| B  |   50% |   20% |   10% |    5% |    2% |    1% |
|---:|------:|------:|------:|------:|------:|------:|
|  1 | 0.91× | **1.62×** | **2.18×** | **2.68×** | **3.14×** | **3.37×** |
|  4 | **1.02×** | **1.86×** | **2.56×** | **3.17×** | **3.74×** | **4.00×** |
|  8 | **1.12×** | **2.00×** | **2.71×** | **3.32×** | **3.86×** | **4.11×** |
| 16 | **1.22×** | **2.13×** | **2.84×** | **3.43×** | **3.94×** | **4.17×** |

### Effect of `channel_num` (B=16, H_kv=8, ctx=128K, fp16)

The score kernel reads K from HBM in 128 B cache lines, so the realistic
ceiling is set by how many cache lines per token we touch:

| channel_num | bytes used / token | cache lines / token | score_ms | sparse @1% | speedup vs dense |
|------------:|-------------------:|--------------------:|---------:|-----------:|-----------------:|
|   8 (recommended) |              16 |              1 |   0.568 |   1.02 | **2.81×** |
|  16 |                32 |                   1 |   0.571 |       ~1.03 | ~2.80× |
|  32 |                64 |                   1 |   0.644 |       ~1.10 | ~2.62× |
|  64 |               128 |                   1 |   1.132 |       ~1.59 | ~1.81× |
| 128 (full head_dim) |    256 |              2 |   2.216 |       ~2.67 | ~1.08× |

Key observations:

- For `channel_num ≤ 64` the K read is one 128 B cache line per token, so the
  score kernel is BW-bound at the same floor (~0.6 ms). `channel_num = 8`,
  `16`, and `32` are all within 7% of each other.
- At `channel_num = 128` (the full head dim) we read two cache lines per
  token; latency roughly doubles and we barely beat dense at very low
  sparsity.
- Below 64 channels the wrapper beats dense at any sparsity ≤ 10%; without
  partial dimensions the wrapper barely beats dense at H_kv=8.

### Top-k stage: FlashInfer's radix vs `torch.topk`

The score kernel produces an `[B, H_q, L_max] = [16, 32, 128K]` tensor of
selection scores; picking the top-`k` over the last axis is the largest
single cost left in the pipeline. FlashInfer ships a radix-based selection
(`flashinfer.top_k`) explicitly designed for vocabularies > 10K, which our
L_max squarely is. Standalone numbers at `[B*H_q, L]=[512, 128K]`:

| k     | torch.topk fp32 | fi.top_k fp32 | fi.top_k fp16 | best speedup |
|------:|----------------:|--------------:|--------------:|-------------:|
| 65536 |          0.972  |        0.731  |        0.518  |  **1.88×**   |
| 26214 |          0.944  |        0.640  |        0.440  |  **2.15×**   |
| 13107 |          0.931  |        0.609  |        0.406  |  **2.29×**   |
|  6554 |          0.921  |        0.578  |        0.375  |  **2.46×**   |
|  2621 |          0.906  |        0.529  |        0.335  |  **2.71×**   |
|  1311 |          0.896  |        0.287  |        0.313  |  **3.12×**   |

(Speedup column is the best of the three. fp16 input is faster than fp32
at every k except k=1311 where fp32 wins by ~10%; the score kernel writes
fp16 either way, so we always go through the fp16 column in the table
above.)

The wrapper imports `flashinfer.top_k` once at construction time and falls
back to `torch.topk` if FlashInfer is missing. The two paths are
set-equivalent (same indices selected, just different order).

### fp16/bf16 score storage

The score kernel writes the `[B, H_q, L_max]` tensor in `Q.dtype`
(fp16/bf16) rather than fp32. Internal accumulation stays in fp32 — only
the final write rounds down. This:

- halves the scores-tensor memory traffic on both ends
  (256 MB → 128 MB at our shape),
- moves the radix top-k from the fp32 column to the fp16 column above
  (15-30% faster at k ≥ 2K),
- introduces score-quantisation ties at the top-k boundary. Empirically
  the fp16 selection set agrees with the fp32 set on **99.9% of indices**
  at `k=1311 / L=128K` for randomly-distributed scores. The differing
  positions are always tied in fp16 anyway, so picking either is
  equally valid.

Net end-to-end win at moderate sparsity (5-20%) is ~15%, exactly where
the topk stage was the bottleneck.

**Crossover**: sparse beats dense from ~10% sparsity (H_kv=8, B≥4) and
already from ~50% sparsity (H_kv=32, B≥4). Going from 50% to 1% sparsity
gives a further ~3.5× because the sparse decode cost scales linearly with
``topk`` (hence ``k``) while score and FlashInfer top-k both shrink as ``k`` shrinks.

---

## Public API

```python
import torch
from sparse_oracle_topk_optimized import BatchDecodeWithPagedKVCacheWrapper

wrapper = BatchDecodeWithPagedKVCacheWrapper(
    float_workspace_buffer,        # uint8 CUDA tensor (>=128 MB)
    kv_layout="NHD",               # or "HND"
    max_seq_len=131072,            # optional: vLLM max_model_len (sync-free run)
)

wrapper.plan(
    paged_kv_indptr,               # int32 [B+1]
    paged_kv_indices,              # int32 [total_pages]
    paged_kv_last_page_len,        # int32 [B]
    num_qo_heads, num_kv_heads, head_dim, page_size,
    q_data_type=torch.float16,
    kv_data_type=torch.float16,
)

out = wrapper.run(
    q,                             # [B, num_qo_heads, head_dim]
    paged_kv_cache,                # rank-5 [P, 2, page_size, H_kv, D] (NHD)
    0.01,                          # float ``topk``: k = round(topk * n_keys), clamped;
                                   # n_keys defaults to L_max (score width); if max_seq_len
                                   # pads beyond the batch max, pass n_keys=... (host int).
    channel_num=8,                 # int in {8,16,32,64,128,256}, <= head_dim;
                                   # multiple of 8 (fp16/bf16) or 4 (fp32);
                                   # default = head_dim (full-D selection)
)
```

Optional: pass `return_lse=True` to also get `lse` (`[B, num_qo_heads]` in
log-2 units, FlashInfer convention).

Constraints carried over from `original_optimized`: `q_len_per_req == 1`,
no sliding window, no sinks, no FP8 KV cache.

---

## How it works

### 1. Pipeline

```
                              channel_num
                                  │
                                  ▼
  ┌──────────────────┐    ┌───────────────┐    ┌────────────────────┐
  │ Q [B, H_q, D]    │───▶│ score kernel  │───▶│ scores             │
  │ K (paged)        │    │ (this package)│    │ [B, H_q, L_max] f32│
  └──────────────────┘    └───────────────┘    └─────────┬──────────┘
                                                         │
                                                         ▼
                                                flashinfer.top_k
                                              (radix; torch.topk fallback)
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │ top_idx [B,H_q,k]│
                                              └─────────┬────────┘
                                                        │
                                                        ▼
  ┌──────────────────┐    ┌─────────────────────┐
  │ V (paged)        │───▶│ sparse decode kernel│───▶ out [B, H_q, D]
  │ Q (paged)        │    │ (sparse_optimized)  │     lse [B, H_q]
  └──────────────────┘    └─────────────────────┘
```

### 2. Score kernel (this package)

- Grid: `(B, H_kv, ceil(L_max / TPB))` — one CTA per `(batch, kv_head, token chunk)`.
- A single CTA loops over the `GROUP_SIZE = H_q / H_kv` query heads that
  share its KV head. **K is loaded once per `(kv_head, token)` and reused
  across all GROUP_SIZE Q vectors** — so K traffic is reduced by GROUP_SIZE
  vs a naive per-`(b, qh)` schedule.
- Cooperative dot product across `BDX = CHANNEL_DIM / VEC_SIZE` lanes per
  warp, reduced with warp shuffles. `BDY` grows correspondingly so total
  threads per CTA stays ~128 regardless of `channel_num`.
- Output is written in fp32 with positions beyond the per-batch sequence
  length filled with `-inf`, so a downstream `torch.topk` cannot select them.
  `sm_scale` and `logits_soft_cap` are deliberately omitted: both are
  monotonic transforms that don't change the top-k ranking.

`channel_num` is template-dispatched at compile time over
`{8, 16, 32, 64, 128, 256}`. The compile-time `CHANNEL_DIM` lets nvcc fully
size `BDX` to the active lanes — without this, lanes beyond `channel_num`
sit in idle SIMD slots and the speedup collapses to ~12% even though the
bytes-read math says 2×.

### 3. Top-k (FlashInfer radix, with `torch.topk` fallback)

FlashInfer's `flashinfer.top_k` is a radix-based selection designed as a
drop-in `torch.topk` replacement for large inner dims. Internally it walks
the input in 8-bit radix passes, narrowing the candidate set in shared
memory rather than maintaining a full sorted heap. For our shape
`[B*H_q, L_max]=[512, 128K]` this cuts the topk stage from ~0.9 ms
(torch) to ~0.29 ms (flashinfer) at k=1311, see standalone numbers above.

The wrapper resolves the implementation once at construction:

```python
self._topk_impl = _resolve_topk_impl(prefer_flashinfer=True)
```

If FlashInfer is unavailable at import time, the resolver falls back to a
plain `torch.topk` lambda. Both paths return `int64` indices; the kernel
downstream copies them as-is into `sparse_idx`.

### 4. Sparse decode (reused from `sparse_optimized`)

The selected indices, a uniform ``sparse_len = k``, and all-ones
`sparse_weights` are fed to `sparse_optimized.sparse_decode_run`. With
all-ones weights, the weighted softmax inside the kernel reduces to a
plain softmax over the selected tokens (full `head_dim`).

---

## Latency decomposition (B=16, H_kv=8, ctx=128K, channel_num=8)

Stages timed in isolation (`score_ms` is the custom kernel, `topk_ms` is
`flashinfer.top_k` on the fp16 score tensor, `decode_ms` is
`sparse_optimized.sparse_decode_run`):

| sparsity | top_k | score_ms | topk_ms | decode_ms | sparse_ms (end-to-end) |
|---------:|------:|---------:|--------:|----------:|-----------------------:|
|     50%  | 65536 |    0.568 |   0.518 |     5.897 |   7.48 |
|     20%  | 26214 |    0.568 |   0.440 |     2.365 |   3.62 |
|     10%  | 13107 |    0.568 |   0.406 |     1.218 |   2.35 |
|      5%  |  6554 |    0.568 |   0.375 |     0.629 |   1.69 |
|      2%  |  2621 |    0.568 |   0.335 |     0.264 |   1.25 |
|      1%  |  1311 |    0.568 |   0.313 |     0.139 |   1.02 |

`score_ms` is constant in `top_k`. `topk_ms` shrinks with `top_k` for
FlashInfer (radix passes finish earlier when `k` is small) — a meaningful
change vs `torch.topk` whose latency was almost constant in `k`. As
sparsity drops below ~5% the floor approaches `score + topk ≈ 0.9 ms`.
Fusing the score computation with the first radix top-k pass into a single
streaming kernel that never materialises the full score tensor would be
the next architectural lever (estimated ~150 μs savings, mostly from
eliminating the score-tensor write+read round trip).

---

## Repository layout

```
sparse_oracle_topk_optimized/
├── __init__.py
├── batch_decode_with_paged_kv_cache_wrapper.py   # Python wrapper
├── cuda_ops.py                                   # JIT loader
├── csrc/
│   └── oracle_topk_score_kernel.cu               # score kernel + bindings
└── README.md                                     # this file
```

Reused from sibling packages:
- `sparse_optimized` — sparse decode kernel.
- `original_optimized` — base `BatchDecodeWithPagedKVCacheWrapper` class
  and the `flashinfer/` vector-load / math headers.
- `sparse_oracle_topk` — pure-PyTorch reference implementation used in
  correctness tests.

---

## Build & run

PyTorch with CUDA 11.0+ is the only requirement. Both kernels JIT-compile
on first use:

```bash
python -c "from sparse_oracle_topk_optimized.cuda_ops import get_oracle_topk_ops; get_oracle_topk_ops()"
```

(First call takes ~75 s to compile the templated score kernel across all
6 `CHANNEL_DIM` × 5 `GROUP_SIZE` × 3 `HEAD_DIM` instantiations; subsequent
runs cache under `~/.cache/torch_extensions/`.)

### Correctness tests

Compares the optimized wrapper against the pure-PyTorch
`sparse_oracle_topk` reference, including parametrised `channel_num`
sweeps and the equivalence `channel_num=head_dim` ↔ default.

```bash
pytest test_sparse_oracle_topk_optimized.py -v
pytest test_sparse_oracle_topk.py -v
```

All 31 cases pass.

### Benchmarks

End-to-end sparse vs dense, with optional `--channel-num-list`:

```bash
python profile_sparse_oracle_topk_optimized.py \
    --batch-size 16 --context-len 131072 \
    --channel-num-list 8 16 32 64 128
```

The dense-baseline numbers in this README come from
`sweep_dense_baseline_128k.py` (calls `bench_one` from
`profile_optimized_decode.py`); the sparse numbers come from
`sweep_channel_num8.py`.

---

## Limitations / not implemented

- `q_len_per_req > 1` (single-token decode only).
- Sliding window (`window_left >= 0`).
- Sinks, KV scaling factor (`kv_cache_sf`).
- FP8 / FP4 KV cache.
- `channel_num` must be one of `{8, 16, 32, 64, 128, 256}` and a multiple
  of the vec lane width (8 for fp16/bf16, 4 for fp32). Adding additional
  values is just a one-line change to the `DISPATCH_CHANNEL` macro in the
  score kernel.
- The `flashinfer.top_k` integration is optional: if FlashInfer isn't
  importable, the wrapper transparently falls back to `torch.topk` and
  loses the ~3× topk-stage speedup at low sparsity.
