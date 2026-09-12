# sparse-backend-flash-infer

FlashInfer-style paged-decode wrappers for long-context LLM serving, plus
two optimized sparse variants that ship custom CUDA kernels:

| Package                          | What it is                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------ |
| `original`                       | Pure-PyTorch reference of FlashInfer's `BatchDecodeWithPagedKVCacheWrapper`.                     |
| `original_optimized`             | Same surface as `original`, backed by an extracted FlashInfer CUDA decode kernel.                |
| `sparse`                         | Pure-PyTorch reference of paged-decode with externally-supplied sparse indices / weights.        |
| `sparse_optimized`               | Custom CUDA sparse paged-decode kernel that consumes `sparse_idx` / `sparse_weights`.            |
| `sparse_oracle_topk`             | Pure-PyTorch reference of oracle-top-k sparsity (computes Q·Kᵀ, picks top-k, runs sparse decode).|
| `sparse_oracle_topk_optimized`   | Optimized oracle-top-k pipeline with custom score kernel + FlashInfer top-k + `sparse_optimized`.|

Each `*_optimized` package's CUDA sources live under `csrc/` and are
JIT-compiled by `torch.utils.cpp_extension.load` on first use.

See [`sparse_oracle_topk_optimized/README.md`](sparse_oracle_topk_optimized/README.md)
for the headline numbers — up to **4.17×** end-to-end speedup over FlashInfer
dense decode at 1% sparsity (B=16, H_kv=32, ctx=128K, fp16 on H100).

---

## Installation

The package depends only on PyTorch (CUDA build) and Ninja for the JIT
compiler. FlashInfer is an optional dependency that unlocks the dense
baseline used by the profiling scripts and the radix `top_k` used by the
oracle-top-k wrapper (a `torch.topk` fallback runs without it).

### Editable install (recommended for development)

```bash
git clone <this-repo>
cd sparse-backend-flash-infer
pip install -e .
```

### Regular install

```bash
pip install .
```

### With FlashInfer extra

```bash
pip install ".[flashinfer]"
```

### With test extras

```bash
pip install ".[test]"
pytest test_sparse_oracle_topk_optimized.py -v
```

The first import of any `*_optimized` package will compile its CUDA kernel
(takes ~30–75 s for the templated score kernel; subsequent runs hit the
`~/.cache/torch_extensions/` cache).

---

## Quick usage

### Dense paged decode

```python
import torch
from original_optimized import BatchDecodeWithPagedKVCacheWrapper

ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
wrapper = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
wrapper.plan(indptr, indices, last_page_len,
             num_qo_heads, num_kv_heads, head_dim, page_size,
             q_data_type=torch.float16, kv_data_type=torch.float16)
out = wrapper.run(q, paged_kv_cache)
```

### Sparse paged decode (caller-provided indices)

```python
from sparse_optimized import BatchDecodeWithPagedKVCacheWrapper
out = wrapper.run(q, paged_kv_cache,
                  sparse_idx=sparse_idx,
                  sparse_len=sparse_len,
                  sparse_weights=sparse_weights)
```

### Oracle top-k sparse decode (full pipeline)

```python
from sparse_oracle_topk_optimized import BatchDecodeWithPagedKVCacheWrapper
out = wrapper.run(q, paged_kv_cache, top_k=1311, channel_num=8)
```

See `sparse_oracle_topk_optimized/README.md` for the channel_num details
and full pipeline diagram.

---

## Repository layout

```
sparse-backend-flash-infer/
├── pyproject.toml
├── README.md                              # this file
├── original/                              # PyTorch reference dense decode
├── original_optimized/                    # CUDA dense decode (FlashInfer-extracted)
│   └── csrc/{paged_decode_fi.cu, flashinfer/}
├── sparse/                                # PyTorch reference sparse decode
├── sparse_optimized/                      # CUDA sparse decode kernel
│   └── csrc/sparse_decode_kernel.cu
├── sparse_oracle_topk/                    # PyTorch reference oracle top-k
├── sparse_oracle_topk_optimized/          # CUDA oracle top-k score kernel + pipeline
│   └── csrc/oracle_topk_score_kernel.cu
├── test_*.py                              # pytest correctness suites
└── profile_*.py, sweep_*.py               # benchmarking scripts
```

The test and profiling scripts at the repo root are **not** part of the
installed wheel; they assume the repo is on disk and are intended to be
run from a checkout (`pytest test_sparse_oracle_topk_optimized.py`,
`python sweep_channel_num8.py`, etc.).

---

## Building distribution artifacts

```bash
pip install build
python -m build           # produces dist/*.whl and dist/*.tar.gz
```

The wheel ships every `csrc/` tree (`.cu`, `.cuh`, `.h`) so JIT
compilation works after install. Verify with:

```bash
unzip -l dist/sparse_backend_flash_infer-*.whl | grep csrc
```

---

## Notes

- Six top-level package names are claimed (`original`, `original_optimized`,
  `sparse`, `sparse_optimized`, `sparse_oracle_topk`,
  `sparse_oracle_topk_optimized`). If you need to embed this in a larger
  codebase that already uses any of those names, vendor the relevant
  subdirectory rather than installing the whole distribution.
- Tested on H100 80GB with PyTorch 2.x and CUDA 12.x. The CUDA kernels
  target SM 90 features (vec128 loads, `cp.async`); compatibility on
  pre-Hopper GPUs is not verified.
