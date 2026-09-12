# Minimal changes in the vLLM fork

The old patch contained pseudo-code and targeted the legacy model runner. This
bundle intentionally separates the verified package from the vLLM-specific
hooks.

## A. Before KV-cache initialization

In the GPU worker, before the V2 runner allocates its KV cache:

```python
from offload_package_hooks import install_all
install_all()
```

The supplied `cache_allocator.py` wraps `vllm.v1.worker.gpu.attn_utils._allocate_kv_cache`.
It adds `OFFLOAD_PACKAGE_STAGING_SLOTS` tail pages to ordinary unshared
`AttentionSpec` caches while leaving `KVCacheConfig.num_blocks` unchanged, so
BlockPool still owns only the normal pages.

The allocator deliberately rejects shared/mixed/non-AttentionSpec caches rather
than silently corrupting their storage geometry.

## B. After V2 KV initialization

In `GPUModelRunner.initialize_kv_cache`, immediately after `init_kv_cache(...)`:

```python
from offload_package.integration.vllm_bridge import initialize_for_runner
from offload_package.config import OffloadConfig

cfg = OffloadConfig.from_env()
if cfg.enabled:
    self._offload_orchestrator = initialize_for_runner(
        self,
        normal_gpu_pages=kv_cache_config.num_blocks,
        block_size=kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size,
    )
```

The runner's `self.kv_caches` must already be populated by that point.

## C. Request lifecycle

The runtime calls are:

```python
orchestrator.sync_request_blocks(req_id, block_ids, current_seq_len)
orchestrator.on_prefill_complete(req_id)
```

for each request after prefill, and:

```python
orchestrator.on_decode_step(active_req_ids, 1)
```

at each decode step. The package keeps W per request and only triggers on a
request's W-th decode token.

On completion/preemption:

```python
orchestrator.on_request_done(req_id)
```

## D. Sparse kernel path

Your existing sparse selector already produces `sparse_idx`, `sparse_len`, and
`sparse_weights`. Before the existing sparse wrapper's `plan()`/`run()` call,
obtain the flat paged-index tensor and its `paged_kv_indptr`, then:

```python
from offload_package.integration.vllm_bridge import prepare_sparse_pages

paged_kv_indices = prepare_sparse_pages(
    orchestrator,
    request_ids,
    sparse_idx,
    sparse_len,
    paged_kv_indices,
    paged_kv_indptr,
)
```

Feed the returned indices to the existing sparse wrapper. `sparse_idx` itself
remains the original token positions.

## E. Do not enable CUDA graphs initially

Run eager decode first. Dynamic staging/residency needs to be validated before
adding graph-capture support.
