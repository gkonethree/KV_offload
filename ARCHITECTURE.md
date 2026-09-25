# KV Cache Offloading Architecture

## Overview

This document describes the scoring-based KV cache offloading system integrated into vLLM v0.29.0. The system:
- **Scores** KV blocks after prefill and every W decode tokens (per-request)
- **Evicts** low-scoring blocks to CPU when GPU budget is exceeded
- **Restores** blocks on-demand from CPU to GPU staging area during attention

## Core Components

### 1. Scoring (`offload_package/scoring/`)

**`base.py`**: Abstract interfaces
- `BlockScorer`: Abstract base for all scorers
- `BlockKVStats`: Per-block K/V statistics (v_norm, k_norm)
- `BlockRef`: Logical block identity (request_id, block_idx)

**`paged_eviction.py`**: Query-agnostic scorer
- Computes block importance as: `mean(V-norm / K-norm) across heads`
- Applies decay: `new_score = decay * current + (1 - decay) * old`
- Selects blocks with lowest scores for eviction

**`norm_kernel.py`**: Efficient norm computation
- Computes L2 norms of K/V blocks per head
- Supports "NHD" and "HND" layouts
- Returns BlockKVStats for scorer.update()

### 2. Residency Tracking (`offload_package/staging/residency.py`)

**`BlockLocation`**: Tracks physical and logical location of a block
- Fields: residency (GPU/CPU/STAGING/IN_FLIGHT), gpu_block_id, cpu_slot, staging_slot, complete

**`ResidencyTable`**: Authoritative map of BlockRef → BlockLocation
- Enforces state machine: GPU → CPU → STAGING → IN_FLIGHT → CPU
- `mark_gpu(ref, gpu_block_id)`: Block allocated on GPU
- `mark_cpu(ref, cpu_slot)`: Block copied to CPU (requires GPU mapping)
- `mark_staging(ref, staging_slot)`: Block in GPU staging (temporary)
- `mark_complete(ref)`: Block filled with data and ready for scoring

### 3. CPU Memory Management (`offload_package/offload/manager.py`)

**`CPUBlockPool`**: Allocates/frees CPU slots
- Pre-allocated fixed number of slots (num_cpu_slots)
- LIFO allocation (last-in-first-out)
- Raises RuntimeError if exhausted

**`KVOffloadManager`**: Coordinates scoring and GPU↔CPU copies
- `score_block(ref, gpu_block_id)`: Compute norms and update scorer
- `on_block_full(ref)`: Score block and mark complete
- `maybe_offload(candidates)`: Offload if GPU blocks exceed budget
- `offload_fraction(candidates, fraction)`: Offload a fraction of blocks
- `_copy_gpu_to_cpu(gpu_block_id)`: Async copy with event tracking
- `wait_cpu_slot(slot)`: Synchronize GPU↔CPU copies
- `free(ref)`: Free CPU slot and remove from scorer

### 4. GPU Staging (`offload_package/staging/pool.py`)

**`KVStagingPool`**: Manages temporary GPU pages for sparse attention
- Tail pages of KV cache reserved for staging (num_staging_slots)
- `ensure_resident(refs)`: Move blocks CPU→staging with LRU eviction
- `_allocate_slot(incoming)`: Allocate staging slot (LRU eviction if full)
- `release(ref)`: Return staging slot to free pool
- On-demand fetching: blocks restored only when attention needs them

### 5. Control Flow (`offload_package/controller.py`)

**`KVCompressionController`**: Per-request decode token tracking
- `on_prefill_complete(request_id)`: Mark request in decode phase
- `on_decode(request_id, num_tokens)`: Increment decode counter
- Returns True when decode_tokens % W == 0 (trigger offload)
- Separate counter per request (enables per-request triggering)

### 6. Orchestration (`offload_package/integration/manager.py`)

**`OffloadOrchestrator`**: Coordinates all components
- Maintains request → blocks mapping (_request_blocks)
- Calls scorer, offload_manager, staging_pool in sequence
- `sync_request_blocks(req_id, gpu_block_ids, seq_len)`: Track blocks
- `on_prefill_complete(req_id)`: Score and offload prefill blocks
- `on_decode_step(req_ids, num_tokens)`: Check W-token triggers
- `prepare_staging(req_ids, sparse_idx, sparse_len)`: Ensure staging
- `patch_paged_kv_indices(...)`: Remap page indices to staging
- `on_request_done(req_id)`: Cleanup (free CPU slots, forget scorer)

### 7. vLLM Integration (`offload_package/integration/vllm_bridge.py`, `vllm_adapter.py`)

**`initialize_for_runner(runner, normal_gpu_pages, block_size)`**:
- Create orchestrator from vLLM runner's KV caches
- Normalize K/V cache tensors via `normalize_runner_kv_caches()`
- Build orchestrator via `build_orchestrator()`

**`build_orchestrator(cfg, runner_kv_caches, staging_base_page_idx)`**:
- Extract K/V from runner caches
- Create CPU cache (pinned memory)
- Create scorer, residency table, offload manager, staging pool
- Create controller and orchestrator

### 8. Model Runner Hooks (`vllm/v1/worker/gpu/model_runner.py`)

**Sparse attention integration** (`_maybe_patch_block_tables_for_offload`):
- Called from `prepare_attn()` after slot mappings computed
- Patches paged_kv_indices to point to staging pages
- On-demand: blocks fetched only when attention accesses them

**Scoring triggers** (`_maybe_trigger_offload_scoring`):
- Called from `sample_tokens()` after postprocess_sampled()
- Detects prefill→decode transitions
- Calls `on_prefill_complete()` for each request that finished prefill
- Calls `on_decode_step()` to trigger W-token scoring

**Request lifecycle hooks**:
- `_maybe_track_new_request_blocks()`: Called in `add_requests()`
- `_maybe_track_cached_request_update()`: Called in `update_requests()`
- `_maybe_track_request_done()`: Called in `finish_requests()`
- Keeps orchestrator synchronized with vLLM's block allocations

## Execution Flow

### Prefill Phase
```
1. Request added: add_requests()
   └─> _maybe_track_new_request_blocks()
       └─> orchestrator.sync_request_blocks(req_id, block_ids, 0)
           └─> residency.mark_gpu() for each block

2. Blocks fill: update_requests() for each scheduling step
   └─> _maybe_track_cached_request_update()
       └─> orchestrator.sync_request_blocks(req_id, all_block_ids, seq_len)
           └─> residency.mark_complete() when block_idx*page_size <= seq_len

3. Prefill sample: sample_tokens()
   └─> Detect prefill→decode transition
   └─> _maybe_trigger_offload_scoring(is_prefill_complete=True)
       └─> orchestrator.on_prefill_complete(req_id)
           └─> scorer.update() for all blocks
           └─> select_for_eviction() (lowest scores)
           └─> offload_manager.offload_fraction()
               └─> _copy_gpu_to_cpu() for each evicted block
               └─> residency.mark_cpu()
```

### Decode Phase
```
1. Each decode token: sample_tokens()
   └─> _maybe_trigger_offload_scoring(is_prefill_complete=False)
       └─> orchestrator.on_decode_step([req_ids], num_tokens=1)
           └─> controller.on_decode(req_id)
           └─> If decode_tokens % W == 0:
               └─> orchestrator.offload_fraction()
                   └─> Offload more blocks following same path

2. Attention: prepare_attn()
   └─> _maybe_patch_block_tables_for_offload()
       └─> prepare_sparse_pages()
           └─> orchestrator.prepare_staging()
               └─> staging_pool.ensure_resident()
                   └─> CPU→staging if needed
           └─> patch_paged_kv_indices()
               └─> Remap indices to staging page IDs
       └─> Sparse attention kernel uses patched indices
           └─> Fetches from staging if block is there
```

### Request Cleanup
```
1. Request done: finish_requests()
   └─> _maybe_track_request_done()
       └─> orchestrator.on_request_done(req_id)
           └─> staging_pool.release(ref) for each block
           └─> offload_manager.free(ref)
               └─> cpu_pool.free(cpu_slot)
               └─> scorer.forget([ref])
           └─> residency.remove(ref)
```

## Configuration

Environment variables (via `OffloadConfig.from_env()`):
- `OFFLOAD_PACKAGE_ENABLED=1`: Enable offloading
- `OFFLOAD_PACKAGE_OFFLOAD_FRACTION=0.20`: Fraction to offload each trigger (0-1)
- `OFFLOAD_PACKAGE_CPU_SLOTS=4096`: Pre-allocated CPU slots
- `OFFLOAD_PACKAGE_STAGING_SLOTS=64`: GPU staging pages
- `OFFLOAD_PACKAGE_INTERVAL=32`: W tokens between offload triggers
- `OFFLOAD_PACKAGE_SCORER_DECAY=1.0`: Score decay (1=newest only, 0=never update)
- `OFFLOAD_PACKAGE_SCORER_HEAD_REDUCTION="mean"`: mean or min across heads

## Error Handling

### CPU Pool Exhaustion
```
Error: "CPU offload pool exhausted (used X / Y slots)"
Fix: Reduce num_cpu_slots or decrease model size
```

### Staging Pool Full
```
Error: "Cannot evict staging block: no free slots"
Fix: Increase num_staging_slots
```

### Residency Corruption
```
Error: "No GPU mapping recorded for BlockRef(...)"
Fix: Internal error—report issue with reproduction steps
```

### Block Not Found
```
Error: "Unknown logical block BlockRef(...)"
Fix: Block was never tracked; check request lifecycle hooks
```

## Debugging

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("offload_package")
logger.setLevel(logging.DEBUG)
```

### Check Residency State
```python
from offload_package.staging.residency import Residency

# In model runner
if hasattr(runner, "_offload_orchestrator"):
    orch = runner._offload_orchestrator
    residency = orch.residency
    
    print(f"Total blocks tracked: {len(residency)}")
    gpu_blocks = residency.all_gpu_blocks()
    print(f"GPU blocks: {len(gpu_blocks)}")
    
    for req_id in orch._request_blocks:
        refs = orch._request_blocks[req_id]
        print(f"Request {req_id}: {len(refs)} blocks")
        for ref in refs:
            loc = residency.get(ref)
            print(f"  {ref}: {loc.residency.name}")
```

### Monitor Scoring
```python
# Check scorer state
scorer = orch.offload_manager.scorer
scores = scorer.scores(some_block_refs)
print(f"Scores: {scores}")
print(f"Scorer has {len(scorer._scores)} blocks cached")
```

### Trace Block Movement
Enable debug logging in KVOffloadManager:
```python
# In offload/manager.py
def _copy_gpu_to_cpu(self, gpu_block_id: int) -> int:
    slot = self._cpu_pool.allocate()
    logger.debug(f"Offloading GPU block {gpu_block_id} → CPU slot {slot}")
    ...
```

## Performance Considerations

1. **Scoring Overhead**: Norm computation is O(num_blocks * num_heads * head_dim)
   - Triggered infrequently (after prefill + every W tokens)
   - Can be batched across multiple blocks

2. **Copy Overhead**: GPU↔CPU copies via DMA
   - Async on dedicated stream to overlap with computation
   - Event tracking ensures synchronization

3. **Block Size Tradeoff**:
   - Larger blocks: fewer copy operations, less fragmentation
   - Smaller blocks: finer-grained offloading, more overhead
   - Default: 16 tokens per block (tunable)

4. **Staging vs. Direct Access**:
   - Staging adds latency (CPU→staging→attention)
   - But enables on-demand fetching without kernel changes
   - Alternative: Modify attention kernel for direct CPU access (future)

## Future Enhancements

1. **Predictive Prefetching**: Anticipate which blocks will be needed
2. **Direct CPU Access**: Modify attention kernel to read from CPU directly
3. **Hierarchical Offloading**: CPU → SSD → S3
4. **Adaptive Decay**: Adjust decay based on access patterns
5. **Per-Layer Budgets**: Offload different layers at different rates
