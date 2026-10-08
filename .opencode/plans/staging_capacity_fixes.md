# Staging Capacity Fixes: Pre-Sized Pool (S2) + Chunked Attention Fallback (S1)

**User-approved.** Sizing depends on `max_num_seqs` (concurrency) — confirmed OK.

Targets the capacity overflow at 64k+ contexts: selected-cold set > staging slots → earlier aliasing/stalls. S2 makes overflow not happen in the warm path; S1 is the exact-attention fallback when capacity is genuinely insufficient.

## Design A (S2) — adaptive staging pool sizing at init

Replace fixed `num_staging_slots` with a policy that sizes the pool to cover the hot selected set, bounded by a pool-fraction cap.

### New config fields (`offload_package/config.py`)
```
staging_policy: str = "fixed"            # "fixed" | "auto"
staging_max_pool_frac: float = 0.25      # cap: staging fraction of GPU KV pool
staging_safety_margin: float = 1.10      # multiplier on estimate
```
Env: `OFFLOAD_PACKAGE_STAGING_POLICY`, `OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC`, `OFFLOAD_PACKAGE_STAGING_SAFETY`.

### Sizing rule (computed in `OffloadPackageConnector.__init__`, both handlers share it)
Inputs at connector init: `vllm_config.model_config.max_model_len`, `vllm_config.scheduler_config.max_num_seqs`, plus the sparse budget envs `SKYLIGHT_SPARSE_TOPK`, `SKYLIGHT_SPARSE_SINK`, `SKYLIGHT_SPARSE_LOCAL` (read directly from env — same source the backend uses).
```
pages_per_req  = ceil( (sink + local + topk * (max_model_len - sink - local)) / page_size )
effective      = ceil( max_num_seqs * pages_per_req * safety_margin )
capped         = min(effective, floor(num_blocks * staging_max_pool_frac))
effective      = max(capped, 64)          # never below the legacy default
```
Then `self.offload_config = dataclasses.replace(offload_config, num_staging_slots=effective)` before constructing the scheduler manager and worker handler — one config object, one answer everywhere.

Call chain check (no fork edit needed):
- `reserve_staging_tail(pool, num_staging_slots)` ← effective value (scheduler bind path)
- `normalize_runner_kv_caches(..., staging_slots=effective)` (worker orchestrator build)
- `staging_base_page = num_blocks - effective` stays consistent (physical tensor already covers it — BlockPool reservation handles allocator exclusion, the tail exists in the KV view by construction).

When policy == "auto" and needs exceed the cap: log a single info line and leave the cap in place; S1 covers the remainder.

## Design B (S1) — chunked exact attention when the selected cold set exceeds capacity

Where the overflow actually surfaces: inside the sparse wrappers' per-step fetch (`_inrun_fetch_selected` in pq_optimized; the same semantic point in oracle is moot — oracle never selects cold pages). Current behavior: admission policy stages only what beats residents → pass just compiles away. New behavior when `len(needed_refs) > staging_pool.num_slots`:

```
for chunk in batched_in_capacity_groups(needed_refs, num_slots):
    staging_map = staging_pool.ensure_resident(chunk)          # never overflows
    patch indices for chunk                                    # staged → staging slot
    out_part, lse_part = sparse_decode_run(...)                # decode chunk
    merge_attn_states(out, prefix_lse, ...)                    # accumulate exact     for ref in chunk_refs: staging_pool.release(ref)   # recycle slots for next chunk
```
- Merge via `vllm/v1/attention/ops/merge_attn_states.py::merge_attn_states` (present in this fork): exact log-sum-exp merge; chunked result is element-wise equal to full-set sparse attention (standard trick; identical to FlashInfer cascade attention's merge).
- Slot recycle: after each chunk's contribution is merged, `staging_pool.release(ref)` each chunk ref and `residency.mark_cpu(ref, loc.cpu_slot)` so the residency map stays truthful (fix release's silent no-op bookkeeping while I'm in there).
- Same object for CPU-resident but unselected chunks? They simply aren't in the chunk list (they're not selected), so they stay null-filled.

### Ordering guards
- Fetches happen with the same stream-ordered visibility (`ensure_resident`'s `wait_stream(copy_stream)`) before each chunk's decode — unchanged.
- `wait_cpu_slot` per chunk's refs before their copy — unchanged.

### Out of scope (flagged)
- Chunked path under CUDA-graph replay (capture can't hold variable chunk loops; graphs rely on S2 sizing). The wrapper keeps its `_dyn_cg` path for graphs.
- Replication/editing `releases` perf — pool.release is cheap dict/set ops.

## Files touched
- `offload_package/config.py` — new fields + env parsing + validate.
- `offload_package/integration/connector.py` — compute effective slots, replace into config before wiring scheduler/worker.
- `offload_package/integration/scheduler_connector.py` — no logic change (reads same config); one log line for effective count.
- `offload_package/integration/worker_connector.py` — no logic change (same config flows in).
- `offload_package/integration/vllm_bridge.py` — `initialize_for_runner` log line for effective slots.
- `offload_package/staging/pool.py` — fix `release()` bookkeeping (residency flip on release) + expose `.num_slots`.
- `skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py` — chunked select→decode path behind `capacity check`, eager-only.
- Tests: new unit tests in `offload_package/tests/test_offload_invariants.py`.

## Tests
1. `test_staging_sizing_auto` — with fake vllm config (max_model_len, max_num_seqs) + env sparse knobs, assert effective slots computed == expected formula and capped fraction respected.
2. `test_staging_tail_outside_allocator` (existing) stays green — reservation just uses the bigger number.
3. `test_chunked_attention_parity` — synthetic: force capacity overflow (tiny pool, big cold set); chunked decode out == dense reference (allclose fp16 tolerance).
4. Full `pytest offload_package/tests -q` green.
5. Live: `bench_64k.py pq+offload` — expect zero "Failed to patch" and materially better TPOT than the current slow run, with identical output text vs the eager baseline (chunking is exact).

## Known open detail to check in implementation
- `merge_attn_states` output shapes for decode tokens [B, H, D] with LSE [B, H] — verify against the fork's op signature before wiring (a test asserts it).
