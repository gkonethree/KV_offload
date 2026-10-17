# KV offload: code walkthrough

How a request flows through `offload_package`, the vLLM fork (`vllm-0.29.0/`) and the Skylight PQ sparse-attention backend (`skylight/`). Each section names the data structures, the exact control flow with `file:line` references, and the invariants worth checking. Section 7 lists known issues.

Line numbers refer to the working tree when this was written. If one has drifted, grep for the function name.

---

## 0. What actually runs

The benchmark entry point is [bench_1req_off99.py](../bench_1req_off99.py). It pins down an environment that much of the code depends on:

| Setting | Where | Why it matters |
|---|---|---|
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | [bench_1req_off99.py:25](../bench_1req_off99.py#L25) | The engine core runs in-process. The scheduler and worker then share one connector object, and they reach each other's state through it. |
| `VLLM_USE_V2_MODEL_RUNNER=0` | [bench_1req_off99.py:26](../bench_1req_off99.py#L26) | Uses the **V1** runner `vllm/v1/worker/gpu_model_runner.py`. All token-offload runner changes are there. With the V2 runner, `_build_orchestrator` raises. |
| `register_backend(CUSTOM, skylight...)` + `attention_config={'backend':'CUSTOM'}` | [bench_1req_off99.py:34](../bench_1req_off99.py#L34) | Attention is Skylight's `SkylightSparseBackend`. With `SKYLIGHT_SPARSE_METHOD=pq` it uses `pq_optimized.BatchDecodeWithPagedKVCacheWrapper`. |
| `enforce_eager=True`, `max_num_seqs=1`, `async_scheduling=False` | `run()` [bench_1req_off99.py:213](../bench_1req_off99.py#L213) | No CUDA graphs, one request, synchronous scheduling. Several invariants rely on this (section 7). |
| prefix caching off for token mode | same | Token mode asserts it is off. |

**Live code paths:**
- **Block offload** (`offload_granularity="block"`, the default). With the default `fetch_granularity="per_layer_tokens"`, cold blocks are fetched through `TokenStagingStore.stage` (the per-layer token store). `KVStagingPool.ensure_resident` (block fetch) runs only when `fetch_granularity="block"`.
- **Token offload** (`offload_granularity="token"`):
  - `token_budget_scope="global"` (**default**, FastKVzip's own policy): `HeadTokenOffloadEngine` with one ranking over all (layer, head, token) scores, and one staging pool shared by all layers and heads. `"layer"`, `"uniform"` and `"cells_shared"` use the same engine with other budget rules;
  - `token_budget_scope="shared"`: the older `TokenOffloadEngine`, the same tokens for every head of a layer, the same count in every layer, per-layer LRU staging. In the bench these are `pq+tok70s` / `pq+tokrand70s`; `pq+tok70` / `pq+tokrand70` now mean the default.

**Not live in the bench:**
- the legacy oracle adapters `integration/sparse_optimized_adapter.py`, `sparse_offload_adapter.py`, `offload_sparse_adapter.py` (reached only with `SKYLIGHT_SPARSE_METHOD=sparse_optimized|offload`);
- the orchestrator's `on_decode_step` family;
- the load-event machinery;
- the wrapper's `token_mode` / straggler branch.

See section 7.4.

### Ownership

```
OffloadPackageConnector                      integration/connector.py:243   (one object, both roles)
 ├─ OffloadPackageWorker                     integration/worker_connector.py
 │    ├─ OffloadOrchestrator                 integration/manager.py:21      (built by shared/types.py:42 build_orchestrator)
 │    │    ├─ ResidencyTable (+device mirror) staging/residency.py:37
 │    │    ├─ KVOffloadManager                offload/manager.py:32          (scorer, pinned CPU block cache)
 │    │    ├─ KVStagingPool                   staging/pool.py:25             (reserved tail pages, pinned rings)
 │    │    └─ token_store: TokenStagingStore  staging/token_store.py:44
 │    └─ token mode: TokenOffloadEngine | HeadTokenOffloadEngine + FastKVzip score hooks
 └─ OffloadPackageScheduler                  integration/scheduler_connector.py (block mode)
      └─ token mode: TokenModeScheduler      integration/token_mode.py:278
Skylight builder class attrs: _offload_orchestrator (worker_connector.py:227), _offload_runner
PQ wrapper: set_offload_orchestrator(...) each build (skylight/src/skylight/backend.py:561)
```

### Configuration
`build_offload_config` ([connector.py:164](integration/connector.py#L164)) maps `kv_connector_extra_config` to a frozen `OffloadConfig` ([config.py](config.py)). The bench's configs map to it like this:

| bench CONFIGS key | extra_config key | OffloadConfig field |
|---|---|---|
| `fraction` | `offload_fraction` | `offload_fraction` |
| `staging_policy`, `num_staging_slots` (or `BENCH_STAGING_SLOTS` → fixed) | same | `staging_policy`, `num_staging_slots` |
| `scorer_type` | `scorer_type` | block scorer (paged_eviction / lru / random) |
| `fetch_granularity` | `fetch_granularity` | `per_layer_tokens` (default) or `block` |
| `granularity` | `offload_granularity` | `block` / `token` |
| `token_scorer` | `token_scorer` | `fastkvzip` / `random` |
| `budget_scope` | `token_budget_scope` | `shared` / `global` / `layer` / `uniform` / `cells_shared` |

Fixed values: `interval_tokens=BENCH_INTERVAL` (256 in the runs), `num_cpu_slots=4096`, `true_free=True`.

With `staging_policy="auto"`, `_auto_staging_slots_for` ([connector.py:80](integration/connector.py#L80)) sizes staging from the sparse top-k/sink/local settings, GQA sharing, `max_model_len` and `max_num_seqs`. For example, at top-k 10% that gives 336 pages; at 20%, 490.

**Check:** precedence is not uniform. Some keys can be overridden by `OFFLOAD_PACKAGE_*` environment variables; `offload_fraction`, `interval_tokens` and `num_cpu_slots` read only `extra_config`.

---

## 1. Startup (once per engine)

1. **Worker connector.** `Worker.initialize_from_config` calls `ensure_kv_transfer_initialized`, which creates the connector from `kv_connector_module_path=...offload_package_connector` → `OffloadPackageConnector.__init__` ([connector.py:243](integration/connector.py#L243)). That builds the config, an `OffloadPackageWorker`, and an `OffloadPackageScheduler` (with its `KVCompressionController`, W = `interval_tokens`).
   - The scheduler computes the **true-free gate** `_true_free_enabled` ([scheduler_connector.py:140](integration/scheduler_connector.py#L140)): enabled, `true_free`, no async scheduling, not multiprocess, world size 1, no CUDA graphs.
2. **Fork, `kv_transfer_state.py:97-99`.** `connector.initialize_from_config(kv_cache_config)` records the KV spec and `staging_base_page = num_blocks - S`.
3. **Fork, `gpu_worker.py:738-742`.** After the V1 runner allocates KV, `connector.set_runner(model_runner)` → `OffloadPackageWorker.set_runner` ([worker_connector.py:516](integration/worker_connector.py#L516)) → `_build_orchestrator` ([worker_connector.py:120](integration/worker_connector.py#L120)):
   - `normalize_runner_kv_caches` ([shared/types.py:166](shared/types.py#L166)) accepts the fork's per-layer combined tensors `[num_blocks, H, P, 2·head_dim]` (K in `[..., :D]`, V in `[..., D:]`). The same tensor object serves as both `k_caches[l]` and `v_caches[l]`.
   - `build_orchestrator` ([shared/types.py:42](shared/types.py#L42)) creates, in order:
     - the pinned CPU block cache `[L, num_cpu_slots, 2, P, H, D]`;
     - the scorer;
     - a `ResidencyTable` with device mirror;
     - `KVOffloadManager`;
     - `KVStagingPool` (the tail pages `[num_blocks-S, num_blocks)`);
     - `OffloadOrchestrator`;
     - `orch.token_store = TokenStagingStore(pool, residency)` when the staging store is `token`.
   - It injects the orchestrator into every Skylight builder **class** (`type(builder)._offload_orchestrator = orch`, [worker_connector.py:227](integration/worker_connector.py#L227)). Each `build()` passes it into the PQ wrapper ([backend.py:561](../skylight/src/skylight/backend.py#L561)).
   - **Token mode:** `_build_token_offload` ([worker_connector.py:255](integration/worker_connector.py#L255)) builds the engine (shared or cell, chosen by `token_budget_scope`) and the gates (FastKVzip from the HF repo `Jang-Hyun/Fast-KVzip`, or random). It then:
     - calls `install_score_hooks` ([token_mode.py:382](integration/token_mode.py#L382));
     - sets `runner.token_offload = engine` (read by the fork's runner code) and `orch.token_offload = engine`;
     - replaces `orch.token_store` with `TokenStagingStore.for_token_engine(pool, engine)`.
4. **Scheduler.** It **reuses** that connector object (fork, [scheduler.py:154](../vllm-0.29.0/vllm/v1/core/sched/scheduler.py#L154)), then:
   - `bind_gpu_block_pool(block_pool, kv_cache_manager)` ([connector.py:291](integration/connector.py#L291) → [scheduler_connector.py:159](integration/scheduler_connector.py#L159)). This calls `reserve_staging_tail` ([cache_allocator.py:22](cache_allocator.py#L22)), which `touch`es the last S free blocks so vLLM never allocates them.
   - In token mode, it creates `TokenModeScheduler` ([connector.py:362](integration/connector.py#L362)).
   - `bind_scheduler(self)` ([connector.py:368](integration/connector.py#L368)) is used by token mode; it asserts prefix caching is off.

**Check:** the scheduler and worker sharing one object is the core in-process assumption. With TP > 1 or a separate worker process, the scheduler's instance would have no orchestrator or engine, and offload would silently do nothing.

---

## 2. vLLM fork changes

| Change | File | What and why |
|---|---|---|
| connector module | `distributed/kv_transfer/kv_connector/v1/offload_package_connector.py`, `v1/__init__.py` | Re-exports `OffloadPackageConnector` for `kv_connector_module_path`. |
| `initialize_from_config` call | `distributed/kv_transfer/kv_transfer_state.py:97-99` | Gives the connector the KV config on creation. |
| `set_runner` call | `v1/worker/gpu_worker.py:738-742` | Builds the orchestrator after KV allocation. |
| scheduler reuses connector, binds block pool and scheduler | `v1/core/sched/scheduler.py:151-160, 308-319` | Scheduler and worker share the object (section 1). |
| `Request.num_offloaded_tokens` | `v1/request.py:184-185` | Logical minus physical KV length (token mode). |
| physical-length allocation | `v1/core/kv_cache_manager.py:460-483` | `num_local_computed_tokens -= num_offloaded`; the full-fit check uses `num_tokens - num_offloaded`. Asserts prefix caching is off. |
| block-table row replacement | `v1/worker/gpu_model_runner.py:1484` | After a round, `take_row_update(req_id)` replaces the request's block-table row with its truncated page list. |
| physical `seq_lens` | `gpu_model_runner.py:2131-2141` | `optimistic_seq_lens_cpu -= offset`; feeds `max_seq_len` and the attention metadata's `seq_lens`. |
| physical slot mapping | `gpu_model_runner.py:2233-2242` | `slot_positions = positions - offset`: KV writes go to physical slots. `positions` stay **logical**, so RoPE is correct. |
| `free_blocks` debug | `v1/core/block_pool.py` | `OFFLOAD_PACKAGE_DEBUG`-gated prints. |
| FlashInfer metadata fields | `v1/attention/backends/flashinfer.py:670, 674` | `paged_kv_indices` and `paged_kv_indptr` are declared twice, and nothing reads them. **Dead.** |
| CUSTOM backend condition | `platforms/cuda.py:415` | `selected_backend is not None or ... is CUSTOM` is a no-op; the bench's monkeypatch is what makes CUSTOM work. |
| RoPE length bump | `config/model.py` | With `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, sets `max_position_embeddings = max_model_len`. |
| V2 runner edits | `v1/worker/gpu/model_runner.py`, `gpu/kv_connector.py` | Not used: the bench forces V1, and offload raises on V2. |

---

## 3. Block mode, per step

### 3.1 Scheduler: `OffloadPackageScheduler.build_connector_meta` ([scheduler_connector.py:196](integration/scheduler_connector.py#L196))
This runs at the end of `Scheduler.schedule()`, after this step's blocks are allocated.

1. **Preemption.** Preempted requests go to `_forget_request` ([scheduler_connector.py:547](integration/scheduler_connector.py#L547)), which drops orchestrator, staging and CPU state.
2. **Block tracking.** For each request it rebuilds the full block list, since vLLM sends only new ids, and calls `orch.sync_request_blocks` ([manager.py:62](integration/manager.py#L62)):
   - a block is **complete** when `(idx+1)*P <= num_computed`;
   - blocks already in CPU or STAGING are never re-bound to a GPU id;
   - newly complete GPU blocks are batch-scored (`KVOffloadManager.score_blocks`, [offload/manager.py:104](offload/manager.py#L104)) with the fused CUDA norm kernel (`scoring/cuda_norms.py`), the Triton kernel, or the torch fallback.
3. **Triggers.**
   - Prefill completion: `num_computed >= prompt + max(num_output-1, 0)` → `controller.on_prefill_complete` and `orch.on_prefill_complete` ([manager.py:114](integration/manager.py#L114)).
   - Decode interval: `controller.on_decode` ([controller.py:30](controller.py#L30)) fires every W tokens.
4. **Offload round.** `_trigger_request_offload` ([scheduler_connector.py:365](integration/scheduler_connector.py#L365)):
   - offloadable = complete and not pinned (`_pinned_block`, [scheduler_connector.py:356](integration/scheduler_connector.py#L356): sink blocks and blocks overlapping the local window);
   - `keep = budget`, or `ceil((1-f)·n_offloadable)`. This is a target, so repeated rounds don't compound;
   - under true free only `ref_cnt == 1` blocks are eligible;
   - `offload_lowest` ([offload/manager.py:236](offload/manager.py#L236)) asks the scorer for the lowest blocks and calls `_copy_gpu_to_cpu_batch` ([offload/manager.py:299](offload/manager.py#L299)): per-layer `index_select`, K/V split from the combined last axis, one non-blocking D2H, then **`event.synchronize()`**, then `mark_cpu`. All of this happens synchronously inside `schedule()`.
5. **Store event.** The step's offloaded refs become one store event in the connector metadata.

### 3.2 Freeing pages (true free)
The round trip:
1. The worker's `wait_for_save` ([worker_connector.py:340](integration/worker_connector.py#L340)) records a CUDA event after the forward.
2. `get_finished` ([worker_connector.py:363](integration/worker_connector.py#L363)) polls it and reports `completed_store_events` in the worker metadata.
3. The scheduler's `update_connector_output` → `_process_store_completion` ([scheduler_connector.py:467](integration/scheduler_connector.py#L467)) → `_free_offloaded_pages` ([scheduler_connector.py:488](integration/scheduler_connector.py#L488)). For each ref:
   - if the table entry still equals the offloaded page id, `req_to_blocks[rid][idx] = null_block`;
   - add it to `orch.freed_pages`;
   - evict its prefix hash;
   - `block_pool.free_blocks`.

With true free off, the CPU copy exists but the page is not reclaimed ("shadow tier").

### 3.3 Residency (`staging/residency.py`)
- **States:** `GPU`, `CPU`, `STAGING` (`staged_mask`: `None` means the whole page; otherwise per-layer token masks).
- **Device mirror** (`enable_device_mirror`, [residency.py:67](staging/residency.py#L67)):
  - `dev_state[S, MB]` int8, `dev_cov[L, S, MB]` and `dev_last_use[S, MB]`, indexed by a stable per-request mirror slot;
  - host transitions mark refs dirty;
  - `flush_device` ([residency.py:137](staging/residency.py#L137)) applies them;
  - `dev_last_use == tick` means "selected by the current layer" and protects a block from eviction.

### 3.4 Attention, per layer (PQ wrapper `run()`, [batch_decode_with_paged_kv_cache_wrapper.py:1400](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1400))
1. **Builder.** Before the forward, the builder's `_patch_indices_for_staging` ([backend.py:613](../skylight/src/skylight/backend.py#L613)) checks that each row's page span matches the tracked block count. It then rewrites `paged_kv_indices`: STAGING blocks point to their staging slot, CPU blocks point to the null page (under true free).
2. **PQ append** ([wrapper:1502](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1502)). Encodes the new key into the PQ code buffer.
3. **PQ select.** `build_luts` → `adc_scores` ([wrapper:1522](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1522)) over all logical positions. With `SKYLIGHT_PQ_GQA_SHARE=1`, `share_across_group` ([wrapper:1530](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1530)) shares scores across a GQA group. `select` then picks sink, local window and top-k, giving `sparse_idx [B, Hq, K]` of **logical** positions.
4. **Fetch.** One of three branches:
   - **(default) token store:** `_token_store_stage` ([wrapper:945](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L945)) → `TokenStagingStore.stage` ([token_store.py:175](staging/token_store.py#L175)):
     - cold = `dev_state[row, pos//P] == CPU`;
     - key = `mirror_slot·mbp + pos`;
     - hits are stamped with the tick;
     - missing keys come back in **one readback**;
     - `fetch` places them in this layer's LRU slots, never evicting current-tick slots (section 4.4);
     - the decode uses physical addresses through identity page tables (`_identity_tables`, [wrapper:1004](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1004));
     - over capacity: the chunked path (section 4.6).
   - **page store, `per_layer_tokens`:** `ensure_resident_tokens` ([pool.py:436](staging/pool.py#L436)), which allocates or gap-fills pages per layer.
   - **block granularity:** `ensure_resident` ([pool.py:262](staging/pool.py#L262)). Victims come from `_victim_order` ([pool.py:221](staging/pool.py#L221)). It copies from the pinned slab ring and patches `paged_kv_indices` once, which serves every layer.
5. **Sparse decode** over the selected positions.

---

## 4. Token mode, per step

### 4.1 State

**Shared engine** `TokenOffloadEngine` ([token_mode.py:48](integration/token_mode.py#L48)). GPU tensors are `[L, S, max_len]` (`_grow`, [token_mode.py:89](integration/token_mode.py#L89)):

| tensor | meaning |
|---|---|
| `score` f32 | FastKVzip score, **max over KV heads** |
| `phys_of_logical` i32 | physical slot in the layer's compacted prefix, -1 = offloaded |
| `logical_of_phys` i32 | inverse map |
| `cpu_slot` i32 | CPU pool slot |

- **CPU pool:** pinned `[L, N, H, 2C]`. A round evicts the same count E in every layer, so one free list serves all layers.
- **`offset(rid)`** is `offloaded`, the same in every layer.

**Cell engine** `HeadTokenOffloadEngine` ([token_mode_heads.py:96](integration/token_mode_heads.py#L96)):
- **`flat_cells`** ([token_mode_heads.py:60](integration/token_mode_heads.py#L60)). vLLM puts all layers in **one backing buffer**: layer l's page of block b sits at `offset + l·layer_stride + b·block_stride`. The buffer is therefore a flat array of `L·num_blocks` equal **units** (one layer's page), each holding `H·P` rows of 2C.
  - `kv_rows = k0.as_strided((n_units·H·P, C2), (C2, 1), 0)` views every row.
  - `unit(l, b) = unit_base[l] + b·unit_stride`.
  - A page stores rows head-major `[H, P]` or token-major `[P, H]`; this model's cache is token-major, strides `(32768, 256, 2048, 1)`. `std_cell` ([token_mode_heads.py:172](integration/token_mode_heads.py#L172)) gives vLLM's own row: `h·P + o` or `o·H + h`.
- **GPU tensors** `[L, S, max_len, H]`: `score` (one per KV head), `cell` (global row of a resident (layer, token, head), -1 = offloaded), `cpu_row`.
- **CPU pool:** pinned `[L·H·num_cpu_tokens, 2C]`, one row per offloaded (layer, head, token). The free list is a stack.
- **`offset = Lg − nb·P`** (vLLM's view, see section 4.3).

**Staging store** `TokenStagingStore` ([token_store.py:44](staging/token_store.py#L44)). `C` ([token_store.py:57](staging/token_store.py#L57)) is:
- shared engine: `num_slots·P` token slots **per layer**, in that layer's own staging pages;
- cell engine: `num_slots·P·H·L` rows in **one pool for all layers and heads**: every layer's unit of every staging page.

Keys: `row·mbp + pos` (shared), or `(row·H + kvh)·mbp + pos` (cell).

Maps:
- `tok_slot [L, S, kps]`: staging slot of a key, -1 = not staged;
- `seen [L, S, kps]`: tick at which a key was last requested (dedup);
- `owner`, `last_use`: per layer `[L, C]` (shared engine), or `[C]` (cell). In cell mode `owner = layer·S·kps + key`, which indexes the all-layer `tok_slot` directly.

### 4.2 Scoring and mapping
- **Score hooks** (`install_score_hooks`, [token_mode.py:382](integration/token_mode.py#L382)). A forward pre-hook on every `self_attn` reads `hidden_states` (post-`input_layernorm`, exactly the gate's input) and the logical `positions`.
  - Rows map to engine request slots through `runner.query_start_loc` and `input_batch.req_ids` (`_token_rows`, [token_mode.py:410](integration/token_mode.py#L410)).
  - Forwards of ≤ 512 tokens (decode) stash every layer's input and score all layers at once with `BatchedFastKVzipGates` ([fastkvzip.py:80](scoring/fastkvzip.py#L80)). Prefill chunks are scored per layer.
  - Shared engine: `amax` over heads. Cell engine: one score per KV head.
- **Gate math** ([fastkvzip.py:66](scoring/fastkvzip.py#L66)): `score = mean_G 1/(1 + Σ_sink exp(logit_sink − logit))`. It depends only on the token's own hidden state, so a token is scored once and never re-scored.
- **`on_step`** ([shared, token_mode.py:156](integration/token_mode.py#L156); [cell, token_mode_heads.py:229](integration/token_mode_heads.py#L229)) maps the tokens vLLM writes this step to physical position `logical − offset`:
  - shared: physical slot `phys`;
  - cell: `unit(l, page)·HP + std_cell(h, o)` in every layer and head.

### 4.3 Offload round

**Driver.** `TokenModeScheduler.build_connector_meta` ([token_mode.py:322](integration/token_mode.py#L322)) runs at the end of `schedule()`:
- per request: `set_pages`, then `on_step(num_computed, 0)`;
- a round is due when `prompt > 0 and num_computed > prompt and (first or num_computed − last ≥ W)` ([token_mode.py:349](integration/token_mode.py#L349)). The first round is one step after prefill, because PQ codes are built from the pages at the first decode step;
- `freed = run_round(rid, extra_tokens=n_sched)`;
- the freed tail blocks are removed from `req_to_blocks` and returned with `block_pool.free_blocks`;
- `request.num_offloaded_tokens = offset`;
- finally `on_step(num_computed, n_sched)`.

The runner then takes `take_row_update`.

**Shared `run_round`** ([token_mode.py:191](integration/token_mode.py#L191)):
1. `K = max(ceil((1−f)·Lg), protected)`; `E = phys_written − K` per layer.
2. Per layer, the lowest-score non-protected tokens (sink `token_sink`, window `token_window`) are chosen by `topk`.
3. D2H into the CPU pool, with a stream sync.
4. Order-preserving compaction of the survivors to physical `[0, K)` in every layer (cloned first, so overlapping moves are safe).
5. Rebuild the maps; `offloaded += E`.
6. Truncate the pages to `ceil((K + extra)/P)` and return the rest.

**Cell `run_round`** ([token_mode_heads.py:302](integration/token_mode_heads.py#L302)):
1. `K` as above. Protected and non-resident cells get score `+inf`.
2. `_select` ([token_mode_heads.py:282](integration/token_mode_heads.py#L282)) evicts the lowest according to the scope:
   - `global`: one ranking over all (layer, token, head), keep `L·H·K` (FastKVzip's own default);
   - `layer`: keep `H·K` per layer;
   - `uniform`: keep `K` per (layer, head);
   - `shared`: per (layer, token), heads combined by max.
3. D2H the evicted rows into new CPU rows, in 64K-row chunks through a pinned bounce buffer.
4. **Pack** the survivors in (layer, head, position) order densely into the units of the request's first `nb = ceil(T/(L·H·P))` blocks, layer-major. A layer may use other layers' units, which is how per-layer totals can differ. `kv_rows[dst] = kv_rows[src]` gathers the source before scattering.
5. `offset = Lg − nb·P` ([token_mode_heads.py:366](integration/token_mode_heads.py#L366)). Keep `nb + ceil(extra/P)` pages and return the rest.

New tokens then land in the standard per-layer layout after the packed blocks (the append region), and the next round repacks them.

### 4.4 Decode, per layer
In the wrapper `run()`:
1. **PQ append** ([wrapper:1502](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1502)) with `phys_lengths = (pages−1)·P + last_page_len`. The code row is the **logical** position; K is read from the newest physical slot. That slot is always in vLLM's standard layout (the tail, or the append region).
2. **PQ select**, as in section 3.4, giving logical `sparse_idx [B, Hq, K]`.
3. **Gate** ([wrapper:1620](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1620)): if `any(engine.offset(r))` and the stream is not capturing, use `_token_offload_stage` ([wrapper:978](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L978)). See issue 7.1.
4. **Discover.** `TokenStagingStore.stage_tokens` ([token_store.py:202](staging/token_store.py#L202)) → `_stage_tokens_kernel` ([token_store.py:269](staging/token_store.py#L269)) → CUDA `tok_discover` ([token_store_kernels.cu:43](staging/csrc/token_store_kernels.cu#L43)). For each entry:
   - `kvh = ((i % N)/ksel)/group` (cell mode);
   - look up the map: resident gives a physical slot or cell;
   - cold and staged: stamp the slot with this tick;
   - cold and missing: emit the key once (`seen` dedup).

   Then **one counter readback** (the per-layer GPU sync): `{n_need, n_hit, err, n_cold}`.
5. **Fetch** ([token_store.py:340](staging/token_store.py#L340), [token_store.py:376](staging/token_store.py#L376)). It fails without doing anything if `n > C − stamped`. Victim choice:
   - shared engine: per-layer **LRU**, `topk` over `last_use`, skipping current-tick slots;
   - cell engine: **ring order** over the shared pool (`_victims`, [token_store.py:255](staging/token_store.py#L255)). It takes the next `n` rows after a hand, skipping only rows stamped this tick (in use by the current layer). The window is `n + stamped`, so it never fails. The hand then advances past the window. There is no recency ordering.
6. **Fill.**
   - `tok_fill` (shared engine, [token_store_kernels.cu:88](staging/csrc/token_store_kernels.cu#L88)): one block per token copies `H × 2C` straight from the device-mapped (UVA) pinned pool.
   - `tok_fill_cells` (cell, [token_store_kernels.cu:143](staging/csrc/token_store_kernels.cu#L143)): one warp per key copies one 2C row into the staging cell `stage_units[v/HP]·HP + v%HP`, and clears the victim's previous owner's `tok_slot` entry, which may belong to another layer.
7. **Address.**
   - `tok_phys` (shared engine): resident = `indices[indptr[b] + p/P]·P + p%P`; cold = `slot + base·P`; unfetched = 0.
   - `tok_phys_cells` (cell): resident = its cell; cold = its staging cell; unfetched = 0.
8. **Decode** (`_decode`, [wrapper:1826](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1826)):
   - shared engine: the layer's own cache with identity tables over its pages;
   - cell: `dec_kv = (kv_rows, kv_rows_v, HP·C2, C2, 0, 0, HP)` ([wrapper:1848](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1848)) with identity tables over `n_units`.

   The kernel computes `page_local = t / page_size; addr = page·stride_page + kv_head·stride_h + in_page·stride_n` ([sparse_decode_kernel.cu:149](../skylight/skylight_kernels/sparse_optimized/csrc/sparse_decode_kernel.cu#L149)). With `page_size = H·P` and `stride_h = 0`, each index is a global row; **no kernel change is needed**. Tested bitwise in `test_token_store.py::test_flat_cell_decode_bitwise_equal`.

### 4.5 Skylight backend hooks
- `_sk_tok_offsets` ([backend.py:859](../skylight/src/skylight/backend.py#L859)) adds the offset back for PQ lengths and `n_keys`.
- `_sk_assert_safe_reencode` ([backend.py:874](../skylight/src/skylight/backend.py#L874)) refuses to re-encode PQ codes from compacted pages. Codes move with their row instead (`_sk_plan_row_moves`, [backend.py:830](../skylight/src/skylight/backend.py#L830)).

### 4.6 Over capacity (chunked)
If the missing keys don't fit ([wrapper:1873](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1873)):
1. A base pass with the missing keys' weights zeroed.
2. Then, per capacity-sized group:
   - `tick += 1`, so earlier chunks become evictable;
   - `fetch`;
   - recompute the addresses;
   - decode with only that group's weights;
   - merge with `_lse2_merge` ([wrapper:1379](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1379)).

Weight 0 becomes `log2(1e-20)` in the kernel: attenuated, not skipped. Address 0 must therefore hold finite data, which it does (vLLM's null block / cell 0).

---

## 5. Benchmark harness
- **Warm-up** (`BENCH_WARMUP=1`): an 8-token request first pays the PQ codebook fit and JIT costs; then `timing.reset()` ([bench_1req_off99.py:314](../bench_1req_off99.py#L314)).
- **Forced decoding** ([bench_force.py](../bench_force.py), `BENCH_FORCE_TOKENS`). A logits processor forces every step to the reference token. Before forcing, it records whether the config's own argmax matched (**agree %**) and the reference token's NLL.
- **Counters** (`timing.note_count`):
  - `tok.sel_entries`: top-k entries;
  - `tok.sel_cold`: entries that were offloaded;
  - `tok.need_keys`: keys fetched from CPU;
  - `pool.d2h_token`: bytes fetched (named before it measured H2D).
- **KV meter.** Free blocks are sampled after each `schedule()`, excluding the reserved staging ([bench_1req_off99.py:381](../bench_1req_off99.py#L381)).

---

## 6. Tests: what each proves
| Test | Proves |
|---|---|
| `test_token_mode_engine.py` | Shared engine: per-layer selection, protected tokens, byte-exact compaction, page freeing. |
| `test_token_mode_head_budget.py` | Cell engine across three layouts: scope budgets, data integrity, layers using other layers' units, append + repack, CPU row release, `cells_shared` = shared engine. |
| `test_token_store.py` | Torch-path staging; **bitwise** equality of identity-table and flat-cell decode (including rows moved into other layers' pages). |
| `test_token_kernels.py`, `test_token_kernels_heads.py` | Kernels == torch path: outputs, maps, staged bytes, over-capacity and chunk fetch, the shared pool letting one layer exceed an equal share, release, missing CPU copy. |
| `test_fastkvzip_batched.py`, `test_fastkvzip_scorer.py` | Batched gates == per-layer gates (max and per head); agreement with the released gates. |
| `test_staging_pool.py`, `test_offload_invariants.py`, `test_core.py`, `test_request_offload_target.py`, `test_chunked_prefill.py`, `test_integration.py` | Block mode: completeness, target budget, staging-tail reservation, index patching, current-selection protection, copy round-trip, LSE2 merge. |

**Gaps:**
- the offset ≤ 0 case (7.1);
- `TokenModeScheduler` end to end, including preemption;
- the wrapper's chunked merge numerics;
- the score hooks against a real runner;
- some permanent `pytest.skip` placeholders in `test_offload_invariants.py`.

---

## 7. Known issues (to verify or fix)

### 7.1 Cell engine: offset can be ≤ 0 after a round (high)
`offset = Lg − nb·P` with `nb = ceil(T/(L·H·P))` rounded up to whole blocks.
- **Offset = 0.** Example: Lg = 1008, P = 16, keep 1000 → nb = 63 → offset = 0. Then `any(offset)` is false at [wrapper:1622](../skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py#L1622), and the decode reads **packed cells as a normal paged layout**. The re-encode guard ([backend.py:881](../skylight/src/skylight/backend.py#L881)) is skipped too.
- **Offset < 0.** Example: Lg = 1001, keep 1000 → nb·P = 1008 → offset = −7. The next token's physical position falls past the allocated pages, and `on_step` raises.

This is only reachable when almost nothing is offloaded (fraction ≈ 0 or a large absolute budget). It is not reached at f = 0.7. The shared engine is safe: its offset = offloaded > 0.

### 7.2 Token mode: environment and lifecycle (high/medium)
- **No gate** for async scheduling, multiprocess, world size or CUDA graphs. Block mode gates true free; token mode compacts KV inside `schedule()` unconditionally.
- **Recompute-prefill after preemption.** `due = num_computed > prompt` can fire a round mid-recompute for a resumed request.
- **Batching** (`max_num_seqs > 1`):
  - [gpu_model_runner.py:1484](../vllm-0.29.0/vllm/v1/worker/gpu_model_runner.py#L1484) replaces the persistent row, but `req_state.block_ids` keeps the old pages. A request re-added to the batch is rebuilt from stale, partly freed pages.
  - `_num_computed_tokens_cpu` stays logical while `seq_lens` is physical.
- **Sync-free path.** `SKYLIGHT_INCR_SYNCFREE=1` (`_sk_fill_slots_syncfree`, [backend.py:910](../skylight/src/skylight/backend.py#L910)) ignores token offsets. The existing guard ([backend.py:346](../skylight/src/skylight/backend.py#L346)) refuses it only with `true_free=True`, so token offload with `true_free=False` slips through.
- **Unscored tokens.** Tokens decoded under CUDA-graph replay, or when a hook layer is skipped, keep score 0, so they are evicted first.

### 7.3 Kernels and staging (medium/low)
- The kernel `err` flag is never cleared: it is reported one layer late and then sticks.
- Cell-mode cross-layer staging eviction relies on all layers' kernels running on one stream (true today).
- Row caches `_tok_rows` / `_tstore_rows` persist across steps, and are refreshed only when the request-id tuple changes.
- Block mode:
  - `RandomScorer` has no budget guard;
  - `_copy_gpu_to_cpu_batch` assumes the combined K‖V layout;
  - with `scorer` / `random` staging eviction, the chunked path could evict a ref resolved earlier in the same call;
  - CPU pool exhaustion is fatal.
- Cell-engine GPU maps grow as `L·S·max_len·H` (≈ 57 MB each at 8k/12k context, `S=4`).

### 7.4 Dead or broken but unreachable in the bench
- The legacy oracle adapters crash if enabled: they call `_dbg` without importing it and read `orchestrator.offload_config`.
- `OffloadOrchestrator.on_decode_step` and related fields.
- The load-event metadata, and interface no-ops (`start_load_kv`, `wait_for_layer_load`, `save_kv_layer`).
- The wrapper's `token_mode` / straggler branch.
- The duplicated FlashInfer metadata fields.
- The no-op `platforms/cuda.py` condition.
- The V2-runner offload edits.
- Stale `.pyc` files of deleted modules.
