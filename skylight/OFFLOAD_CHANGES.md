# Skylight changes for KV offload

What the KV-offload project adds to Skylight, measured against two baselines:

| Part | Baseline | How to diff |
|---|---|---|
| `skylight/src/skylight/backend.py` | upstream Skylight, branch `ak/pq`, commit `88105e8` (github.com/skylight-org/skylight) | `curl -s https://raw.githubusercontent.com/skylight-org/skylight/88105e8191d952ad90d6b007151df775b3c71e9c/src/skylight/backend.py \| diff -u - skylight/src/skylight/backend.py` |
| `skylight/skylight_kernels` (its own git repo) | `00e151a` | `cd skylight/skylight_kernels && git diff 00e151a` |

Current size of the changes:
- `backend.py`: +392 / −12 over upstream's PQ backend.
- `skylight_kernels`: 9 files, +886 / −53. The PQ wrapper accounts for +599.

Before the cleanup, `skylight_kernels` was +1,760, including +1,472 for the PQ wrapper and 11 tracked `.pyc` files. Our own `backend.py` was +611 over the old `20d5255` copy, before being replaced by upstream's.

Line numbers refer to the current tree.

Each change belongs to one of four groups:
- **offload integration:** needed for block or token offload to work and stay correct;
- **bug fix:** a defect in the original code that the offload path hit;
- **build / environment:** toolchain fixes, unrelated to offload;
- **deterministic mode:** opt-in reproducibility (`SKYLIGHT_PQ_DETERMINISTIC=1`).

---

## 1. `skylight/src/skylight/backend.py`

This file is now upstream Skylight's backend (github.com/skylight-org/skylight, branch `ak/pq`, commit `88105e8191d9`, which has its own PQ support), plus the changes below: +392 / −12 lines, from the `curl … | diff` command in the table above. The PQ wiring this project once wrote itself is gone; upstream's version is used.

**Run settings.** Upstream drives PQ slot codes only on its FULL-cg slot path, and calls the production method `pq_optimized` (its `pq` is the pure-torch reference). For `method='pq'` the benches (`bench_1req_off99.py`, `bench_2req_composition.py`) set `SKYLIGHT_SPARSE_METHOD=pq_optimized`, `SKYLIGHT_INCR_FULLCG=1`, `SKYLIGHT_INCR_SLOT=1` and `SKYLIGHT_SPARSE_DYNAMIC_TOPK=1`, still running eager.

### Compatibility with this vLLM fork
- `get_flashinfer_layout_string(self.kv_cache_layout)` replaces `get_kv_cache_layout()`, which the fork doesn't have (`:37`, plus its two call sites).
- `_get_decode_wrapper` (`:1054`): `paged_kv_indices` is a plain tensor in the fork (upstream reads `.gpu`). Only reached on the CUDA-graph path.

### Offload integration
- **Optional `offload_package.timing`** (`:46`), with a no-op stand-in.
- **Class-level hooks** (`:249`):
  - `_offload_orchestrator`, `_offload_runner` (injected by `offload_package/integration/worker_connector.py`);
  - `_patch_bail_reason`, `_step_serial` / `_pattern_cached_serial`, `_cached_request_ids`.

  Plus a per-engine reset (`:281`) when the engine has no `OffloadPackageConnector`.
- **Guards:**
  - `SKYLIGHT_INCR_SYNCFREE=1` is refused with true freeing or token-level offload (`:451`);
  - PQ with offload requires the slot path, `SKYLIGHT_INCR_FULLCG=1` and `SKYLIGHT_INCR_SLOT=1` (`:467`). Without it `pq_optimized` encodes from the block table.
- **`build()`** (`:530`):
  - `_sk_offload_prebuild` (`:569`): bumps the step serial and records this step's decode row ids from `runner.input_batch` (with the `SKYLIGHT_OFFLOAD_DISABLE_RUNNER_IDS` fault switch);
  - `_sk_offload_postbuild` (`:605`): the fail-loud backstop on an index-patch bail under true freeing, and the orchestrator / ids hand-off to the PQ wrapper;
  - `n_keys` adds the token-offload offset back (`:540`), because `seq_lens` is physical.
- **`_compute_flashinfer_kv_metadata` override** (`:635`), followed by `_patch_indices_for_staging` (`:655`): row alignment using the host `indptr`, then `patch_paged_kv_indices_with_staging` (STAGING → staging slot, CPU → null page under true freeing).
- **`_sk_fill_slots`** (`:800`): minimal changes, explained by the comment block at `:784`.
  - Logical lengths (`seq_lens + _sk_tok_offsets`).
  - A row whose request moved gets its codes **copied** from the old row instead of re-encoded. That uses `_sk_plan_row_moves` (`:926`) and rows owned by request id (`_sk_current_row_ids` `:917`, `_sk_record_row_owners` `:999`).
  - Any remaining re-encode is checked by `_sk_assert_safe_reencode` (`:970`), which refuses compacted pages (`engine.remapped`) and freed pages.

  Upstream's occupancy detection and codebook-fit retry are unchanged.
- **`_note_selection`** (`:1142`, called after each layer's forward at `:1138`). Once per step it hands the PQ wrapper's `_last_topk_idx` to `orch.note_selection` (LRU recency for block offload).

---

## 2. `skylight/skylight_kernels/pq_optimized/batch_decode_with_paged_kv_cache_wrapper.py`

### Offload integration
- **Optional offload hooks** (`:45-59`): `offload_package.debug` / `timing`, with no-op stand-ins, so the kernels package still imports without `offload_package`.
- **Setters:** `set_offload_orchestrator` (`:211`) and `set_offload_request_ids` (`:215`), called from `backend.py` each build.
- **Per-step caches** reset in `plan()` (`:373-377`): the discovery cache plus the token-store and token-offload row maps. A released and re-added request can get a different slot.
- **Block offload, block-granularity fetch:**
  - `_discover_inrun_fetch` (`:427`) holds the true-freeing guards and the row-alignment check;
  - `_discover_vec` (`:524`) classifies selected blocks on the residency device mirror and stamps `dev_last_use` (protection plus recency);
  - `_apply_inrun_fetch` (`:555`) stages CPU blocks into `KVStagingPool` and patches the indices.
- **Block offload, per-layer token fetch (the default `per_layer_tokens`):** `_token_store_stage` (`:618`) uses `TokenStagingStore`.
- **Token-level offload:** `_token_offload_stage` (`:651`) stages each (position, KV head) entry in per-head (cell) mode.
- **Physical addressing:**
  - `_identity_tables` (`:677`): `indptr = 0`, `indices = arange`, so `sparse_idx` carries physical addresses;
  - in per-head (cell) mode, `dec_kv` (`:979`, `:998`) gives the kernel the flat KV buffer with `page_size = H·P` and `stride_h = 0`, so indices are global rows.

  The sparse decode kernel itself is unchanged.
- **PQ append under token offload** (`:798-812`): the newest token is read from its physical tail (`phys_lengths`), and its code is written at the logical position.
- **Dispatch** (`:905-950`):
  - a remapped token-offload request → token-offload stage;
  - CPU-resident blocks with the token store → token-store stage;
  - otherwise → block-granularity discovery and fetch.
- **Over-capacity (chunked) decode.** A base pass without the unfetched entries, then capacity-sized chunks, merged with `_lse2_merge` (`:693`):
  - token store / cells: `:1020-1040`;
  - block pages: `:1050-1110`.
- **Selection export:** `_last_topk_idx` (`:888`) is read by `backend._note_selection`.
- **Timing stages** (`pq.append`, `pq.luts`, `pq.adc`, `pq.select`, `wrap.*`): no-ops unless `SKYLIGHT_OFFLOAD_TIMING=1`.

### Bug fixes
- **Zero-weight padding** (`:875`): padding picks point at token 0 instead of arbitrary positions. They used to read stale, freed or reused page ids.
- **Chunked page loop** (`:1083`): the loop variable is now `chunk`, not `group`. `group` is the GQA ratio, reused afterwards by the attention-error measurement.

### Deterministic mode
- `SKYLIGHT_PQ_DETERMINISTIC=1` (`:876`): sorts the selection, so the softmax gathers in a canonical order.

## 3. Other `skylight_kernels` files

| File | Change | Group |
|---|---|---|
| `original_optimized/batch_decode_with_paged_kv_cache_wrapper.py` (`_unpack_paged_kv_cache`) | KV strides are taken from the tensors, not derived from shapes; non-contiguous innermost dimensions raise. The fork hands interleaved `[P, N, H, 2C]` K/V views, where shape-derived strides read the wrong elements. | bug fix (offload integration) |
| `test_original_optimized.py` | Regression tests for the stride fix; `FLASHINFER_DISABLE_VERSION_CHECK`. | test |
| `pq_optimized/slot_codes.py` | `phys_lengths` in `append` (token offload); combined `[P, H, N, 2C]` K view in `_norm_K`; optional `offload_package.timing` around the codebook fit; `[PQ FIT]` printed only when timing is on. | offload integration |
| `pq_optimized/cuda_ops.py` | ninja and `CUDA_HOME` on `PATH` so the CUDA scorer actually builds (otherwise a silent fallback ~17× slower); one-time fallback warning. | build / environment |
| `pq_optimized/selection.py` | Torch fallback scorer expands codes one subspace at a time (less memory, identical result). | build / environment (memory) |
| `pq/codebook.py` | Fixed-order Lloyd update under `SKYLIGHT_PQ_DETERMINISTIC`. | deterministic mode |
| `pqis_optimized/csrc/pqis_sampler_kernel.cu` | Local `MaxOp` replaces `cub::Max`, removed in CUDA 13. | build / environment |
| `setup.py` | `-std=c++20`, needed by the current torch headers. | build / environment |

Caveat, unchanged: `slot_codes._norm_K` decides it has the combined layout when `shape[1] == h_kv`. That's correct for this fork's caches, but it would misread a separate `[P, H, N, D]` cache.

---

## Removed in the cleanup
- **`backend.py`:**
  - the 1-step-lag prefetch (`_sparse_pattern_cache`, `_cached_sparse_idx/len`, the host copy of the pattern; only the removed `sparse_optimized` / `offload` methods and an oracle path PQ never takes used it);
  - `SKYLIGHT_SYNC_STAGING` and its `set_orchestrator` hand-off (no implementer);
  - the debug attributes `_last_patched_indices`, `_last_decode_wrapper`;
  - redundant `supports_attn_type` ×2, `is_sparse`, `supports_sparse`;
  - an always-on `sys.stdout.flush()` on every layer's `run()`, and all `_dbg` prints;
  - the per-step GPU sync for `indptr` in the index patcher.

  Also restored the deleted comment on `get_name() == "CUSTOM"`.
- **PQ wrapper:**
  - the `per_layer_tokens` page-store fetch path (`_apply_inrun_fetch_tokens`, the per-layer discovery branch, the token-pair chunks) and its host-loop discovery;
  - the dead `token_mode` / straggler branch;
  - the debug tools: `SKYLIGHT_PQ_VERIFY` probe, `SKYLIGHT_PQ_FETCH_STATS`, `SKYLIGHT_SELECTION_DUMP`, `[IDENT DBG]`, `[TOK LAYER MAP]`, `[PQ PATH]`.
- **`slot_codes.py`:** `[APPEND DBG]`.
- **pqis kernel:** an unused `<cuda/std/functional>` include.
- **Build artifacts:** tracked `__pycache__/*.pyc` restored to the base commit. They reappear when Python recompiles, so untrack them if they should never show up.
- **`skylight/skylight_kernels_blockminmax`:** an orphan gitlink (no `.gitmodules`), untracked. The directory stays on disk.
- **Paired removals in `offload_package`:**
  - `KVStagingPool.ensure_resident_tokens`, and the residency partial-page masks (`staged_mask`, `dev_cov`, `merge_staged_mask`, `staged_coverage`);
  - the `staging_store="page"` option with `per_layer_tokens`, which now always uses the token store;
  - the selection score dump;
  - the `SKYLIGHT_PQ_VERIFY` end-to-end gates;
  - `tests/test_token_fetch.py`, plus the page-store parts of `test_staging_pool.py` and `test_residency.py`.
