# KV Offload Bug-Fix Implementation Plan

**Status — CURRENT: 64k benchmarks incl. CUDA-graph runs, all landed on the rebuilt stack.**

## Kernel/toolchain + fork fixes this phase
- Rebuilt `skylight_kernels` AOT `.so`s for root venv (torch 2.13.0+cu130, venv nvcc 13.0.88); symbol-gate zero `CUDAErrorLogCapture` refs; all wrappers import in root venv; editable finder mapping re-extended post-reinstall.
- Fork fix `vllm/config/model.py`: `VLLM_ALLOW_LONG_MAX_MODEL_LEN` now also bumps `hf_config.max_position_embeddings` — the piecewise compile path's `forward_static` rope table no longer bakes a `< 2048` bounds assert at longer max_model_len. Without this, graphs at 64k OOMed with device-side asserts even in plain FLASHINFER (fork bug, reproducible).
- Skylight backend graph-safety fixes: `paged_kv_indices` is a plain Tensor (was incorrectly `.gpu`-accessed); capture guards on `_cache_sparse_pattern` / `_patch_indices_for_staging` / `_sk_fill_slots` via `torch.cuda.is_current_stream_capturing()`; frozen-pattern guard (batch mismatch → skip fetch) in `_patch_indices_for_staging`.
- scheduler_connector gate: `true_free` now also disengages when cudagraphs are active (`_graphs_active`), because in-run fetch can't execute under replay; shadow tier keeps pages valid instead of risking stale-pattern mismatches.

## 64k numbers (max_model_len=66000, ~119.6k tokens in, 256 out, TPOT=ms/token)

| config | eager TPOT | graph TPOT |
|---|---|---|
| oracle (no offload) | 29.6 | 8.2 (graphs ×3.6) |
| pq (no offload) | 124.3 | 103.7 (graphs ×1.2) |
| oracle+offload | 55.0 | 44.6 (graphs ×1.2) |
| pq+offload | 160.6 | 128.7 (graphs ×1.24) |

PQ numbers above include the rebuilt CUDA ADC path plus the restored integration edits; the earlier slot_codes `init_layer` crash (14952 vs 59802) was caused by the branch checkout erasing my combined-`[P,H,N,2C]` `_norm_K` fix — re-applied.

## Outstanding
- Promotion frequency is high under graphs (no fetch trouble; it's pre-step host work): 1000+ promotions/run at 64k suggests the offloaded set is genuinely re-needed each step — evaluation point for P4/selection-aware offload.
- Deterministic perf A/B vs saved baselines lives in `/tmp/opencode/bench64k_*.log`.
- Suites all green after graph changes.


## Task: skylight_kernels AOT rebuild against root-venv torch (user-approved target)

**Diagnosis (measured):** the user's 16:12–16:19 rebuild produced `.so`s that import-fail in the benchmark venv with `undefined symbol: _ZN3c104cuda19CUDAErrorLogCaptureC1Ev` (`c10::cuda::CUDAErrorLogCapture::CUDAErrorLogCapture()` and `c10_cuda_check_implementation(...)`). `nm -D` on root venv's `libc10_cuda.so` shows 0 such symbols, and the header `torch/include/c10/cuda/CUDAErrorLogCapture.h` is absent in *both* venvs — i.e., the fresh build was compiled against a THIRD torch install (neither root `.venv` (2.13.0+cu130) nor `skylight/.venv` (2.11.0+cu130)). So the .so's are ABI-incompatible with the runtime they must serve.

**What already works, untouched:** the PQ ADC JIT kernel (`pq_adc_score_kernel.cu`), last cached at `~/.cache/torch_extensions/py312_cu130/pq_adc_ops/pq_adc_ops.so`, built by root venv's nvcc+torch, bit-exact vs torch reference.

**Execution steps:**
1. Rebuild: `cd skylight/skylight_kernels && CUDA_HOME=/home/gaurav_kumar/KV_offload/.venv/lib/python3.12/site-packages/nvidia/cu13 PATH="$CUDA_HOME/bin:$PATH" /home/gaurav_kumar/KV_offload/.venv/bin/python -m pip install -e . --force-reinstall --no-deps --no-build-isolation`. Clean `build/` first if stale objects survive (`rm -rf skylight/skylight_kernels/build` is contained and regenerable).
2. Verify: `nm -D` each rebuilt `.so` — no `CUDAErrorLogCapture` undefined refs; import every wrapper module under root venv (oracle/oracle-sink-local, sparse_optimized, original, pq_optimized, pqrs_optimized, pqis_optimized, block_minmax*). If `original_optimized` still carries the symbol, its csrc is using an API whose caller choose by TORCH_VERSION check — adjust minimally in the kernel source, do NOT edit vllm fork.
3. Re-extend the editable finder mapping if the reinstall regenerates it and drops the late-add packages (pq*, pqrs*, block_minmax*). (Same fix applied twice already; idempotent script exists in history.)
4. Re-run pair: (a) `pytest offload_package/tests -q` — suite green, (b) `bench_64k.py` sweep oracle / pq / oracle+offload / pq+offload, (c) compare TPOT to the pre-rebuild table (oracle 29.6ms / pq 124.3ms / oa 55.0ms / pa 160.6ms at 64k no-graph).
5. Then proceed to the blocked workstream: graph runs (oracle first) at 64k — fork fix for long-context rope override already landed earlier today; the NvCC-path piecewise failure is a fork bug separate from this.

**Rollback:** git checkout of `skylight/` subdir, or re-run of the rebuild once against skylight venv's torch if you later need those benches.


**Status:** B1 ✅ B2 ✅ B5 ✅ B4 ✅ B3 ✅ B14.1 ✅ B14.2 ✅ B14.3 ✅ B13 ✅ — current phase set complete.

## B14.2 — 0-lag in-run fetch (LANDED)
- `pq_optimized/.../batch_decode_with_paged_kv_cache_wrapper.py`: `_inrun_fetch_selected(sparse_idx, page_size, request_ids)` runs between `select` and `sparse_decode_run` — device-side per-row unique-block bitmap (single sync), host residency filter via new cheap `ResidencyTable.num_cpu_blocks` counter, `staging_pool.ensure_resident` for the delta, in-place `index_copy_` patch of the aliased persistent indices buffer. Setters: `set_offload_orchestrator` / `set_offload_request_ids`; env off-switch `SKYLIGHT_PQ_INRUN_FETCH=0`. Inherited by `pqrs_optimized` for free.
- `skylight/backend.py::build()`: hands orchestrator + request ids to the cached decode wrapper per step.
- **Live evidence (PQ + offload, sync mode):** `entered=1386` probe calls (all 64 steps × 22 layers), `orch_none=0` (wiring complete). With the lagged build-time fetch stubbed out, the in-run path alone fetched exactly 20 blocks = the 13+7 that had been true-freed — the standalone correctness proof.

## B14.3 — recall measurement (LANDED)
- `test_pq_recall.py` at repo root: drives `_attn_err` (`SKYLIGHT_PQ_ERR_DBG=1`) per variant → prints attention-mass / recall / rel_l2 / cos.
- **Recorded numbers (TinyLlama, same budget):** PQ mass=0.9240 recall=0.9072; PQRS(heavy_frac=0.5) mass=0.8715 recall=0.7992. Default stays `pq`; pqrs is env-selectable.

## B13 — hygiene (LANDED)
- New `offload_package/debug.py` (`dbg`, gated by `OFFLOAD_PACKAGE_DEBUG=1`); ~110 hot-path `print`s converted across scheduler_connector, worker_connector, integration/manager, offload/manager, paged_eviction, shared/types, vllm_bridge, legacy adapters, `skylight/backend.py`, fork `block_pool.py` (local gates in the two cross-package sites). Default logs are now quiet (verified: 0 debug lines on a full run).
- Dead code removed: `staging/pool.py` duplicate `OffloadPackageWorker` class (lines 166–572), `registration.py`, `sparse_registration.py`.
- Stale suites repaired to current APIs/layout: `test_offload_manager.py` (scored_blocks param + combined layout), `test_integration.py` (combined KVCacheView + current on_prefill_complete contract), `test_global_budget_eviction.py` (4 fixture sites converted, shared inline-double use), `test_scoring.py` (+inf-for-unknown = never-evict contract documented), `test_offload_invariants.py` (2 more skip→live).
- **Suite: 89 passed / 7 skipped(→Phase C) / 0 failed. skylight/tests: 29 passed.**

## Phase B15 — Oracle + true freeing accuracy (R2) — LANDED & measured

**Config:** `OffloadConfig.true_free` (intent knob; runtime gate still guards async/MP/world>1) + `promote_per_step` (default 4). Connector reads both via `kv_connector_extra_config` with env fallback.

**Mechanism landed:**
- `OffloadOrchestrator.promote_hot_cpu_blocks(request_id, max_count)`: ranks the request's CPU-resident refs by current scorer scores (highest=most important), `staging_pool.ensure_resident(top_P)` → they become STAGING → B4's patch makes their real K visible to the oracle scorer before the score pass.
- Scheduler calls it per decode step per active request in `build_connector_meta` (guarded by `num_cpu_blocks > 0`).
- Explicitly NOT ported `_inrun_fetch_selected` into the oracle wrapper — vacuous there by construction (cold pages are invisible to the scorer until staged).
- The oracle wrapper already carried the `_attn_err` harness (`SKYLIGHT_PQ_ERR_DBG=1`); confirmed live.

**Live evidence:**
- Promotion fires at schedule time: `[ORCH PROMOTE] req=...: promoted 4/13 cold blocks (cpu_blocks=9)`; frees still occur (`True-freed 13/7`).
- Recall table (same budget, TinyLlama; harness `mass` = true softmax mass covered, recall = exact top-k overlap):
  - `oracle_shadow` (copies, no free): mass 0.361–0.393, recall ~0.32–0.35
  - `oracle_free` (true free, no promotion): mass 0.3646 — statistically identical to shadow
  - `oracle_free_promo` (R2): mass 0.3644 — no regression vs shadow; promotion preserves selection integrity
  - `pq`: mass 0.924 / recall 0.907; `pqrs`: mass 0.871 / recall 0.799 (from B14.3)
  - Read: at these settings, the oracle topk's own attention-mass coverage (~0.36) is the binding constraint; freeing+promotion adds ~0 error on top of it. PQ selection covers ≫ more mass and composes with freeing+retrieval cleanly.
- Suites: 94 passed / 7 skipped / 0 failed (5 new promotion tests: ordering, no-op-when-nothing-cold, max_count bound, zero-disables, content parity through a real offload+promote).

## Phase C — remaining (scheduling-level, after user review)
- Deterministic e2e output-equality guard, preempt-cleanup tests, async-mode freeing via deferred-free queue.

**Hard fact driving the design:** oracle's selection pass reads K pages via paged indices. A freed page is null-filled there → scores ~0 → can never be re-selected → in-run fetch can never fire for it. So for oracle, retrieval must be staged *ahead of* the score pass, driven by the query-agnostic offload scorer. Sync fetch still helps only for values on already-visible pages.

**Design:**
- `OffloadOrchestrator.promote_hot_cpu_blocks(request_id, max_count)`: collect the request's CPU-resident refs, rank by current scorer scores (highest = most important; the same PagedEvictionScorer used for offload decisions), call `staging_pool.ensure_resident(top_P)` → they become STAGING-resident → B4's per-step patch maps them into the indices → oracle scorer sees their *real* K content → correct selection pressure restored.
- Scheduler hook: in `OffloadPackageScheduler.build_connector_meta`, per decode step for each active request, call promote with `max_count = OFFLOAD_PACKAGE_PROMOTE_PER_STEP` (default 4). Cheap short-circuit on `residency.num_cpu_blocks == 0` (the counter added in B14.2). Staging churn is self-limiting (LRU + score-aware eviction to CPU).
- Because promotion runs at schedule time (before that step's forward), even the first decode step after a prefill-complete offload sees its highest-priority cold blocks. The residual accuracy risk is bounded: a page that spikes in true importance but stays under-promoted misses at most one step (its residency flips next step if it keeps mattering, since re-scoring every W updates priorities).
- Deliberately NOT porting `_inrun_fetch_selected` into the oracle wrapper (it is vacuous there); leaving an explicit comment in the oracle wrapper so a future reader doesn't add it expecting cold saves.

**_attn_err port to oracle:**
- Reuse `pq_optimized/attn_error.py` as the shared harness (import from the oracle wrapper's run(), same env gates `SKYLIGHT_PQ_ERR_DBG=1`, `SKYLIGHT_PQ_ERR_EVERY`); measure() recomputes exact dense attention from the same pages/selection and reports attention-mass/recall/rel_l2/cos.
- Extend `test_pq_recall.py` table to: `oracle_shadow` (offload copies, no free → exact-selection upper bound under offload), `oracle_truefree_promo` (R2), keeping `pq` / `pqrs` rows. This gives the accuracy story: oracle+free+promotion vs the exact shadow bound vs PQ.

**Sizing/knobs:** `OFFLOAD_PACKAGE_PROMOTE_PER_STEP` default 4 (tunable); promotion only for active decode rows; never promotes blocks still in-flight for a pending store event (store copies are sync, so residency CPU implies data is on CPU; safe).

**Tests:** unit — promotion picks highest-scored CPU refs and marks them STAGING; promotion respects `num_cpu_blocks==0` fast path; staging slot bound honored under contention. Live — free deltas still occur with promotion on; recall numbers collected across the four-method table.

## Kernel rebuild (user-fixed toolchain) — LANDED
- Diagnosis: earlier `nvcc not found` was a wrong `CUDA_HOME` pointing at system site-packages (no compiler); the user's rebuild produced `.so`s with undefined symbol `c10::cuda::CUDAErrorLogCapture::CUDAErrorLogCapture()` — built against *another* torch (skylight/.venv's 2.11), incompatible with the benchmark venv's torch 2.13.0+cu130.
- Fix: reinstall editable with root venv python + `CUDA_HOME=<root venv>/lib/python3.12/site-packages/nvidia/cu13` (nvcc 13.0.88 = torch's CUDA). Verified every needed package imports in root venv; finder mapping re-extended (regenerates on reinstall).
- The PQ ADC CUDA kernel (`pq_adc_score_kernel.cu`) JIT-compiles on first call now; bit-exact vs the torch fallback (max abs diff 0.0); kernel engages at runtime.

## Final measurement state (after rebuild, GPU2, torch 2.13+cu130)
| config | total | TPOT | peak GPU | offload evidence |
|---|---|---|---|---|
| oracle | 15.1s | 29.6ms | 120.41GB | — |
| pq | 63.6s | 124.3ms | 132.37GB | — |
| oracle+offload | 28.2s | 55.0ms | 120.41GB | 4 true-freed events |
| pq+offload | 82.2s | 160.6ms | 132.37GB | 4 true-freed, 510 promotions |

Profiler story: PQ vs oracle self-CUDA equal (~140ms on shared small scale); PQ's residual gap is host-side per-layer orchestration (bmm LUT build, extra copy_/index/einsum per step). PQ at 64k adds the ADC kernel scan (~0.19ms×22 layers/step). Good enough now; deeper gap closure = P4/cudagraph scope.


**Edits:**
- `selection/adapter.py::collect_needed_block_refs` — all-tensor: 1 host sync per call (was ~64), device `unique` per row, single host transfer per row.
- `selection/adapter.py::patch_paged_kv_indices_with_staging` loop-2 — gathered `(positions, values)` → one `index_put_`; per-row validation retained.
- `staging/pool.py::ensure_resident` — phase-1 batch slot allocation with `exclude` protection for in-batch slots (prevents duplicate-slot in a batched copy); phase-2 per-layer gather → permute+cat → single H2D `index_copy_`; ONE stream sync per batch.
- `offload/manager.py::score_blocks` — batched norm scoring (1 index_select×layers, 1 host sync). Wired into scheduler rescore loop, `sync_request_blocks`, `on_prefill_complete`.
- Cycle fixes forced by order changes: `staging/pool.py` shed its now-dead heavy vllm imports; `offload_package/integration/__init__.py` and fork `vllm/.../v1/__init__.py` moved to lazy `__getattr__` exports (true import cycle eliminated structurally).

**Micro-bench (synthetic 64k shapes, `bench_p1_p3.py`):** collect 389→28ms (13.8×), patch 84→9.7ms (8.6×), fetch×22layers 36.4→0.1ms (423×), score 2000 blk 325→0.4ms (835×).

**Live 64k bench (119,604 prompt tokens, 2 prompts × 256 new tokens, sync mode, GPU2 (free card); TPOT = total/decode-tokens):**
| config | total | TPOT | peak GPU | frees |
|---|---|---|---|---|
| oracle | 26.9s | 52.6ms | 120.41GB | — |
| pq | 68.6s | 133.9ms | 132.37GB | — |
| oracle+offload | 37.7s | 73.6ms | 120.41GB | 4 events (true-freed) |
| pq+offload | 109.7s | 214.3ms | 132.37GB | 4 events, 510 promotions |

**Pre-fix reference:** oracle+offload was 225.1s (439.7ms/tok) — now **6× faster**; pq+offload previously could not finish (~9s/step, killed at 1800s ≈ 167/256 steps) — now 214ms/step. **~37× step-time improvement for the PQ path.**
PQ's residual +110ms/token vs oracle is the torch-fallback ADC scorer (CUDA kernel unbuildable on this host: nvcc 12.0 vs torch cu13); P4 would let promotion/adaptive staging size absorb it.
Suites: 94 passed / 7 skipped / 0 failed after all edits; cycle fixes included.


**Root-cause evidence (64k measured):** pq+offload ran ~9 s/step (never finished a 256-token decode inside 1800s); oracle+offload held 0.44 s/step. Deltas isolate the Python hot paths:
1. `collect_needed_block_refs`: B×Hq per-head loop with `.item()` reads of `sparse_len` (a GPU sync each) + `torch.unique().tolist()` per head — thousands of host ops + ~64 syncs/step.
2. `patch_paged_kv_indices_with_staging` remediation loop: ~7.5k `BlockRef` ctor + `dict.get` per step + per-request `sorted()` in the validation check.
3. `ensure_resident`: per block × 22 layers: CPU `permute().contiguous()` + `cat` + one tiny H2D launch each — fetches scale × nblocks × layers.
4. Scorer: re-scores ALL resident blocks per W-window with per-block × per-layer `.cpu().tolist()` syncs — ~82k syncs per trigger at 64k.
5. The 64k *thrash* (fetch set > staging capacity) is P4 territory — deferred with user sign-off; P1 keeps it bounded but not eliminated.

**P1 edits (pure constant-factor, no semantic change):**
- `selection/adapter.py::collect_needed_block_refs` — all-tensor: `blocks = sparse_idx // page_size`; per-row validity from ONE `sparse_len[:,0,:].cpu()` (fabricated or provided); per-row dedup via `torch.unique` on device; single `(row, block)` pair tensor → one `.cpu().tolist()` → BlockRef construction. Same dedup union semantics; out-of-range refs still filtered downstream by residency.
- `selection/adapter.py::patch_paged_kv_indices_with_staging` — replacement of the per-position remediation loop with a gathered tensor scatter: build once per call a `(positions, new_ids)` pair list from residency/staging map (O(#patched), not O(#pages)), then one `index_put_`/fancy assign. Identical output values.
- `staging/pool.py::ensure_resident` — batch fetch: allocate/evict for the whole batch first (preserve score-aware eviction order), then per layer: `cpu_cache[layer, slot_ids]` gather → single permute+contiguous+cat → single `index_copy_` into `k_cache[base+slots]`. Same semantics, ~22 kernels/batch instead of 22×nblocks.
- Worker `_launch_cpu_to_staging_copy` (dormant path) uses the same batched helper.

**P3 edits (scorer batching):**
- `KVOffloadManager.score_blocks(refs)`: per-layer `index_select` over ALL blocks at once → norms → stack → mean over layers → ONE `.cpu().tolist()` → single `scorer.update`. Keep `score_block` for single callers.
- Recsite updates: `_trigger_global_offload` rescore loop, `sync_request_blocks` new-complete scoring, `on_prefill_complete` — collect then batch-score.

**Verification:**
- Existing suites must stay green (89+5 tests) — no semantic change allowed.
- New micro-bench: standalone script timing collect/patch/ensure_resident/score_blocks at 64k-shaped tensors (n=3.7k pages/req, 2 reqs) before vs after — target: fetch+patch python host time < 50ms/step.
- Live rerun of bench_64k configs with the script's log evidence; compare TPOT vs recorded baselines.

## Phase C — remaining (scheduling-level, after user review)
- Deterministic e2e output-equality guard, preempt-cleanup tests, async-mode freeing via deferred-free queue.
- P4 (selection-aware offload targeting) — deferred per user.




## Environment note (workspace fix, recorded)
Root `.venv`'s editable finder for `skylight_kernels` was extended to cover packages added post-install (`pq*`, `pqrs*`, `block_minmax*`, ...). CUDA rebuild impossible on this host (nvcc 12.0 vs torch cu13); prebuilt `.so` artifacts are in-tree, so map-extension is equivalent. `pq_optimized` ADC scorer falls back to its torch path until the CUDA kernel is built elsewhere. `skylight/.venv` was left untouched.

## B14.1 — PQ cold-page retrieval (LANDED)
- `skylight/backend.py`: registered `pq`/`pqrs` in `_import_sparse_wrapper_cls`; new `_sk_pq` mode constructing `SlotCodeMaintainer` (n_slots=max_num_seqs, l_max=max_model_len, M=SKYLIGHT_PQ_SUBSPACES=16) with `_ch` pinned to the resolved channel_num BEFORE codebook sizing; `build()` dispatches `_sk_fill_slots` for `pq|pqrs` as well (hook is maintainer-agnostic by design).
- `pq_optimized/slot_codes.py::_norm_K`: now handles this fork's combined 4-D `[P, H, N, 2C]` cache → K-half slice + page-major permute; separate `[P, N, H, D]` passes through.
- `pq_optimized/.../batch_decode_with_paged_kv_cache_wrapper.py::run()`: exposes `self._last_topk_idx = sparse_idx` → feeds `_cache_sparse_pattern` → build-time staging fetch (B4 path) now sees offloaded blocks as selectable.
- **Live evidence (TinyLlama, 2 distinct prompts, 96 tok, offload_fraction=0.5, sync mode):** codebooks FITTED on all 22 layers (386 prompt keys each); `True-freed 28/12/6` GPU blocks; `prepare_staging` staged exactly those 28/12/6 blocks back from CPU on the following steps — offloaded pages selected again by ADC scores and fetched to staging. The "never comes back" failure mode is closed.
- Oracle path unaffected (shared-prefix prompts correctly yield only ref_cnt==1 frees).

## Suite state after B14.1
- Invariants + core + residency + scoring: **37 passed / 2 failed (pre-existing stale `test_scoring.py` reset/forget expectations) / 7 skipped**.

## Remaining
- B14.2 — 0-lag in-run sync fetch between `select` and `sparse_decode` (approved design: D2H candidate probe + H2D fetch of delta + in-place index patch; cost only on cache miss).
- B14.3 — attention-mass recall measurement (`_attn_err` harness) to pick pq default vs pqrs split mode; tune M / SKYLIGHT_SPARSE_DENSE_UNTIL.
- B13 hygiene: debug prints behind env flag; delete dead code (`staging/pool.py` duplicate worker, dead adapters/backends/registration); repair stale test fixtures (test_integration combined-copy path, test_offload_manager, test_global_budget_eviction, test_scoring reset/forget semantics).
- Phase C: deterministic-harness e2e equality guard; preemption cleanup tests.

## B4 — Staging reaches kernel (LANDED)
- `worker_connector.py`: hardened orchestrator injection (iterates `runner.attn_groups` → `get_metadata_builder()`, sets class attr, post-init assert when skylight importable); `start_load_kv` early-returns when builder armed (kills the double-copy).
- `skylight/backend.py`: moved patch from post-plan (`attn_metadata` field, never consumed) to **pre-plan** via `_compute_flashinfer_kv_metadata` override — patches the persistent buffer content that `fast_plan_decode` passes to `plan()`. Includes alignment guard (row page count must equal tracked block count per request, else skip patching) and `null_id` fill gated on `orchestrator.true_freeing`.
- `selection/adapter.py::patch_paged_kv_indices_with_staging`: sparse_idx optional; added residency remediation loop (STAGING→staging slot; CPU→null when freeing); safe skips on batch-composition drift instead of IndexError.
- **Invariant tests:** `test_patch_applied_in_decode_metadata` (CPU-only: GPU stays / CPU→null / STAGING→slot / shadow-mode no-fill) and `test_offload_fetch_roundtrip_parity` (GPU: offload→fetch roundtrip content equal across layers) both green.

## B3 — True freeing + table surgery (LANDED)
- `_trigger_global_offload`: ref_cnt==1 filter (prefix-shared blocks never offloaded when freeing on); `touch()` deleted (it was canceling the free); refs carried through `_step_store_acc` → `_pending_store_events`.
- `_process_store_completion`: per-ref surgery — `mgr.req_to_blocks[req_id][block_idx] = mgr._null_block` (mirrors fork's own `_remove_blocks_in_range` pattern), mismatch-safe (skip+log if table entry doesn't match, e.g. finished/preempted requests), prefix-hash evict before free, then `free_blocks` (ref 1→0 → real reclaim). `_req_block_table` mirror updated so sync never resurrects freed ids.
- `manager.py::sync_request_blocks`: CPU/STAGING refs never re-marked GPU (old comment-code claimed otherwise implicitly; stale runner ids bounce off).
- `build_connector_meta` publishes `true_freeing` onto orchestrator (drives null-fill).
- Async/MP gate: `async_scheduling` or multiprocess or world_size>1 → shadow-tier mode + log (freeing unsafe there without deferred-free plumbing).
- Fork wiring: `scheduler.py:311` passes `kv_cache_manager` via inspect.signature gating (other connectors unaffected); `OffloadPackageConnector.bind_gpu_block_pool(pool, kv_cache_manager=None)` delegates.

## B3/B4 live evidence
- Sync run, 4 distinct prompts (no prefix sharing): `True-freed 79 GPU blocks` → `11157`, then `24 → 11173`, `9 → 11174` (monotonic modulo decode allocations; surgery zero mismatches).
- Identical-prompts run: only ref_cnt==1 blocks freed (2–3/event) — shared blocks correctly protected.
- Baseline-vs-baseline in-process runs diverge (steps 1/6): token-compare harness is nondeterministic across engine instances — documented; e2e equality test deferred to Phase C (needs nondeterminism control).
- Degenerate output text ("pueblo…") is pre-existing oracle-sparse behavior at topk=0.1 on TinyLlama — reproduced with the connector entirely disabled.

## Suite state
- `pytest offload_package/tests -q`: **77 passed / 12 failed (pre-existing stale fixtures) / 7 skipped(→Phase C)**.
- Invariants: 5 green / 7 skipped.

## Phase B14 — Cold-page retrieval via GPU-resident PQ codes (DESIGN — from user direction; branch studied)

**Goal:** offloaded blocks must remain *score-selectable* so they can be fetched back when they become relevant. (Null-remap alone ⇒ cold pages score ~0 ⇒ never selected ⇒ permanent eviction.)

**Mechanism (pq_optimized / pqrs_optimized in skylight_kernels, PQRS branch):**
- `SlotCodeMaintainer`: persistent GPU buffer `code_buf[n_slots+1, H_kv, L_max, M]` (uint8 codes, ~1/32 of KV bytes). Slot = request's batch position (production path). `init_layer` (eager, prefill) fits+freezes per-layer codebook (k-means, 256 centroids, tiled if short) and encodes prompt keys; `append` folds each new token per decode step (capturable; reads the always-resident tail page via paged indices ✓).
- run(): `build_luts(q, centroids)` → `adc_scores(code_buf, lut, slots, lengths)` → `select(...)` → `sparse_idx/sparse_weights` → `sparse_decode_run(...)` reads pages only for selected tokens. **Selection never reads KV pages** ⟹ offloaded pages stay scorable.

**Integration plan:**
- **B14.1 — wiring (fetches at build-time, 1-step lag; kills the liveness death):**
  - Register `pq`/`pqrs` in `skylight/backend.py::_import_sparse_wrapper_cls`.
  - Drive `SlotCodeMaintainer` from `SkylightSparseMetadataBuilder` mirroring the existing `_sk_fullcg`/`_sk_fill_slots` block_minmax hook (`_slot_maint`, `_slot_buf`, `_slot_slots`, `_slot_seqlens`, `pending` list, owner-change detect).
  - Wrapper exposes selected pattern (`self._last_topk_idx = sparse_idx`) so the existing build-time staging fetch (`B4._patch_indices_for_staging` → `prepare_staging`) stages selected-offloaded pages for the NEXT step. Residual gap: a block entering top-k at step t reads null/zeros at step t (one-step churn noise — bounded by selection locality).
- **B14.2 — 0-lag correctness (in-run fetch):**
  - In pq run() between `select` and `sparse_decode`: derive unique page candidates `(tokens // page_size)`; host-map against `orchestrator.residency` (small D2H of candidate ints); `staging_pool.ensure_resident` for the delta (usually empty); in-place patch `self._kv_indices_int32` (aliases persistent buffer in eager ✓ via B4 mechanics). Then decode reads staged K/V with this step's own selection — fully consistent.
  - Wrapper needs an orchestrator handle: add `SkylightSparseMetadataBuilder` → wrapper handoff (like `_make_sparse_decode_wrapper` already does via `set_orchestrator` duck-type).
- **B14.3 — measurement:** use built-in `_attn_err` instrumentation (attention-mass recall vs exact) to pick pq vs pqrs and set M / SKYLIGHT_SPARSE_DENSE_UNTIL defaults.

**Locked decisions (user):** pq first, pqrs-ready (env-switchable, no extra wiring work until recall measured); in-run sync fetch APPROVED (0-lag correctness, cost only on cache miss); size **tests only** ≤4k ctx / ≤8 reqs for B14.1; production sizing (<32k ctx) deferred with assert-on-undersize.
**Correctness invariants preserved:** tail page never offloaded (keeps `append` valid); codebook fit deferred until ≥C finite keys (SKYLIGHT_SPARSE_DENSE_UNTIL path); surgery/free (B3) untouched — codes outlive pages by design.



## LANDED — verification results

### B1 — Layout normalization (Bug 3)
- `shared/types.py`: deleted `% 2 == 0` alternating-guess; per-layer combined `[B, H, N, 2C]` tensors now shared as `k_caches[i] is v_caches[i]` with strict shape validation; `kv_layout="HND"`; raises on unknown formats.
- `vllm_adapter.py`: removed shadowed `_get_attention_spec`/`_split_kv_cache_with_spec` duplicates and dead `normalize_runner_kv_caches`.
- **Live evidence:** CPU cache shape now `(22, 2048, 2, 16, 4, 64)` — all 22 TinyLlama layers (was 11). 
- **Test:** `test_normalize_pairs_one_tensor_per_layer` ✅ green.
- Bonus: fixed stale imports in `test_integration.py` / `test_vllm_adapter.py` → they collect again.

### B2 — Scoring dims (Bug 4)
- `offload/manager.py::score_block`: detects combined pages (`k_cache is v_cache`), splits `[..., :head_dim]`/`[..., head_dim:]`, computes with `layout="HND"` → per-head mean-over-token `[B, H]` norms.
- `norm_kernel.py`: added assert tying norm last-dim to the heads axis (guards axis drift).
- Corrected the invariant fixture (it paired `[B,H,N,D]` with the wrong layout flag and wrong reference math).
- **Test:** `test_scorer_uses_true_kv_norms` ✅ green.

### B5 — Staging tail isolation (Bugs 2/8) — APPROACH CHANGED (simpler)
- Dropped the tensor-extension hook (its target symbol `attn_utils._allocate_kv_cache` no longer exists in the fork → was broken/dead).
- New: `cache_allocator.reserve_staging_tail(block_pool, num_slots)` — `touch()`es the last S block ids once at startup → blocks pinned at ref_cnt=1, removed from the free queue, permanently unallocatable, yet still physically addressable for staging writes.
- Wired in `OffloadPackageScheduler.bind_gpu_block_pool` (called by fork `scheduler.py:311` for BOTH runners — zero fork edits needed).
- **Live evidence:** `pool_blocks=11039, free_before=11038 → free_after=10974` = exactly 64 reserved; after the full run the pool settles at 10974 (reserved stay reserved).
- **Test:** `test_staging_tail_outside_allocator` ✅ green (reserve 8/100 → only 91 allocatable, reserved ids never appear).

### Suite state
- `pytest offload_package/tests -q`: **75 passed / 12 failed / 9 skipped** — failures are all pre-existing stale suites (`test_integration` combined-copy-path fixtures, `test_offload_manager`, `test_global_budget_eviction` scored-filter drift, `test_scoring` reset/forget semantics); none caused by B1/B2/B5. Tracked for B13 repair.
- Invariants: 3 passed / 9 skipped (skips flip per phase).

## Not yet done: B4 (staging→kernel) then B3 (true freeing), then B13 hygiene, B14 cold-page policy.

---

## Phase B1 — Layout normalization (Bug 3) — LANDED *see above*

## Phase B2 — Scoring dims (Bug 4) — LANDED *see above*

## Phase B5 — Staging tail isolation (Bugs 2/8) — LANDED *see above*


**Scope (confirmed with user):** uni-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), TP=1, sync scheduling, V1+V2 runners. Async/MP modes get a loud gate disabling true freeing.
**Open design item (deferred by user):** cold-page retrieval after true free (null-remap makes cold pages un-scoreable → never staged back). Options: K-only restore into scoring scratch, GPU-resident per-block summaries, offload-scorer-driven stage-back. Scheduled after core fixes land (Phase B14).

Evidence collected (Phase A, all CONFIRMED live):
- Bug 1: `test_block_free_check.py` → free delta 0, ref_cnt stays 1 (touch+free cancel).
- Bug 4/3: `[BUILD DEBUG] CPU cache shape (11, ...)` for 22-layer TinyLlama (alternate-pairing); invariant test red.
- Bug 5: `[SKYLIGHT DEBUG] build: orchestrator=False` — injection never fires.
- Bug 2: `GPU k_cache shape[0]=11039=num_blocks` — no staged tail visible on V1 path.

---

## Phase B1 — Layout normalization (Bug 3)

Files: `offload_package/shared/types.py`, `offload_package/vllm_adapter.py`

`normalize_runner_kv_caches` rewrite:
1. DELETE the `len() % 2 == 0` "flat alternating K/V" guess branch (the actual mis-pairing source).
2. Combined-format branch: each entry a 4-D tensor with `shape[1]==spec.num_kv_heads`, `shape[2]==spec.block_size`, `shape[-1]==2*spec.head_size` → this fork's `[num_blocks, H, N, 2C]` per-layer page; set `k_caches[i] = v_caches[i] = cache` (same object; `_copy_gpu_to_cpu`, `ensure_resident`, `_launch_cpu_to_staging_copy` already split `[..., :C]`/`[..., C:]` internally — NO changes needed there).
3. Return `KVCacheView(..., kv_layout="HND")` — fork's `[B, H, N, C]` order (was hardcoded "NHD").
4. Keep `[(K,V), ...]` tuple-pairs branch; raise on anything else (no silent guessing).
5. `vllm_adapter.py`: delete duplicated/shadowed `_get_attention_spec` ×1 and `_split_kv_cache_with_spec` ×2 (second def rejects 4D and would crash), and the dead `normalize_runner_kv_caches` there (imports nonexistent names from shared.types).

Gate: `test_normalize_pairs_one_tensor_per_layer` green; startup prints 22-layer CPU cache.

## Phase B2 — Scoring dims (Bug 4)

Files: `offload_package/offload/manager.py`, `offload_package/scoring/norm_kernel.py`, `offload_package/tests/test_offload_invariants.py`

1. `KVOffloadManager.score_block`: when `k_cache is v_cache` (combined), split `k_in = cache[..., :head_dim]`, `v_in = cache[..., head_dim:]` and call `compute_block_kv_norms(..., layout="HND")`. HND branch `norm(dim=-1).mean(dim=2)` → `[B, H]` per-head mean-over-tokens (correct for `[B, H, N, C]` splits).
2. `norm_kernel.py`: add assert that returned last dim equals the heads axis for the given layout (guards axis drift).
3. Fix invariant fixture: my current test built `[B, H, N, D]` but asserted against NHD branch — repoint fixture at HND path and correct expected math (`||ones(64)||=8`, ratio V/K=2 with v=2·k).

Gate: `test_scorer_uses_true_kv_norms` green; live scores vary per block.

## Phase B5 — Staging tail isolation (Bugs 2/8) — APPROACH CHANGED (simpler, verified against fork)

Old plan (wrap `vllm.../attn_utils._allocate_kv_cache`) is invalid: that symbol no longer exists in the fork (grep confirms only `allocate_kv_cache`/`bind_kv_cache` in `worker/utils.py`), so the current V2-only hook is dead/broken, and V1 never had it.

New approach — reserve tail pages at the allocator, no tensor surgery:
1. Rewrite `offload_package/cache_allocator.py`:
   - `reserve_staging_tail(block_pool, num_slots) -> list[int]`: `touch()` the last `num_slots` blocks (ids `N-S .. N-1`) once at startup. `touch()` pops them from the free queue and sets ref_cnt=1 → BlockPool never hands them to a request. They stay physically addressable (views cover all N pages) for staging writes at `staging_base = N - S`.
   - Keep a deprecated no-op `install()` so the existing fork call at `vllm-0.29.0/vllm/v1/worker/gpu/model_runner.py:550` doesn't break.
2. Wire: `OffloadPackageScheduler.bind_gpu_block_pool` → call `reserve_staging_tail(pool, self.offload_config.num_staging_slots)` when `offload_config.enabled` and slots > 0; log reserved range. `scheduler.py:311` already calls bind for our connector → covers BOTH runners, no fork edits.
3. Cross-check: worker-side `staging_base` (= `view.shape[0] - S` = `N - S`) equals reserved range start (`N - S`); add one-time assert in `OffloadPackageScheduler.bind_gpu_block_pool` comparing against `self.kv_cache_config.num_blocks - S`.
4. Note: `reset_prefix_cache` will warn (reserved blocks count as "used") — acceptable, documented.

Gate: `test_staging_tail_outside_allocator` (BlockPool-level: reserve 8 of 100 → only 91 allocatable, reserved ids exactly tail) green; e2e log shows staging pages `>=` last allocatable id.

---

## Phase B4 (next, after review) — Staging reaches kernel (Bugs 5/7)
1. Harden `worker_connector._build_orchestrator` injection onto `SkylightSparseMetadataBuilder._offload_orchestrator` + post-init assert.
2. `bind_connector_metadata` already writes `_cached_request_ids` (keep); also set builder class attrs in `_cache_sparse_pattern`.
3. Single fetch owner: `start_load_kv` skips fetch when builder path armed (kills double copy).
4. `_prepare_staging_for_sparse_attention`: patch staged→`staging_base+slot` AND resident-CPU-unstaged→null(0) in per-step indices copy; validate req-id↔row alignment.

## Phase B3 (after B4) — True freeing + table surgery (Bugs 1/6)
Predicated on confirmed scope: uni-proc, TP=1, sync.
1. `offload_fraction` predicate → only `ref_cnt==1` blocks offloadable (skip prefix-shared; kills duplicate copies).
2. Delete `touch()` (`scheduler_connector.py:367`) — sync copies already serialize; touch is the net-zero culprit.
3. Surgery at store completion: `mgr.req_to_blocks[req_id][block_idx] = pool.null_block`; `pool.evict_blocks({bid})` if hashed; mirror in `_req_block_table`; then `free_blocks` (ref 1→0 → real reclaim). Pass `kv_cache_manager` into `bind_gpu_block_pool` (fork `scheduler.py:311`).
4. Null-out cold CPU-resident unstaged positions in per-step indices (B4.4) so scorer/attention never read reallocated pages. Interim only; cold-page retrieval policy is Phase B14.
5. Async/MP gate → shadow-tier fallback + log-once.

## Phase B6–B12 — Robustness batch
- B6 ref_cnt==1 filter (folded in B3.1). B7 via B4.3. B8a fix `staging_pool.release(req_id)` type (pass BlockRefs or drop dead branch, `scheduler_connector.py:399`). B8b delete duplicated loop `worker_connector.py:516-536`. B9 `_step_store_acc` cleared after flush. B10 single CPU allocator (`offload_manager._cpu_pool`), delete shadow `cpu_coordinator`/`_derive_cpu_config` (kills crash-after-mutation). B11 `_copy_gpu_to_cpu` try/except→`_cpu_pool.free(slot)`. B12 preemption cleanup wired (`on_request_preempted` + `handle_preemptions`).

## Phase B13 — Hygiene
DEBUG prints behind `OFFLOAD_PACKAGE_DEBUG`; `_cache_sparse_pattern` once/step (not 22×); vectorize `collect_needed_block_refs`; delete dead code (pool.py duplicate worker class, unused adapters/backends/registration); repair stale test imports/fixtures.

## Phase B14 — Cold-page retrieval (discussed separately with user)
K-only restore / per-block GPU summaries / scorer-driven stage-back. Not started here.

## Phase C — Tests
Flip skips per phase; e2e greedy TinyLlama equality offload-on vs off + memory watermark. Final gate: `pytest offload_package/tests -q` green + `test_block_free_check.py` real deltas.

---

## Now executing (user approved): B1 + B2 + B5, then pause for review before B4/B3.
