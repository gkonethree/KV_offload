"""
Invariant tests for KV offload mechanism.

These tests assert externally observable properties that must hold regardless
of internal implementation. Each test is red on the current codebase and
will turn green only after the corresponding fix is applied.
"""

from __future__ import annotations

import pytest
import torch

from offload_package.integration.scheduler_connector import OffloadPackageScheduler
from offload_package.integration.worker_connector import OffloadPackageWorker
from offload_package.offload.manager import KVOffloadManager, CPUBlockPool
from offload_package.scoring.base import BlockRef, BlockKVStats
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable, BlockLocation
from offload_package.controller import KVCompressionController
from offload_package.shared.types import build_orchestrator, normalize_runner_kv_caches, OrchestratorConfig
from offload_package.vllm_adapter import KVCacheView
from offload_package.config import OffloadConfig


class TestTrueFreeAccounting:
    """BlockPool accounting for true GPU block freeing (Bug 1)."""

    def test_offloaded_block_returns_to_free_pool(self, monkeypatch):
        """Offloaded blocks must return to BlockPool free list (ref_cnt == 0)."""
        pytest.skip("Requires vLLM BlockPool integration - run after fix B2")

    def test_no_double_free_at_request_finish(self):
        """After offload+free, request finish must not double-free blocks."""
        pytest.skip("Requires scheduler + BlockPool integration - run after fix B2")

    def test_shared_prefix_block_never_offloaded(self):
        """Prefix-shared blocks (ref_cnt > 1) must be excluded from offload candidates."""
        pytest.skip("Requires orchestrator+scorer+BlockPool integration - run after fix B2")


class TestLayoutIntegrity:
    """Per-layer tensor layout correctness (Bug 4)."""

    def test_normalize_pairs_one_tensor_per_layer(self):
        """normalize_runner_kv_caches must map 1 input tensor -> 1 K/V pair per layer."""
        num_layers = 22
        head_dim = 64
        num_kv_heads = 4
        page_size = 16
        num_blocks = 100

        # Build synthetic combined tensors with unique sentinel per layer
        tensors = []
        for layer in range(num_layers):
            # Combined [num_blocks, H, N, 2*head_dim] with K in [:,:,:,:head_dim], V in [:,:,:,head_dim:]
            t = torch.zeros(num_blocks, num_kv_heads, page_size, 2 * head_dim, dtype=torch.float16)
            # Stamp layer ID in K channel 0, token 0, head 0
            t[:, 0, 0, 0] = float(layer)
            # Stamp layer ID + 100 in V channel 0, token 0, head 0  
            t[:, 0, 0, head_dim] = float(layer + 100)
            tensors.append(t)

        from offload_package.shared.types import normalize_runner_kv_caches

        # Minimal fake KVCacheConfig
        class FakeSpec:
            block_size = 16
            num_kv_heads = 4
            head_size = 64

        class FakeGroup:
            layer_names = [f"layer_{i}" for i in range(num_layers)]
            kv_cache_spec = FakeSpec()

        class FakeConfig:
            kv_cache_groups = [FakeGroup()]
            kv_cache_tensors = []

        view = normalize_runner_kv_caches(tensors, FakeConfig(), staging_slots=0)

        # Must produce exactly 22 layers, 22 K, 22 V
        assert len(view.layers) == num_layers, f"Expected {num_layers} layers, got {len(view.layers)}"
        assert len(view.k_caches) == num_layers, f"Expected {num_layers} k_caches, got {len(view.k_caches)}"
        assert len(view.v_caches) == num_layers, f"Expected {num_layers} v_caches, got {len(view.v_caches)}"

        # Combined layout: k_caches[i] and v_caches[i] are the same tensor, and
        # it must be layer i's tensor (K sentinel at [.., :head_dim], V at [.., head_dim:]).
        for i in range(num_layers):
            assert view.k_caches[i] is view.v_caches[i], f"layer {i}: expected combined tensor shared by k/v"
            k_layer_id = int(view.k_caches[i][0, 0, 0, 0].item())
            v_layer_id = int(view.v_caches[i][0, 0, 0, head_dim].item()) - 100
            assert k_layer_id == i, f"k_caches[{i}] has layer {k_layer_id}, expected {i}"
            assert v_layer_id == i, f"v_caches[{i}] has layer {v_layer_id}, expected {i}"

        # Layout must be HND (fork's [B, H, N, C] page ordering)
        assert view.kv_layout == "HND", f"kv_layout={view.kv_layout}"

    def test_offload_fetch_roundtrip_parity(self):
        """Offload a block to CPU, fetch to staging; staging must equal original page."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.integration.manager import OffloadOrchestrator
        from offload_package.offload.manager import KVOffloadManager
        from offload_package.scoring.paged_eviction import PagedEvictionScorer
        from offload_package.staging.pool import KVStagingPool
        from offload_package.staging.residency import Residency, ResidencyTable
        from offload_package.vllm_adapter import KVCacheView

        num_layers, num_pages = 3, 16
        H, N, C = 2, 16, 64           # heads, page tokens, head_dim
        staging_slots = 4
        staging_base = num_pages - staging_slots
        head_dim = C

        # Combined per-layer tensors [pages, H, N, 2C] with distinct content per layer/page
        combined_layers = []
        for l in range(num_layers):
            t = torch.zeros(num_pages, H, N, 2 * C, dtype=torch.float16, device="cuda")
            for p in range(num_pages):
                t[p].fill_(float(l * 100 + p) / 100.0)
            combined_layers.append(t)
        k_caches = list(combined_layers)
        v_caches = list(combined_layers)
        view = KVCacheView(
            layers=combined_layers,
            k_caches=k_caches,
            v_caches=v_caches,
            block_size=N,
            num_kv_heads=H,
            head_dim=head_dim,
            kv_layout="HND",
            staging_base_page=staging_base,
        )

        cpu_cache = torch.empty(
            (num_layers, 8, 2, N, H, C), dtype=torch.float16,
            device="cpu", pin_memory=True,
        )
        residency = ResidencyTable()
        scorer = PagedEvictionScorer()
        offload_manager = KVOffloadManager(
            scorer=scorer, residency=residency,
            gpu_k_caches=k_caches, gpu_v_caches=v_caches,
            cpu_cache=cpu_cache, gpu_budget_blocks=staging_base, kv_layout="HND",
        )
        staging_pool = KVStagingPool(
            num_slots=staging_slots, staging_base_page=staging_base,
            gpu_k_caches=k_caches, gpu_v_caches=v_caches, cpu_cache=cpu_cache,
            residency=residency, offload_manager=offload_manager,
        )

        ref = BlockRef("req-x", 0)
        original_pages = [combined_layers[l][3].clone() for l in range(num_layers)]
        residency.mark_gpu(ref, 3, complete=True)
        offload_manager.score_block(ref, 3)  # candidates must be scored pre-offload

        offloaded = offload_manager.offload_fraction([ref], 0.99, {ref})
        assert offloaded == [ref], "block should have been offloaded"
        loc = residency.get(ref)
        assert loc.residency == Residency.CPU and loc.cpu_slot is not None

        staging_map, skipped = staging_pool.ensure_resident([ref])
        assert not skipped
        assert ref in staging_map
        loc = residency.get(ref)
        assert loc.residency == Residency.STAGING

        slot = staging_map[ref]
        for l in range(num_layers):
            staged = combined_layers[l][staging_base + slot]
            assert torch.equal(staged, original_pages[l]), (
                f"layer {l}: staged page content mismatch after offload+fetch"
            )

    def test_offload_roundtrip_multi_block_batched(self):
        """Batched store event: N blocks in one call; fetch-back content parity.

        Guards the batched store path (_copy_gpu_to_cpu_batch): distinct CPU
        slots, distinct staging slots, byte-exact content across all layers and
        blocks.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.offload.manager import KVOffloadManager
        from offload_package.scoring.paged_eviction import PagedEvictionScorer
        from offload_package.scoring.base import BlockKVStats
        from offload_package.staging.pool import KVStagingPool
        from offload_package.staging.residency import Residency, ResidencyTable

        num_layers, num_pages = 3, 64
        H, N, C = 2, 16, 64
        staging_slots = 16
        staging_base = num_pages - staging_slots
        n_blocks = 8

        combined_layers = []
        for l in range(num_layers):
            t = torch.randn(num_pages, H, N, 2 * C, dtype=torch.float16, device="cuda")
            combined_layers.append(t)

        cpu_cache = torch.empty(
            (num_layers, 64, 2, N, H, C), dtype=torch.float16, device="cpu", pin_memory=True
        )
        residency = ResidencyTable()
        scorer = PagedEvictionScorer()
        mgr = KVOffloadManager(
            scorer=scorer, residency=residency,
            gpu_k_caches=combined_layers, gpu_v_caches=combined_layers,
            cpu_cache=cpu_cache, gpu_budget_blocks=staging_base, kv_layout="HND",
        )
        pool = KVStagingPool(
            num_slots=staging_slots, staging_base_page=staging_base,
            gpu_k_caches=combined_layers, gpu_v_caches=combined_layers,
            cpu_cache=cpu_cache, residency=residency, offload_manager=mgr,
        )

        refs = [BlockRef("req0", i) for i in range(n_blocks)]
        stats = BlockKVStats(
            v_norm=torch.arange(1, n_blocks + 1, dtype=torch.float32).unsqueeze(1),
            k_norm=torch.ones(n_blocks, 1, dtype=torch.float32),
            block_ids=refs,
        )
        scorer.update(stats)
        for ref in refs:
            residency.mark_gpu(ref, gpu_block_id=ref.block_idx, complete=True)

        originals = {
            ref: [combined_layers[l][ref.block_idx].clone() for l in range(num_layers)]
            for ref in refs
        }

        offloaded = mgr.offload_fraction(refs, 0.99, set(refs))
        assert len(offloaded) == n_blocks
        slots = {residency.get(r).cpu_slot for r in offloaded}
        assert len(slots) == n_blocks, "CPU slots must be distinct per block"

        staging_map, skipped = pool.ensure_resident(offloaded)
        assert not skipped
        assert len(staging_map) == n_blocks
        stage_slots = set(staging_map.values())
        assert len(stage_slots) == n_blocks, "staging slots must be distinct per block"

        for ref, slot in staging_map.items():
            for l in range(num_layers):
                staged = combined_layers[l][staging_base + slot]
                assert torch.equal(staged, originals[ref][l]), (
                    f"{ref} layer {l}: content mismatch after batched store+fetch"
                )

    def test_scorer_uses_true_kv_norms(self):
        """compute_block_kv_norms must produce per-head mean-over-token norms.

        HND layout [B, H, N, D]: norm over head_dim (D), then mean over tokens
        (N), keeping one value per head per block.
        """
        from offload_package.scoring.norm_kernel import compute_block_kv_norms
        from offload_package.scoring.base import BlockRef

        num_blocks = 1
        num_kv_heads = 4
        page_size = 16
        head_dim = 64

        k_cache = torch.zeros(num_blocks, num_kv_heads, page_size, head_dim, dtype=torch.float32)
        v_cache = torch.zeros(num_blocks, num_kv_heads, page_size, head_dim, dtype=torch.float32)
        k_cache.fill_(1.0)
        v_cache.fill_(2.0)

        stats = compute_block_kv_norms(
            k_cache, v_cache, [0],
            logical_block_ids=[BlockRef("test", 0)],
            layout="HND",
        )

        # Per (block, head): ||K_token|| = sqrt(head_dim) = 8; mean over 16 tokens = 8.
        assert stats.k_norm.shape == (1, num_kv_heads), f"k_norm shape {stats.k_norm.shape}"
        assert stats.v_norm.shape == (1, num_kv_heads), f"v_norm shape {stats.v_norm.shape}"

        expected_k = head_dim ** 0.5          # 8.0
        expected_v = 2.0 * head_dim ** 0.5    # 16.0
        assert torch.allclose(stats.k_norm, torch.full_like(stats.k_norm, expected_k), atol=1e-3)
        assert torch.allclose(stats.v_norm, torch.full_like(stats.v_norm, expected_v), atol=1e-3)

        ratio = stats.v_norm / stats.k_norm
        assert torch.allclose(ratio, torch.full_like(ratio, 2.0), atol=1e-3)


class TestStagingReachesAttention:
    """Staged data must be visible to attention kernel (Bug 2)."""

    def test_patch_applied_in_decode_metadata(self):
        """After staging fetch, paged_kv_indices must point to staging_base + slot.

        CPU-only adapter-level test over patch_paged_kv_indices_with_staging:
        staged refs -> staging slots; CPU-resident unstaged refs -> null (when
        true freeing is on); GPU-resident refs -> untouched.
        """
        from offload_package.selection.adapter import patch_paged_kv_indices_with_staging
        from offload_package.staging.residency import ResidencyTable

        residency = ResidencyTable()
        staging_base = 1000
        # One request, 6 blocks (pages) originally at GPU ids 10..15
        gpu_ids = [10, 11, 12, 13, 14, 15]
        for idx, gid in enumerate(gpu_ids):
            residency.mark_gpu(BlockRef("r0", idx), gid, complete=True)

        # Block 2 offloaded (CPU, slot 7); block 4 offloaded then staged (slot 5).
        residency.mark_cpu(BlockRef("r0", 2), cpu_slot=7)
        residency.mark_cpu(BlockRef("r0", 4), cpu_slot=3)
        residency.mark_staging(BlockRef("r0", 4), staging_slot=5)

        indices = torch.tensor(gpu_ids, dtype=torch.int32)
        indptr = torch.tensor([0, 6], dtype=torch.int32)

        staged = patch_paged_kv_indices_with_staging(
            indices, indptr, ["r0"], sparse_idx=None, sparse_len=None,
            page_size=16, staging_base_page_idx=staging_base,
            residency=residency, staging_map={},
            null_block_id=0,
        )

        assert int(staged[0]) == 10 and int(staged[1]) == 11          # GPU stays
        assert int(staged[2]) == 0                                    # CPU -> null
        assert int(staged[3]) == 13                                   # GPU stays
        assert int(staged[4]) == staging_base + 5                     # staged slot
        assert int(staged[5]) == 15                                   # GPU stays

        # Without null fill (shadow mode), CPU-resident ids stay as-is.
        nofill = patch_paged_kv_indices_with_staging(
            indices, indptr, ["r0"], sparse_idx=None, sparse_len=None,
            page_size=16, staging_base_page_idx=staging_base,
            residency=residency, staging_map={},
            null_block_id=None,
        )
        assert int(nofill[2]) == 12                                   # CPU -> unchanged
        assert int(nofill[4]) == staging_base + 5                     # staged slot still applied

    def test_sparse_output_matches_dense_on_offloaded_blocks(self):
        """Oracle wrapper output on staged blocks must match dense reference."""
        pytest.skip("Requires kernel-level integration - run after fix B3")

    def test_staging_fetch_visible_before_attention(self):
        """Fetch copy event must complete before attention kernel launch."""
        pytest.skip("Requires CUDA event integration - run after fix B3")


class TestAllocatorIsolation:
    """Staging tail must not overlap allocator space (Bug 5)."""

    def test_staging_tail_outside_allocator(self):
        """Reserved tail pages must never be handed to a request by BlockPool."""
        from vllm.v1.core.block_pool import BlockPool
        from offload_package.cache_allocator import reserve_staging_tail

        pool = BlockPool(num_gpu_blocks=100, enable_caching=False, hash_block_size=16)
        # 100 blocks; null_block (id 0) is popped at init -> 99 allocatable.
        reserved = reserve_staging_tail(pool, 8)
        assert reserved == list(range(92, 100))
        assert pool.get_num_free_blocks() == 91

        # Drain the pool: the reserved tail must never appear in an allocation.
        got = pool.get_new_blocks(91)
        got_ids = {b.block_id for b in got}
        assert got_ids.isdisjoint(reserved)
        assert max(got_ids) < 92

        # One more allocation must fail (reserved blocks are truly unavailable).
        with pytest.raises(ValueError):
            pool.get_new_blocks(1)


class TestFusedScorer:
    """K2: fused scorer parity vs the per-layer torch path (CUDA and Triton)."""

    @staticmethod
    def _parity_asserts(score_blocks_fused):
        from offload_package.scoring.norm_kernel import compute_block_kv_norms

        num_layers, num_pages, H, N, C2 = 2, 32, 4, 16, 128   # 128 == 2*64
        torch.manual_seed(0)
        caches = [torch.randn(num_pages, H, N, C2, dtype=torch.float16, device="cuda")
                  for _ in range(num_layers)]
        ids = [1, 3, 5, 17]

        k_fused, v_fused = score_blocks_fused(caches, ids)

        # Reference: per-layer split + compute_block_kv_norms (this fork's
        # combined layout => split last dim into K/V halves, HND page order).
        all_k, all_v = [], []
        for c in caches:
            k_in, v_in = c[..., : C2 // 2], c[..., C2 // 2 :]
            st = compute_block_kv_norms(k_in, v_in, ids, layout="HND")
            all_k.append(st.k_norm)
            all_v.append(st.v_norm)
        k_ref, v_ref = torch.stack(all_k), torch.stack(all_v)

        assert k_fused.shape == k_ref.shape == (num_layers, len(ids), H)
        assert torch.allclose(k_fused, k_ref, atol=1e-2), (k_fused - k_ref).abs().max()
        assert torch.allclose(v_fused, v_ref, atol=1e-2), (v_fused - v_ref).abs().max()

    def test_score_blocks_fused_parity_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.scoring.cuda_norms import available, score_blocks_fused
        if not available():
            pytest.skip("cuda scorer unavailable")
        self._parity_asserts(score_blocks_fused)

    def test_score_blocks_fused_empty_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.scoring.cuda_norms import available, score_blocks_fused
        if not available():
            pytest.skip("cuda scorer unavailable")
        cache = [torch.randn(8, 4, 16, 128, dtype=torch.float16, device="cuda")]
        k, v = score_blocks_fused(cache, [])
        assert k.shape == (1, 0, 4) and v.shape == (1, 0, 4)

    def test_score_blocks_fused_parity_triton(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.offload.triton_score import score_blocks_fused, _HAS_TRITON
        if not _HAS_TRITON:
            pytest.skip("triton not available")
        self._parity_asserts(score_blocks_fused)

    def test_score_blocks_fused_empty_triton(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        from offload_package.offload.triton_score import score_blocks_fused, _HAS_TRITON
        if not _HAS_TRITON:
            pytest.skip("triton not available")
        cache = [torch.randn(8, 4, 16, 128, dtype=torch.float16, device="cuda")]
        k, v = score_blocks_fused(cache, [])
        assert k.shape == (1, 0, 4) and v.shape == (1, 0, 4)
    """Preemption and request finish must not leak CPU slots (Bug 6)."""

    def test_preemption_releases_cpu_slots(self):
        """Preempted request's CPU slots + residency + scorer entries must be freed."""
        pytest.skip("Requires scheduler preemption hook - run after fix B6")


class TestEndToEndGuard:
    """Full integration smoke test."""

    def test_offload_integrated_output_equality(self):
        """Greedy output with offload must equal dense baseline (token-for-token)."""
        pytest.skip("End-to-end validation - run after all fixes")


class TestStagingSizing:
    """S2: auto staging-pool sizing must follow the documented formula and
    respect the pool-fraction cap."""

    @staticmethod
    def _fake_configs(max_model_len: int, max_num_seqs: int,
                      block_size: int, num_blocks: int,
                      num_kv_heads: int = 4, num_q_heads: int = 32,
                      tp_size: int = 1):
        from types import SimpleNamespace
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=max_model_len,
                hf_config=SimpleNamespace(
                    num_attention_heads=num_q_heads * tp_size),
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tp_size),
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=num_blocks,
            kv_cache_groups=[SimpleNamespace(
                kv_cache_spec=SimpleNamespace(
                    block_size=block_size, num_kv_heads=num_kv_heads),
            )],
        )
        return vllm_config, kv_cache_config

    @staticmethod
    def _union_factor(topk: float, g: int) -> float:
        return 1.0 - (1.0 - topk) ** g

    @staticmethod
    def _expected(max_len, sink, local, page, union, f, max_seqs, safety=1.10):
        """Documented formula: only offloaded blocks are staged, sink/local
        are pinned, never more than the offloadable set."""
        import math
        mid_pages = -(-(max_len - sink - local) // page)
        staged = min(math.ceil(f * union * mid_pages), math.ceil(f * mid_pages))
        return max(64, int(max_seqs * staged * safety))

    def test_staging_sizing_auto(self, monkeypatch):
        """Union-aware sizing: no GQA-share -> G_eff = H_q (32 independent
        q-heads), so the pool covers 1-(1-topk)^32 of the middle range."""
        from offload_package.integration.connector import _auto_staging_slots_for

        monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
        monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "512")
        monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "512")
        monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "pq")
        monkeypatch.delenv("SKYLIGHT_PQ_GQA_SHARE", raising=False)
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25")

        max_len, max_seqs, page, num_blocks = 65536, 4, 16, 10_000_000
        vc, kc = self._fake_configs(max_len, max_seqs, page, num_blocks)

        union = self._union_factor(0.1, 32)
        expected = self._expected(max_len, 512, 512, page, union, 0.5, max_seqs)
        assert _auto_staging_slots_for(vc, kc, {"offload_fraction": 0.5}) == expected

    def test_staging_sizing_auto_gqa_share(self, monkeypatch):
        """GQA-share: G_eff = H_kv (4 kv-heads), shrinking the union ~4x
        versus 32 independent q-heads at the same topk."""
        from offload_package.integration.connector import _auto_staging_slots_for

        monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.05")
        monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "64")
        monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "64")
        monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "pq")
        monkeypatch.setenv("SKYLIGHT_PQ_GQA_SHARE", "1")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25")

        vc, kc = self._fake_configs(18000, 1, 16, 10_000_000)
        x = {"offload_fraction": 0.99}
        shared = self._expected(18000, 64, 64, 16, self._union_factor(0.05, 4), 0.99, 1)
        assert _auto_staging_slots_for(vc, kc, x) == shared

        # Share flag on a NON-pq method must not shrink: oracle selects
        # per q-head regardless of the pq env var.
        monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "oracle")
        unshared = self._expected(18000, 64, 64, 16, self._union_factor(0.05, 32), 0.99, 1)
        assert _auto_staging_slots_for(vc, kc, x) == unshared
        assert unshared > shared

    def test_staging_sizing_extra_config_knobs(self, monkeypatch):
        """kv_connector_extra_config overrides the env knobs (previously
        dead: stored in OffloadConfig but never consumed by the sizer)."""
        from offload_package.integration.connector import _auto_staging_slots_for

        monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.1")
        monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "0")
        monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "0")
        monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "pq")
        monkeypatch.setenv("SKYLIGHT_PQ_GQA_SHARE", "1")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25")

        vc, kc = self._fake_configs(8192, 1, 16, 10_000_000)
        union = self._union_factor(0.1, 4)
        base = self._expected(8192, 0, 0, 16, union, 0.99, 1, 1.10)
        assert _auto_staging_slots_for(vc, kc, {"offload_fraction": 0.99}) == base
        got = _auto_staging_slots_for(
            vc, kc, {"offload_fraction": 0.99, "staging_safety_margin": 2.0})
        assert got == self._expected(8192, 0, 0, 16, union, 0.99, 1, 2.0)
        assert got != base

    def test_staging_sizing_counts_only_offloadable_blocks(self, monkeypatch):
        """Staging covers selected AND offloaded blocks, capped at the
        offloadable set; the per-request block budget maps to a fraction."""
        from offload_package.integration.connector import _auto_staging_slots_for

        monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "1.0")
        monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "64")
        monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "64")
        monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "pq")
        monkeypatch.setenv("SKYLIGHT_PQ_GQA_SHARE", "1")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.0")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "1.0")
        monkeypatch.delenv("OFFLOAD_PACKAGE_GPU_BLOCK_BUDGET", raising=False)
        vc, kc = self._fake_configs(16512, 1, 16, 10_000_000, num_kv_heads=8)
        mid_pages = (16512 - 128) // 16                     # 1024
        # union == 1: the estimate is exactly the offloadable set, f * mid_pages
        assert _auto_staging_slots_for(vc, kc, {"offload_fraction": 0.25}) == 256
        assert _auto_staging_slots_for(vc, kc, {"offload_fraction": 0.5}) == 512
        # keep 768 of 1024 on GPU -> f = 0.25 -> same as above
        assert _auto_staging_slots_for(
            vc, kc, {"gpu_block_budget_per_request": 768}) == 256
        # nothing offloaded -> the 64-slot floor
        assert _auto_staging_slots_for(vc, kc, {"offload_fraction": 0.0}) == 64
        assert mid_pages == 1024

    def test_staging_sizing_auto_capped(self, monkeypatch):
        from offload_package.integration.connector import _auto_staging_slots_for

        monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.5")
        monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "512")
        monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "512")
        monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "pq")
        monkeypatch.setenv("SKYLIGHT_PQ_GQA_SHARE", "1")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25")

        # Tiny pool: estimate >> cap; result must be capped at 25% of pool.
        vc, kc = self._fake_configs(65536, 4, 16, 1000)
        assert _auto_staging_slots_for(vc, kc) == 250

    def test_staging_sizing_auto_floor(self, monkeypatch):
        from offload_package.integration.connector import _auto_staging_slots_for

        monkeypatch.setenv("SKYLIGHT_SPARSE_TOPK", "0.0")
        monkeypatch.setenv("SKYLIGHT_SPARSE_SINK", "0")
        monkeypatch.setenv("SKYLIGHT_SPARSE_LOCAL", "0")
        monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "pq")
        monkeypatch.setenv("SKYLIGHT_PQ_GQA_SHARE", "1")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_SAFETY", "1.10")
        monkeypatch.setenv("OFFLOAD_PACKAGE_STAGING_MAX_POOL_FRAC", "0.25")

        # Degenerate estimate (0 pages) on a big pool -> floor of 64 slots.
        vc, kc = self._fake_configs(65536, 1, 16, 10_000_000)
        assert _auto_staging_slots_for(vc, kc) == 64


class TestChunkedDecode:
    """S1: exact-attention fallback when the selected cold set exceeds staging
    capacity. Two properties:

    1. `_lse2_merge` is the exact online-softmax decomposition: merged output
       over disjoint subsets == dense attention over the union.
    2. `pool.release()` flips residency back to CPU so slots recycle.
    """

    def test_lse2_merge_parity(self):
        torch.manual_seed(0)
        from pq_optimized.batch_decode_with_paged_kv_cache_wrapper import (
            BatchDecodeWithPagedKVCacheWrapper as W,
        )

        N, D = 128, 32
        scores = torch.randn(1, 3, N, dtype=torch.float32)          # logits
        vals = torch.randn(1, 3, N, D, dtype=torch.float32) * 0.5
        high = torch.tensor(float("inf"))
        cold = torch.zeros(1, 3, N, dtype=torch.bool)
        cold[:, :, N // 2 :] = True  # second half is "cold" (goes in chunk pass)

        def partial(x, v, keep):
            m = x.masked_fill(~keep, -high)
            mx = m.amax(dim=-1, keepdim=True)                      # log2 domain
            e = torch.exp2(m - mx)
            d = e.sum(dim=-1, keepdim=True)
            # out[1,3,D] = sum_n (e/d)[1,3,n] * v[1,3,n,D]
            w = (e / d).unsqueeze(-1)                              # [1,3,N,1]
            out = (w * v).sum(dim=2)                               # [1,3,D]
            lse = (mx + torch.log2(d)).squeeze(-1)                 # [1, 3]
            return out, lse

        out_a, lse_a = partial(scores, vals, ~cold)                # "base" pass
        out_b, lse_b = partial(scores, vals, cold)                 # "chunk" pass
        out_full, _lse_f = partial(scores, vals, torch.ones_like(cold))

        merged, _ = W._lse2_merge(out_a, lse_a, out_b, lse_b)
        assert torch.allclose(merged, out_full, atol=1e-5), (
            merged - out_full
        ).abs().max()

    def test_chunked_capacity_overflow_keeps_residents(self):
        """When cold set > num_slots, the wrapper chunks into capacity-sized
        groups. Since the F6 change the chunked refs are NOT released per
        layer: later groups stage by EVICTING earlier groups' residents
        (score-aware), and the last group stays resident -- layers 2..N then
        fetch only their delta instead of re-fetching everything."""
        from offload_package.offload.manager import KVOffloadManager
        from offload_package.scoring.paged_eviction import PagedEvictionScorer
        from offload_package.staging.pool import KVStagingPool
        from offload_package.staging.residency import Residency, ResidencyTable
        from offload_package.tests.conftest import combined_caches

        num_pages, N, H, C = 32, 8, 2, 32
        staging_slots = 4
        combined = combined_caches(
            [torch.randn(num_pages, N, H, C, dtype=torch.float16, device="cuda")],
            [torch.randn(num_pages, N, H, C, dtype=torch.float16, device="cuda")],
        )
        cpu_cache = torch.zeros(
            (1, 32, 2, N, H, C), dtype=torch.float16, device="cpu", pin_memory=True
        )
        residency = ResidencyTable()
        mgr = KVOffloadManager(
            scorer=PagedEvictionScorer(), residency=residency,
            gpu_k_caches=combined, gpu_v_caches=combined,
            cpu_cache=cpu_cache, gpu_budget_blocks=0, kv_layout="HND",
        )
        pool = KVStagingPool(
            num_slots=staging_slots, staging_base_page=num_pages - staging_slots,
            gpu_k_caches=combined, gpu_v_caches=combined,
            cpu_cache=cpu_cache, residency=residency, offload_manager=mgr,
        )

        refs = [BlockRef("r", i) for i in range(10)]  # 10 > 4 slots
        for ref in refs:
            residency.mark_gpu(ref, gpu_block_id=ref.block_idx, complete=True)
        mgr.score_blocks(refs, [ref.block_idx for ref in refs])
        offloaded = mgr.offload_fraction(refs, 0.99, set(refs))
        assert len(offloaded) == 10
        for ref in refs:
            assert residency.get(ref).residency == Residency.CPU

        # Chunk like the wrapper: capacity-size groups, NO per-chunk release.
        groups = [refs[i:i + staging_slots] for i in range(0, 10, staging_slots)]
        for grp in groups:
            m, skipped = pool.ensure_resident(grp)
            assert not skipped
            assert len(m) == len(grp)
            for ref in grp:
                loc = residency.get(ref)
                if loc.residency == Residency.STAGING:
                    assert residency.get(ref).staging_slot == m[ref]

        # After the cycle exactly `staging_slots` refs are still resident (no
        # release): the last group plus the survivors of earlier groups that
        # later groups did not need to evict. Everything else is back on CPU.
        still_staged = [
            r for r in refs
            if residency.get(r).residency == Residency.STAGING
        ]
        assert len(still_staged) == staging_slots
        assert len(pool._lru) == staging_slots
        assert set(groups[-1]) <= set(still_staged)

        # The request-cleanup path (explicit release) still works.
        for ref in still_staged:
            pool.release(ref)
            loc = residency.get(ref)
            assert loc.residency == Residency.CPU, (
                f"after release, {ref} should be back on CPU, got {loc.residency}"
            )
        assert len(pool._lru) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])