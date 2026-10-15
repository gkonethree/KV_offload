"""Factory + facade integration tests for token-granularity mode."""
from __future__ import annotations

import pytest
import torch

from offload_package.shared.types import OrchestratorConfig, build_orchestrator
from offload_package.vllm_adapter import KVCacheView

L = 2
PAGES = 8
PAGE = 4
H = 2
D = 4
FREQ = D // 2


def _write_stats(tmp_path) -> str:
    g = torch.Generator().manual_seed(0)
    payload = {
        "metadata": {
            "num_attention_heads": H,
            "num_kv_heads": H,
            "head_dim": D,
            "num_layers": 1,
            "rope_theta": 10000.0,
            "rope_style": "half",
        },
        "layer_stats": {
            0: {
                "q_mean_complex": torch.randn(H, FREQ, 2, generator=g),
                "q_abs_mean": torch.rand(H, FREQ, generator=g) + 0.5,
                "freq_scale_sq": torch.ones(H, FREQ),
            }
        },
    }
    path = tmp_path / "token_stats.pt"
    torch.save(payload, path)
    return str(path)


def _make_view(torch_device):
    caches = [
        torch.zeros(PAGES, H, PAGE, 2 * D, dtype=torch.float16, device=torch_device)
        for _ in range(L)
    ]
    return caches, KVCacheView(
        layers=caches,
        k_caches=caches,
        v_caches=caches,
        block_size=PAGE,
        num_kv_heads=H,
        head_dim=D,
        kv_layout="HND",
        staging_base_page=PAGES - 2,
    )


class TestTokenModeFactory:

    def _cfg(self, stats_path, **over):
        base = dict(
            offload_fraction=0.2,
            num_cpu_slots=4,
            num_staging_slots=2,
            W=32,
            page_size=PAGE,
            dtype=torch.float16,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            gpu_budget_blocks=PAGES - 2,
            granularity="token",
            scorer_type="triattention",
            triattention_stats_path=stats_path,
            token_budget=8,
            window_size=4,
            sink_tokens=2,
            protect_prefill=False,
        )
        base.update(over)
        return OrchestratorConfig(**base)

    def test_requires_triattention_scorer(self, torch_device, tmp_path):
        _, view = _make_view(torch_device)
        with pytest.raises(ValueError, match="triattention"):
            build_orchestrator(
                self._cfg(_write_stats(tmp_path), scorer_type="paged_eviction"),
                kv_cache_view=view,
            )

    def test_requires_stats_path(self, torch_device):
        _, view = _make_view(torch_device)
        with pytest.raises(ValueError, match="stats_path"):
            build_orchestrator(
                self._cfg("", ), kv_cache_view=view
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="token mode targets CUDA runners"
    )
    def test_round_lifecycle_through_facade(self, torch_device, tmp_path):
        caches, view = _make_view(torch_device)
        orch = build_orchestrator(
            self._cfg(_write_stats(tmp_path)), kv_cache_view=view
        )
        assert orch.token_mode
        engine = orch.token_engine
        assert engine.token_pool.num_slots == 4 * PAGE
        assert engine.token_budget == 8

        g = torch.Generator(device="cuda").manual_seed(0)
        for t in range(20):
            for layer in range(L):
                caches[layer][t // PAGE, :, t % PAGE, :] = torch.randn(
                    H, 2 * D, generator=g, device="cuda"
                ).to(torch.float16)

        # New request: 20 tokens over 5 pages.
        orch.sync_request_blocks("r1", list(range(5)), 20)
        state = engine.state("r1")
        assert state.physical_len == 20 and state.logical_len == 20

        orch.on_prefill_complete("r1", 20)
        result = engine.on_round("r1")
        assert result is not None
        assert result.kept_count == 8
        assert result.evicted_count == 12
        assert result.freed_block_ids == [2, 3, 4]
        assert state.block_ids == [0, 1]
        assert len(state.cpu_slot_of_logical) == 12
        assert state.logical_of_physical[:8].tolist() == sorted(
            state.logical_of_physical[:8].tolist()
        )

        # Three more tokens arrive with three new pages appended at the tail.
        orch.sync_request_blocks("r1", state.block_ids + [5, 6, 7], 23)
        assert state.physical_len == 11
        assert state.logical_len == 23
        assert state.block_ids == [0, 1, 5, 6, 7]

        # The shrink provider reads the CPU-resident count.
        assert state.logical_len - state.physical_len == 12

        orch.on_request_done("r1")
        assert engine.state("r1") is None
        assert engine.token_pool.num_free == 4 * PAGE

    def test_block_mode_unchanged(self, torch_device, tmp_path):
        """Block mode still builds the full block-level machinery."""
        caches, view = _make_view(torch_device)
        cfg = self._cfg(
            _write_stats(tmp_path),
            granularity="block",
            scorer_type="paged_eviction",
        )
        orch = build_orchestrator(cfg, kv_cache_view=view)
        assert not orch.token_mode
        assert orch.token_engine is None
        assert orch.offload_manager is not None
        assert orch.staging_pool is not None
        assert orch.residency is not None
