"""Token-granularity mode: config validation + block-mode factory unchanged.

Token mode is built by the worker connector (``_build_token_offload``:
TokenOffloadEngine + FastKVzip gates + score hooks), not by
``build_orchestrator``; the engine itself is covered by
test_token_mode_engine.py.
"""
from __future__ import annotations

import types

import pytest
import torch

from offload_package.config import OffloadConfig
from offload_package.integration.token_mode import TokenModeScheduler
from offload_package.shared.types import OrchestratorConfig, build_orchestrator
from offload_package.vllm_adapter import KVCacheView

L, PAGES, PAGE, H, D = 2, 8, 4, 2, 4


def test_token_config_defaults_and_validation():
    cfg = OffloadConfig(offload_granularity="token")
    assert (cfg.token_scorer, cfg.token_sink, cfg.token_window) == ("fastkvzip", 64, 128)
    assert OffloadConfig().offload_granularity == "block"
    with pytest.raises(ValueError, match="offload_granularity"):
        OffloadConfig(offload_granularity="page")
    with pytest.raises(ValueError, match="fastkvzip"):
        OffloadConfig(offload_granularity="token", token_scorer="triattention")


def test_token_scheduler_requires_prefix_caching_off():
    sched = TokenModeScheduler(interval_tokens=32, engine_getter=lambda: None)
    on = types.SimpleNamespace(cache_config=types.SimpleNamespace(enable_prefix_caching=True))
    with pytest.raises(AssertionError, match="prefix_caching"):
        sched.bind(scheduler=on)
    off = types.SimpleNamespace(cache_config=types.SimpleNamespace(enable_prefix_caching=False))
    sched.bind(scheduler=off)
    assert sched.sched is off


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_block_mode_factory_unchanged():
    """Block mode still builds the full block-level machinery."""
    caches = [torch.zeros(PAGES, H, PAGE, 2 * D, dtype=torch.float16, device="cuda")
              for _ in range(L)]
    view = KVCacheView(layers=caches, k_caches=caches, v_caches=caches,
                       block_size=PAGE, num_kv_heads=H, head_dim=D,
                       kv_layout="HND", staging_base_page=PAGES - 2)
    cfg = OrchestratorConfig(
        offload_fraction=0.2, num_cpu_slots=4, num_staging_slots=2, W=32,
        page_size=PAGE, dtype=torch.float16, device=torch.device("cuda"),
        gpu_budget_blocks=PAGES - 2, kv_layout="HND")
    orch = build_orchestrator(cfg, kv_cache_view=view)
    assert orch.offload_manager is not None
    assert orch.staging_pool is not None
    assert orch.residency is not None
