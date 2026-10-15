"""Per-request offload targeting in the scheduler handler.

Covers: the (1-f) per-request GPU target (no compounding across rounds), the
absolute gpu_block_budget_per_request knob, sink/local-window pinning, that a
round only touches the triggering request, and preemption cleanup.
"""
from __future__ import annotations

import math

import pytest
import torch

from offload_package.config import OffloadConfig
from offload_package.controller import KVCompressionController
from offload_package.integration.manager import OffloadOrchestrator
from offload_package.integration.scheduler_connector import OffloadPackageScheduler
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.base import BlockRef
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import Residency, ResidencyTable
from offload_package.tests.conftest import combined_caches

PAGE = 16
NUM_PAGES = 256
STAGING_BASE = 240

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA")


def _orchestrator():
    k = [torch.randn(NUM_PAGES, PAGE, 4, 64, dtype=torch.float16, device="cuda")
         for _ in range(2)]
    v = [torch.randn_like(t) for t in k]
    caches = combined_caches(k, v)
    cpu = torch.empty((2, 512, 2, PAGE, 4, 64), dtype=torch.float16,
                      pin_memory=True)
    residency = ResidencyTable()
    mgr = KVOffloadManager(
        scorer=PagedEvictionScorer(), residency=residency,
        gpu_k_caches=caches, gpu_v_caches=caches, cpu_cache=cpu,
        gpu_budget_blocks=STAGING_BASE, kv_layout="HND",
    )
    pool = KVStagingPool(
        num_slots=NUM_PAGES - STAGING_BASE, staging_base_page=STAGING_BASE,
        gpu_k_caches=caches, gpu_v_caches=caches, cpu_cache=cpu,
        residency=residency, offload_manager=mgr,
    )
    orch = OffloadOrchestrator(
        offload_manager=mgr, staging_pool=pool, residency=residency,
        controller=KVCompressionController(W=32, page_size=PAGE),
        page_size=PAGE, staging_base_page_idx=STAGING_BASE,
        offload_fraction=0.5,
    )
    return orch


def _scheduler(orch, *, fraction=0.5, budget=None, sink=0, local=0):
    """A scheduler handler with only the state _trigger_request_offload
    touches (no vLLM engine): shadow mode, so no block-pool surgery."""
    s = OffloadPackageScheduler.__new__(OffloadPackageScheduler)
    s.offload_config = OffloadConfig(
        offload_fraction=fraction, gpu_block_budget_per_request=budget)
    s._orchestrator = orch
    s._controller = KVCompressionController(W=32, page_size=PAGE)
    s._sink_tokens, s._local_tokens = sink, local
    s._true_free_enabled = False
    s._gpu_block_pool = None
    s._step_store_acc = []
    s._pending_store_events = {}
    s._num_computed = {}
    s._prefill_completed = set()
    s._req_block_table = {}
    s._prompt_lens = {}
    return s


def _add_request(sched, orch, rid, n_blocks, first_page, num_computed=None):
    num_computed = n_blocks * PAGE if num_computed is None else num_computed
    ids = list(range(first_page, first_page + n_blocks))
    orch.sync_request_blocks(rid, ids, num_computed)
    sched._num_computed[rid] = num_computed


def _counts(orch, rid):
    out = {Residency.GPU: 0, Residency.CPU: 0, Residency.STAGING: 0}
    for ref in orch.residency.refs_for_request(rid):
        out[orch.residency.get(ref).residency] += 1
    return out


def test_fraction_target_does_not_compound():
    orch = _orchestrator()
    sched = _scheduler(orch, fraction=0.5)
    _add_request(sched, orch, "r", 20, 0)
    sched._trigger_request_offload("r")
    assert _counts(orch, "r")[Residency.GPU] == 10
    # A second round with no new blocks must not offload further (the old
    # global path took another 50% of what was left every round).
    sched._trigger_request_offload("r")
    assert _counts(orch, "r")[Residency.GPU] == 10
    # Four new complete blocks: target ceil(0.5 * 24) = 12 GPU.
    _add_request(sched, orch, "r", 24, 0)
    sched._trigger_request_offload("r")
    assert _counts(orch, "r")[Residency.GPU] == 12


def test_zero_fraction_offloads_nothing():
    orch = _orchestrator()
    sched = _scheduler(orch, fraction=0.0)
    _add_request(sched, orch, "r", 10, 0)
    sched._trigger_request_offload("r")
    assert _counts(orch, "r")[Residency.CPU] == 0


def test_budget_overrides_fraction():
    orch = _orchestrator()
    sched = _scheduler(orch, fraction=0.1, budget=3)
    _add_request(sched, orch, "r", 12, 0)
    sched._trigger_request_offload("r")
    assert _counts(orch, "r")[Residency.GPU] == 3


def test_sink_and_local_blocks_are_pinned():
    orch = _orchestrator()
    # 2 sink blocks, local window of 40 tokens over a 20-block (320 token)
    # request whose tail block is partial (num_computed=312).
    sched = _scheduler(orch, fraction=0.9, sink=2 * PAGE, local=40)
    _add_request(sched, orch, "r", 20, 0, num_computed=312)
    sched._trigger_request_offload("r")
    gpu = {ref.block_idx for ref in orch.residency.refs_for_request("r")
           if orch.residency.get(ref).residency == Residency.GPU}
    local_start = 312 - 40
    pinned = {i for i in range(20)
              if i * PAGE < 2 * PAGE or (i + 1) * PAGE > local_start}
    assert pinned <= gpu
    offloadable = [i for i in range(19) if i not in pinned]  # 19 complete
    assert len(gpu - pinned) == math.ceil(0.1 * len(offloadable))


def test_round_touches_only_triggering_request():
    orch = _orchestrator()
    sched = _scheduler(orch, fraction=0.5)
    _add_request(sched, orch, "a", 10, 0)
    _add_request(sched, orch, "b", 10, 10)
    sched._trigger_request_offload("a")
    assert _counts(orch, "a")[Residency.GPU] == 5
    assert _counts(orch, "b")[Residency.CPU] == 0
    assert sched._step_store_acc and sched._step_store_acc[0][3] == ["a"]


def test_forget_request_strips_pending_store_refs():
    orch = _orchestrator()
    sched = _scheduler(orch, fraction=0.5)
    _add_request(sched, orch, "a", 8, 0)
    _add_request(sched, orch, "b", 8, 8)
    sched._trigger_request_offload("a")
    sched._trigger_request_offload("b")
    sched._pending_store_events[0] = (
        [r for e in sched._step_store_acc for r in e[0]],
        [g for e in sched._step_store_acc for g in e[1]], [], ["a", "b"])
    sched._forget_request("a")
    refs, gids, _, rids = sched._pending_store_events[0]
    assert all(r.request_id == "b" for r in refs) and len(refs) == len(gids)
    assert rids == ["b"]
    assert orch.residency.refs_for_request("a") == []
