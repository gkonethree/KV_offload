"""TokenOffloadEngine: per-layer eviction + order-preserving compaction."""
from __future__ import annotations

import math

import pytest
import torch

from offload_package.integration.token_mode import TokenOffloadEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
L, PAGE, PAGES, H, C2 = 3, 4, 32, 2, 8


def _enc(layer, logical):
    return float(layer * 1000 + logical)


def _make(fraction=0.5, sink=2, window=3, cpu=256):
    caches = [torch.full((PAGES, H, PAGE, C2), -1.0, device="cuda") for _ in range(L)]
    eng = TokenOffloadEngine(kv_caches=caches, page_size=PAGE, max_len=128,
                             num_req_slots=1, num_cpu_tokens=cpu,
                             offload_fraction=fraction, sink_tokens=sink,
                             window_tokens=window)
    return eng, caches


def _write(eng, caches, req, lo, hi):
    """Simulate vLLM writing KV of logical [lo, hi) at their physical slots."""
    s = eng.slot_of(req)
    pages = eng.pages(req)
    for l in range(L):
        for t in range(lo, hi):
            p = int(eng.phys_of_logical[l, s, t])
            caches[l][pages[p // PAGE], :, p % PAGE, :] = _enc(l, t)


def _check(eng, caches, req, written):
    s = eng.slot_of(req)
    r = eng._reqs[req]
    pages = eng.pages(req)
    for l in range(L):
        resident = 0
        for t in range(written):
            p = int(eng.phys_of_logical[l, s, t])
            if p >= 0:
                resident += 1
                v = caches[l][pages[p // PAGE], 0, p % PAGE, 0].item()
                assert v == _enc(l, t), (l, t, p, v)
                assert int(eng.logical_of_phys[l, s, p]) == t
            else:
                c = int(eng.cpu_slot[l, s, t])
                assert c >= 0, (l, t)
                assert eng.cpu_pool[l, c, 0, 0].item() == _enc(l, t)
        assert resident == r.phys_written, (l, resident, r.phys_written)
    # physical prefix is logically ordered in every layer
    for l in range(L):
        lp = eng.logical_of_phys[l, s, :r.phys_written].tolist()
        assert lp == sorted(lp)


def test_round_per_layer_selection_and_compaction():
    eng, caches = _make(fraction=0.5, sink=2, window=3)
    req = "r0"
    eng.add_request(req)
    eng.set_pages(req, list(range(8)))                 # 32 token slots
    eng.on_step(req, 0, 20)                            # prefill 20 tokens
    _write(eng, caches, req, 0, 20)
    eng.on_step(req, 20, 0)
    s = eng.slot_of(req)
    torch.manual_seed(0)
    eng.score[:, s, :20] = torch.rand(L, 20, device="cuda")   # different per layer
    scores = eng.score[:, s, :20].clone()
    freed = eng.run_round(req)
    K = math.ceil(0.5 * 20)                            # 10 kept per layer
    assert eng._reqs[req].phys_written == K
    assert freed == list(range(math.ceil(K / PAGE), 8))
    _check(eng, caches, req, 20)
    kept_sets = []
    for l in range(L):
        kept = {t for t in range(20) if int(eng.phys_of_logical[l, s, t]) >= 0}
        prot = {0, 1, 17, 18, 19}
        assert prot <= kept
        cand = [t for t in range(20) if t not in prot]
        expect = prot | set(sorted(cand, key=lambda t: -float(scores[l, t]))[:K - len(prot)])
        assert kept == expect, l
        kept_sets.append(kept)
    assert len({frozenset(k) for k in kept_sets}) > 1      # layers differ


def test_second_round_after_decode_growth_and_release():
    eng, caches = _make(fraction=0.5, sink=2, window=3)
    req = "r1"
    eng.add_request(req)
    eng.set_pages(req, list(range(8)))
    eng.on_step(req, 0, 16); _write(eng, caches, req, 0, 16); eng.on_step(req, 16, 0)
    s = eng.slot_of(req)
    eng.score[:, s, :16] = torch.rand(L, 16, device="cuda")
    freed1 = eng.run_round(req)
    eng.set_pages(req, eng.pages(req))
    # decode 6 tokens: mapped at the new physical tail in every layer; like
    # vLLM, allocate a page whenever the physical tail crosses a boundary
    # (freed pages are recycled first).
    spare = list(freed1) + list(range(8, PAGES))
    for t in range(16, 22):
        r = eng._reqs[req]
        if (t - r.offloaded) + 1 > len(r.pages) * PAGE:     # token t's phys slot
            eng.set_pages(req, r.pages + [spare.pop(0)])
        eng.on_step(req, t, 1)
        _write(eng, caches, req, t, t + 1)
        eng.score[:, s, t] = torch.rand(L, device="cuda")
    eng.on_step(req, 22, 0)
    _check(eng, caches, req, 22)
    held = list(eng.pages(req))
    freed2 = eng.run_round(req)
    K2 = math.ceil(0.5 * 22)
    assert eng._reqs[req].phys_written == K2
    _check(eng, caches, req, 22)
    keep_pages = math.ceil(K2 / PAGE)
    assert eng.pages(req) == held[:keep_pages] and freed2 == held[keep_pages:]
    free_before = len(eng._free_cpu)
    eng.release(req)
    assert len(eng._free_cpu) == free_before + eng.stats["offloaded"]
    assert not eng.has(req)


def test_no_round_when_budget_covers_everything():
    eng, caches = _make(fraction=0.0)
    req = "r2"
    eng.add_request(req); eng.set_pages(req, list(range(8)))
    eng.on_step(req, 0, 12); _write(eng, caches, req, 0, 12); eng.on_step(req, 12, 0)
    assert eng.run_round(req) == []
    assert eng._reqs[req].offloaded == 0
