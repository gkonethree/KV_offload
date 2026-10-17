"""Token-staging CUDA kernels == the torch path (token-engine mode).

Two identical worlds (pool + TokenOffloadEngine after one offload round) run
the same selection sequence, one through the kernels and one through torch;
every output, map and staged byte must match.
"""
from __future__ import annotations

import pytest
import torch

from offload_package.integration.token_mode import TokenOffloadEngine
from offload_package.staging import token_kernels
from offload_package.staging.token_store import TokenStagingStore
from offload_package.tests.test_staging_pool import _make_pool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
P, SLOTS, PAGES, LAYERS, PROMPT = 8, 4, 20, 2, 100      # _make_pool page size is 8


def _world(kernels: bool):
    torch.manual_seed(0)
    pool, _, _ = _make_pool(num_pages=PAGES, num_slots=SLOTS, layers=LAYERS)
    eng = TokenOffloadEngine(kv_caches=pool.gpu_k_caches, page_size=P, max_len=128,
                             num_req_slots=2, num_cpu_tokens=256, offload_fraction=0.6,
                             sink_tokens=2, window_tokens=3)
    eng.add_request("r")
    eng.set_pages("r", list(range(13)))                 # 104 token slots
    eng.on_step("r", 0, PROMPT)
    eng.on_step("r", PROMPT, 0)
    s = eng.slot_of("r")
    eng.score[:, s, :PROMPT] = torch.rand(LAYERS, PROMPT, device="cuda")
    eng.run_round("r")
    store = TokenStagingStore.for_token_engine(pool, eng)
    if not kernels:
        store._kern = False
    rows = torch.tensor([[s]], device="cuda")
    return pool, eng, store, rows


def _kern_or_skip():
    if token_kernels.get() is None:
        pytest.skip(f"token kernels unavailable: {token_kernels.why_unavailable()}")


def _cold_positions(eng, layer):
    s = eng.slot_of("r")
    return (eng.phys_of_logical[layer, s, :PROMPT] < 0).nonzero().flatten().tolist()


def _compare(a, b, layer, idx):
    (pa, ea, sa, ra), (pb, eb, sb, rb) = a, b
    out_a = sa.stage_tokens(layer, idx, ra)
    out_b = sb.stage_tokens(layer, idx, rb)
    if hasattr(sa, "last_counts"):                    # kernel-side cold count
        assert sa.last_counts[:2] == (idx.numel(), int(out_b[1].sum()))
    for x, y in zip(out_a[:3], out_b[:3]):
        assert torch.equal(x, y)
    ma, mb = out_a[3], out_b[3]
    assert (ma is None) == (mb is None)
    if ma is not None:
        assert torch.equal(ma.cpu(), mb.cpu())
    assert torch.equal(sa.tok_slot, sb.tok_slot)
    assert torch.equal(sa.owner, sb.owner)
    assert torch.equal(sa.last_use, sb.last_use)
    pages = torch.tensor(ea.pages("r"), dtype=torch.int32, device="cuda")
    indptr = torch.tensor([0, pages.numel()], dtype=torch.int32, device="cuda")
    da = sa.phys_tokens(layer, out_a[0], out_a[1], out_a[2], indptr, pages)
    db = sb.phys_tokens(layer, out_b[0], out_b[1], out_b[2], indptr, pages)
    assert torch.equal(da, db)
    torch.cuda.synchronize()
    ka, kb = pa.gpu_k_caches[layer], pb.gpu_k_caches[layer]
    assert torch.equal(ka[pa.base:], kb[pb.base:])          # staging tails byte-equal
    # Every staged cold token holds exactly its CPU-pool row.
    s = ea.slot_of("r")
    flat = sa.tok_slot[layer].view(-1)
    for pos in set(idx.flatten().tolist()):
        if int(ea.phys_of_logical[layer, s, pos]) >= 0:
            continue
        v = int(flat[s * sa.mbp + pos])
        if v < 0:
            continue
        cs = int(ea.cpu_slot[layer, s, pos])
        got = ka[pa.base + v // P, :, v % P].cpu()
        assert torch.equal(got, ea.cpu_pool[layer, cs]), (layer, pos)
    return out_a, out_b


def test_kernels_match_torch_path():
    _kern_or_skip()
    a, b = _world(True), _world(False)
    assert a[2]._kernels() is not None and b[2]._kernels() is None
    g = torch.Generator(device="cuda").manual_seed(1)
    for step in range(12):
        for layer in range(LAYERS):
            idx = torch.randint(0, PROMPT, (1, 64), device="cuda", generator=g)
            _compare(a, b, layer, idx)


def test_lru_protection_and_over_capacity_match():
    _kern_or_skip()
    a, b = _world(True), _world(False)
    cold = _cold_positions(a[1], 0)
    C = SLOTS * P
    assert len(cold) > C                         # selection can overflow staging
    sel = lambda xs: torch.tensor([xs], dtype=torch.long, device="cuda")
    _compare(a, b, 0, sel(cold[:20]))            # 20 staged
    _compare(a, b, 0, sel(cold[20:30]))          # 10 more (30 / 32 used)
    # 20 hits protected this tick + 16 new > 12 evictable: over capacity.
    out_a, _ = _compare(a, b, 0, sel(cold[:20] + cold[30:46]))
    assert out_a[3] is not None and out_a[3].numel() == 16
    # Chunked path as the wrapper drives it: new tick per chunk, then fetch.
    for st in (a[2], b[2]):
        st.tick += 1
        assert st.fetch(0, out_a[3].to("cuda"), out_a[3])
    assert torch.equal(a[2].tok_slot, b[2].tok_slot)
    assert torch.equal(a[2].owner, b[2].owner)
    torch.cuda.synchronize()
    assert torch.equal(a[0].gpu_k_caches[0][a[0].base:], b[0].gpu_k_caches[0][b[0].base:])


def test_release_and_missing_cpu_copy():
    _kern_or_skip()
    pool, eng, store, rows = _world(True)
    cold = _cold_positions(eng, 1)
    store.stage_tokens(1, torch.tensor([cold[:5]], device="cuda"), rows)
    assert int((store.tok_slot[1] >= 0).sum()) == 5
    store.release_slot(eng.slot_of("r"))
    assert int((store.tok_slot >= 0).sum()) == 0 and int((store.owner >= 0).sum()) == 0
    # A cold position whose CPU copy is missing must raise (reported at the
    # next counter readback).
    s = eng.slot_of("r")
    eng.cpu_slot[1, s, cold[6]] = -1
    store.stage_tokens(1, torch.tensor([[cold[6]]], device="cuda"), rows)
    with pytest.raises(RuntimeError, match="no CPU copy"):
        store.stage_tokens(1, torch.tensor([[cold[7]]], device="cuda"), rows)
