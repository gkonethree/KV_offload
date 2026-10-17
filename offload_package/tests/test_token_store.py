"""TokenStagingStore: per-layer packed token staging + physical addressing."""
from __future__ import annotations

import pytest
import torch

from offload_package.scoring.base import BlockRef
from offload_package.staging.token_store import TokenStagingStore
from offload_package.tests.test_staging_pool import _make_pool, _offload_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

P = 8          # _make_pool page size
NBLK = 6


def _setup(num_slots=4, layers=2):
    pool, mgr, res = _make_pool(num_pages=16, num_slots=num_slots, layers=layers)
    res.enable_device_mirror(num_layers=layers, max_blocks=8, num_slots=2, device="cuda")
    refs = [BlockRef("r", i) for i in range(NBLK)]
    assert _offload_blocks(mgr, res, refs) == refs          # all cold, pages 0..5
    store = TokenStagingStore(pool, res)
    rows = torch.tensor([[res.slots_for(["r"])[0]]], device="cuda")
    indptr = torch.tensor([0, NBLK], dtype=torch.int32, device="cuda")
    indices = torch.arange(NBLK, dtype=torch.int32, device="cuda")
    return pool, res, store, rows, indptr, indices


def _pos(*p):
    return torch.tensor([list(p)], dtype=torch.long, device="cuda")


def test_fetch_places_exact_bytes_per_layer():
    pool, res, store, rows, indptr, indices = _setup()
    idx = _pos(0, 9, 17, 40, 9)
    key, cold, missing = store.stage(0, idx, rows)
    assert missing is None and bool(cold.all())
    phys = store.phys(0, key, cold, idx, indptr, indices)
    torch.cuda.synchronize()
    k = pool.gpu_k_caches[0]
    page, off = phys // P, phys % P
    assert bool((page >= pool.base).all())
    # Offloaded pages are still intact in the test, so they are ground truth.
    assert torch.equal(k[page[0], :, off[0]], k[idx[0] // P, :, idx[0] % P])
    assert int((store.tok_slot[1] >= 0).sum()) == 0          # layer 1 untouched


def test_hits_do_not_refetch():
    _, _, store, rows, _, _ = _setup()
    calls = []
    real = store._copy
    store._copy = lambda *a: (calls.append(a[1].numel()), real(*a))
    store.stage(0, _pos(1, 2, 3), rows)
    store.stage(0, _pos(1, 2, 3, 4), rows)
    assert calls == [3, 1]


def test_lru_eviction_and_current_layer_protection():
    _, _, store, rows, _, _ = _setup()               # C = 4 * 8 = 32 slots
    assert store.stage(0, _pos(*range(33)), rows)[2].numel() == 33   # > C
    store.stage(0, _pos(*range(8)), rows)            # oldest
    store.stage(0, _pos(*range(8, 20)), rows)
    store.stage(0, _pos(*range(20, 40)), rows)       # 12 free + tokens 0..7 evicted
    flat = store.tok_slot[0].view(-1)
    base = int(rows) * store.mbp
    staged = (flat[base:base + 48] >= 0).nonzero().flatten().tolist()
    assert staged == list(range(8, 40))
    # 20 hits are protected this tick; 16 new need more than the 12 evictable.
    key, _, missing = store.stage(0, _pos(*range(20, 40), *range(40, 48), *range(0, 8)), rows)
    assert missing is not None and missing.numel() == 16
    assert (flat[base + 20:base + 40] >= 0).all()    # nothing changed


def test_release_on_request_removal():
    _, res, store, rows, _, _ = _setup()
    with torch.inference_mode():          # decode allocates the maps here,
        store.stage(0, _pos(1, 2), rows)
        store.stage(1, _pos(3), rows)
    # ...the scheduler releases outside inference mode.
    for i in range(NBLK):
        res.remove(BlockRef("r", i))
    assert int((store.tok_slot >= 0).sum()) == 0
    assert int((store.owner >= 0).sum()) == 0


def test_identity_tables_decode_bitwise_equal():
    ops_mod = pytest.importorskip("sparse_optimized.cuda_ops")
    ops = ops_mod.get_sparse_decode_ops()
    torch.manual_seed(0)
    H, Hq, D, PS, pages, K = 2, 4, 128, 16, 20, 32
    kv = torch.randn(pages, H, PS, 2 * D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(1, Hq, D, dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([7, 3, 11, 5], dtype=torch.int32, device="cuda")
    indptr = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    idx = torch.randint(0, 4 * PS, (1, Hq, K), device="cuda")
    w = torch.ones(1, Hq, K, device="cuda")
    slen = torch.full((1, Hq, 1), K, dtype=torch.int32, device="cuda")
    ks = int(ops.pick_k_split(1, Hq, K, 64))

    def run(ip, ind, sidx):
        return ops.sparse_decode_run(
            q, kv, kv, H * PS * 2 * D, 2 * D, PS * 2 * D, D,
            ip, ind, slen, sidx, w, H, PS, D ** -0.5, 0.0, ks, True)

    out_a, lse_a = run(indptr, indices, idx)
    phys = indices[idx // PS].long() * PS + idx % PS
    out_b, lse_b = run(torch.zeros(2, dtype=torch.int32, device="cuda"),
                       torch.arange(pages, dtype=torch.int32, device="cuda"), phys)
    assert torch.equal(out_a, out_b) and torch.equal(lse_a, lse_b)
