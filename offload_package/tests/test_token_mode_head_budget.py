"""Per-(layer, head) token budgets: HeadTokenOffloadEngine on synthetic caches.

Every (layer, position, head) gets a distinct KV row, so after a round we can
check that each resident cell still holds its own token, each offloaded one
sits byte-exact in the CPU pool, and the scope's budget was applied."""
import pytest
import torch

from offload_package.integration.token_mode import TokenOffloadEngine
from offload_package.integration.token_mode_heads import HeadTokenOffloadEngine, flat_cells

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, H, P, C2, NB, MAXLEN = 3, 2, 4, 8, 48, 96


def _caches(layout):
    unit = H * P * C2
    buf = torch.zeros(L * NB * unit, dtype=torch.float32, device="cuda")
    if layout == "layer_outer":
        return [buf[l * NB * unit:(l + 1) * NB * unit].view(NB, H, P, C2) for l in range(L)]
    if layout == "token_major":                    # vLLM's own: pages stored [P, H]
        return list(buf.view(L, NB, P, H, C2).permute(0, 1, 3, 2, 4).unbind(0))
    v = buf.view(NB, L, H, P, C2)
    return [v[:, l] for l in range(L)]


def _row(l, p, h):
    return torch.tensor([l, p, h, 1000 * l + 10 * p + h, 0, 0, 0, 0], dtype=torch.float32,
                        device="cuda")


def _engine(layout="layer_outer", scope="global", **kw):
    kw.setdefault("offload_fraction", 0.5)
    return HeadTokenOffloadEngine(kv_caches=_caches(layout), page_size=P, max_len=MAXLEN,
                                  num_req_slots=2, num_cpu_tokens=MAXLEN, sink_tokens=2,
                                  window_tokens=4, scope=scope, **kw)


def _write(eng, s, lo, hi):
    """Write each newly mapped (layer, pos, head) its own row."""
    for l in range(L):
        for p in range(lo, hi):
            for h in range(H):
                eng.kv_rows[int(eng.cell[l, s, p, h])] = _row(l, p, h)


def _check(eng, s, n):
    """Resident cells hold their token; offloaded ones are in the CPU pool."""
    for l in range(L):
        for p in range(n):
            for h in range(H):
                c, r = int(eng.cell[l, s, p, h]), int(eng.cpu_row[l, s, p, h])
                want = _row(l, p, h)
                if c >= 0:
                    assert r < 0
                    assert torch.equal(eng.kv_rows[c], want), (l, p, h)
                else:
                    assert r >= 0
                    assert torch.equal(eng.cpu_pool[r].cuda(), want), (l, p, h)
    cells = eng.cell[:, s, :n]
    live = cells[cells >= 0]
    assert live.unique().numel() == live.numel(), "two tokens share a cell"


def _prefill(eng, n, first_page=3, seed=0):
    s = eng.add_request("r")
    pages = list(range(first_page, first_page + -(-n // P)))
    eng.set_pages("r", pages)
    eng.on_step("r", 0, n)
    eng.on_step("r", n, 0)
    _write(eng, s, 0, n)
    g = torch.Generator(device="cuda").manual_seed(seed)
    eng.score[:, s, :n] = torch.rand((L, n, H), device="cuda", generator=g)
    return s, pages


LAYOUTS = ["layer_outer", "block_outer", "token_major"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_flat_cells_address_every_layer_page(layout):
    caches = _caches(layout)
    rows, ub, us, n_units, head_major = flat_cells(caches)
    assert n_units == L * NB and head_major == (layout != "token_major")
    for l in (0, L - 1):
        for b in (0, 7, NB - 1):
            for h in range(H):
                for o in range(P):
                    caches[l][b, h, o] = float(1 + l * 100000 + b * 100 + h * 10 + o)
                    c = (ub[l] + b * us) * H * P + (h * P + o if head_major else o * H + h)
                    assert torch.equal(rows[c], caches[l][b, h, o])


@pytest.mark.parametrize("layout", LAYOUTS)
def test_global_round_budget_data_and_pages(layout):
    n = 40
    eng = _engine(layout)
    s, pages = _prefill(eng, n)
    freed = eng.run_round("r")
    K = max(eng.keep_count(n), eng._n_protected(n))
    cells = eng.cell[:, s, :n]
    assert int((cells >= 0).sum()) == L * H * K
    pos = torch.arange(n, device="cuda").view(1, -1, 1).expand(L, n, H)
    prot = (pos < 2) | (pos >= n - 4)
    assert bool((cells[prot] >= 0).all())
    # Global threshold: every evicted score <= every kept unprotected score.
    sc = eng.score[:, s, :n]
    kept_sc = sc[(cells >= 0) & ~prot]
    ev_sc = sc[cells < 0]
    assert float(ev_sc.max()) <= float(kept_sc.min())
    nb = -(-L * H * K // (L * H * P))
    assert freed == pages[nb:]
    assert eng.offset("r") == n - nb * P
    _check(eng, s, n)


def test_skewed_scores_make_layers_borrow_units():
    n = 40
    eng = _engine()
    s, pages = _prefill(eng, n)
    eng.score[0, s, :n] += 10.0                      # layer 0 wins every comparison
    eng.run_round("r")
    kept = (eng.cell[:, s, :n] >= 0).sum((1, 2))
    K = max(eng.keep_count(n), eng._n_protected(n))
    assert int(kept[0]) == H * n                     # layer 0 keeps everything
    assert int(kept.sum()) == L * H * K
    # Layer 0 now needs more units than its own share of the request's blocks.
    nb = -(-L * H * K // (L * H * P))
    own = {eng.unit(0, b) for b in pages[:nb]}
    used = {int(c) // (H * P) for c in eng.cell[0, s, :n].reshape(-1).tolist()}
    assert used - own, "expected layer 0 cells in other layers' units"
    _check(eng, s, n)


@pytest.mark.parametrize("scope", ["layer", "uniform"])
def test_layer_and_uniform_budgets(scope):
    n = 40
    eng = _engine(scope=scope)
    s, _ = _prefill(eng, n)
    eng.run_round("r")
    K = max(eng.keep_count(n), eng._n_protected(n))
    kept = eng.cell[:, s, :n] >= 0
    if scope == "layer":
        assert kept.sum((1, 2)).tolist() == [H * K] * L
    else:
        assert kept.sum(1).reshape(-1).tolist() == [K] * (L * H)
    _check(eng, s, n)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_append_then_second_round_repacks(layout):
    n = 40
    eng = _engine(layout)
    s, pages = _prefill(eng, n)
    eng.run_round("r", extra_tokens=1)
    kept_pages = eng.pages("r")
    # This step's write lands at the physical tail (standard layout).
    new_pages = kept_pages + [40, 41]
    eng.set_pages("r", new_pages)
    eng.on_step("r", n, 6)
    eng.on_step("r", n + 6, 0)
    phys = n - eng.offset("r")
    for l in range(L):
        for h in range(H):
            c = int(eng.cell[l, s, n, h])
            b = new_pages[phys // P]
            assert c == eng.unit(l, b) * H * P + eng.std_cell(h, phys % P)
            assert torch.equal(eng.kv_rows[c], eng.kv_caches[l][b, h, phys % P])
    _write(eng, s, n, n + 6)
    eng.score[:, s, n:n + 6] = 0.5
    eng.run_round("r")
    K = max(eng.keep_count(n + 6), eng._n_protected(n + 6))
    assert int((eng.cell[:, s, :n + 6] >= 0).sum()) == L * H * K
    _check(eng, s, n + 6)


def test_release_returns_cpu_rows():
    eng = _engine()
    top = eng._top
    _prefill(eng, 40)
    eng.run_round("r")
    assert eng._top < top
    eng.release("r")
    assert eng._top == top
    assert sorted(eng._free_rows[:top].tolist()) == list(range(top))


def test_shared_scope_matches_layer_engine():
    """scope='shared' keeps, for every head, exactly the tokens the current
    (one-decision-per-layer) engine keeps."""
    n = 40
    eng = _engine(scope="shared")
    s, pages = _prefill(eng, n)
    old = TokenOffloadEngine(kv_caches=_caches("layer_outer"), page_size=P, max_len=MAXLEN,
                             num_req_slots=2, num_cpu_tokens=MAXLEN, offload_fraction=0.5,
                             sink_tokens=2, window_tokens=4)
    so = old.add_request("r")
    old.set_pages("r", pages)
    old.on_step("r", 0, n)
    old.on_step("r", n, 0)
    old.score[:, so, :n] = eng.score[:, s, :n].amax(-1)
    eng.run_round("r")
    old.run_round("r")
    for l in range(L):
        want = set(old.logical_of_phys[l, so, :n].tolist()) - {-1}
        for h in range(H):
            got = set(torch.nonzero(eng.cell[l, s, :n, h] >= 0).view(-1).tolist())
            assert got == want, (l, h)
    _check(eng, s, n)


def test_round_that_frees_no_page_is_skipped():
    """Keeping 37 of 40 tokens per head packs into 10 blocks = 40 slots, so
    the offset would be 0 (packed cells read as a standard layout): skip."""
    n = 40
    eng = _engine(token_budget=37)
    s, pages = _prefill(eng, n)
    before = (eng.cell.clone(), eng.cpu_row.clone(), eng._top)
    assert eng.run_round("r") == []
    assert not eng.remapped("r") and eng.offset("r") == 0
    assert torch.equal(eng.cell, before[0]) and torch.equal(eng.cpu_row, before[1])
    assert eng._top == before[2] and eng.pages("r") == pages
    _check(eng, s, n)


@pytest.mark.parametrize("scope", ["global", "layer", "uniform", "shared"])
def test_applied_round_always_leaves_positive_offset(scope):
    for n in (9, 17, 31, 40, 63, 64, 65):
        for f in (0.0, 0.02, 0.1, 0.3, 0.7, 0.95):
            eng = _engine(scope=scope, offload_fraction=f)
            _prefill(eng, n, seed=n)
            freed = eng.run_round("r")
            if eng.remapped("r"):
                assert eng.offset("r") > 0, (n, f)
                assert len(eng.pages("r")) * P >= n - eng.offset("r")
            else:
                assert freed == [] and eng.offset("r") == 0


class _Blk:
    def __init__(self, i):
        self.block_id = i


def _fake_sched(eng, n_pages=24):
    """Minimal scheduler / KV-manager stand-ins for TokenModeScheduler."""
    from types import SimpleNamespace as N
    from offload_package.integration.token_mode import TokenModeScheduler
    freed = []
    stm = N(req_to_blocks={})
    kv = N(coordinator=N(single_type_managers=[stm]),
           block_pool=N(free_blocks=lambda bs: freed.extend(b.block_id for b in bs)))
    req = N(num_tokens=0, num_offloaded_tokens=0)
    sched = N(requests={"r": req}, cache_config=N(enable_prefix_caching=False))
    ts = TokenModeScheduler(interval_tokens=8, engine_getter=lambda: eng)
    ts.bind(scheduler=sched, kv_cache_manager=kv)
    return ts, stm, req, freed


def _step(ts, new=None, cached=None, n_sched=1, preempted=()):
    from types import SimpleNamespace as N
    so = N(preempted_req_ids=set(preempted), num_scheduled_tokens={"r": n_sched},
           scheduled_new_reqs=[new] if new else [],
           scheduled_cached_reqs=N(req_ids=[cached[0]] if cached else [],
                                   num_computed_tokens=[cached[1]] if cached else []))
    ts.build_connector_meta(so)


def test_token_scheduler_rounds_only_while_decoding():
    from types import SimpleNamespace as N
    eng = _engine()
    ts, stm, req, freed = _fake_sched(eng)
    prompt = 40
    stm.req_to_blocks["r"] = [_Blk(i) for i in range(3, 3 + 11)]   # 44 slots
    req.num_tokens = prompt
    _step(ts, new=N(req_id="r", prompt_len=prompt, num_computed_tokens=0), n_sched=prompt)
    s = eng.slot_of("r")
    _write(eng, s, 0, prompt)
    eng.score[:, s, :prompt] = torch.rand((L, prompt, H), device="cuda")
    # First decode step (num_computed == prompt): no round yet.
    req.num_tokens = prompt + 1
    _step(ts, cached=("r", prompt))
    assert not eng.remapped("r")
    _write(eng, s, prompt, prompt + 1)
    # Second decode step: the round fires and frees pages back to vLLM.
    req.num_tokens = prompt + 2
    _step(ts, cached=("r", prompt + 1))
    assert eng.remapped("r") and freed and req.num_offloaded_tokens == eng.offset("r") > 0
    assert [b.block_id for b in stm.req_to_blocks["r"]] == eng.pages("r")
    _check(eng, s, prompt + 1)
    # Preempted, then resumed: recompute-prefill of prompt + outputs in chunks.
    _step(ts, preempted=("r",), n_sched=0)
    assert not eng.has("r") and req.num_offloaded_tokens == 0
    stm.req_to_blocks["r"] = [_Blk(i) for i in range(3, 3 + 13)]   # 52 slots
    req.num_tokens = prompt + 10                                       # 49 to recompute
    _step(ts, cached=("r", 0), n_sched=24)
    _step(ts, cached=("r", 24), n_sched=18)
    # num_computed = 42 > prompt, but 7 tokens of the recompute remain.
    _step(ts, cached=("r", 42), n_sched=7)
    assert not eng.remapped("r"), "round fired mid recompute-prefill"
