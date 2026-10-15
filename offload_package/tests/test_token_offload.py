"""Unit tests for token-granular offloading: compactor, pool, round engine."""
from __future__ import annotations

import pytest
import torch

from offload_package.offload.compactor import (
    KVCompactor,
    resolve_token_addresses,
)
from offload_package.offload.token_pool import CPUTokenPool
from offload_package.integration.token_rounds import (
    TokenRoundEngine,
    TokenRequestState,
)
from offload_package.scoring.token_base import TokenScorer

PAGE_SIZE = 4
NUM_PAGES = 16
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4
COMBINED = 2 * HEAD_DIM


def make_combined_cache(device, num_pages=NUM_PAGES):
    """Per-layer combined KV caches: list of [num_pages, H, page, 2D]."""
    return [
        torch.zeros(
            num_pages,
            NUM_HEADS,
            PAGE_SIZE,
            COMBINED,
            dtype=torch.float32,
            device=device,
        )
        for _ in range(NUM_LAYERS)
    ]


def write_tokens(caches, block_ids, num_tokens, *, base=0.0):
    """Write deterministic K/V so each token's value encodes (layer, logical).

    K[..., :head_dim] = base + 1000*layer + logical; V half adds 0.5.
    """
    for layer, layer_cache in enumerate(caches):
        for t in range(num_tokens):
            block = block_ids[t // PAGE_SIZE]
            off = t % PAGE_SIZE
            value = base + 1000.0 * layer + t
            layer_cache[block, :, off, :HEAD_DIM] = value
            layer_cache[block, :, off, HEAD_DIM:] = value + 0.5


def token_value(layer, logical, base=0.0):
    return base + 1000.0 * layer + logical


class StubScorer(TokenScorer):
    """Content-driven scorer: score = transform(K[layer0, head0, dim0]).

    write_tokens() encodes each token's logical position into its K value,
    so this stub derives stable per-logical scores regardless of residency.
    """

    def __init__(self, transform=None):
        self.scored_layers = [0]
        self.transform = transform

    def score(self, k_by_layer, round_start):
        k = k_by_layer[0][:, 0, 0]
        if self.transform is not None:
            k = self.transform(k)
        return k.float()


def make_engine(
    device,
    cache,
    *,
    budget=10,
    window=4,
    sink=2,
    protect_prefill=True,
    num_slots=512,
    transform=None,
):
    compactor = KVCompactor(cache, PAGE_SIZE)
    pool = CPUTokenPool(
        num_layers=NUM_LAYERS,
        num_slots=num_slots,
        num_kv_heads=NUM_HEADS,
        combined_head_dim=COMBINED,
        dtype=torch.float32,
        device=device,
    )
    scorer = StubScorer(transform=transform)
    engine = TokenRoundEngine(
        scorer=scorer,
        compactor=compactor,
        token_pool=pool,
        page_size=PAGE_SIZE,
        token_budget=budget,
        window_size=window,
        sink_tokens=sink,
        protect_prefill=protect_prefill,
    )
    return engine, scorer


# ---------------------------------------------------------------------------
# resolve_token_addresses / KVCompactor
# ---------------------------------------------------------------------------


class TestResolveAddresses:

    def test_basic_mapping(self, torch_device):
        page_ids = [7, 3, 9]
        slots = torch.tensor([0, 5, 6, 11], dtype=torch.long)
        blocks, offsets = resolve_token_addresses(slots, page_ids, PAGE_SIZE)
        assert blocks.tolist() == [7, 3, 3, 9]
        assert offsets.tolist() == [0, 1, 2, 3]

    def test_empty(self, torch_device):
        slots = torch.tensor([], dtype=torch.long, device=torch_device)
        blocks, offsets = resolve_token_addresses(slots, [], PAGE_SIZE)
        assert blocks.numel() == 0 and offsets.numel() == 0

    def test_out_of_span_raises(self, torch_device):
        slots = torch.tensor([12], dtype=torch.long)
        with pytest.raises(IndexError, match="exceeds the request's page span"):
            resolve_token_addresses(slots, [1, 2], PAGE_SIZE)


class TestCompactor:

    def test_gather_k_and_kv(self, torch_device):
        cache = make_combined_cache(torch_device)
        block_ids = [5, 2]
        write_tokens(cache, block_ids, 8)
        compactor = KVCompactor(cache, PAGE_SIZE)
        blocks = torch.tensor([5, 5, 2, 2], dtype=torch.long)
        offsets = torch.tensor([0, 2, 1, 3], dtype=torch.long)
        k = compactor.gather_k(blocks, offsets)[0]
        assert k.shape == (4, NUM_HEADS, HEAD_DIM)
        # logical: (p5,o0)=0, (p5,o2)=2, (p2,o1)=5, (p2,o3)=7
        for row, logical in enumerate([0, 2, 5, 7]):
            assert torch.all(k[row] == token_value(0, logical)).item()
        kv = compactor.gather_kv(blocks, offsets)[0]
        assert kv.shape == (4, NUM_HEADS, COMBINED)
        assert torch.all(kv[..., HEAD_DIM:] == kv[..., :HEAD_DIM] + 0.5).item()

    def test_permute_overlap_safe(self, torch_device):
        cache = make_combined_cache(torch_device)
        block_ids = list(range(4))
        write_tokens(cache, block_ids, 16)
        compactor = KVCompactor(cache, PAGE_SIZE)

        # Shift survivors 0,1,3,4,5... (drop logical 2) to the prefix.
        keep = [0, 1, 3, 4, 5, 6, 7, 8]
        src_slots = torch.tensor(keep, dtype=torch.long)
        dst_slots = torch.arange(len(keep), dtype=torch.long)
        s_blocks, s_off = resolve_token_addresses(src_slots, block_ids, PAGE_SIZE)
        d_blocks, d_off = resolve_token_addresses(dst_slots, block_ids, PAGE_SIZE)
        compactor.permute(s_blocks, s_off, d_blocks, d_off)

        layer = cache[0]
        for new_slot, old_slot in enumerate(keep):
            expected = token_value(0, old_slot)
            block = block_ids[new_slot // PAGE_SIZE]
            got = layer[block, :, new_slot % PAGE_SIZE, :HEAD_DIM]
            assert torch.all(got == expected).item(), (new_slot, old_slot)

    def test_shape_validation(self, torch_device):
        cache = make_combined_cache(torch_device)
        with pytest.raises(ValueError, match="page_size mismatch"):
            KVCompactor(cache, PAGE_SIZE + 2)
        with pytest.raises(ValueError, match="at least one KV layer"):
            KVCompactor([], PAGE_SIZE)


# ---------------------------------------------------------------------------
# CPUTokenPool
# ---------------------------------------------------------------------------


class TestCPUTokenPool:

    def test_store_fetch_roundtrip(self, torch_device):
        cache = make_combined_cache(torch_device)
        block_ids = [3, 4]
        write_tokens(cache, block_ids, 7)
        pool = CPUTokenPool(
            num_layers=NUM_LAYERS,
            num_slots=64,
            num_kv_heads=NUM_HEADS,
            combined_head_dim=COMBINED,
            dtype=torch.float32,
            device=torch_device,
        )
        slots_phys = torch.tensor([1, 4, 6], dtype=torch.long)
        blocks, offsets = resolve_token_addresses(
            slots_phys, block_ids, PAGE_SIZE
        )
        cpu_slots = pool.store_from_gpu(cache, blocks, offsets)
        assert len(cpu_slots) == 3

        chunks, event = pool.fetch_to_gpu(cpu_slots)
        if event is not None:
            event.synchronize()
        assert len(chunks) == NUM_LAYERS
        for layer, chunk in enumerate(chunks):
            assert chunk.shape == (3, NUM_HEADS, COMBINED)
            for row, phys in enumerate(slots_phys.tolist()):
                expected_k = token_value(layer, phys)
                assert torch.all(chunk[row, :, :HEAD_DIM] == expected_k).item()
                assert torch.all(
                    chunk[row, :, HEAD_DIM:] == expected_k + 0.5
                ).item()

    def test_exhaustion_and_free(self, torch_device):
        pool = CPUTokenPool(
            num_layers=1,
            num_slots=4,
            num_kv_heads=NUM_HEADS,
            combined_head_dim=COMBINED,
            dtype=torch.float32,
            device=torch_device,
        )
        slots = pool.allocate(4)
        assert pool.num_free == 0
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.allocate(1)
        pool.free(slots[:2])
        assert pool.num_free == 2
        assert pool.allocate(2) == sorted(slots[:2], reverse=True)

    def test_fetch_after_immediate_store(self, torch_device):
        """A token stored this round must be fetchable without manual syncs."""
        cache = make_combined_cache(torch_device)
        write_tokens(cache[:1], [0], PAGE_SIZE)
        pool = CPUTokenPool(
            num_layers=1,
            num_slots=16,
            num_kv_heads=NUM_HEADS,
            combined_head_dim=COMBINED,
            dtype=torch.float32,
            device=torch_device,
        )
        blocks = torch.tensor([0, 0], dtype=torch.long)
        offsets = torch.tensor([1, 3], dtype=torch.long)
        cpu_slots = pool.store_from_gpu(cache[:1], blocks, offsets)
        chunks, event = pool.fetch_to_gpu(cpu_slots)
        if event is not None:
            event.synchronize()
        assert torch.all(
            chunks[0][0, :, :HEAD_DIM] == token_value(0, 1)
        ).item()
        assert torch.all(
            chunks[0][1, :, :HEAD_DIM] == token_value(0, 3)
        ).item()


# ---------------------------------------------------------------------------
# TokenRoundEngine
# ---------------------------------------------------------------------------


class TestTokenRoundEngine:

    def _setup(self, torch_device, num_tokens=24, *, transform=None, **engine_kwargs):
        """24 tokens over 6 pages; stub scores equal the logical position."""
        cache = make_combined_cache(torch_device)
        block_ids = list(range(6))
        write_tokens(cache, block_ids, num_tokens)
        engine, scorer = make_engine(
            torch_device, cache, transform=transform, **engine_kwargs
        )
        engine.add_request("r1", block_ids, num_tokens)
        return engine, cache, scorer, block_ids

    def test_round_budget_window_sink(self, torch_device):
        engine, cache, scorer, block_ids = self._setup(torch_device)
        engine.on_prefill_complete("r1", 0)  # no prefill protection
        # budget=10, sink=2, window=4 (defaults from make_engine)
        result = engine.on_round("r1")

        assert result is not None
        assert result.evicted_count == 14
        assert result.kept_count == 10
        # 24 tokens = 6 pages; 10 kept -> 3 pages; 3 freed.
        assert result.freed_block_ids == [3, 4, 5]
        state = engine.state("r1")
        assert state.block_ids == [0, 1, 2]
        assert state.physical_len == 10

        # Kept logical set: sinks 0,1 + window 20..23 + top scores 16..19.
        kept_logical = sorted(state.logical_of_physical[:10].tolist())
        assert kept_logical == [0, 1, 16, 17, 18, 19, 20, 21, 22, 23]

        # Physical prefix now holds kept tokens in logical order.
        for slot, logical in enumerate(kept_logical):
            block = state.block_ids[slot // PAGE_SIZE]
            for layer in range(NUM_LAYERS):
                got = cache[layer][block, :, slot % PAGE_SIZE, :HEAD_DIM]
                assert torch.all(got == token_value(layer, logical)).item()

        # Evicted tokens are recoverable from the CPU pool.
        evicted_logical = [t for t in range(24) if t not in kept_logical]
        cpu_slots = [
            state.cpu_slot_of_logical[t] for t in evicted_logical
        ]
        chunks, event = engine.token_pool.fetch_to_gpu(cpu_slots)
        if event is not None:
            event.synchronize()
        for row, logical in enumerate(evicted_logical):
            for layer, chunk in enumerate(chunks):
                assert torch.all(
                    chunk[row, :, :HEAD_DIM] == token_value(layer, logical)
                ).item()

    def test_prefill_protection_dominates(self, torch_device):
        engine, cache, scorer, block_ids = self._setup(torch_device)
        engine.on_prefill_complete("r1", 10)
        result = engine.on_round("r1")
        # Protected prefill (10) alone exceeds the budget of 10, so nothing
        # else survives.
        assert result.kept_count == 10
        state = engine.state("r1")
        kept_logical = sorted(state.logical_of_physical[:10].tolist())
        assert kept_logical == list(range(10))

    def test_window_larger_than_budget_keeps_newest(self, torch_device):
        engine, cache, scorer, block_ids = self._setup(
            torch_device, budget=6, window=8, sink=0
        )
        engine.on_prefill_complete("r1", 0)
        result = engine.on_round("r1")
        state = engine.state("r1")
        kept_logical = sorted(state.logical_of_physical[:6].tolist())
        # Window of 8 but only 6 budgeted: newest 6 logical tokens kept.
        assert kept_logical == [18, 19, 20, 21, 22, 23]

    def test_noop_when_under_budget(self, torch_device):
        engine, cache, scorer, block_ids = self._setup(
            torch_device, budget=100
        )
        engine.on_prefill_complete("r1", 0)
        assert engine.on_round("r1") is None
        assert engine.state("r1").physical_len == 24

    def test_tokens_written_after_round(self, torch_device):
        engine, cache, scorer, block_ids = self._setup(torch_device)
        engine.on_prefill_complete("r1", 0)
        engine.on_round("r1")
        state = engine.state("r1")
        assert state.physical_len == 10 and state.logical_len == 24

        # Simulate 2 new decode tokens written at the physical tail.
        new_blocks = [6, 7]
        engine.sync_blocks("r1", state.block_ids + new_blocks)
        for t in range(2):
            logical = 24 + t
            slot = 10 + t
            block = state.block_ids[slot // PAGE_SIZE]
            for layer in range(NUM_LAYERS):
                cache[layer][block, :, slot % PAGE_SIZE, :HEAD_DIM] = (
                    token_value(layer, logical)
                )
                cache[layer][block, :, slot % PAGE_SIZE, HEAD_DIM:] = (
                    token_value(layer, logical) + 0.5
                )
        engine.on_tokens_written("r1", 2)
        assert state.physical_len == 12
        assert state.logical_len == 26
        assert state.logical_of_physical[10].item() == 24
        assert state.logical_of_physical[11].item() == 25

        # Second round with the same budget: stub scores equal logical
        # positions, so the 2 lowest-scoring residents (16, 17) are evicted.
        result = engine.on_round("r1")
        assert result is not None
        assert result.evicted_count == 2
        kept_logical = sorted(state.logical_of_physical[:10].tolist())
        assert kept_logical == [0, 1, 18, 19, 20, 21, 22, 23, 24, 25]

    def test_remove_request_frees_pool(self, torch_device):
        engine, cache, scorer, block_ids = self._setup(torch_device)
        engine.on_prefill_complete("r1", 0)
        engine.on_round("r1")
        free_before = engine.token_pool.num_free
        engine.remove_request("r1")
        assert engine.token_pool.num_free == free_before + 14
        assert engine.state("r1") is None

    def test_pool_exhaustion_raises(self, torch_device):
        cache = make_combined_cache(torch_device)
        block_ids = list(range(6))
        write_tokens(cache, block_ids, 24)
        engine, scorer = make_engine(
            torch_device, cache, budget=2, window=0, sink=0, num_slots=4
        )
        engine.add_request("r1", block_ids, 24)
        engine.on_prefill_complete("r1", 0)
        with pytest.raises(RuntimeError, match="token pool exhausted"):
            engine.on_round("r1")

    def test_promotion_round_brings_cpu_tokens_back(self, torch_device):
        """Round 2 with a flipped scorer must promote previously evicted
        tokens back into the resident prefix and evict the ex-keepers."""
        engine, cache, scorer, block_ids = self._setup(torch_device)
        engine.on_prefill_complete("r1", 0)
        first = engine.on_round("r1")
        assert first is not None and first.evicted_count == 14
        state = engine.state("r1")
        assert len(state.cpu_slot_of_logical) == 14
        free_before = engine.token_pool.num_free

        # Flip the ranking: now the OLDEST tokens score highest.
        scorer.transform = lambda k: (24.0 - k)
        second = engine.on_round("r1")
        assert second is not None
        assert second.promoted_count == 4
        assert second.evicted_count == 4
        # Sinks + window + the four oldest mid tokens are resident now.
        kept_logical = sorted(state.logical_of_physical[:10].tolist())
        assert kept_logical == [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]
        assert len(state.cpu_slot_of_logical) == 14  # 14 - 4 + 4
        assert engine.token_pool.num_free == free_before  # 4 freed, 4 used

        # The promoted tokens' data must be back on GPU, byte-exact.
        for slot, logical in enumerate(kept_logical):
            block = state.block_ids[slot // PAGE_SIZE]
            for layer in range(NUM_LAYERS):
                got = cache[layer][block, :, slot % PAGE_SIZE, :HEAD_DIM]
                assert torch.all(got == token_value(layer, logical)).item()

        # Evicted ex-keepers must be recoverable from the pool.
        evicted = [t for t in range(24) if t not in kept_logical]
        cpu_slots = [state.cpu_slot_of_logical[t] for t in evicted]
        chunks, event = engine.token_pool.fetch_to_gpu(cpu_slots)
        if event is not None:
            event.synchronize()
        for row, logical in enumerate(evicted):
            for layer, chunk in enumerate(chunks):
                assert torch.all(
                    chunk[row, :, :HEAD_DIM] == token_value(layer, logical)
                ).item()

    def test_real_triattention_scorer_round(self, torch_device, tmp_path):
        """Smoke: the round pipeline runs end-to-end with the real scorer."""
        import sys
        from pathlib import Path

        tests_dir = Path(__file__).parent
        sys.path.insert(0, str(tests_dir))
        try:
            from test_triattention_scorer import (
                _make_q_stats,
                _write_tri_stats,
                N_HEADS as S_HEADS,
                HEAD_DIM as S_DIM,
            )
        finally:
            sys.path.pop(0)

        # Build a cache matching the scorer's stats geometry: the scorer
        # uses 1 layer, N_HEADS kv heads, HEAD_DIM head dim.
        cache = torch.zeros(
            8, S_HEADS, PAGE_SIZE, 2 * S_DIM, dtype=torch.float32,
            device=torch_device,
        )
        block_ids = list(range(6))
        g = torch.Generator().manual_seed(3)
        for t in range(24):
            block = block_ids[t // PAGE_SIZE]
            cache[block, :, t % PAGE_SIZE, :] = torch.randn(
                S_HEADS, 2 * S_DIM, generator=g
            )
        q = _make_q_stats(0)
        path = _write_tri_stats(tmp_path, q)
        from offload_package.scoring.triattention import TriAttentionTokenScorer

        scorer = TriAttentionTokenScorer(path, torch_device)
        compactor = KVCompactor([cache], PAGE_SIZE)
        pool = CPUTokenPool(
            num_layers=1,
            num_slots=256,
            num_kv_heads=S_HEADS,
            combined_head_dim=2 * S_DIM,
            dtype=torch.float32,
            device=torch_device,
        )
        engine = TokenRoundEngine(
            scorer=scorer,
            compactor=compactor,
            token_pool=pool,
            page_size=PAGE_SIZE,
            token_budget=8,
            window_size=4,
            sink_tokens=2,
            protect_prefill=False,
        )
        engine.add_request("r1", block_ids, 24)
        result = engine.on_round("r1")
        assert result is not None
        assert result.kept_count == 8
        state = engine.state("r1")
        # Sink tokens are always kept.
        kept_logical = sorted(state.logical_of_physical[:8].tolist())
        assert kept_logical[:2] == [0, 1]
        # Map stays strictly increasing (order-preserving repack).
        mapped = state.logical_of_physical[:8]
        assert bool((mapped[1:] > mapped[:-1]).all().item())
