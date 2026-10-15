"""Per-request token offload rounds.

A round runs every ``divide_length`` decode tokens (gated by the controller)
and implements the token-level offloading pipeline:

1. **Score** all GPU-resident tokens with the ``TokenScorer`` (v1: the
   TriAttention frequency scorer; offloaded tokens are not rescored until
   promotion lands).
2. **Protect** prefill/sink tokens outright; recent-window tokens get a
   +inf score but still consume budget (TriAttention semantics).
3. **Offload** the evicted tokens GPU -> CPU token pool (batched D2H).
4. **Compact** survivors into the dense physical prefix with an
   order-preserving permute (holes drift to the tail).
5. **Reclaim** the emptied tail pages so the scheduler can free them back
   to the vLLM block allocator.

After a round, physical slot ``i`` holds the ``i``-th surviving token in
logical order; ``logical_of_physical`` tracks which logical position each
physical slot holds, and new decode tokens keep appending at the physical
tail.  Everything the attention path needs (positions are baked into the
cached keys; the page-table span shrinks with the prefix) stays consistent
without touching the kernels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from ..offload.compactor import KVCompactor, resolve_token_addresses
from ..offload.token_pool import CPUTokenPool
from ..scoring.token_base import TokenScorer


class TokenCodeAdapter(Protocol):
    """Optional hooks for PQ-code maintenance under token offloading.

    The skylight PQ selector keeps per-request code rows indexed by position
    within the request's span.  Compaction permutes positions, offloaded
    tokens leave the span, and promotions re-enter it, so the code table
    needs the same moves as the KV data.  The adapter is implemented by the
    attention-side integration; ``None`` disables code maintenance (tests,
    oracle-family backends).
    """

    def on_evict(
        self, request_id: str, phys_slots: torch.Tensor, cpu_slots: list[int]
    ) -> None: ...

    def on_permute(
        self,
        request_id: str,
        src_phys: torch.Tensor,
        dst_phys: torch.Tensor,
    ) -> None: ...

    def on_promote(
        self, request_id: str, cpu_slots: list[int], dst_phys: torch.Tensor
    ) -> None: ...


@dataclass
class TokenRequestState:
    """Physical/logical bookkeeping for one request's token layout."""

    request_id: str
    block_ids: list[int]
    physical_len: int
    logical_len: int
    prefill_len: int = 0
    logical_of_physical: torch.Tensor | None = None  # int32 [capacity]
    cpu_slot_of_logical: dict[int, int] = field(default_factory=dict)
    # logical -> kernel-visible position: physical slot for resident tokens,
    # staging virtual position for fetched stragglers (>= pages_kept*page),
    # -1 for CPU-resident unfetched tokens.  Maintained at round boundaries
    # (resident scatter) and per step (staged scatter by the fetch path).
    virtual_of_logical: torch.Tensor | None = None  # int32 [capacity]


@dataclass
class TokenRoundResult:
    request_id: str
    evicted_count: int
    kept_count: int
    promoted_count: int
    freed_block_ids: list[int]


class TokenRoundEngine:
    """Orchestrates score -> offload -> compact -> reclaim per request."""

    def __init__(
        self,
        *,
        scorer: TokenScorer,
        compactor: KVCompactor,
        token_pool: CPUTokenPool,
        page_size: int,
        token_budget: int,
        window_size: int = 128,
        sink_tokens: int = 64,
        protect_prefill: bool = True,
        initial_capacity: int = 1024,
        code_adapter: TokenCodeAdapter | None = None,
        score_cpu_tokens: bool = True,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        self.scorer = scorer
        self.compactor = compactor
        self.token_pool = token_pool
        self.page_size = int(page_size)
        self.token_budget = int(token_budget)
        self.window_size = max(0, int(window_size))
        self.sink_tokens = max(0, int(sink_tokens))
        self.protect_prefill = bool(protect_prefill)
        self.code_adapter = code_adapter
        self.score_cpu_tokens = bool(score_cpu_tokens)
        self.device = compactor.device
        self._initial_capacity = max(int(initial_capacity), self.page_size)
        self._states: dict[str, TokenRequestState] = {}
        # Cumulative counters (survive request cleanup; read by benchmarks).
        self.stats = {
            "rounds": 0,
            "evicted": 0,
            "promoted": 0,
            "freed_pages": 0,
        }
        # Optional straggler-fetch coordinator (installed by the worker
        # connector in token mode): owns packed staging slots and per-step
        # remaps.  The engine notifies it of promotions/removals.
        self.straggler = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add_request(
        self, request_id: str, block_ids: list[int], num_tokens: int
    ) -> TokenRequestState:
        if request_id in self._states:
            raise ValueError(f"request {request_id} already tracked")
        if num_tokens > len(block_ids) * self.page_size:
            raise ValueError(
                f"{num_tokens} tokens exceed the page span "
                f"({len(block_ids)} pages x {self.page_size})"
            )
        capacity = max(
            self._initial_capacity,
            num_tokens * 2,
            len(block_ids) * self.page_size,
        )
        token_map = torch.arange(
            capacity, dtype=torch.int32, device=self.device
        )
        state = TokenRequestState(
            request_id=request_id,
            block_ids=list(block_ids),
            physical_len=int(num_tokens),
            logical_len=int(num_tokens),
            logical_of_physical=token_map,
            virtual_of_logical=token_map.clone(),
        )
        self._states[request_id] = state
        return state

    def remove_request(self, request_id: str) -> TokenRequestState | None:
        state = self._states.pop(request_id, None)
        if state is not None:
            if state.cpu_slot_of_logical:
                self.token_pool.free(list(state.cpu_slot_of_logical.values()))
            if self.straggler is not None:
                self.straggler.release_request(request_id)
        return state

    def on_prefill_complete(self, request_id: str, prefill_len: int) -> None:
        state = self._states.get(request_id)
        if state is not None:
            state.prefill_len = int(prefill_len)

    def sync_blocks(self, request_id: str, block_ids: list[int]) -> None:
        """Adopt a (possibly extended) page list for the request."""
        state = self._states.get(request_id)
        if state is None:
            return
        needed = (state.physical_len + self.page_size - 1) // self.page_size
        if len(block_ids) < needed:
            raise ValueError(
                f"sync_blocks for {request_id}: {len(block_ids)} pages cannot "
                f"hold {state.physical_len} tokens"
            )
        state.block_ids = list(block_ids)

    def on_tokens_written(self, request_id: str, count: int) -> None:
        """Record ``count`` newly written tokens at the physical tail."""
        if count <= 0:
            return
        state = self._states.get(request_id)
        if state is None:
            return
        token_map = state.logical_of_physical
        virt_map = state.virtual_of_logical
        assert token_map is not None
        new_physical = state.physical_len + count
        new_logical = state.logical_len + count
        if new_physical > token_map.numel():
            token_map = self._grow_map(token_map, new_physical)
            state.logical_of_physical = token_map
            virt_map = token_map.new_full(
                token_map.shape, -1, dtype=torch.int32
            )
            # Rebuild resident entries from the (possibly compacted) map.
            virt_map.scatter_(0, token_map[: state.physical_len].long(),
                              torch.arange(
                                  state.physical_len, dtype=torch.int32,
                                  device=token_map.device))
            state.virtual_of_logical = virt_map
        base_logical = state.logical_len
        tail = torch.arange(
            base_logical,
            base_logical + count,
            dtype=torch.int32,
            device=token_map.device,
        )
        token_map[state.physical_len : new_physical] = tail
        if virt_map is not None:
            virt_map[state.physical_len : new_physical] = tail
        state.physical_len = new_physical
        state.logical_len = new_logical
        needed = (new_physical + self.page_size - 1) // self.page_size
        if len(state.block_ids) < needed:
            raise ValueError(
                f"request {request_id}: {new_physical} physical tokens "
                f"exceed its page span ({len(state.block_ids)} pages)"
            )

    # ------------------------------------------------------------------
    # Accessors (for the attention-side integration)
    # ------------------------------------------------------------------

    def state(self, request_id: str) -> TokenRequestState | None:
        return self._states.get(request_id)

    @property
    def states(self) -> dict[str, TokenRequestState]:
        return self._states

    def cpu_slot_of(self, request_id: str, logical_pos: int) -> int | None:
        state = self._states.get(request_id)
        if state is None:
            return None
        return state.cpu_slot_of_logical.get(int(logical_pos))

    # ------------------------------------------------------------------
    # Round
    # ------------------------------------------------------------------

    def on_round(self, request_id: str) -> TokenRoundResult | None:
        """Run one offload round. Returns None when nothing changes.

        Re-scores every token -- GPU-resident and CPU-offloaded alike -- and
        moves the request toward the target residency: the global top
        ``token_budget`` logical tokens live on GPU (in logical order in the
        dense physical prefix), everything else lives in the CPU token pool.
        Transfers at a round boundary are: D2H for newly evicted tokens, H2D
        for promoted (re-entering) tokens, plus the in-GPU compaction permute.
        """
        state = self._states.get(request_id)
        if state is None:
            return None
        phys_len = state.physical_len
        logical_len = state.logical_len
        if logical_len == 0 or not state.block_ids:
            return None

        keep_logical = self._select_keep(state)
        if keep_logical is None:
            return None
        keep_count = int(keep_logical.numel())
        # Skip when the keep set equals the current residency (same tokens,
        # same order): no permute, no transfers.
        token_map = state.logical_of_physical
        assert token_map is not None
        resident_logical = token_map[:phys_len].to(torch.long)
        if keep_count == phys_len and torch.equal(
            keep_logical.to(self.device), resident_logical
        ):
            return None

        keep_logical = keep_logical.to(self.device)

        # Classify keep-set members by current residency. The old physical
        # order is strictly increasing in logical position, so searchsorted
        # recovers each stayer's old physical slot.
        resident_sorted = resident_logical
        stay_pos = torch.searchsorted(
            resident_sorted, keep_logical
        ).clamp(max=phys_len)
        stay_hit = (
            stay_pos < phys_len
        ) & (resident_sorted[stay_pos.clamp(max=phys_len - 1)] == keep_logical)
        # Promotees: keep-set tokens that currently live in the CPU pool.
        cpu_resident = {
            int(logical): slot
            for logical, slot in state.cpu_slot_of_logical.items()
            if logical < logical_len
        }
        promote_mask = ~stay_hit
        for i in range(keep_count):
            if bool(stay_hit[i]):
                continue
            if int(keep_logical[i]) not in cpu_resident:
                raise RuntimeError(
                    f"keep-set token {int(keep_logical[i])} is neither "
                    "GPU-resident nor in the CPU pool"
                )

        # 1) Offload newly evicted GPU tokens while they are still in place.
        keep_set = set(keep_logical.tolist())
        evict_logical = [
            int(logical)
            for logical in resident_logical.tolist()
            if logical not in keep_set
        ]
        evicted_count = 0
        if evict_logical:
            evict_pos = torch.searchsorted(
                resident_sorted,
                torch.tensor(evict_logical, dtype=torch.long, device=self.device),
            )
            e_blocks, e_offsets = resolve_token_addresses(
                evict_pos, state.block_ids, self.page_size
            )
            cpu_slots = self.token_pool.store_from_gpu(
                self.compactor.kv_caches, e_blocks, e_offsets
            )
            for logical, slot in zip(evict_logical, cpu_slots):
                state.cpu_slot_of_logical[int(logical)] = int(slot)
            evicted_count = len(evict_logical)
            if self.code_adapter is not None:
                self.code_adapter.on_evict(request_id, evict_pos, cpu_slots)

        # 2) Fetch promoted tokens CPU -> GPU.
        promote_logical = [
            int(keep_logical[i]) for i in range(keep_count) if bool(promote_mask[i])
        ]
        promote_slots = [cpu_resident[int(l)] for l in promote_logical]
        promoted_count = len(promote_logical)
        promote_chunks: list[torch.Tensor] | None = None
        if promoted_count:
            promote_chunks, _ = self.token_pool.fetch_to_gpu(promote_slots)
            for logical in promote_logical:
                state.cpu_slot_of_logical.pop(int(logical), None)
            self.token_pool.free(promote_slots)

        # 3) Build the target layout: keep set in logical order at the
        #    prefix. Stayers permute from their old slots; promotees scatter
        #    from the fetched chunks into their target slots.
        dst_phys = torch.arange(
            keep_count, dtype=torch.long, device=self.device
        )
        d_blocks, d_offsets = resolve_token_addresses(
            dst_phys, state.block_ids, self.page_size
        )

        stay_dst = dst_phys[stay_hit]
        stay_src = stay_pos[stay_hit]
        if int(stay_dst.numel()) > 0:
            s_blocks, s_offsets = resolve_token_addresses(
                stay_src, state.block_ids, self.page_size
            )
            self.compactor.permute(s_blocks, s_offsets, d_blocks[stay_hit], d_offsets[stay_hit])
            if self.code_adapter is not None:
                self.code_adapter.on_permute(request_id, stay_src, stay_dst)

        if promote_chunks is not None:
            promote_dst = dst_phys[~stay_hit]
            self.compactor.scatter_kv(
                d_blocks[~stay_hit], d_offsets[~stay_hit], promote_chunks
            )
            if self.code_adapter is not None:
                self.code_adapter.on_promote(request_id, promote_slots, promote_dst)

        # 4) Update the token map, the virtual map, and the page span.
        token_map[:keep_count] = keep_logical.to(torch.int32)
        virt_map = state.virtual_of_logical
        if virt_map is not None:
            # Clear the logical range, then scatter residents; staged
            # stragglers re-enter via the per-step fetch path's scatter.
            virt_map[:logical_len].fill_(-1)
            virt_map.scatter_(
                0,
                keep_logical,
                torch.arange(
                    keep_count, dtype=torch.int32, device=self.device
                ),
            )
        state.physical_len = keep_count
        pages_needed = (keep_count + self.page_size - 1) // self.page_size
        freed_block_ids = state.block_ids[pages_needed:]
        state.block_ids = state.block_ids[:pages_needed]

        # Promoted tokens' staging copies are superseded by the resident
        # prefix; release their packed staging slots.
        if promote_logical and self.straggler is not None:
            self.straggler.release_tokens(request_id, promote_logical)

        if evicted_count == 0 and promoted_count == 0 and keep_count == phys_len:
            return None
        self.stats["rounds"] += 1
        self.stats["evicted"] += evicted_count
        self.stats["promoted"] += promoted_count
        self.stats["freed_pages"] += len(freed_block_ids)
        return TokenRoundResult(
            request_id=request_id,
            evicted_count=evicted_count,
            kept_count=keep_count,
            promoted_count=promoted_count,
            freed_block_ids=freed_block_ids,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_keep(
        self, state: TokenRequestState
    ) -> torch.Tensor | None:
        """Sorted logical positions of the keep set, or None when unchanged.

        Scores every logical token -- GPU-resident (gathered from pages) and
        CPU-offloaded (gathered from the token pool) -- applies protections
        (prefill/sink hard-protected; recent window kept newest-first within
        the budget), and keeps the top ``token_budget`` tokens overall so the
        GPU always holds the global heavy hitters.
        """
        phys_len = state.physical_len
        logical_len = state.logical_len
        token_map = state.logical_of_physical
        assert token_map is not None

        gpu_scores, gpu_logical = self._score_resident(state)

        cpu_items = sorted(state.cpu_slot_of_logical.items())
        if self.score_cpu_tokens and cpu_items:
            cpu_logical = torch.tensor(
                [logical for logical, _ in cpu_items],
                dtype=torch.long,
            )
            cpu_slots = [slot for _, slot in cpu_items]
            cpu_scores = self._score_cpu(state, cpu_slots)
        else:
            cpu_logical = torch.zeros(0, dtype=torch.long)
            cpu_scores = torch.zeros(0, dtype=torch.float32)

        all_logical = torch.cat([gpu_logical.cpu(), cpu_logical])
        all_scores = torch.cat([gpu_scores.cpu(), cpu_scores])
        if all_logical.numel() == 0:
            return None

        # De-duplicate (a token must never be both resident and pooled).
        order = torch.argsort(all_logical)
        all_logical = all_logical[order]
        all_scores = all_scores[order]
        if all_logical.numel() > 1:
            dup = all_logical[1:] == all_logical[:-1]
            if bool(dup.any()):
                raise RuntimeError(
                    "token scored as both GPU-resident and CPU-offloaded"
                )

        protected = torch.zeros_like(all_logical, dtype=torch.bool)
        if self.protect_prefill and state.prefill_len > 0:
            protected |= all_logical < state.prefill_len
        if self.sink_tokens > 0:
            protected |= all_logical < self.sink_tokens

        n_hard = int(protected.sum())
        budget_unprotected = max(0, self.token_budget - n_hard)
        n_total = int(all_logical.numel())

        keep_mask = protected.clone()
        if budget_unprotected > 0 and n_total > n_hard:
            cand_logical = all_logical[~protected]
            cand_scores = all_scores[~protected]
            keep_local = torch.zeros(
                cand_logical.numel(), dtype=torch.bool
            )
            if self.window_size > 0:
                window_mask = cand_logical >= logical_len - self.window_size
                window_idx = window_mask.nonzero(as_tuple=False).squeeze(-1)
                if window_idx.numel() > 0:
                    window_take = min(budget_unprotected, window_idx.numel())
                    order_w = torch.argsort(
                        cand_logical[window_idx], descending=True
                    )
                    keep_local[window_idx[order_w[:window_take]]] = True
            remaining = budget_unprotected - int(keep_local.sum())
            if remaining > 0:
                rest_scores = cand_scores.masked_fill(
                    keep_local, float("-inf")
                )
                take = min(remaining, int(cand_logical.numel()))
                top = torch.topk(rest_scores, take).indices
                keep_local[top] = True
            keep_mask[~protected] |= keep_local

        keep_logical = all_logical[keep_mask]
        # Sanity: protections alone may exceed the budget; that is allowed
        # (hard-protected tokens are never evicted), so keep_mask can hold
        # more than token_budget entries.
        if int(keep_logical.numel()) >= logical_len:
            # Everything kept and already resident in order: nothing to do.
            if phys_len == logical_len:
                return None
        return keep_logical

    def _score_resident(
        self, state: TokenRequestState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every GPU-resident token with the TokenScorer.

        Returns (scores [phys_len], logical positions [phys_len]) in the
        current physical order.
        """
        phys_len = state.physical_len
        phys_slots = torch.arange(
            phys_len, dtype=torch.long, device=self.device
        )
        blocks, offsets = resolve_token_addresses(
            phys_slots, state.block_ids, self.page_size
        )
        head_dim = self.compactor.head_dim
        k_by_layer: dict[int, torch.Tensor] = {}
        num_caches = len(self.compactor.kv_caches)
        for layer in self.scorer.scored_layers:
            if 0 <= layer < num_caches:
                cache = self.compactor.kv_caches[layer]
                k_by_layer[layer] = cache[blocks, :, offsets, :head_dim]
        scores = self.scorer.score(k_by_layer, round_start=state.logical_len)
        if scores.numel() != phys_len:
            raise RuntimeError(
                f"scorer returned {scores.numel()} scores for "
                f"{phys_len} tokens"
            )
        token_map = state.logical_of_physical
        assert token_map is not None
        logical = token_map[:phys_len].to(torch.long)
        return scores.to(dtype=torch.float32), logical

    def _score_cpu(
        self, state: TokenRequestState, cpu_slots: list[int]
    ) -> torch.Tensor:
        """Score CPU-offloaded tokens directly from the pinned token pool."""
        head_dim = self.compactor.head_dim
        k_by_layer: dict[int, torch.Tensor] = {}
        slot_ids = torch.tensor(cpu_slots, dtype=torch.long)
        num_layers = self.token_pool.num_layers
        for layer in self.scorer.scored_layers:
            if 0 <= layer < num_layers:
                k_by_layer[layer] = self.token_pool.pool[layer].index_select(
                    0, slot_ids
                )[..., :head_dim]
        scores = self.scorer.score(k_by_layer, round_start=state.logical_len)
        if scores.numel() != len(cpu_slots):
            raise RuntimeError(
                f"scorer returned {scores.numel()} scores for "
                f"{len(cpu_slots)} CPU tokens"
            )
        return scores.to(dtype=torch.float32)

    def _grow_map(
        self, old_map: torch.Tensor, min_capacity: int
    ) -> torch.Tensor:
        new_capacity = max(min_capacity, old_map.numel() * 2)
        new_map = torch.empty(
            new_capacity, dtype=torch.int32, device=old_map.device
        )
        new_map[: old_map.numel()] = old_map
        return new_map
