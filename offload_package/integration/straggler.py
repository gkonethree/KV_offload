"""Per-step straggler fetch: packed staging + logical->virtual remap.

Token mode keeps the geometric top-``token_budget`` tokens GPU-resident
(round-level promotion), but the per-step PQ top-k is query-dependent and
may select tokens that live in the CPU token pool.  This module makes those
stragglers attendable the same step:

1. Selection runs over the LOGICAL-keyed code table, producing sparse
   indices in logical-position space.
2. Each request's ``virtual_of_logical`` map translates them: physical slot
   for residents, staging virtual position for already-staged stragglers,
   -1 for tokens that must be fetched.
3. Missing entries are placed into packed ``(staging_page, offset)`` slots
   (LRU-reused across steps), fetched from the pinned CPU token pool in one
   batched H2D, and scattered into the staging pages.
4. The wrapper's flat page-table rows are extended with each request's
   staging pages so the decode kernel resolves the remapped positions
   through the unchanged page-table contract.

When the fetch set exceeds staging capacity, the caller falls back to the
chunked LSE-merge decode (exact online-softmax decomposition); this module
exposes per-chunk place/fetch/release for that path.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from ..offload.compactor import KVCompactor
from ..offload.token_pool import CPUTokenPool
from .token_rounds import TokenRoundEngine


@dataclass
class SelectionPlan:
    """Result of remapping one layer's selection."""

    sparse_idx: torch.Tensor          # [B, Hq, K] kernel positions
    sparse_idx_logical: torch.Tensor   # [B, Hq, K] logical (for masks)
    indptr: torch.Tensor              # extended row offsets (int32, GPU)
    indices: torch.Tensor             # extended flat page ids (int32, GPU)
    fetch_missing: list[tuple[str, int, int]]  # (req, logical, cpu_slot)
    staged_any: bool                   # any request has staged tokens
    capacity: int                      # total staging token slots


class TokenStagingPool:
    """Packed ``(page, offset)`` staging slots with per-token LRU.

    A slot id is ``sp_local * page_size + offset`` into the reserved
    staging tail pages.  A request's staging pages are tracked in
    insertion order so each page's row index (and therefore each staged
    token's virtual position) is stable while it stays staged.
    """

    def __init__(
        self, *, staging_base_page: int, num_pages: int, page_size: int
    ) -> None:
        if num_pages <= 0 or page_size <= 0:
            raise ValueError("num_pages and page_size must be > 0")
        self.staging_base_page = int(staging_base_page)
        self.num_pages = int(num_pages)
        self.page_size = int(page_size)
        self.total_slots = self.num_pages * self.page_size
        self._free = list(range(self.total_slots - 1, -1, -1))
        self._slot_of: dict[tuple[str, int], int] = {}
        self._owner: dict[int, tuple[str, int]] = {}
        # req -> insertion-ordered staging page list (sp_local).
        self._pages_of: dict[str, dict[int, None]] = {}
        # (req, sp_local) -> live slot count in that page.
        self._page_occ: dict[tuple[str, int], int] = {}
        self._lru: OrderedDict[tuple[str, int], None] = OrderedDict()

    @property
    def num_staged(self) -> int:
        return len(self._slot_of)

    @property
    def num_free(self) -> int:
        return len(self._free)

    def slot_of(self, req: str, logical: int, *, touch: bool = True) -> int | None:
        key = (req, int(logical))
        slot = self._slot_of.get(key)
        if slot is not None and touch:
            self._lru.move_to_end(key)
        return slot

    def pages_of(self, req: str) -> list[int]:
        return list(self._pages_of.get(req, ()))

    def _claim_slot(self, req: str, logical: int) -> int:
        slot = self._free.pop()
        key = (req, int(logical))
        self._slot_of[key] = slot
        self._owner[slot] = key
        self._lru[key] = None
        sp_local = slot // self.page_size
        pages = self._pages_of.setdefault(req, {})
        pages.setdefault(sp_local, None)
        occ_key = (req, sp_local)
        self._page_occ[occ_key] = self._page_occ.get(occ_key, 0) + 1
        return slot

    def place_many(
        self,
        entries: list[tuple[str, int]],
        protect: set[tuple[str, int]],
    ) -> tuple[dict[tuple[str, int], int], list[tuple[str, int]]]:
        """Place tokens, LRU-evicting unprotected staged tokens when full.

        Returns (placed: key -> slot, unplaced).  ``unplaced`` is non-empty
        only when even evicting every unprotected staged token cannot make
        room -- the caller must chunk those.
        """
        placed: dict[tuple[str, int], int] = {}
        unplaced: list[tuple[str, int]] = []
        for key in entries:
            existing = self._slot_of.get(key)
            if existing is not None:
                self._lru.move_to_end(key)
                placed[key] = existing
                continue
            if not self._free:
                evicted = False
                for victim in list(self._lru.keys()):
                    if victim in protect or victim in placed:
                        continue
                    self._release_key(victim)
                    evicted = True
                    break
                if not evicted:
                    unplaced.append(key)
                    continue
            placed[key] = self._claim_slot(*key)
        return placed, unplaced

    def release_tokens(self, req: str, logicals: list[int]) -> None:
        for logical in logicals:
            key = (req, int(logical))
            if key in self._slot_of:
                self._release_key(key)

    def release_request(self, req: str) -> None:
        for key in [k for k in self._slot_of if k[0] == req]:
            self._release_key(key)

    def _release_key(self, key: tuple[str, int]) -> None:
        slot = self._slot_of.pop(key)
        self._owner.pop(slot, None)
        self._lru.pop(key, None)
        self._free.append(slot)
        req = key[0]
        sp_local = slot // self.page_size
        occ_key = (req, sp_local)
        remaining = self._page_occ.get(occ_key, 0) - 1
        if remaining > 0:
            self._page_occ[occ_key] = remaining
            return
        # Last slot of this page: drop the page from the request's list.
        self._page_occ.pop(occ_key, None)
        pages = self._pages_of.get(req)
        if pages is not None:
            pages.pop(sp_local, None)
            if not pages:
                self._pages_of.pop(req, None)


class StragglerFetchCoordinator:
    """Per-step glue between the token engine and the PQ wrapper."""

    def __init__(
        self,
        *,
        engine: TokenRoundEngine,
        token_pool: CPUTokenPool,
        compactor: KVCompactor,
        staging_pool: TokenStagingPool,
        page_size: int,
    ) -> None:
        self.engine = engine
        self.token_pool = token_pool
        self.compactor = compactor
        self.staging = staging_pool
        self.page_size = int(page_size)
        self._step_serial_seen = -1
        self._request_ids: list[str] = []
        # Per-step placements (protected from LRU eviction within the step).
        self._step_placed: set[tuple[str, int]] = set()
        self._ext_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Cumulative counters (survive request cleanup; read by benchmarks).
        self.stats = {
            "fetch_batches": 0,       # H2D batches issued
            "fetched_tokens": 0,      # tokens fetched CPU -> staging
            "chunked_steps": 0,       # decode steps that fell back chunked
            "chunk_groups": 0,        # per-chunk fetch+decode passes
        }

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def begin_step(
        self, step_serial: int, request_ids: list[str] | None
    ) -> None:
        """Per-step prep: adopt ids and re-scatter staged virtuals.

        A request's physical page count changes as decode appends pages, so
        every staged token's virtual position (``(n_phys_pages + j) * page
        + offset``) is recomputed from the staging pool's records.
        Idempotent within a step (layers call it redundantly).
        """
        if step_serial == self._step_serial_seen:
            return
        self._step_serial_seen = int(step_serial)
        self._request_ids = [str(r) for r in (request_ids or [])]
        self._step_placed = set()
        self._rescatter_staged_virtuals()

    def _rescatter_staged_virtuals(self) -> None:
        for req in self.staging._pages_of:
            state = self.engine.state(req)
            if state is None:
                continue
            self._scatter_virtuals(req, state, keys_to_touch=False)

    # ------------------------------------------------------------------
    # Selection remap + fetch
    # ------------------------------------------------------------------

    def apply_selection(
        self,
        sparse_idx_logical: torch.Tensor,
        request_ids: list[str],
        orig_indptr: torch.Tensor,
        orig_indices: torch.Tensor,
    ) -> SelectionPlan:
        """Remap one layer's logical selection; fetch what is missing.

        Fast path: every missing token places and fetches; the plan's
        ``fetch_missing`` is then empty and a single decode attends all
        selected tokens.  Overflow: ``fetch_missing`` holds the entries
        that could not be staged even after full LRU eviction -- the caller
        decodes those in capacity-sized chunks (base pass masks them out).
        """
        ids = [str(r) for r in request_ids]
        device = sparse_idx_logical.device
        states = [self.engine.state(r) for r in ids]

        virt_rows = []
        missing: list[tuple[str, int, int]] = []
        any_staged = False
        protect: set[tuple[str, int]] = set(self._step_placed)
        for b, state in enumerate(states):
            row = sparse_idx_logical[b]
            if state is None or state.virtual_of_logical is None:
                virt_rows.append(row.to(torch.int32))
                continue
            vol = state.virtual_of_logical
            # Current-step (in-flight) tokens: logical positions in
            # [logical_len, selection_len) were just written at the
            # physical tail [physical_len, physical_len + n) by the
            # runner's slot rewrite; they are always resident.  Positions
            # below logical_len come from the virtual map.  The vol read is
            # clamped for safety; the in-flight side MUST use the
            # unclamped position (the newest token can sit exactly at the
            # map's capacity).
            inflight = state.physical_len + (row - state.logical_len)
            virt = torch.where(
                row < state.logical_len,
                vol[row.clamp(max=vol.numel() - 1)],
                inflight.to(torch.int32),
            )
            n_phys_pages = (
                (state.physical_len + self.page_size - 1) // self.page_size
            )
            if bool((virt >= n_phys_pages * self.page_size).any()):
                any_staged = True
                # Touch LRU and protect: a token staged (possibly an earlier
                # step) and selected by this layer must not be evicted to
                # make room -- its virtual position is baked into this
                # layer's remapped indices.
                for logical in self._touch_staged(state, row, virt, n_phys_pages):
                    protect.add((state.request_id, int(logical)))
            if bool((virt < 0).any()):
                miss = row[(virt < 0)].unique().tolist()
                for logical in miss:
                    cpu_slot = state.cpu_slot_of_logical.get(int(logical))
                    if cpu_slot is None:
                        import os as _os
                        if _os.environ.get("SKYLIGHT_PQ_STRAGGLER_DBG") == "1":
                            print(
                                f"[STRAGGLER DBG] req={ids[b]} logical={logical} "
                                f"state.logical={state.logical_len} "
                                f"physical={state.physical_len} "
                                f"cpu_tokens={len(state.cpu_slot_of_logical)} "
                                f"blocks={len(state.block_ids)} "
                                f"row_len={int(row.shape[-1]) if row.dim() else 1} "
                                f"sel_len={int(sparse_idx_logical.shape[-1])} ",
                                flush=True,
                            )
                        raise RuntimeError(
                            f"selected token {logical} of {ids[b]} is "
                            "neither resident nor in the CPU pool"
                        )
                    missing.append((ids[b], int(logical), int(cpu_slot)))
                    protect.add((ids[b], int(logical)))
            virt_rows.append(virt)

        # Place + fetch the missing set (LRU reuse keeps repeated selections
        # staged).  Overflow: recycle the partial placements and hand the
        # WHOLE missing set to the chunked path -- its base pass masks every
        # unfetched token and each chunk re-places from the same capacity,
        # exactly like the block-mode chunked fallback.
        entries = [(req, logical) for req, logical, _ in missing]
        placed, unplaced = self.staging.place_many(entries, protect)
        if placed and unplaced:
            for (req, logical) in list(placed):
                self.staging.release_tokens(req, [logical])
                self._step_placed.discard((req, logical))
                st = self.engine.state(req)
                if st is not None and st.virtual_of_logical is not None:
                    st.virtual_of_logical[int(logical)] = -1
            placed = {}
        if placed:
            self._fetch_and_scatter(placed, states, ids)
            self._step_placed.update(placed)
            any_staged = True
            self.stats["fetch_batches"] += 1
            self.stats["fetched_tokens"] += len(placed)
            fetch_missing: list[tuple[str, int, int]] = []
        else:
            fetch_missing = missing
            if fetch_missing:
                self.stats["chunked_steps"] += 1

        # Kernel positions: remap via the (now-updated) virtual maps;
        # unfetched entries clamp to position 0 (their weights are zeroed
        # by the chunked base pass; the read is harmless).  In-flight
        # positions again resolve to the physical tail arithmetically.
        kernel_rows = []
        for b, state in enumerate(states):
            row = sparse_idx_logical[b]
            if state is None or state.virtual_of_logical is None:
                kernel_rows.append(row.to(torch.int32))
            else:
                vol = state.virtual_of_logical
                inflight = state.physical_len + (
                    row - state.logical_len
                )
                virt = torch.where(
                    row < state.logical_len,
                    vol[row.clamp(max=vol.numel() - 1)],
                    inflight.to(torch.int32),
                )
                kernel_rows.append(virt.clamp_(min=0))
        sparse_idx_kernel = torch.stack(kernel_rows, dim=0)

        need_ext = any_staged or bool(placed)
        if need_ext:
            indptr, indices = self._build_ext_rows(
                ids, orig_indptr, orig_indices, device
            )
        else:
            indptr, indices = orig_indptr, orig_indices

        return SelectionPlan(
            sparse_idx=sparse_idx_kernel,
            sparse_idx_logical=sparse_idx_logical,
            indptr=indptr,
            indices=indices,
            fetch_missing=fetch_missing,
            staged_any=need_ext,
            capacity=self.staging.total_slots,
        )

    def fetch_chunk(
        self,
        entries: list[tuple[str, int, int]],
    ) -> list[tuple[str, int]]:
        """Place + fetch one chunk of overflow entries (chunked path).

        Slots for a chunk are released again via :meth:`release_chunk` so
        consecutive chunks reuse the same staging capacity.  Returns the
        entries that could not be placed (caller must fail loud).
        """
        ids_seen = {r for r, _, _ in entries}
        states = {r: self.engine.state(r) for r in ids_seen}
        keys = [(req, logical) for req, logical, _ in entries]
        protect = set(self._step_placed) | set(keys)
        placed, unplaced = self.staging.place_many(keys, protect)
        if placed:
            self._fetch_and_scatter(placed, states, ids_seen)
            self._step_placed.update(placed)
            self.stats["fetch_batches"] += 1
            self.stats["fetched_tokens"] += len(placed)
            self.stats["chunk_groups"] += 1
        return unplaced

    def chunk_group_size(self) -> int:
        """Staging slots available to one chunk (base placements protected)."""
        return max(1, self.staging.total_slots - len(self._step_placed))

    def remap_kernel_positions(
        self,
        sparse_idx_logical: torch.Tensor,
        request_ids: list[str],
    ) -> torch.Tensor:
        """Re-gather kernel positions after a chunk placement updated vols."""
        ids = [str(r) for r in request_ids]
        rows = []
        for b in range(sparse_idx_logical.shape[0]):
            if b < len(ids):
                state = self.engine.state(ids[b])
                if state is not None and state.virtual_of_logical is not None:
                    vol = state.virtual_of_logical
                    row = sparse_idx_logical[b]
                    inflight = state.physical_len + (
                        row - state.logical_len
                    )
                    virt = torch.where(
                        row < state.logical_len,
                        vol[row.clamp(max=vol.numel() - 1)],
                        inflight.to(torch.int32),
                    )
                    rows.append(virt.clamp_(min=0))
                    continue
            rows.append(sparse_idx_logical[b].to(torch.int32))
        return torch.stack(rows, dim=0)

    def release_request(self, req: str) -> None:
        """Release every staged token of a finished/removed request."""
        self.staging.release_request(req)
        self._step_placed = {
            key for key in self._step_placed if key[0] != req
        }

    def release_tokens(self, req: str, logicals: list[int]) -> None:
        """Release staged copies of promoted tokens.

        Called by the engine after a round rebuilt the virtual map, so vol
        entries for these logicals already point at their new physical
        slots; only the staging slots are recycled here.  A defensive vol
        clear covers callers that did not rebuild.
        """
        self.staging.release_tokens(req, logicals)
        state = self.engine.state(req)
        if state is None or state.virtual_of_logical is None:
            return
        n_phys_pages = (
            (state.physical_len + self.page_size - 1) // self.page_size
        )
        vol = state.virtual_of_logical
        for logical in logicals:
            logical = int(logical)
            if int(vol[logical].item()) >= n_phys_pages * self.page_size:
                vol[logical] = -1

    def release_chunk(self, entries: list[tuple[str, int, int]]) -> None:
        """Recycle a chunk's staging slots and clear its virtual entries."""
        for req, logical, _ in entries:
            self.staging.release_tokens(req, [logical])
            self._step_placed.discard((req, int(logical)))
            state = self.engine.state(req)
            if state is not None and state.virtual_of_logical is not None:
                state.virtual_of_logical[int(logical)] = -1

    def ext_rows_for(
        self,
        request_ids: list[str],
        orig_indptr: torch.Tensor,
        orig_indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [str(r) for r in request_ids]
        return self._build_ext_rows(ids, orig_indptr, orig_indices, device)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _touch_staged(
        self,
        state,
        row: torch.Tensor,
        virt: torch.Tensor,
        n_phys_pages: int,
    ) -> list[int]:
        """LRU-touch staged tokens hit by this selection row; return them."""
        req = state.request_id
        staged_mask = virt >= n_phys_pages * self.page_size
        if not bool(staged_mask.any()):
            return []
        logicals = row[staged_mask].unique().tolist()
        for logical in logicals:
            self.staging.slot_of(req, int(logical))
        return [int(l) for l in logicals]

    def _fetch_and_scatter(
        self,
        placed: dict[tuple[str, int], int],
        states,
        ids,
    ) -> None:
        """H2D-fetch placed tokens and scatter them into staging pages."""
        cpu_slots = []
        blocks = []
        offsets = []
        virt_updates: dict[str, tuple[list[int], list[int]]] = {}
        for (req, logical), slot in placed.items():
            state = (
                states[req]
                if isinstance(states, dict)
                else states[ids.index(req)]
            )
            cpu_slots.append(state.cpu_slot_of_logical[int(logical)])
            sp_local = slot // self.page_size
            blocks.append(self.staging.staging_base_page + sp_local)
            offsets.append(slot % self.page_size)
            n_phys_pages = (
                len(state.block_ids)
            )
            j = self.staging.pages_of(req).index(sp_local)
            virtual = (n_phys_pages + j) * self.page_size + slot % self.page_size
            logicals, virts = virt_updates.setdefault(req, ([], []))
            logicals.append(int(logical))
            virts.append(virtual)

        chunks, event = self.token_pool.fetch_to_gpu(cpu_slots)
        blocks_t = torch.tensor(blocks, dtype=torch.long, device=device_of(self.compactor.kv_caches[0]))
        offsets_t = torch.tensor(offsets, dtype=torch.long, device=blocks_t.device)
        self.compactor.scatter_kv(blocks_t, offsets_t, chunks)
        if event is not None:
            # The scatter is stream-ordered after the H2D; the event only
            # gates pinned-source reuse, which the pool already tracks.
            pass

        for req, (logicals, virts) in virt_updates.items():
            state = self.engine.state(req)
            if state is None or state.virtual_of_logical is None:
                continue
            idx_t = torch.tensor(logicals, dtype=torch.long, device=state.virtual_of_logical.device)
            val_t = torch.tensor(virts, dtype=torch.int32, device=state.virtual_of_logical.device)
            state.virtual_of_logical.index_copy_(0, idx_t, val_t)

    def _scatter_virtuals(
        self, req: str, state, *, keys_to_touch: bool = True
    ) -> None:
        """Recompute every staged token's virtual for ``req``."""
        vol = state.virtual_of_logical
        if vol is None:
            return
        n_phys_pages = len(state.block_ids)
        pages = self.staging.pages_of(req)
        logicals: list[int] = []
        virts: list[int] = []
        for (r, logical), slot in self.staging._slot_of.items():
            if r != req:
                continue
            sp_local = slot // self.page_size
            j = pages.index(sp_local)
            logicals.append(int(logical))
            virts.append((n_phys_pages + j) * self.page_size + slot % self.page_size)
        if not logicals:
            return
        # Clear then rewrite: tokens released since the last scatter must
        # not keep a stale virtual.
        vol[: state.logical_len].fill_(-1)
        phys_map = state.logical_of_physical
        if phys_map is not None:
            vol.scatter_(
                0,
                phys_map[: state.physical_len].long(),
                torch.arange(
                    state.physical_len, dtype=torch.int32, device=vol.device
                ),
            )
        idx_t = torch.tensor(logicals, dtype=torch.long, device=vol.device)
        val_t = torch.tensor(virts, dtype=torch.int32, device=vol.device)
        vol.index_copy_(0, idx_t, val_t)

    def _build_ext_rows(
        self,
        ids: list[str],
        orig_indptr: torch.Tensor,
        orig_indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat page rows: [physical pages | request's staging pages]."""
        total_rows = int(orig_indptr.numel()) - 1
        nd = len(ids)
        orig_indptr_host = orig_indptr.tolist()
        orig_host = orig_indices.tolist()

        flat: list[int] = []
        lens: list[int] = []
        for b in range(total_rows):
            if b < nd:
                state = self.engine.state(ids[b])
                row = list(state.block_ids) if state is not None else []
                row += [
                    self.staging.staging_base_page + p
                    for p in self.staging.pages_of(ids[b])
                ]
            else:
                start = int(orig_indptr_host[b])
                end = int(orig_indptr_host[b + 1])
                row = orig_host[start:end]
            flat.extend(row)
            lens.append(len(row))

        indptr_host = [0]
        for n in lens:
            indptr_host.append(indptr_host[-1] + n)
        total = indptr_host[-1]
        cached = self._ext_cache.get(total)
        if cached is None:
            indptr_t = torch.empty(total_rows + 1, dtype=torch.int32, device=device)
            indices_t = torch.empty(
                max(total, 1), dtype=torch.int32, device=device
            )
            cached = (indptr_t, indices_t)
            self._ext_cache[total] = cached
            if len(self._ext_cache) > 4:
                self._ext_cache.pop(next(iter(self._ext_cache)))
        indptr_t, indices_t = cached
        indptr_t.copy_(torch.as_tensor(indptr_host, dtype=torch.int32))
        if total > 0:
            indices_t.copy_(torch.as_tensor(flat, dtype=torch.int32))
        return indptr_t[: total_rows + 1], indices_t[:total]


def device_of(tensor: torch.Tensor) -> torch.device:
    return tensor.device
