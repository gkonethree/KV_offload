# SPDX-License-Identifier: Apache-2.0
"""Token-level offloading with per-(layer, head) budgets ("cell" mode).

``TokenOffloadEngine`` keeps the same tokens for every KV head of a layer and
the same count in every layer. Here each (layer, KV head) keeps its own
tokens, and with ``scope="global"`` (FastKVzip's own policy) its own amount:
one ranking over all (layer, head, token) scores of a request.

Cells. In the KV view ``[pages, H, P, 2C]`` every (page, head, offset) is a
contiguous 2C row, and a page is a contiguous block of H * P rows (stored
head-major ``[H, P]`` or token-major ``[P, H]``). vLLM places all layers in ONE
backing buffer (layer ``l``'s page of block ``b`` at ``offset + l *
layer_stride + b * block_stride``), so the buffer is a flat array of equal
"units" (one layer's page of one block) and of cells: ``cell = unit * H * P +
c`` with ``c = h * P + o`` (head-major) or ``o * H + h`` (token-major) for
vLLM's own writes; packed cells use any ``c``. Viewed as rows
``kv_rows [n_cells, 2C]``, every token of every head of every layer has one
global cell index. The sparse decode kernel reads the flat buffer with
``page_size = H * P``, ``stride_h = 0`` and identity page tables, so
``sparse_idx`` carries global cells directly -- no kernel change.

After a round a request's survivors (all layers, all heads) are packed
densely, layer-major then head-major, into the units of its first ``nb``
blocks, where ``nb = ceil(kept / (L * H * P))``. A layer may use another
layer's unit of the request's blocks, so per-layer totals can differ while
vLLM keeps one block table. Tokens written after a round land, as usual, at
the physical tail in the standard per-layer layout (the append region) and
are repacked at the next round.

State (GPU, ``[layer, request slot, position, kv head]``):
  score   float32  keep score (FastKVzip per KV head)
  cell    int32    global cell of a resident token, -1 = offloaded
  cpu_row int32    row of an offloaded token in the CPU pool, -1 = none

CPU pool: pinned ``[N, 2C]``, one row per offloaded (layer, head, token).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from offload_package import timing as _timing

SCOPES = ("global", "layer", "uniform", "shared")
_D2H_CHUNK = 1 << 16          # rows per device->host chunk of a round


@dataclass
class _Req:
    slot: int
    pages: list[int] = field(default_factory=list)
    written: int = 0          # logical tokens whose KV is in the cache
    mapped: int = 0           # logical tokens with a cell assigned
    offset: int = 0           # logical - physical length (vLLM's view)
    remapped: bool = False    # a round has packed this request's cells
    cpu_rows: list[torch.Tensor] = field(default_factory=list)


def flat_cells(kv_caches: list[torch.Tensor]):
    """``(kv_rows, unit_base, unit_stride, n_units, head_major)`` for
    per-layer views of one backing buffer: ``kv_rows`` [cells, 2C] over the
    whole buffer, the unit of (layer ``l``, block ``b``) = ``unit_base[l] + b *
    unit_stride``, and whether a page stores its rows ``[H, P]`` (else
    ``[P, H]``)."""
    k0 = kv_caches[0]
    _, H, P, C2 = (int(x) for x in k0.shape)
    unit = H * P * C2
    storage = k0.untyped_storage()
    es = k0.element_size()
    sbase = storage.data_ptr()
    ub, us = [], None
    head_major = k0.stride(1) == P * C2 and k0.stride(2) == C2
    for l, k in enumerate(kv_caches):
        if (k.untyped_storage().data_ptr() != sbase or k.dtype != k0.dtype
                or tuple(k.shape[1:]) != (H, P, C2)):
            raise RuntimeError("per-head token offload: KV caches must be views of one "
                               "backing buffer with equal [H, P, 2C] pages")
        page_rows = (P * C2, C2) if head_major else (C2, H * C2)
        if k.stride(3) != 1 or (k.stride(1), k.stride(2)) != page_rows:
            raise RuntimeError(f"per-head token offload: layer {l} pages are not "
                               f"contiguous [H, P] or [P, H] rows (strides {k.stride()})")
        off = (k.data_ptr() - sbase) // es
        if off % unit or k.stride(0) % unit:
            raise RuntimeError(f"per-head token offload: layer {l} is not unit-aligned")
        if us is None:
            us = k.stride(0) // unit
        elif k.stride(0) // unit != us:
            raise RuntimeError("per-head token offload: layers have different block strides")
        ub.append(off // unit)
    n_units = storage.nbytes() // es // unit
    kv_rows = k0.as_strided((n_units * H * P, C2), (C2, 1), 0)
    return kv_rows, ub, int(us), int(n_units), head_major


class HeadTokenOffloadEngine:
    headwise = True

    def __init__(
        self,
        *,
        kv_caches: list[torch.Tensor],       # per layer [pages, H, P, 2C]
        page_size: int,
        max_len: int,
        num_req_slots: int,
        num_cpu_tokens: int,
        offload_fraction: float = 0.0,
        token_budget: int | None = None,
        sink_tokens: int = 64,
        window_tokens: int = 128,
        scope: str = "global",
    ) -> None:
        if not 0.0 <= offload_fraction < 1.0:
            raise ValueError("offload_fraction must be in [0, 1)")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        self.kv_caches = list(kv_caches)
        self.L = len(self.kv_caches)
        self.P = int(page_size)
        self.max_len = int(max_len)
        self.dev = self.kv_caches[0].device
        self.H = int(self.kv_caches[0].shape[1])
        self.C2 = int(self.kv_caches[0].shape[3])
        if int(self.kv_caches[0].shape[2]) != self.P:
            raise ValueError("KV cache page size != page_size")
        self.HP = self.H * self.P
        self.scope = scope
        self.fraction = float(offload_fraction)
        self.budget = None if token_budget is None else int(token_budget)
        self.sink = max(0, int(sink_tokens))
        self.window = max(0, int(window_tokens))
        (self.kv_rows, ub, self.unit_stride, self.n_units,
         self.head_major) = flat_cells(self.kv_caches)
        self.unit_base = ub
        self._ub = torch.tensor(ub, dtype=torch.long, device=self.dev)
        self.kv_rows_v = self.kv_rows[:, self.C2 // 2:]    # V half: kernel V pointer
        self._reqs: dict[str, _Req] = {}
        self._row_updates: dict[str, list[int]] = {}
        self.on_release: list = []
        self._free_req_slots: list[int] = []
        self.S = 0
        self._grow(max(1, int(num_req_slots)))
        n_rows = self.L * self.H * int(num_cpu_tokens)
        self.cpu_pool = torch.empty((n_rows, self.C2), dtype=self.kv_caches[0].dtype,
                                    device="cpu", pin_memory=True)
        self._free_rows = torch.arange(n_rows - 1, -1, -1, dtype=torch.long)
        self._top = n_rows                      # free rows are _free_rows[:_top]
        self._d2h = None                        # pinned bounce buffer for rounds
        self.stats = {"rounds": 0, "offloaded": 0, "freed_pages": 0}
        self.layer_keep_log: list[tuple[float, float, float]] = []

    # ---- state tensors ----------------------------------------------------

    def _grow(self, n: int) -> None:
        old = (getattr(self, "score", None), getattr(self, "cell", None),
               getattr(self, "cpu_row", None))
        shape = (self.L, n, self.max_len, self.H)
        with torch.inference_mode(False):
            score = torch.zeros(shape, dtype=torch.float32, device=self.dev)
            cell = torch.full(shape, -1, dtype=torch.int32, device=self.dev)
            cpu = torch.full(shape, -1, dtype=torch.int32, device=self.dev)
        if old[0] is not None:
            k = old[0].shape[1]
            score[:, :k].copy_(old[0]); cell[:, :k].copy_(old[1]); cpu[:, :k].copy_(old[2])
        self.score, self.cell, self.cpu_row = score, cell, cpu
        self._free_req_slots.extend(range(n - 1, self.S - 1, -1))
        self.S = n

    def unit(self, layer: int, block: int) -> int:
        return self.unit_base[layer] + int(block) * self.unit_stride

    def std_cell(self, head, offset):
        """In-unit cell of (head, page offset) in vLLM's own page layout."""
        return head * self.P + offset if self.head_major else offset * self.H + head

    # ---- request lifecycle --------------------------------------------------

    def add_request(self, req_id: str) -> int:
        r = self._reqs.get(req_id)
        if r is not None:
            return r.slot
        if not self._free_req_slots:
            self._grow(self.S * 2)
        slot = self._free_req_slots.pop()
        self._reqs[req_id] = _Req(slot=slot)
        return slot

    def has(self, req_id: str) -> bool:
        return req_id in self._reqs

    def slot_of(self, req_id: str) -> int:
        r = self._reqs.get(req_id)
        return -1 if r is None else r.slot

    def offset(self, req_id: str) -> int:
        """Logical - physical length (vLLM's block table covers the physical)."""
        r = self._reqs.get(req_id)
        return 0 if r is None else r.offset

    def remapped(self, req_id: str) -> bool:
        """True once a round has packed this request: its KV no longer sits
        at vLLM's standard (logical) positions, whatever the offset."""
        r = self._reqs.get(req_id)
        return r is not None and r.remapped

    def take_row_update(self, req_id: str) -> list[int] | None:
        return self._row_updates.pop(req_id, None)

    def _free_cpu_rows(self, rows: torch.Tensor) -> None:
        n = int(rows.numel())
        self._free_rows[self._top:self._top + n] = rows
        self._top += n

    def release(self, req_id: str) -> None:
        self._row_updates.pop(req_id, None)
        r = self._reqs.pop(req_id, None)
        if r is None:
            return
        for rows in r.cpu_rows:
            self._free_cpu_rows(rows)
        s = r.slot
        for cb in self.on_release:
            cb(s)
        self.score[:, s].zero_()
        self.cell[:, s].fill_(-1)
        self.cpu_row[:, s].fill_(-1)
        self._free_req_slots.append(s)

    def set_pages(self, req_id: str, pages: list[int]) -> None:
        self._reqs[req_id].pages = list(pages)

    def pages(self, req_id: str) -> list[int]:
        return self._reqs[req_id].pages

    def on_step(self, req_id: str, num_computed: int, num_scheduled: int) -> None:
        """Map newly written tokens to their standard-layout cells (physical
        position ``logical - offset`` of the request's block table), in every
        layer and head."""
        r = self._reqs[req_id]
        r.written = max(r.written, int(num_computed))
        end = int(num_computed) + int(num_scheduled)
        if end > self.max_len:
            raise RuntimeError(f"{req_id}: {end} tokens exceed max_len {self.max_len}")
        if end <= r.mapped:
            return
        lo = r.mapped
        phys = torch.arange(lo - r.offset, end - r.offset, dtype=torch.long)
        bi = torch.div(phys, self.P, rounding_mode="floor")
        if int(bi[-1]) >= len(r.pages):
            raise RuntimeError(f"{req_id}: position {end - 1} has no page "
                               f"({len(r.pages)} pages)")
        blk = torch.tensor(r.pages, dtype=torch.long)[bi].to(self.dev, non_blocking=True)
        o = (phys - bi * self.P).to(self.dev, non_blocking=True)
        unit = self._ub.view(-1, 1) + blk.view(1, -1) * self.unit_stride       # [L, n]
        h = torch.arange(self.H, device=self.dev).view(1, 1, -1)
        cells = unit.unsqueeze(-1) * self.HP + self.std_cell(h, o.view(1, -1, 1))  # [L, n, H]
        self.cell[:, r.slot, lo:end] = cells.to(torch.int32)
        r.mapped = end

    # ---- scoring --------------------------------------------------------------

    def write_scores(self, layer: int, slots: torch.Tensor,
                     positions: torch.Tensor, scores: torch.Tensor) -> None:
        """``scores`` [N, H] for tokens at logical ``positions`` of ``slots``."""
        self.score[layer, slots, positions] = scores

    # ---- offload round ----------------------------------------------------------

    def keep_count(self, logical_len: int) -> int:
        if self.budget is not None:
            k = self.budget
        else:
            k = math.ceil((1.0 - self.fraction) * logical_len)
        return max(0, int(k))

    def _n_protected(self, n: int) -> int:
        sink = min(n, self.sink)
        return sink + min(self.window, max(0, n - sink))

    @staticmethod
    def _lowest(sc: torch.Tensor, n_evict: torch.Tensor) -> torch.Tensor:
        """Mask of the ``n_evict[g]`` lowest entries of each row ``sc[g]``."""
        order = torch.argsort(sc, dim=1)
        rank_ok = (torch.arange(sc.shape[1], device=sc.device).view(1, -1)
                   < n_evict.to(sc.device).view(-1, 1))
        return torch.zeros_like(rank_ok).scatter_(1, order, rank_ok)

    def _select(self, sc: torch.Tensor, resid: torch.Tensor, K: int) -> torch.Tensor:
        """Evict mask [L, Lg, H] over resident cells, keeping (with protected
        cells' scores at +inf) the budget of the engine's scope."""
        L, Lg, H = sc.shape
        if self.scope == "global":
            n = (resid.sum() - L * H * K).clamp(min=0).view(1)
            return self._lowest(sc.reshape(1, -1), n).view(L, Lg, H)
        if self.scope == "layer":
            n = (resid.sum((1, 2)) - H * K).clamp(min=0)
            return self._lowest(sc.reshape(L, -1), n).view(L, Lg, H)
        if self.scope == "uniform":
            n = (resid.sum(1) - K).clamp(min=0).reshape(-1)               # [L*H]
            ev = self._lowest(sc.permute(0, 2, 1).reshape(L * H, Lg), n)
            return ev.view(L, H, Lg).permute(0, 2, 1)
        # shared: one decision per (layer, token), heads combined by max.
        tok = sc.amax(-1)                                                  # [L, Lg]
        n = (resid.any(-1).sum(1) - K).clamp(min=0)
        return self._lowest(tok, n).unsqueeze(-1).expand(L, Lg, H) & resid

    @torch.no_grad()
    def run_round(self, req_id: str, extra_tokens: int = 0) -> list[int]:
        """Offload the lowest-scored resident cells under the scope's budget,
        pack all survivors densely into the request's first blocks, and return
        the now-unused tail page ids (the caller frees them in vLLM)."""
        r = self._reqs.get(req_id)
        if r is None:
            return []
        Lg, s = r.written, r.slot
        if r.mapped != Lg:
            raise RuntimeError(f"{req_id}: run_round needs mapped == written "
                               f"(mapped={r.mapped}, written={Lg})")
        if Lg <= 0:
            return []
        K = max(self.keep_count(Lg), self._n_protected(Lg))
        cells = self.cell[:, s, :Lg]                                       # [L, Lg, H]
        resid = cells >= 0
        pos = torch.arange(Lg, device=self.dev)
        protect = ((pos < self.sink) | (pos >= Lg - self.window)).view(1, -1, 1)
        sc = (self.score[:, s, :Lg].masked_fill(protect, float("inf"))
              .masked_fill(~resid, float("inf")))
        evict = self._select(sc, resid, K) & resid
        ev = evict.nonzero()                                               # [E, 3] (l, p, h)
        E = int(ev.shape[0])
        if E == 0:
            return []
        # Survivors fill nb whole blocks; a round that frees no page (offset
        # would be <= 0) is skipped, so an applied round always leaves
        # offset > 0 and the next token a page at physical nb * P.
        T = int(resid.sum()) - E
        nb = -(-T // (self.L * self.HP))
        if Lg - nb * self.P <= 0:
            return []
        if E > self._top:
            raise RuntimeError(f"CPU token pool exhausted: need {E} rows, free {self._top}")
        _timing.start("tok.round")

        # 1) D2H the evicted cells into freshly allocated CPU rows.
        rows = self._free_rows[self._top - E:self._top].clone()
        self._top -= E
        r.cpu_rows.append(rows)
        src = cells[ev[:, 0], ev[:, 1], ev[:, 2]].long()
        if self._d2h is None:
            self._d2h = torch.empty((_D2H_CHUNK, self.C2), dtype=self.cpu_pool.dtype,
                                    pin_memory=True)
        stream = torch.cuda.current_stream(self.dev)
        for a in range(0, E, _D2H_CHUNK):
            b = min(E, a + _D2H_CHUNK)
            self._d2h[: b - a].copy_(self.kv_rows[src[a:b]], non_blocking=True)
            stream.synchronize()
            self.cpu_pool.index_copy_(0, rows[a:b], self._d2h[: b - a])
        self.cpu_row[:, s][ev[:, 0], ev[:, 1], ev[:, 2]] = rows.to(self.dev, torch.int32)

        # 2) Pack survivors layer-major, head-major, in logical order into the
        #    units (layer-major) of the request's first nb blocks.
        keep = (resid & ~evict).permute(0, 2, 1)                          # [L, H, Lg]
        kp = keep.nonzero()                                                # (l, h, p)
        assert int(kp.shape[0]) == T
        pages = torch.tensor(r.pages[:nb], dtype=torch.long, device=self.dev)
        units = (self._ub.view(-1, 1) + pages.view(1, -1) * self.unit_stride).reshape(-1)
        d = torch.arange(T, device=self.dev)
        dst = units[torch.div(d, self.HP, rounding_mode="floor")] * self.HP + d % self.HP
        src = cells[kp[:, 0], kp[:, 2], kp[:, 1]].long()
        moved = src != dst
        if bool(moved.any()):
            self.kv_rows[dst[moved]] = self.kv_rows[src[moved]]      # gather copies first
        cv = self.cell[:, s]
        cv[ev[:, 0], ev[:, 1], ev[:, 2]] = -1
        cv[kp[:, 0], kp[:, 2], kp[:, 1]] = dst.to(torch.int32)

        # 3) Bookkeeping: survivors fill nb whole blocks in vLLM's view.
        r.offset = Lg - nb * self.P
        r.remapped = True
        r.mapped = Lg
        keep_pages = min(len(r.pages), nb + -(-int(extra_tokens) // self.P))
        freed = r.pages[keep_pages:]
        r.pages = r.pages[:keep_pages]
        self._row_updates[req_id] = list(r.pages)
        self.stats["rounds"] += 1
        self.stats["offloaded"] += E
        self.stats["freed_pages"] += len(freed)
        if len(self.layer_keep_log) < 4:
            per_layer = keep.sum((1, 2)).float() / self.H
            self.layer_keep_log.append((float(per_layer.min()), float(per_layer.mean()),
                                        float(per_layer.max())))
            print(f"[TOKEN HEADS] round {self.stats['rounds']} scope={self.scope}: "
                  f"kept tokens/layer (avg over heads) min {self.layer_keep_log[-1][0]:.0f} "
                  f"mean {self.layer_keep_log[-1][1]:.0f} max {self.layer_keep_log[-1][2]:.0f}"
                  f" (uniform budget {K}); blocks {nb}", flush=True)
        _timing.end("tok.round")
        return freed
