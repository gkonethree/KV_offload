# SPDX-License-Identifier: Apache-2.0
"""Token-level offloading: per-layer eviction + order-preserving compaction.

Every offload round, each layer keeps the same NUMBER of tokens (same budget,
same protected sink + recent window) but chooses ITS OWN lowest-scored tokens
to offload. After the round every layer's survivors are packed, in logical
order, into the same physical prefix length ``P``. So vLLM keeps ONE block
table / slot mapping / page allocation per request (new tokens land at
physical slot ``P`` in every layer); only the logical -> physical token map is
per layer.

State (GPU, indexed ``[layer, request slot, position]``):
  score          float32  query-agnostic keep score (FastKVzip, written at
                          token write time by a hook on each layer's attention)
  phys_of_logical int32   physical slot of a logical token, -1 = offloaded
  logical_of_phys int32   inverse map over the physical prefix
  cpu_slot        int32   CPU token-pool slot of an offloaded token, -1 = none

CPU token pool: pinned ``[L, N, H, 2*head_dim]``. A round offloads exactly
``E`` tokens in every layer, so one slot index ``s`` holds "layer l's i-th
evicted token" for all layers at once -- a single free list.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

from offload_package import timing as _timing


@dataclass
class _Req:
    slot: int
    pages: list[int] = field(default_factory=list)
    written: int = 0          # logical tokens whose KV is in the cache
    mapped: int = 0           # logical tokens with a physical slot assigned
    offloaded: int = 0        # per layer (equal in every layer)
    cpu_slots: list[int] = field(default_factory=list)

    @property
    def phys_written(self) -> int:
        return self.written - self.offloaded


class TokenOffloadEngine:
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
    ) -> None:
        if not 0.0 <= offload_fraction < 1.0:
            raise ValueError("offload_fraction must be in [0, 1)")
        self.kv_caches = list(kv_caches)
        self.L = len(self.kv_caches)
        self.P = int(page_size)
        self.max_len = int(max_len)
        self.dev = self.kv_caches[0].device
        self.H = int(self.kv_caches[0].shape[1])
        self.C2 = int(self.kv_caches[0].shape[3])
        self.fraction = float(offload_fraction)
        self.budget = None if token_budget is None else int(token_budget)
        self.sink = max(0, int(sink_tokens))
        self.window = max(0, int(window_tokens))
        self._reqs: dict[str, _Req] = {}
        self._row_updates: dict[str, list[int]] = {}
        self.on_release: list = []      # callbacks(slot) when a request is released
        self._free_req_slots: list[int] = []
        self.S = 0
        self._grow(max(1, int(num_req_slots)))
        self.cpu_pool = torch.empty(
            (self.L, int(num_cpu_tokens), self.H, self.C2),
            dtype=self.kv_caches[0].dtype, device="cpu", pin_memory=True)
        self._free_cpu = list(range(int(num_cpu_tokens) - 1, -1, -1))
        self.stats = {"rounds": 0, "offloaded": 0, "freed_pages": 0}

    # ---- state tensors ----------------------------------------------------

    def _grow(self, n: int) -> None:
        old = (getattr(self, "score", None), getattr(self, "phys_of_logical", None),
               getattr(self, "logical_of_phys", None), getattr(self, "cpu_slot", None))
        shape = (self.L, n, self.max_len)
        with torch.inference_mode(False):
            score = torch.zeros(shape, dtype=torch.float32, device=self.dev)
            p2l = torch.full(shape, -1, dtype=torch.int32, device=self.dev)
            l2p = torch.full(shape, -1, dtype=torch.int32, device=self.dev)
            cpu = torch.full(shape, -1, dtype=torch.int32, device=self.dev)
        if old[0] is not None:
            k = old[0].shape[1]
            score[:, :k].copy_(old[0]); p2l[:, :k].copy_(old[1])
            l2p[:, :k].copy_(old[2]); cpu[:, :k].copy_(old[3])
        self.score, self.phys_of_logical, self.logical_of_phys, self.cpu_slot = score, p2l, l2p, cpu
        self._free_req_slots.extend(range(n - 1, self.S - 1, -1))
        self.S = n

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
        """Tokens offloaded per layer = logical - physical length."""
        r = self._reqs.get(req_id)
        return 0 if r is None else r.offloaded

    def remapped(self, req_id: str) -> bool:
        """True once a round has compacted this request's KV."""
        r = self._reqs.get(req_id)
        return r is not None and r.offloaded > 0

    def take_row_update(self, req_id: str) -> list[int] | None:
        """Runner side: this step's full page list for a request whose pages
        a round truncated (None = append this step's new blocks as usual)."""
        return self._row_updates.pop(req_id, None)

    def release(self, req_id: str) -> None:
        self._row_updates.pop(req_id, None)
        r = self._reqs.pop(req_id, None)
        if r is None:
            return
        self._free_cpu.extend(r.cpu_slots)
        s = r.slot
        for cb in self.on_release:
            cb(s)
        self.score[:, s].zero_()
        self.phys_of_logical[:, s].fill_(-1)
        self.logical_of_phys[:, s].fill_(-1)
        self.cpu_slot[:, s].fill_(-1)
        self._free_req_slots.append(s)

    def set_pages(self, req_id: str, pages: list[int]) -> None:
        self._reqs[req_id].pages = list(pages)

    def pages(self, req_id: str) -> list[int]:
        return self._reqs[req_id].pages

    def on_step(self, req_id: str, num_computed: int, num_scheduled: int) -> None:
        """``num_computed`` logical tokens now have KV; ``num_scheduled`` more
        are written this step at the physical tail. Map them in every layer
        (they are resident everywhere until a round offloads them)."""
        r = self._reqs[req_id]
        r.written = max(r.written, int(num_computed))
        end = int(num_computed) + int(num_scheduled)
        if end > self.max_len:
            raise RuntimeError(f"{req_id}: {end} tokens exceed max_len {self.max_len}")
        if end <= r.mapped:
            return
        lo = r.mapped
        logical = torch.arange(lo, end, dtype=torch.int32, device=self.dev)
        self.phys_of_logical[:, r.slot, lo:end] = logical - r.offloaded
        self.logical_of_phys[:, r.slot, lo - r.offloaded:end - r.offloaded] = logical
        r.mapped = end

    # ---- scoring --------------------------------------------------------------

    def write_scores(self, layer: int, slots: torch.Tensor,
                     positions: torch.Tensor, scores: torch.Tensor) -> None:
        """``scores`` [N] for tokens at logical ``positions`` of request
        ``slots`` (all [N], on device). Called by the attention-input hook."""
        self.score[layer, slots, positions] = scores

    # ---- offload round ----------------------------------------------------------

    def keep_count(self, logical_len: int) -> int:
        if self.budget is not None:
            k = self.budget
        else:
            k = math.ceil((1.0 - self.fraction) * logical_len)
        return max(0, int(k))

    @torch.no_grad()
    def run_round(self, req_id: str, extra_tokens: int = 0) -> list[int]:
        """Offload each layer's lowest-scored tokens, compact the survivors in
        order, and return the request's now-unused tail page ids (the caller
        frees them in vLLM). Only tokens whose KV is written are considered;
        pages for ``extra_tokens`` more (this step's writes) are kept."""
        r = self._reqs.get(req_id)
        if r is None:
            return []
        Pw, Lg, s = r.phys_written, r.written, r.slot
        if r.mapped != Lg:
            raise RuntimeError(f"{req_id}: run_round needs mapped == written "
                               f"(mapped={r.mapped}, written={Lg})")
        if Pw <= 0:
            return []
        resid = self.logical_of_phys[:, s, :Pw].long()                   # [L, Pw]
        protect = (resid < self.sink) | (resid >= Lg - self.window)
        n_prot = int(protect[0].sum())
        K = max(self.keep_count(Lg), n_prot)
        E = Pw - K
        if E <= 0:
            return []
        if E > len(self._free_cpu):
            raise RuntimeError(f"CPU token pool exhausted: need {E}, free {len(self._free_cpu)}")
        _timing.start("tok.round")
        sc = self.score[:, s].gather(1, resid).masked_fill(protect, float("inf"))
        evict = torch.topk(sc, E, dim=1, largest=False).indices        # [L, E]
        evict, _ = torch.sort(evict, dim=1)
        keep = torch.ones_like(resid, dtype=torch.bool)
        keep.scatter_(1, evict, False)
        surv = keep.nonzero(as_tuple=True)[1].view(self.L, K)          # [L, K] sorted
        evict_logical = resid.gather(1, evict)                          # [L, E]
        surv_logical = resid.gather(1, surv)                            # [L, K]
        pages = torch.tensor(r.pages, dtype=torch.long, device=self.dev)

        # 1) D2H the evicted tokens (each layer its own), one slot index per i.
        cpu_slots = [self._free_cpu.pop() for _ in range(E)]
        slots_cpu = torch.tensor(cpu_slots, dtype=torch.long)
        host = []
        for l in range(self.L):
            e = evict[l]
            g = self.kv_caches[l][pages[e // self.P], :, e % self.P]     # [E, H, 2C]
            host.append(g.to("cpu", non_blocking=True))
        torch.cuda.current_stream(self.dev).synchronize()
        for l in range(self.L):
            self.cpu_pool[l].index_copy_(0, slots_cpu, host[l])
        r.cpu_slots.extend(cpu_slots)
        self.cpu_slot[:, s].scatter_(
            1, evict_logical,
            torch.tensor(cpu_slots, dtype=torch.int32, device=self.dev).expand(self.L, E).contiguous())

        # 2) Order-preserving compaction: survivors -> physical [0, K).
        dst = torch.arange(K, device=self.dev)
        for l in range(self.L):
            src = surv[l]
            moved = src != dst
            if not bool(moved.any()):
                continue
            first = int(moved.nonzero()[0])
            src_m, dst_m = src[first:], dst[first:]
            data = self.kv_caches[l][pages[src_m // self.P], :, src_m % self.P].clone()
            self.kv_caches[l][pages[dst_m // self.P], :, dst_m % self.P] = data

        # 3) Maps.
        p2l = self.phys_of_logical[:, s]
        p2l[:, :Lg] = -1
        p2l.scatter_(1, surv_logical, dst.to(torch.int32).expand(self.L, K).contiguous())
        self.logical_of_phys[:, s, :Pw] = -1
        self.logical_of_phys[:, s, :K] = surv_logical.to(torch.int32)
        r.offloaded += E
        r.mapped = Lg                        # tokens >= Lg get mapped at the new tail
        keep_pages = min(len(r.pages), -(-(K + int(extra_tokens)) // self.P))
        freed = r.pages[keep_pages:]
        r.pages = r.pages[:keep_pages]
        self._row_updates[req_id] = list(r.pages)
        self.stats["rounds"] += 1
        self.stats["offloaded"] += E
        self.stats["freed_pages"] += len(freed)
        _timing.end("tok.round")
        return freed

    # ---- decode-time CPU source -------------------------------------------------

    def gather_cpu(self, layer: int, cpu_slots: torch.Tensor) -> torch.Tensor:
        """Host gather of offloaded tokens for one layer -> [n, H, 2C] (CPU)."""
        return self.cpu_pool[layer].index_select(0, cpu_slots)


class TokenModeScheduler:
    """Scheduler-side driver: decides rounds, frees compacted tail pages in
    vLLM, and keeps the engine's per-request page lists / maps in sync.

    Runs inside ``build_connector_meta`` (end of ``schedule()``): this step's
    pages are already allocated, the previous step's KV writes are complete
    (synchronous scheduling), and the runner reads the results in this same
    step's ``_update_states`` / ``_prepare_inputs``.
    """

    def __init__(self, *, interval_tokens: int, engine_getter) -> None:
        self.W = int(interval_tokens)
        self._engine = engine_getter
        self.sched = None
        self.kv_mgr = None
        self._prompt_len: dict[str, int] = {}
        self._last_round: dict[str, int] = {}

    def bind(self, *, scheduler=None, kv_cache_manager=None) -> None:
        if scheduler is not None:
            self.sched = scheduler
            assert not scheduler.cache_config.enable_prefix_caching, (
                "token-level offloading requires enable_prefix_caching=False")
        if kv_cache_manager is not None:
            self.kv_mgr = kv_cache_manager

    def _mgr(self):
        return self.kv_mgr.coordinator.single_type_managers[0]

    def _pages(self, rid: str) -> list[int]:
        return [b.block_id for b in self._mgr().req_to_blocks.get(rid, [])]

    def forget(self, rid: str) -> None:
        eng = self._engine()
        if eng is not None:
            eng.release(rid)
        self._last_round.pop(rid, None)
        if self.sched is not None and rid in self.sched.requests:
            self.sched.requests[rid].num_offloaded_tokens = 0

    def request_finished(self, rid: str) -> None:
        self.forget(rid)
        self._prompt_len.pop(rid, None)

    def build_connector_meta(self, so) -> None:
        eng = self._engine()
        if eng is None or self.kv_mgr is None:
            return
        for rid in (getattr(so, "preempted_req_ids", None) or ()):
            self.forget(rid)
        sched_tokens = so.num_scheduled_tokens
        for nr in so.scheduled_new_reqs:
            rid = nr.req_id
            if str(rid).startswith("_warmup_"):
                continue
            self._prompt_len[rid] = int(nr.prompt_len)
            eng.add_request(rid)
            eng.set_pages(rid, self._pages(rid))
            eng.on_step(rid, int(nr.num_computed_tokens), int(sched_tokens.get(rid, 0)))
        cached = so.scheduled_cached_reqs
        for rid, num_computed in zip(cached.req_ids, cached.num_computed_tokens):
            n_sched = int(sched_tokens.get(rid, 0))
            num_computed = int(num_computed)
            if not eng.has(rid):          # resumed after preemption
                eng.add_request(rid)
            eng.set_pages(rid, self._pages(rid))
            eng.on_step(rid, num_computed, 0)
            prompt = self._prompt_len.get(rid, 0)
            last = self._last_round.get(rid)
            # First round one step after prefill (the PQ codes are built from
            # the pages at the first decode step), then every W tokens. Only
            # while decoding: a request resumed after preemption recomputes
            # its prompt + outputs, and compacting mid-recompute would make
            # the rest of that prefill attend over a compacted prefix.
            req = self.sched.requests.get(rid) if self.sched is not None else None
            decoding = req is None or num_computed >= req.num_tokens - 1
            due = prompt > 0 and num_computed > prompt and decoding and (
                last is None or num_computed - last >= self.W)
            if due:
                if last is None and getattr(eng, "prompt_scorer", None) is not None \
                        and req is not None:
                    # Reconstruction scorers (KVzip) rank the finished prompt
                    # once, just before its first round.
                    eng.prompt_scorer(rid, list(req.prompt_token_ids))
                self._last_round[rid] = num_computed
                freed = eng.run_round(rid, extra_tokens=n_sched)
                if freed:
                    blocks = self._mgr().req_to_blocks[rid]
                    keep = len(eng.pages(rid))
                    tail = blocks[keep:]
                    assert [b.block_id for b in tail] == freed, "page list out of sync"
                    del blocks[keep:]
                    self.kv_mgr.block_pool.free_blocks(reversed(tail))
                if self.sched is not None and rid in self.sched.requests:
                    self.sched.requests[rid].num_offloaded_tokens = eng.offset(rid)
            eng.on_step(rid, num_computed, n_sched)


def _decoder_layers(model):
    """The decoder layer list of a vLLM causal-LM (``model.model.layers``)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("token offloading: model has no .model.layers")
    return list(layers)


# Forwards up to this many tokens (decode steps) stash each layer's gate input
# and score all layers in one batched call at the last layer; larger ones
# (prefill chunks) score per layer, so no extra [L, N, D] stack is held.
BATCHED_GATE_MAX_TOKENS = 512
_GATE_CHECK = os.environ.get("OFFLOAD_PACKAGE_TOKEN_GATE_CHECK", "0") == "1"


def install_score_hooks(runner, engine: TokenOffloadEngine, gates,
                        batched=None) -> list:
    """Score every written token with the layer's FastKVzip gate.

    A forward pre-hook on each decoder layer's ``self_attn`` receives exactly
    the gate's input: the post-``input_layernorm`` hidden states and the
    tokens' LOGICAL positions. Tokens map to request slots through the
    runner's query_start_loc / req_ids (computed once per forward, at the
    first layer). Heads are combined by max: a token matters to the layer if
    any of its KV heads needs it -- except for a per-head engine
    (``engine.headwise``), which keeps one score per KV head.

    ``batched`` (``[L, N, D] -> [L, N]``, e.g. BatchedFastKVzipGates) scores
    small forwards for all layers at once: run per layer, the gate is launch
    bound (~16 ms per decode step for 36 layers). The stashed tensors are the
    attention inputs themselves (references, no copy): vLLM never writes them
    after the attention call, and holding the reference keeps the allocator
    from reusing them before the last layer's hook scores them.
    """
    layers = _decoder_layers(runner.model)
    if len(layers) != len(gates) or len(layers) != engine.L:
        raise RuntimeError(f"token offloading: {len(layers)} layers, "
                           f"{len(gates)} gates, {engine.L} KV caches")
    step = {"rows": None, "pos": None, "stash": []}
    last = len(layers) - 1
    per_head = bool(getattr(engine, "headwise", False))
    combine = (lambda s: s.t()) if per_head else (lambda s: s.amax(0))   # [H, n] ->

    def _token_rows(n_tokens: int, device):
        ib = runner.input_batch
        num_reqs = int(ib.num_reqs)
        if num_reqs == 0:
            return None
        qsl = runner.query_start_loc.np[: num_reqs + 1]
        if int(qsl[-1]) != n_tokens:
            n_tokens = min(n_tokens, int(qsl[-1]))
        slots = [engine.slot_of(str(r)) for r in ib.req_ids[:num_reqs]]
        counts = torch.from_numpy((qsl[1:] - qsl[:-1]).astype("int64"))
        per_tok = torch.repeat_interleave(torch.tensor(slots, dtype=torch.long), counts)
        per_tok = per_tok[:n_tokens].to(device, non_blocking=True)
        return per_tok, n_tokens, all(x >= 0 for x in slots)

    handles = []
    for li, layer in enumerate(layers):
        gate = gates[li]

        def hook(module, args, kwargs, li=li, gate=gate):
            if torch.cuda.is_current_stream_capturing():
                return None
            hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
            positions = kwargs.get("positions", args[0] if args else None)
            if hidden is None or positions is None:
                return None
            if li == 0:
                step["rows"] = _token_rows(int(hidden.shape[0]), hidden.device)
                step["stash"] = []
            rows = step["rows"]
            if rows is None:
                return None
            slot_t, n, all_tracked = rows
            if batched is not None and n <= BATCHED_GATE_MAX_TOKENS:
                step["stash"].append(hidden[:n])
                if li != last:
                    return None
                stash, step["stash"] = step["stash"], []
                if len(stash) != len(layers):                 # a layer was skipped
                    return None
                score = batched(torch.stack(stash))           # [L, n]
                if _GATE_CHECK and step.setdefault("checks", 0) < 8:
                    # Debug: the stashed inputs must still be the layers'
                    # attention inputs, i.e. batched == per-layer gates.
                    ref = torch.stack([combine(gates[i](h)) for i, h in enumerate(stash)])
                    step["checks"] += 1
                    print(f"[TOKEN GATE CHECK] n={n} batched vs per-layer "
                          f"max|diff|={float((score - ref).abs().max()):.2e}", flush=True)
                pos = positions[:n].long()
                if not all_tracked:
                    keep = slot_t >= 0
                    slot_t, pos, score = slot_t[keep], pos[keep], score[:, keep]
                engine.score[:, slot_t, pos] = score
                return None
            score = combine(gate(hidden[:n]))                 # [n] ([n, H] per head)
            pos = positions[:n].long()
            if not all_tracked:                               # host-side flag: no sync
                keep = slot_t >= 0
                slot_t, pos, score = slot_t[keep], pos[keep], score[keep]
            engine.write_scores(li, slot_t, pos, score)
            return None

        handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    return handles
