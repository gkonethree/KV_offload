# SPDX-License-Identifier: Apache-2.0
"""Per-layer token staging: cold tokens packed densely into each layer's
own staging tail, addressed by physical token slot.

The page store (``KVStagingPool``) gives a cold block a whole staging page
shared by every layer, so layers whose top-k touch different tokens of the
same blocks evict each other cyclically. Here layer L keeps ITS selected
cold tokens in ITS KV tensor's tail (``num_slots * page_size`` token slots
per layer) with its own LRU, so layers never compete.

The sparse kernel reads ``page = indices[indptr[b] + t // P]``; with
``indptr = 0`` and ``indices = arange`` that is ``t // P``, so ``sparse_idx``
can carry physical slots (``page * P + offset``) and each layer gets its own
token -> address map without a kernel change.

Keys are ``mirror_slot * (MB * P) + logical_position`` (mirror slot from the
residency device mirror, stable per request). All maps live on the GPU.
"""

from __future__ import annotations

import torch

from offload_package import timing as _timing
from offload_package.scoring.base import BlockRef
from offload_package.staging.residency import DEV_CPU

_NEVER = torch.iinfo(torch.int64).max


class TokenStagingStore:
    def __init__(self, pool, residency) -> None:
        if not residency.has_device_mirror:
            raise ValueError("TokenStagingStore needs the residency device mirror")
        self.pool = pool                      # pinned ring + copy stream + caches
        self.res = residency
        self.P = int(pool._page_size)
        self.C = int(pool.num_slots) * self.P  # token slots per layer
        self.L = len(pool.gpu_k_caches)
        self.mbp = int(residency._mb) * self.P
        self.dev = pool.gpu_k_caches[0].device
        self._tick = 0
        self._stamped = 0        # slots stamped with the current tick (kernel path)
        self._S = 0
        self._kern = None        # resolved on first token-engine stage (see _kernels)
        residency.on_slot_release.append(self.release_slot)

    @property
    def tick(self) -> int:
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:
        # A new tick makes every earlier stamp evictable again.
        self._tick = int(value)
        self._stamped = 0

    def _kernels(self):
        """The CUDA token-staging kernels (token-engine mode only), or None.

        Needs the engine's pinned CPU token pool to be device-mapped (UVA) and
        16-byte aligned staging rows; otherwise warns once and uses torch."""
        if self._kern is None:
            self._kern = False
            eng = getattr(self.res, "engine", None)
            if eng is not None:
                from offload_package.staging import token_kernels
                mod = token_kernels.get()
                if mod is not None:
                    why = self._kernel_blocker(mod, eng)
                    if why is None:
                        self._kern = mod
                    else:
                        print(f"[TOKEN KERNELS] FALLBACK to torch token staging: {why}",
                              flush=True)
        return self._kern or None

    def _kernel_blocker(self, mod, eng):
        self._pool_dev = int(mod.device_alias(eng.cpu_pool))
        if not self._pool_dev:
            return "CPU token pool is not device-mapped pinned memory"
        k = self.pool.gpu_k_caches[0]
        es = k.element_size()
        if (k.stride(3) != 1 or any((k.stride(d) * es) % 16 for d in range(3))
                or (k.shape[3] * es) % 16 or k.data_ptr() % 16
                or (eng.cpu_pool.stride(0) * es) % 16):
            return "staging rows are not 16-byte aligned"
        if tuple(eng.cpu_pool.shape[2:]) != (k.shape[1], k.shape[3]):
            return "CPU token pool row shape != KV cache row shape"
        return None

    @classmethod
    def for_token_engine(cls, pool, engine) -> "TokenStagingStore":
        """Store whose cold tokens are TOKEN-offloaded ones (``engine``'s
        per-layer maps / CPU token pool) instead of offloaded blocks."""
        return cls(pool, _EngineSlots(engine))

    def _ensure(self) -> None:
        """(Re)size the maps to the mirror's request-slot count."""
        S = int(self.res._num_slots)
        if S <= self._S:
            return
        # Runs inside the forward's inference_mode, but release_slot runs
        # from the scheduler outside it: allocate normal tensors.
        with torch.inference_mode(False):
            self._alloc(S)

    def _alloc(self, S: int) -> None:
        tok =torch.full((self.L, S, self.mbp), -1, dtype=torch.int32, device=self.dev)
        # seen: tick at which each key was last requested (kernel-path dedup);
        # ticks only grow, so stale stamps never need clearing.
        seen = torch.zeros((self.L, S, self.mbp), dtype=torch.int64, device=self.dev)
        if self._S:
            tok[:, : self._S].copy_(self.tok_slot)
            seen[:, : self._S].copy_(self.seen)
        else:
            # counters {n_need, n_hit_new, err, pad} + pinned host copy.
            self._ctr = torch.zeros(4, dtype=torch.int32, device=self.dev)
            self._ctr_host = torch.zeros(4, dtype=torch.int32, pin_memory=True)
            self._need = torch.empty(0, dtype=torch.int64, device=self.dev)
            # owner: key held by each slot (-1 free); last_use: LRU tick.
            self.owner = torch.full((self.L, self.C), -1, dtype=torch.int64, device=self.dev)
            self.last_use = torch.zeros((self.L, self.C), dtype=torch.int64, device=self.dev)
        self.tok_slot = tok
        self.seen = seen
        self._S = S

    def release_slot(self, s: int) -> None:
        """Forget every token of mirror slot ``s`` in all layers."""
        if s >= self._S:
            return
        self.tok_slot[:, s] = -1
        gone = (self.owner >= 0) & (torch.div(self.owner, self.mbp, rounding_mode="floor") == s)
        self.owner[gone] = -1
        self.last_use[gone] = 0

    def stage(self, layer: int, idx: torch.Tensor, rows: torch.Tensor):
        """Classify ``idx`` [B, N] (logical positions; ``rows`` [B, 1] mirror
        slots), stamp this layer's hits, and fetch the missing cold tokens.

        Returns ``(key, cold, missing)``: ``missing`` is None when everything
        is staged, else the host tensor of keys that did not fit (the caller
        fetches them in chunks with ``fetch``).
        """
        _timing.start("wrap.discover")
        self.res.flush_device()
        self._ensure()
        blk = torch.div(idx, self.P, rounding_mode="floor")
        cold = self.res.dev_state[rows, blk] == DEV_CPU
        key = rows * self.mbp + idx
        slot = self.tok_slot[layer].view(-1)[key]
        self.tick += 1
        self.last_use[layer][slot[cold & (slot >= 0)].long()] = self.tick
        need_d = torch.unique(key[cold & (slot < 0)])
        need = need_d.cpu()                    # the one per-layer readback
        _timing.end("wrap.discover")
        if not need.numel():
            return key, cold, None
        _timing.start("wrap.fetch")
        ok = self.fetch(layer, need_d, need)
        _timing.end("wrap.fetch")
        return key, cold, None if ok else need

    def stage_tokens(self, layer: int, idx: torch.Tensor, rows: torch.Tensor):
        """Token-engine mode: like ``stage`` but per-layer token residency.

        Returns ``(key, cold, phys_pos, missing)``: ``phys_pos`` is each
        position's physical slot in the request's (compacted) pages, -1 when
        offloaded."""
        eng = self.res.engine
        kern = self._kernels()
        if kern is not None:
            return self._stage_tokens_kernel(kern, eng, layer, idx, rows)
        _timing.start("wrap.discover")
        self._ensure()
        phys_pos = eng.phys_of_logical[layer][rows, idx]
        cold = phys_pos < 0
        key = rows * self.mbp + idx
        slot = self.tok_slot[layer].view(-1)[key]
        self.tick += 1
        self.last_use[layer][slot[cold & (slot >= 0)].long()] = self.tick
        need_d = torch.unique(key[cold & (slot < 0)])
        need = need_d.cpu()
        _timing.end("wrap.discover")
        if not need.numel():
            return key, cold, phys_pos, None
        _timing.start("wrap.fetch")
        ok = self.fetch(layer, need_d, need)
        _timing.end("wrap.fetch")
        return key, cold, phys_pos, None if ok else need

    def _stage_tokens_kernel(self, kern, eng, layer, idx, rows):
        """``stage_tokens`` on the CUDA kernels: one classify/stamp/dedup
        launch and ONE counter readback, then (if anything is missing) one
        victim topk and one map-update + zero-copy fill launch."""
        _timing.start("wrap.discover")
        self._ensure()
        self.tick += 1
        idx = idx.contiguous()
        total = idx.numel()
        if self._need.numel() < total:
            self._need = torch.empty(max(total, 2 * self._need.numel()),
                                     dtype=torch.int64, device=self.dev)
        phys_pos = torch.empty(idx.shape, dtype=torch.int32, device=self.dev)
        cold = torch.empty(idx.shape, dtype=torch.bool, device=self.dev)
        key = torch.empty(idx.shape, dtype=torch.int64, device=self.dev)
        kern.tok_discover(idx, rows.reshape(-1).long().contiguous(),
                          eng.phys_of_logical[layer], self.tok_slot[layer].view(-1),
                          self.last_use[layer], self.seen[layer].view(-1), self.tick,
                          self.mbp, phys_pos, cold, key, self._need, self._ctr)
        self._ctr_host.copy_(self._ctr, non_blocking=True)
        torch.cuda.current_stream(self.dev).synchronize()   # the one per-layer readback
        n_need, n_hit, err, n_cold = (int(x) for x in self._ctr_host.tolist())
        if err:
            raise RuntimeError("token fetch: selected token has no CPU copy")
        self._stamped = n_hit
        self.last_counts = (total, n_cold, n_need)
        _timing.note_count("tok.sel_entries", total)
        _timing.note_count("tok.sel_cold", n_cold)
        _timing.note_count("tok.need_keys", n_need)
        _timing.end("wrap.discover")
        if not n_need:
            return key, cold, phys_pos, None
        # Same key order as torch.unique, so victim assignment matches the
        # torch path exactly.
        need_d = torch.sort(self._need[:n_need]).values
        _timing.start("wrap.fetch")
        ok = self.fetch(layer, need_d, need_d)
        _timing.end("wrap.fetch")
        return key, cold, phys_pos, None if ok else need_d

    def phys_tokens(self, layer, key, cold, phys_pos, indptr, indices) -> torch.Tensor:
        """Physical address per position in token-engine mode: resident
        tokens through the row's compacted page list, cold ones from the
        staged slot (0 if unfetched -- only read with weight 0)."""
        kern = self._kernels()
        if kern is not None:
            if indices.dtype != indptr.dtype:
                indices = indices.to(indptr.dtype)
            return kern.tok_phys(phys_pos.contiguous(), key.contiguous(), cold.contiguous(),
                                 indptr.contiguous(), indices.contiguous(),
                                 self.tok_slot[layer].view(-1), int(self.pool.base), self.P)
        B = phys_pos.shape[0]
        p = phys_pos.clamp(min=0).long()
        blk = torch.div(p, self.P, rounding_mode="floor")
        gpu = indices[indptr[:B].view(B, 1).long() + blk].long() * self.P + (p - blk * self.P)
        slot = self.tok_slot[layer].view(-1)[key].long()
        staged = torch.where(slot >= 0, slot + self.pool.base * self.P, 0)
        return torch.where(cold, staged, gpu)

    def fetch(self, layer: int, keys_d: torch.Tensor, keys: torch.Tensor) -> bool:
        """Place ``keys`` into this layer's least-recently-used slots, never
        evicting a slot stamped with the current tick. False (nothing done)
        when they do not fit."""
        n = int(keys.numel())
        if n > self.C:
            return False
        kern = self._kernels()
        if kern is not None:
            return self._fetch_kernel(kern, layer, keys_d, n)
        lu = self.last_use[layer]
        vals, victims = torch.topk(lu.masked_fill(lu == self.tick, _NEVER), n, largest=False)
        if int(vals.max()) == _NEVER:
            return False
        flat = self.tok_slot[layer].view(-1)
        old = self.owner[layer][victims]
        flat[old[old >= 0]] = -1
        self.owner[layer][victims] = keys_d
        flat[keys_d] = victims.to(torch.int32)
        lu[victims] = self.tick
        self._copy(layer, keys, victims)
        return True

    def _fetch_kernel(self, kern, layer: int, keys_d: torch.Tensor, n: int) -> bool:
        # Victims among slots not stamped this tick; the stamped count is
        # known on the host, so no ``vals.max()`` readback.
        if n > self.C - self._stamped:
            return False
        eng = self.res.engine
        lu = self.last_use[layer]
        _, victims = torch.topk(lu.masked_fill(lu == self.tick, _NEVER), n, largest=False)
        k_cache = self.pool.gpu_k_caches[layer]
        _timing.note_bytes("pool.d2h_token",
                           n * k_cache.shape[1] * k_cache.shape[3] * k_cache.element_size())
        _timing.start("pool.h2d")
        kern.tok_fill(keys_d.contiguous(), victims.contiguous(), self.owner[layer],
                      self.tok_slot[layer].view(-1), lu, self.tick, eng.cpu_slot[layer],
                      self.mbp, eng.cpu_pool[layer],
                      self._pool_dev + layer * eng.cpu_pool.stride(0) * eng.cpu_pool.element_size(),
                      k_cache, int(self.pool.base), self.P, self._ctr)
        _timing.end("pool.h2d")
        self._stamped += n
        return True

    def phys(self, layer, key, cold, idx, indptr, indices) -> torch.Tensor:
        """Physical token address per position: the paged location for GPU
        blocks, the staged slot for cold tokens (0 for an unfetched one --
        only ever read with weight 0 by the chunked fallback)."""
        B = idx.shape[0]
        blk = torch.div(idx, self.P, rounding_mode="floor")
        gpu = indices[indptr[:B].view(B, 1).long() + blk].long() * self.P + (idx - blk * self.P)
        slot = self.tok_slot[layer].view(-1)[key].long()
        staged = torch.where(slot >= 0, slot + self.pool.base * self.P, 0)
        return torch.where(cold, staged, gpu)

    def _host_tokens(self, layer: int, keys: torch.Tensor) -> torch.Tensor:
        """``[n, H, 2C]`` host KV for ``keys`` (block-offload source)."""
        P, MB = self.P, self.mbp // self.P
        s = torch.div(keys, self.mbp, rounding_mode="floor")
        pos = keys - s * self.mbp
        blk = torch.div(pos, P, rounding_mode="floor")
        off = pos - blk * P
        ub, inv = torch.unique(s * MB + blk, return_inverse=True)
        rid_of = {v: k for k, v in self.res._slot_of.items()}
        cpu = []
        for u in ub.tolist():
            loc = self.res.get(BlockRef(rid_of[u // MB], u % MB))
            if loc is None or loc.cpu_slot is None:
                raise RuntimeError(f"token fetch: block {(rid_of[u // MB], u % MB)} has no CPU copy")
            self.pool.offload_manager.wait_cpu_slot(loc.cpu_slot)
            cpu.append(loc.cpu_slot)
        slot_t = torch.tensor(cpu, dtype=torch.long)[inv]
        _timing.start("pool.cpu_gather")
        gathered = self.pool.cpu_cache[layer][slot_t, :, off, :, :]   # [n, 2, H, C]
        _timing.end("pool.cpu_gather")
        _timing.start("pool.transform")
        out = gathered.permute(0, 2, 1, 3).reshape(int(keys.numel()), gathered.shape[2], -1)
        _timing.end("pool.transform")
        return out

    def _copy(self, layer: int, keys: torch.Tensor, victims: torch.Tensor) -> None:
        """CPU source -> this layer's staging slots, via the pool's pinned ring."""
        P = self.P
        src = getattr(self.res, "host_tokens", None)
        host = src(layer, keys) if src is not None else self._host_tokens(layer, keys)
        pool, n = self.pool, int(keys.numel())
        k_cache = pool.gpu_k_caches[layer]          # [pages, H, P, 2C]
        _timing.note_bytes("pool.d2h_token",
                           n * k_cache.shape[1] * k_cache.shape[3] * k_cache.element_size())
        try:
            slab = pool._acquire_tok_slab(n)
            slab[:n].copy_(host)
            # WAR: the slots may still be read by queued kernels (earlier
            # layers' chunks / the previous step).
            pool.copy_stream.wait_stream(torch.cuda.current_stream(device=self.dev))
            with torch.cuda.stream(pool.copy_stream):
                _timing.start("pool.h2d")
                gpu_slab = slab[:n].to(self.dev, non_blocking=True)
                page = torch.div(victims, P, rounding_mode="floor") + pool.base
                k_cache[page, :, victims - (page - pool.base) * P] = gpu_slab
                _timing.end("pool.h2d")
                event = torch.cuda.Event()
                event.record(pool.copy_stream)
            torch.cuda.current_stream(device=self.dev).wait_stream(pool.copy_stream)
            pool._tok_events[pool._tok_cur] = event
            pool._tok_pending[pool._tok_cur].append(gpu_slab)
        except Exception:
            pool._reset_tok_ring()
            raise


class _EngineSlots:
    """Adapter giving ``TokenStagingStore`` the request-slot / CPU-source view
    of a ``TokenOffloadEngine`` (in place of the block residency mirror)."""

    has_device_mirror = True

    def __init__(self, engine) -> None:
        self.engine = engine
        self.on_slot_release: list = []
        engine.on_release.append(lambda s: [cb(s) for cb in self.on_slot_release])

    @property
    def _mb(self) -> int:
        return -(-self.engine.max_len // self.engine.P)

    @property
    def _num_slots(self) -> int:
        return self.engine.S

    def flush_device(self) -> None:
        pass

    def host_tokens(self, layer: int, keys: torch.Tensor) -> torch.Tensor:
        """Keys -> CPU token-pool rows of this layer (``[n, H, 2C]``)."""
        eng = self.engine
        mbp = self._mb * eng.P
        s = torch.div(keys, mbp, rounding_mode="floor")
        pos = keys - s * mbp
        slots = eng.cpu_slot[layer][s.to(eng.dev), pos.to(eng.dev)].long().cpu()
        if bool((slots < 0).any()):
            raise RuntimeError("token fetch: selected token has no CPU copy")
        return eng.gather_cpu(layer, slots)
