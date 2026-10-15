"""Pinned CPU store for individually offloaded KV tokens.

Block-level offloading stores whole pages in a page-shaped
``[L, S, 2, page, H, D]`` pool.  Token-level offloading needs a
token-granular store: after compaction the surviving prefix is dense and
the evicted tail is a sparse set of individual tokens, so the pool holds
one token per slot with layout ``[num_layers, num_slots, H, 2*head_dim]``
-- exactly the shape the GPU-side gather produces, which keeps both the
D2H offload and the H2D fetch a single contiguous burst per layer.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


class CPUTokenPool:
    """Free-list allocator + batched D2H/H2D over a pinned token store."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_kv_heads: int,
        combined_head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if num_layers <= 0 or num_slots <= 0 or num_kv_heads <= 0:
            raise ValueError("num_layers/num_slots/num_kv_heads must be > 0")
        if combined_head_dim <= 0 or combined_head_dim % 2:
            raise ValueError("combined_head_dim must be a positive even int")
        if num_slots > 1 << 31:
            raise ValueError("num_slots too large")
        self.num_layers = int(num_layers)
        self.num_slots = int(num_slots)
        self.dtype = dtype
        self.device = device
        self.pool = torch.empty(
            (self.num_layers, self.num_slots, num_kv_heads, combined_head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        self._free = list(range(self.num_slots - 1, -1, -1))
        self._store_events: dict[int, torch.cuda.Event] = {}
        if device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=device)
        else:
            self._copy_stream = None

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n < 0:
            raise ValueError("n must be >= 0")
        if n > len(self._free):
            raise RuntimeError(
                f"CPU token pool exhausted: need {n}, free {len(self._free)}"
            )
        return [self._free.pop() for _ in range(n)]

    def free(self, slots: Sequence[int]) -> None:
        for slot in slots:
            slot = int(slot)
            if slot < 0 or slot >= self.num_slots:
                raise ValueError(f"slot out of range: {slot}")
            self._free.append(slot)
            self._store_events.pop(slot, None)

    def store_from_gpu(
        self,
        kv_caches: Sequence[torch.Tensor],
        block_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> list[int]:
        """Offload a batch of tokens GPU -> CPU, one D2H burst per layer.

        Mirrors the block pool's batched store: per-layer advanced-index
        gather on the current stream, one contiguous slab D2H, an event
        sync, then a host scatter into the pinned pool.  Slots stay
        guarded by a store event until the first fetch waits on them.

        Returns the allocated pool slots, aligned with the input tokens.
        """
        n = int(block_ids.numel())
        if n != int(offsets.numel()):
            raise ValueError(
                f"block_ids ({n}) and offsets ({offsets.numel()}) must "
                "address the same number of tokens"
            )
        if n == 0:
            return []
        slots = self.allocate(n)
        slot_ids_t = torch.tensor(slots, dtype=torch.long)
        dev = block_ids.device

        gathered_hold: list[torch.Tensor] = []
        slabs_hold: list[torch.Tensor] = []
        host_hold: list[torch.Tensor] = []
        try:
            for layer, cache in enumerate(kv_caches):
                if cache.shape[0] <= int(block_ids.max().item()):
                    raise IndexError(
                        f"layer {layer}: block id out of cache range"
                    )
                gathered = cache[block_ids.to(dev), :, offsets.to(dev), :]
                slab_host = gathered.to("cpu", non_blocking=True)
                gathered_hold.append(gathered)
                slabs_hold.append(gathered)
                host_hold.append(slab_host)
            event = torch.cuda.Event()
            event.record()
            event.synchronize()
            for layer, slab_host in enumerate(host_hold):
                self.pool[layer].index_copy_(0, slot_ids_t, slab_host)
        except Exception:
            for slot in slots:
                self._free.append(slot)
            raise
        finally:
            del gathered_hold, slabs_hold, host_hold

        for slot in slots:
            self._store_events[slot] = event
        return slots

    def wait_slots(self, slots: Sequence[int]) -> None:
        """Block until every pending D2H store of ``slots`` has landed."""
        seen: set[torch.cuda.Event] = set()
        for slot in slots:
            event = self._store_events.get(int(slot))
            if event is not None and event not in seen:
                event.synchronize()
                seen.add(event)

    def fetch_to_gpu(
        self,
        slots: Sequence[int],
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[list[torch.Tensor], torch.cuda.Event | None]:
        """Gather a batch of tokens CPU -> GPU, one H2D burst per layer.

        Waits on any pending store events first so a token stored this
        round can be fetched immediately.  Returns per-layer
        ``[n, H, 2*head_dim]`` GPU tensors and the completion event
        (None on CPU-only setups).
        """
        if not slots:
            return [], None
        self.wait_slots(slots)
        slot_ids_cpu = torch.tensor(
            [int(s) for s in slots], dtype=torch.long
        )
        if self.device.type != "cuda":
            chunks = [
                self.pool[layer].index_select(0, slot_ids_cpu).to(
                    self.device, non_blocking=False
                )
                for layer in range(self.num_layers)
            ]
            return chunks, None

        # H2D on a side stream; the current stream is ordered behind it so
        # consumers can use the chunks immediately after this call.
        active_stream = stream or self._copy_stream
        assert active_stream is not None
        with torch.cuda.stream(active_stream):
            chunks = [
                self.pool[layer]
                .index_select(0, slot_ids_cpu)
                .to(self.device, non_blocking=True)
                for layer in range(self.num_layers)
            ]
            torch.cuda.current_stream(self.device).wait_stream(active_stream)
        event = torch.cuda.Event()
        event.record(active_stream)
        return chunks, event
