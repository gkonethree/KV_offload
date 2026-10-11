from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.logger import init_logger

from ..controller import KVCompressionController
from ..offload.manager import KVOffloadManager
from ..selection.adapter import (
    collect_needed_block_refs,
    patch_paged_kv_indices_with_staging,
)
from ..scoring.base import BlockRef
from ..staging.pool import KVStagingPool
from ..staging.residency import Residency, ResidencyTable
from offload_package.debug import dbg as _dbg
from offload_package import timing as _timing

logger = init_logger(__name__)


class OffloadOrchestrator:
    """Runtime coordinator between vLLM, scorer, offloader and sparse attention."""

    def __init__(
        self,
        *,
        offload_manager: KVOffloadManager,
        staging_pool: KVStagingPool,
        residency: ResidencyTable,
        controller: KVCompressionController,
        page_size: int,
        staging_base_page_idx: int,
        offload_fraction: float,
    ) -> None:
        self.offload_fraction = float(offload_fraction)
        self.offload_manager = offload_manager
        self.staging_pool = staging_pool
        self.residency = residency
        self.controller = controller
        self.page_size = int(page_size)
        self.staging_base_page_idx = int(staging_base_page_idx)
        self._request_blocks: dict[str, list[BlockRef]] = {}
        self._scored: set[BlockRef] = set()
        self._sparse_mode = True  # Enable offloading by default
        self._last_offload_step = 0  # Track offload steps for pattern caching
        self._active_request_ids: list[str] = []  # Track active requests for staging
        # Set by the scheduler: with true freeing, offloaded pages are
        # recycled, so CPU-resident unstaged indices must be null-filled.
        self.true_freeing = False
        # Last residency CPU-version seen by the lag prefetch (gate state).
        self._lag_seen_cpu_version = 0

    def _enable_sparse_mode(self) -> None:
        """Enable sparse attention offloading mode."""
        self._sparse_mode = True

    # FIX — track last seen block count per request and only process new tail blocks:
    def sync_request_blocks(self, request_id, gpu_block_ids, num_computed_tokens: int):
        # Skip warmup requests
        if str(request_id).startswith("_warmup_"):
            return []
            
        prev_count = len(self._request_blocks.get(request_id, []))
        refs = []
        for block_idx, gpu_id in enumerate(gpu_block_ids):
            ref = BlockRef(str(request_id), int(block_idx))
            complete = (block_idx + 1) * self.page_size <= int(num_computed_tokens)
            old = self.residency.get(ref)
            if old is not None and old.residency in (Residency.CPU, Residency.STAGING):
                # Offloaded blocks keep their CPU copy as the source of truth
                # and never rebind to a GPU page: after true-free surgery the
                # table entry is a placeholder (or a stale id from the runner's
                # block table), never a valid live page.
                if complete and not old.complete:
                    self.residency.mark_complete(ref)
                refs.append(ref)
                continue
            if old is None or old.gpu_block_id != int(gpu_id):
                self.residency.mark_gpu(ref, int(gpu_id), complete=complete)
            elif complete and not old.complete:
                self.residency.mark_complete(ref)
            refs.append(ref)

        # Score newly added blocks (batched: one GPU pass, one host sync)
        _score_refs, _score_gids = [], []
        for ref in refs[prev_count:]:
            loc = self.residency.get(ref)
            if loc is not None and loc.complete and ref not in self._scored:
                if loc.residency == Residency.GPU:
                    _score_refs.append(ref)
                    _score_gids.append(loc.gpu_block_id)
                    self._scored.add(ref)
        # Also score previously tracked blocks that just became complete
        for ref in refs[:prev_count]:
            loc = self.residency.get(ref)
            if loc is not None and loc.complete and ref not in self._scored:
                if loc.residency == Residency.GPU:
                    _score_refs.append(ref)
                    _score_gids.append(loc.gpu_block_id)
                    self._scored.add(ref)
        # Mark completes BEFORE the batch score (scoring reads GPU pages).
        for ref in _score_refs:
            self.residency.mark_complete(ref)
        if _score_refs:
            self.offload_manager.score_blocks(_score_refs, _score_gids)
        
        self._request_blocks[request_id] = refs
        return refs

    def on_prefill_complete(
        self, request_id: str, num_computed_tokens: int = 0
    ) -> list[BlockRef]:
        """Mark this request's FULL blocks complete and score them.

        Offloading itself is NOT done here: the scheduler-side
        _trigger_global_offload() runs right after prefill completion and is
        the single offload path (it sets store events + touches GPU blocks so
        completions are tracked). Offloading here as well would double-copy
        to different CPU slots and leak slots.

        Returns an empty list (kept for call-site compatibility).
        """
        # Skip warmup requests
        if str(request_id).startswith("_warmup_"):
            return []

        if not self._sparse_mode:
            return []

        # Completeness is per FILL STATE, not per phase: the tail block that
        # prefill left partially filled is still being written by decode, so
        # marking it complete would let the offloader snapshot it with
        # unfilled rows and free the GPU page mid-write -- every later decode
        # token in that block would attend to a stale/garbage CPU copy. Only
        # blocks with (idx+1)*page <= num_computed_tokens graduate.
        refs = self._request_blocks.get(request_id, [])
        _score_refs, _score_gids = [], []
        for ref in refs:
            loc = self.residency.get(ref)
            if loc is None:
                continue
            filled = (ref.block_idx + 1) * self.page_size <= int(num_computed_tokens)
            if not filled or loc.complete:
                continue
            self.residency.mark_complete(ref)
            # Score blocks that became complete now
            if loc.residency == Residency.GPU and ref not in self._scored:
                _score_refs.append(ref)
                _score_gids.append(loc.gpu_block_id)
                self._scored.add(ref)
        if _score_refs:
            self.offload_manager.score_blocks(_score_refs, _score_gids)
        return []

    def on_decode_step(
        self,
        request_ids: Sequence[str],
        num_tokens: int = 1,
    ) -> list[BlockRef]:

        triggered_requests = []

        for request_id in request_ids:
            if self.controller.on_decode(
                request_id,
                num_tokens,
            ):
                triggered_requests.append(request_id)

        if not triggered_requests:
            return []

        # Track active request IDs for staging
        self._active_request_ids = list(request_ids)

        # Debug: Print block residency for each active request
        for request_id in request_ids:
            refs = self.residency.refs_for_request(request_id)
            if refs:
                _dbg(f"[ORCH DECODE] req={request_id}:")
                for bref in refs:
                    loc = self.residency.get(bref)
                    if loc:
                        _dbg(f"  block_idx={bref.block_idx}, gpu_block_id={loc.gpu_block_id}, residency={loc.residency.name}, complete={loc.complete}")

        # Global candidate pool for all triggered requests
        all_gpu_complete = self.residency.all_gpu_blocks(complete_only=True)
        
        # Increment offload step counter for pattern caching
        if triggered_requests:
            self._last_offload_step += 1
            _dbg(f"[ORCH DEBUG] Offload triggered! Step={self._last_offload_step}")

        return self.offload_manager.offload_fraction(
            all_gpu_complete, self.offload_fraction, self._scored
        )

    def prepare_staging(
        self,
        request_ids: Sequence[str],
        sparse_idx: torch.Tensor,
        sparse_len: torch.Tensor,
    ) -> dict[BlockRef, int]:
        _dbg(f"[ORCHESTRATOR DEBUG] prepare_staging called: request_ids={request_ids}, sparse_idx.shape={sparse_idx.shape}")
        # Lag-prefetch gate: skip the (host-sync + per-token loop) discovery
        # pass when no block transitioned into or out of CPU-only residency
        # since the last prefetch. With no transitions the previously staged
        # set is still valid; any newly-selected cold block is fetched by the
        # same-step in-run path, which is the correctness path anyway. The
        # gate re-arms on every transition (offload triggers, fetches,
        # evictions), so a freshly offloaded block is prefetched on the next
        # build after its transition.
        version = self.residency.cpu_version
        if version == self._lag_seen_cpu_version:
            _dbg("[ORCHESTRATOR DEBUG] prepare_staging: skipped (no residency transitions)")
            return {}
        self._lag_seen_cpu_version = version

        _timing.start("lag.collect")
        needed = collect_needed_block_refs(
            request_ids, sparse_idx, sparse_len, self.page_size
        )
        _timing.end("lag.collect")
        cpu_needed = [
            ref for ref in needed
            if (loc := self.residency.get(ref)) is not None
            and loc.residency == Residency.CPU
        ]
        _dbg(f"[ORCHESTRATOR DEBUG] needed {len(needed)} blocks, cpu_needed {len(cpu_needed)}")
        _timing.start("lag.stage")
        result, skipped = self.staging_pool.ensure_resident(cpu_needed)
        _timing.end("lag.stage")
        if skipped:
            logger.warning(
                "prepare_staging (1-step-lag prefetch): %d/%d selected cold "
                "blocks could not be staged (staging exhausted); they are "
                "skipped for this prefetch pass -- the same-step in-run fetch "
                "remains responsible for them",
                len(skipped), len(cpu_needed),
            )
        _dbg(f"[ORCHESTRATOR DEBUG] ensure_resident staged {len(result)} slots")
        return result

    def note_selection(
        self,
        request_ids: Sequence[str],
        sparse_idx: torch.Tensor,
        page_size: int,
    ) -> None:
        """Record last-use ticks for the blocks this step's top-k selected.

        Consumes the per-step selection pattern (layer 0's, union over
        heads). The div+unique run on the pattern's device (a ~50us kernel
        over [B, Hq, S]); only the ~tens-of-blocks unique result is read
        back, so this adds no meaningful sync cost. Scorers without a
        ``note_usage`` hook (paged_eviction, random) make this a no-op.
        """
        # Feed EVERY scorer that consumes recency: the offload scorer (when
        # scorer_type=lru) AND the staging pool's independent eviction scorer
        # (when staging_eviction=lru). Identity-guarded: when they are the
        # same object (legacy "scorer" inheritance) tick it only once.
        scorers = []
        offload_scorer = self.offload_manager.scorer
        evict_scorer = getattr(self.staging_pool, "_eviction_scorer", None)
        if callable(getattr(offload_scorer, "note_usage", None)):
            scorers.append(offload_scorer)
        if (
            evict_scorer is not None
            and evict_scorer is not offload_scorer
            and callable(getattr(evict_scorer, "note_usage", None))
        ):
            scorers.append(evict_scorer)
        if not scorers:
            return
        if sparse_idx.numel() == 0 or page_size <= 0:
            return
        if len(request_ids) < sparse_idx.shape[0]:
            return  # row/request mapping would be wrong
        _timing.start("lru.note")
        blocks_all = torch.div(sparse_idx, page_size, rounding_mode="floor")
        for b in range(sparse_idx.shape[0]):
            rid = str(request_ids[b])
            blocks = blocks_all[b].unique().tolist()
            for scorer in scorers:
                scorer.note_usage(rid, blocks)
        _timing.end("lru.note")

    def patch_paged_kv_indices(
        self,
        paged_kv_indices: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        request_ids: Sequence[str],
        sparse_idx: torch.Tensor,
        sparse_len: torch.Tensor,
        staging_map: dict[BlockRef, int],
    ) -> torch.Tensor:
        return patch_paged_kv_indices_with_staging(
            paged_kv_indices,
            paged_kv_indptr,
            request_ids,
            sparse_idx,
            sparse_len,
            self.page_size,
            self.staging_base_page_idx,
            self.residency,
            staging_map,
        )

    def on_request_done(self, request_id: str) -> None:
        self._cleanup_request(request_id)

    def on_request_preempted(self, request_id: str) -> None:
        """Handle preempted request - same cleanup as finished request."""
        self._cleanup_request(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        refs = self._request_blocks.pop(request_id, [])
        for ref in refs:
            self.staging_pool.release(ref)
            self.offload_manager.free(ref)
            self._scored.discard(ref)
        self.controller.remove(request_id)
