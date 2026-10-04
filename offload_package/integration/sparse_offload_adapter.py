"""Sparse Attention + Offload Integration.

This adapter uses:
1. Oracle/Block-minmax scorer to compute sparse pattern (top-k blocks)
2. sparse_optimized kernel for FAST decode with pre-computed pattern
3. Offload controller to know which blocks are on CPU vs GPU
4. Sync staging: fetch top-k blocks from CPU to GPU (lag 0)

The flow:
- Every decode step: get sparse pattern from scorer (or cached)
- Convert block indices to token indices (block_idx * page_size)  
- Fetch top-k blocks from CPU to GPU staging if offloaded
- Pass sparse_idx/sparse_len to sparse_optimized kernel (FAST)
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union, List, Tuple

import torch

from original_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    _unpack_paged_kv_cache,
)
from original_optimized import BatchDecodeWithPagedKVCacheWrapper as DenseOptimizedWrapper


class SparseOffloadAdapterWrapper:
    """Unified adapter: Sparse scorer (oracle/block-minmax) + sparse_optimized kernel + Offload.
    
    Uses sparse_optimized kernel for FAST decode with pre-computed sparse pattern.
    Offload controller scores blocks; top-k blocks fetched from CPU to GPU.
    """
    
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[list[Any]] = None,
        topk: Optional[float] = None,
        channel_num: Optional[int] = None,
        sink_size: Optional[int] = None,
        local_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        # Internal sparse_optimized wrapper (FAST decode path)
        self._inner = DenseOptimizedWrapper(
            float_workspace_buffer=float_workspace_buffer,
            kv_layout=kv_layout,
            use_cuda_graph=use_cuda_graph,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
            backend=backend,
            jit_args=jit_args,
        )
        
        # Config
        self._topk = topk if topk is not None else float(os.environ.get("SKYLIGHT_SPARSE_TOPK", "0.1"))
        self._sink_size = sink_size if sink_size is not None else int(os.environ.get("SKYLIGHT_SPARSE_SINK", "64"))
        self._local_size = local_size if local_size is not None else int(os.environ.get("SKYLIGHT_SPARSE_LOCAL", "64"))
        self._channel_num = channel_num if channel_num is not None else int(os.environ.get("SKYLIGHT_SPARSE_CHANNEL_NUM", "8"))
        self._max_seq_len = max_seq_len if max_seq_len is not None else int(os.environ.get("SKYLIGHT_SPARSE_MAX_SEQ_LEN", "8192"))
        
        # Offload adapter (uses offload controller's scores)
        self._offload_adapter = None
        
        # Oracle scorer (for computing sparse pattern) - LAZY import
        self._scorer = None
        
        # State
        self._indptr: Optional[torch.Tensor] = None
        self._indices: Optional[torch.Tensor] = None
        self._last_page_len: Optional[torch.Tensor] = None
        self._kv_cache_spec: Optional[Tuple[int, int, int]] = None
        self._layout = kv_layout
        self._last_sparse_idx: Optional[torch.Tensor] = None
        self._last_sparse_len: Optional[torch.Tensor] = None
        self._last_sparse_weights: Optional[torch.Tensor] = None
        
        # Sync staging (lag 0): fetch blocks in same step
        self._sync_staging = os.environ.get("SKYLIGHT_SYNC_STAGING") == "1"
        self._orchestrator = None
        
        # Cache pattern for reuse (avoid recomputing every step)
        self._cached_pattern = None
        self._pattern_step = -1
    
    def _get_scorer(self):
        """Lazy import of scorer to avoid circular imports."""
        from offload_package.scoring.sparse_scorer import SparseAttentionScorer
        return SparseAttentionScorer(
            topk=self._topk,
            sink_size=self._sink_size,
            local_size=self._local_size,
            channel_num=self._channel_num,
            max_seq_len=self._max_seq_len,
        )
    
    def _get_offload_adapter(self):
        """Lazy import of offload adapter to avoid circular imports."""
        if self._offload_adapter is None and self._orchestrator is not None:
            from offload_package.integration.offload_sparse_adapter import OffloadSparseAdapter
            self._offload_adapter = OffloadSparseAdapter(
                orchestrator=self._orchestrator,
                kv_layout=self._layout,
                page_size=self._kv_cache_spec[2] if self._kv_cache_spec else 16,
                num_kv_heads=self._kv_cache_spec[0] if self._kv_cache_spec else 4,
                head_dim=self._kv_cache_spec[1] if self._kv_cache_spec else 64,
                device="cuda",
            )
            self._offload_adapter.set_kernel_wrapper(self._inner)
        return self._offload_adapter
    
    def set_orchestrator(self, orchestrator):
        """Set offload orchestrator for staging and scoring."""
        self._orchestrator = orchestrator
    
    def _do_sync_staging(self, paged_kv_cache: Union[torch.Tensor, tuple], pattern):
        """Perform synchronous staging: fetch blocks to GPU, patch indices."""
        if self._orchestrator is None:
            return None
        
        # Prepare staging using current pattern
        request_ids = getattr(self, '_cached_request_ids', None)
        if request_ids is None:
            return None
        
        try:
            staging_map = self._orchestrator.prepare_staging(
                request_ids,
                pattern.sparse_idx.to(self._orchestrator.staging_pool.gpu_k_caches[0].device),
                pattern.sparse_len.to(self._orchestrator.staging_pool.gpu_k_caches[0].device),
            )
            
            # Launch async copies for each block
            for ref, staging_slot in staging_map.items():
                loc = self._orchestrator.residency.get(ref)
                if loc and loc.cpu_slot is not None:
                    self._launch_cpu_to_staging_copy(loc.cpu_slot, staging_slot)
            
            # Wait for copies to complete
            torch.cuda.synchronize()
            
            # Patch paged_kv_indices to point to staging slots
            from offload_package.selection.adapter import patch_paged_kv_indices_with_staging
            
            patched_indices = patch_paged_kv_indices_with_staging(
                self._indices,
                self._indptr,
                request_ids,
                pattern.sparse_idx,
                pattern.sparse_len,
                self._kv_cache_spec[2],  # page_size
                self._orchestrator.staging_base_page_idx,
                self._orchestrator.residency,
                staging_map,
            )
            
            return patched_indices
        except Exception as e:
            _dbg(f"[SYNC STAGING] Failed: {e}")
            return None
    
    def _launch_cpu_to_staging_copy(self, cpu_slot: int, staging_slot: int):
        """Launch async copy from CPU cache to GPU staging slot."""
        if self._orchestrator is None:
            return
        staging_pool = self._orchestrator.staging_pool
        for layer, (k_cache, v_cache) in enumerate(zip(
            staging_pool.gpu_k_caches, staging_pool.gpu_v_caches
        )):
            k_src = staging_pool.cpu_cache[layer, cpu_slot, 0]
            v_src = staging_pool.cpu_cache[layer, cpu_slot, 1]
            k_src = k_src.permute(1, 0, 2).contiguous()
            v_src = v_src.permute(1, 0, 2).contiguous()
            combined = torch.cat([k_src, v_src], dim=-1)
            k_cache[staging_pool.base + staging_slot].copy_(combined, non_blocking=False)
    
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> None:
        """Forward plan to inner wrapper and cache metadata for scorer."""
        self._inner.plan(
            indptr, indices, last_page_len,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=block_tables,
            seq_lens=seq_lens,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        
        # Cache for scorer
        self._indptr = indptr
        self._indices = indices
        self._last_page_len = last_page_len
        self._kv_cache_spec = (num_kv_heads, head_dim, page_size)
    
    def _unpack_paged_kv_cache(
        self,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract K and V caches from paged KV cache."""
        k_cache, v_cache, *_ = _unpack_paged_kv_cache(paged_kv_cache, self._layout)
        return k_cache, v_cache
    
    def _get_kv_strides(
        self,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[int, int, int, int]:
        """Get KV cache strides for scorer."""
        _, _, stride_page, stride_n, stride_h, v_offset, _, _ = _unpack_paged_kv_cache(paged_kv_cache, self._layout)
        return (int(stride_page), int(stride_n), int(stride_h), int(v_offset))
    
    def _compute_sparse_pattern(self, q: torch.Tensor, paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute sparse pattern using oracle scorer."""
        if self._indptr is None or self._last_page_len is None:
            raise RuntimeError("plan() must be called before run().")
        
        k_cache, v_cache = self._unpack_paged_kv_cache(paged_kv_cache)
        
        # Use oracle scorer to compute sparse pattern
        scorer = self._get_scorer()
        pattern = scorer.compute_pattern(
            q=q,
            k_cache=k_cache,
            indptr=self._indptr,
            indices=self._indices,
            last_page_len=self._last_page_len,
            kv_strides=self._get_kv_strides(paged_kv_cache),
            page_size=self._kv_cache_spec[2],
            num_kv_heads=self._kv_cache_spec[0],
            head_dim=self._kv_cache_spec[1],
            layout=self._layout,
        )
        
        # Cache for reuse
        self._cached_pattern = (pattern.sparse_idx, pattern.sparse_len, pattern.sparse_weights)
        
        return pattern.sparse_idx, pattern.sparse_len, pattern.sparse_weights
    
    def _should_recompute_pattern(self) -> bool:
        """Check if we should recompute the sparse pattern.
        
        Recompute when:
        1. No cached pattern yet
        2. Offload controller triggered a new scoring round
        """
        if self._cached_pattern is None:
            _dbg(f"[ADAPTER DEBUG] Recompute: no cached pattern")
            return True
        
        # Check if offload controller triggered a new scoring
        _dbg(f"[ADAPTER DEBUG] Check: _orchestrator is None: {self._orchestrator is None}")
        if self._orchestrator is not None:
            _dbg(f"[ADAPTER DEBUG] Orchestrator type: {type(self._orchestrator)}, id: {id(self._orchestrator)}")
            _dbg(f"[ADAPTER DEBUG] Has _last_offload_step: {hasattr(self._orchestrator, '_last_offload_step')}")
            if hasattr(self._orchestrator, '_last_offload_step'):
                last_step = self._orchestrator._last_offload_step
                _dbg(f"[ADAPTER DEBUG] Check: _last_offload_step={last_step}, _pattern_step={self._pattern_step}, equal={last_step == self._pattern_step}")
                if last_step != self._pattern_step:
                    self._pattern_step = last_step
                    _dbg(f"[ADAPTER DEBUG] Recompute: offload step changed to {self._pattern_step}")
                    return True
        
        _dbg(f"[ADAPTER DEBUG] Reuse cached pattern (step={self._pattern_step})")
        return False
    
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args: Any,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        q_len_per_req: Optional[int] = 1,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        kv_cache_sf: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run sparse decode with offload integration.
        
        1. Compute (or reuse) sparse pattern from oracle scorer
        2. Convert block indices to token indices
        3. Fetch top-k blocks from CPU to GPU staging (if offloaded)
        4. Call sparse_optimized kernel with sparse pattern (FAST)
        """
        _dbg(f"[ADAPTER DEBUG] run() called, _cached_pattern is None: {self._cached_pattern is None}")
        if self._indptr is None:
            raise RuntimeError("plan() must be called before run().")
        
        if q_len_per_req is None:
            q_len_per_req = 1
        if q_len_per_req != 1:
            raise NotImplementedError("Only q_len_per_req == 1 supported.")
        
        # Compute or reuse sparse pattern
        if self._should_recompute_pattern():
            sparse_idx, sparse_len, sparse_weights = self._compute_sparse_pattern(q, paged_kv_cache)
        else:
            sparse_idx, sparse_len, sparse_weights = self._cached_pattern
        
        # Sync staging: fetch top-k blocks from CPU to GPU (lag 0)
        if self._sync_staging and self._orchestrator is not None and self._get_offload_adapter() is not None:
            try:
                # Get active request IDs
                request_ids = getattr(self._orchestrator, '_active_request_ids', [])
                if request_ids:
                    sparse_idx, sparse_len, sparse_weights = self._cached_pattern
                    staging_map = self._orchestrator.prepare_staging(
                        request_ids,
                        sparse_idx.to(self._orchestrator.staging_pool.gpu_k_caches[0].device),
                        sparse_len.to(self._orchestrator.staging_pool.gpu_k_caches[0].device),
                    )
                    
                    # Launch async copies for each block
                    for ref, staging_slot in staging_map.items():
                        loc = self._orchestrator.residency.get(ref)
                        if loc and loc.cpu_slot is not None:
                            self._launch_cpu_to_staging_copy(loc.cpu_slot, staging_slot)
                    
                    # Wait for copies to complete (lag 0)
                    torch.cuda.synchronize()
            except Exception as e:
                _dbg(f"[SYNC STAGING] Failed: {e}")
        
        # Call sparse_optimized kernel with sparse pattern (FAST!)
        sparse_idx, sparse_len, sparse_weights = self._cached_pattern
        return self._inner.run(
            q, paged_kv_cache,
            sparse_len, sparse_idx, sparse_weights,
            *args,
            q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
            out=out, lse=lse, return_lse=return_lse,
            enable_pdl=enable_pdl, window_left=window_left,
            sinks=sinks, q_len_per_req=q_len_per_req,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            kv_cache_sf=kv_cache_sf,
        )
    
    def _launch_cpu_to_staging_copy(self, cpu_slot: int, staging_slot: int):
        """Launch async copy from CPU cache to GPU staging slot."""
        if self._orchestrator is None:
            return
        staging_pool = self._orchestrator.staging_pool
        for layer, (k_cache, v_cache) in enumerate(zip(
            staging_pool.gpu_k_caches, staging_pool.gpu_v_caches
        )):
            k_src = staging_pool.cpu_cache[layer, cpu_slot, 0]
            v_src = staging_pool.cpu_cache[layer, cpu_slot, 1]
            k_src = k_src.permute(1, 0, 2).contiguous()
            v_src = v_src.permute(1, 0, 2).contiguous()
            combined = torch.cat([k_src, v_src], dim=-1)
            k_cache[staging_pool.base + staging_slot].copy_(combined, non_blocking=False)
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
    
    # For offload system: expose cached pattern
    def get_cached_sparse_pattern(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._last_sparse_idx, self._last_sparse_len