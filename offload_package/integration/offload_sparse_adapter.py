"""Unified Offload Sparse Adapter.

Integrates OffloadController's block scores with sparse kernels:
- sparse_optimized: expects external sparse_idx/sparse_len/sparse_weights
- block_minmax_incr_optimized: has internal scoring via block min/max summary + top-k

This adapter:
1. Gets top-k blocks from OffloadController (scored every W tokens)
2. Converts to sparse_idx/sparse_len/sparse_weights for sparse_optimized
3. Or configures block_minmax's topk = offload_fraction for its internal scorer
"""

from __future__ import annotations

import os
from typing import Any, Optional, Union, Tuple, List

import torch


class OffloadSparseAdapter:
    """Unified adapter: OffloadController scores -> sparse kernels."""
    
    def __init__(
        self,
        orchestrator,  # OffloadOrchestrator with scorer/offload_manager
        kv_layout: str = "NHD",
        page_size: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 64,
        device: str = "cuda",
    ):
        self.orchestrator = orchestrator
        self.kv_layout = kv_layout
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        
        # Config from orchestrator
        self.offload_fraction = orchestrator.offload_config.offload_fraction
        self.sink_size = orchestrator.offload_config.sink_size if hasattr(orchestrator.offload_config, 'sink_size') else 64
        self.local_size = orchestrator.offload_config.local_size if hasattr(orchestrator.offload_config, 'local_size') else 64
        self.channel_num = orchestrator.offload_config.channel_num if hasattr(orchestrator.offload_config, 'channel_num') else -1
        if self.channel_num == -1:
            self.channel_num = head_dim
        
        # State
        self._cached_sparse_idx: Optional[torch.Tensor] = None
        self._cached_sparse_len: Optional[torch.Tensor] = None
        self._cached_sparse_weights: Optional[torch.Tensor] = None
        self._cached_request_ids: List[str] = []
        self._plan_called = False
        
        # Kernel reference (set by backend)
        self._kernel_wrapper = None
    
    def set_kernel_wrapper(self, wrapper):
        """Store reference to kernel wrapper for direct score passing."""
        self._kernel_wrapper = wrapper
    
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        **kwargs,
    ) -> None:
        """Cache plan metadata for scoring."""
        self._indptr = indptr
        self._indices = indices
        self._last_page_len = last_page_len
        self._kv_cache_spec = (num_kv_heads, head_dim, page_size)
        self._plan_called = True
    
    def _get_topk_blocks_from_controller(self, request_ids: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k block indices and lengths from OffloadController for given requests."""
        if not request_ids or self.orchestrator is None:
            return None, None
        
        # Get scored blocks from orchestrator's offload_manager
        scored_blocks = self.orchestrator._scored  # set of BlockRef
        
        if not scored_blocks:
            return None, None
        
        # Group by request_id
        req_to_blocks = {}
        for ref in scored_blocks:
            if ref.request_id in request_ids:
                if ref.request_id not in req_to_blocks:
                    req_to_blocks[ref.request_id] = []
                req_to_blocks[ref.request_id].append(ref.block_idx)
        
        if not req_to_blocks:
            return None, None
        
        # Build per-request block lists
        B = len(request_ids)
        H_q = self.num_kv_heads  # assuming H_q == H_kv for now
        
        # For each request, get its blocks and convert to token indices
        # Each block = page_size tokens
        all_indices = []
        all_lens = []
        
        for req_id in request_ids:
            block_indices = req_to_blocks.get(req_id, [])
            if not block_indices:
                all_indices.append(torch.empty(0, dtype=torch.int64, device=self.device))
                all_lens.append(0)
                continue
            
            # Convert block indices to token indices (block_idx * page_size)
            # The sparse kernel expects token indices, not block indices
            token_indices = []
            for b_idx in block_indices:
                start = b_idx * self.page_size
                end = start + self.page_size
                token_indices.extend(range(start, end))
            
            if token_indices:
                all_indices.append(torch.tensor(token_indices, dtype=torch.int64, device=self.device))
                all_lens.append(len(token_indices))
            else:
                all_indices.append(torch.empty(0, dtype=torch.int64, device=self.device))
                all_lens.append(0)
        
        # Pad to same length
        max_len = max(lens) if (lens := [len(x) for x in all_indices]) else 0
        if max_len == 0:
            return None, None
        
        padded_indices = []
        for idx_tensor in all_indices:
            if len(idx_tensor) < max_len:
                pad = torch.zeros(max_len - len(idx_tensor), dtype=torch.int64, device=self.device)
                padded_indices.append(torch.cat([idx_tensor, pad]))
            else:
                padded_indices.append(idx_tensor[:max_len])
        
        sparse_idx = torch.stack(padded_indices).view(B, 1, max_len).expand(B, H_q, max_len)
        sparse_len = torch.tensor(lens, dtype=torch.int32, device=self.device).view(B, 1, 1).expand(B, H_q, 1)
        sparse_weights = torch.ones((B, H_q, max_len), dtype=torch.float32, device=self.device)
        
        return sparse_idx, sparse_len, sparse_weights
    
    def _build_sparse_pattern_from_controller(self, request_ids: List[str]):
        """Build sparse pattern from offload controller's scored blocks."""
        # Debug: print what's in _scored
        scored = getattr(self.orchestrator, '_scored', set())
        _dbg(f"[ADAPTER DEBUG] _scored has {len(scored)} blocks, request_ids={request_ids}")
        for ref in list(scored)[:5]:
            _dbg(f"  BlockRef: req={ref.request_id}, block={ref.block_idx}")
        
        return self._get_topk_blocks_from_controller(request_ids)
    
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run sparse decode using offload controller's scores.
        
        For sparse_optimized: pass sparse_idx/sparse_len/sparse_weights
        For block_minmax: pass topk=sink_size/local_size via kwargs
        """
        if not self._plan_called:
            raise RuntimeError("plan() must be called before run().")
        
        # Get request_ids from orchestrator (current step's active requests)
        # The metadata builder sets this during build()
        request_ids = getattr(self.orchestrator, '_active_request_ids', [])
        
        if not request_ids:
            # Fallback: dense decode
            return self._run_dense(q, paged_kv_cache, *args, **kwargs)
        
        # Build sparse pattern from offload scores
        sparse_idx, sparse_len, sparse_weights = self._build_sparse_pattern_from_controller(request_ids)
        
        if sparse_idx is None:
            # No scores available, fallback to dense
            return self._run_dense(q, paged_kv_cache, *args, **kwargs)
        
        # Cache for worker connector (staging)
        self._cached_sparse_idx = sparse_idx.detach().cpu()
        self._cached_sparse_len = sparse_len.detach().cpu()
        self._cached_sparse_weights = sparse_weights.detach().cpu()
        self._cached_request_ids = request_ids
        
        # Write to module-level cache for worker connector
        try:
            import skylight.backend as skylight_backend
            skylight_backend._sparse_pattern_cache['sparse_idx'] = sparse_idx.detach().cpu()
            skylight_backend._sparse_pattern_cache['sparse_len'] = sparse_len.detach().cpu()
            skylight_backend._sparse_pattern_cache['request_ids'] = request_ids
        except Exception:
            pass
        
        # Detect which kernel wrapper we're using
        if hasattr(self._kernel_wrapper, '_sparse_ops'):
            # sparse_optimized wrapper
            return self._kernel_wrapper.run(
                q, paged_kv_cache,
                sparse_len, sparse_idx, sparse_weights,
                *args, **kwargs
            )
        else:
            # block_minmax or oracle - pass topk/sink/local via kwargs
            return self._kernel_wrapper.run(
                q, paged_kv_cache,
                topk=self.offload_fraction,
                sink_size=self.sink_size,
                local_size=self.local_size,
                channel_num=self.channel_num,
                *args, **kwargs
            )
    
    def _run_dense(self, q, paged_kv_cache, *args, **kwargs):
        """Fallback to dense decode."""
        if self._kernel_wrapper:
            return self._kernel_wrapper.run(q, paged_kv_cache, *args, **kwargs)
        raise RuntimeError("No kernel wrapper available")
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel_wrapper, name)
    
    # For worker connector: expose cached pattern
    def get_cached_sparse_pattern(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._cached_sparse_idx, self._cached_sparse_len


def create_offload_sparse_adapter(
    orchestrator,
    kernel_wrapper,
    **kwargs,
) -> OffloadSparseAdapter:
    """Factory: create adapter and bind to kernel wrapper."""
    adapter = OffloadSparseAdapter(orchestrator, **kwargs)
    kernel_wrapper._offload_adapter = adapter
    adapter.set_kernel_wrapper(kernel_wrapper)
    return adapter