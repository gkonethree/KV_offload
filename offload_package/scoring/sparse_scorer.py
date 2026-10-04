"""Sparse attention scorer using fused Oracle kernels.

Computes sparse attention pattern for sparse_optimized kernel by:
1. Computing compact scores using Oracle's score kernel
2. Top-k selection using FlashInfer's radix topk
3. Combining with sink + local tokens
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class SparsePattern:
    """Sparse attention pattern for sparse_optimized kernel."""
    sparse_idx: torch.Tensor          # int64 [B, H_q, max_S]
    sparse_len: torch.Tensor          # int32 [B, H_q, 1]
    sparse_weights: torch.Tensor      # float32 [B, H_q, max_S]


class SparseAttentionScorer:
    """Computes sparse attention pattern using Oracle's fused score + topk.
    
    Uses:
    - oracle_topk_compute_scores_compact: compact Q@K^T scores over middle range
    - FlashInfer's top_k: warp-level radix top-k selection
    """
    
    def __init__(
        self,
        topk: float = 0.1,
        sink_size: int = 64,
        local_size: int = 64,
        channel_num: int = 8,
        max_seq_len: int = 8192,
    ):
        self.topk = topk
        self.sink_size = sink_size
        self.local_size = local_size
        self.channel_num = channel_num
        self.max_seq_len = max_seq_len
        
        # Load Oracle scorer ops
        from sparse_oracle_topk_sink_local_optimized.cuda_ops import get_scorer_ops
        self._ops = get_scorer_ops()
        
        # Try to import FlashInfer for fast top-k
        self._flashinfer_topk = None
        try:
            import flashinfer
            self._flashinfer_topk = getattr(flashinfer, "top_k", None)
        except ImportError:
            pass
    
    def _topk_impl(self, scores_3d: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k implementation: prefer FlashInfer radix, fallback to torch.topk."""
        if self._flashinfer_topk is not None:
            B, H_q, L = scores_3d.shape
            scores_2d = scores_3d.reshape(B * H_q, L)
            _, idx = self._flashinfer_topk(scores_2d, k=k, sorted=False)
            return idx.view(B, H_q, k)
        return torch.topk(scores_3d, k=k, dim=-1, sorted=False).indices
    
    def compute_pattern(
        self,
        q: torch.Tensor,                   # [B, H_q, D]
        k_cache: torch.Tensor,             # paged KV cache
        indptr: torch.Tensor,              # [B+1] int32
        indices: torch.Tensor,             # [total_pages] int32
        last_page_len: torch.Tensor,       # [B] int32
        kv_strides: tuple,                 # (stride_page, stride_n, stride_h, v_offset)
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        layout: str = "NHD",
    ) -> SparsePattern:
        """
        Compute sparse attention pattern.
        
        Returns:
            SparsePattern with:
            - sparse_idx: int64 [B, H_q, max_S]
            - sparse_len: int32 [B, H_q, 1]
            - sparse_weights: float32 [B, H_q, max_S]
        """
        B = indptr.shape[0] - 1
        H_q = q.shape[1]
        
        # 1. Compute per-batch sequence lengths from indptr + last_page_len
        pages_count = indptr[1:] - indptr[:-1]
        L_per = (pages_count - 1).clamp(min=0).to(torch.int64) * page_size + last_page_len.to(torch.int64)
        L_max = int(L_per.max().item())
        
        # 2. Resolve token range for scoring (exclude sink + local)
        token_start = self.sink_size
        token_end = L_max - self.local_size
        if token_end <= token_start:
            token_end = L_max
            token_start = 0
        
        # 3. Compute compact scores over middle range
        stride_page, stride_n, stride_h, v_offset = kv_strides
        q_contig = q if q.is_contiguous() else q.contiguous()
        # Ensure all tensors are on CUDA
        indptr_cuda = indptr if indptr.is_cuda else indptr.cuda()
        indices_cuda = indices if indices.is_cuda else indices.cuda()
        last_page_len_cuda = last_page_len if last_page_len.is_cuda else last_page_len.cuda()
        scores = self._ops.oracle_topk_compute_scores_compact(
            q_contig, k_cache, stride_page, stride_n, stride_h,
            indptr_cuda, indices_cuda, last_page_len_cuda,
            num_kv_heads, page_size,
            token_start, token_end, self.channel_num
        )  # [B, H_q, L_mid]
        
        # 4. Top-k selection
        n_keys_mid = max(0, token_end - token_start)
        k_eff = max(1, int(n_keys_mid * self.topk))
        topk_idx = self._topk_impl(scores, k_eff)  # [B, H_q, k_eff]
        
        # Free scores tensor early
        del scores
        
        # 5. Add mid_offset (token_start) to make indices global
        topk_idx = topk_idx + token_start
        
        # 6. Combine: [sink] + topk + [local]
        parts = []
        if self.sink_size > 0:
            sink_idx = torch.arange(self.sink_size, device=q.device, dtype=torch.int64)
            sink_idx = sink_idx.view(1, 1, -1).expand(B, H_q, -1)
            parts.append(sink_idx)
        parts.append(topk_idx)
        if self.local_size > 0:
            local_start = (L_per.to(q.device) - self.local_size).clamp(min=self.sink_size)
            local_offsets = torch.arange(self.local_size, device=q.device, dtype=torch.int64)
            local_idx = local_start.view(B, 1, 1) + local_offsets.view(1, 1, -1)
            local_idx = local_idx.expand(B, H_q, -1)
            parts.append(local_idx)
        
        sparse_idx = torch.cat(parts, dim=-1).contiguous()  # [B, H_q, total_k]
        total_k = sparse_idx.shape[-1]
        
        # 7. sparse_len = clamp(L_per, total_k) per request
        sparse_len = L_per.to(q.device).clamp(max=total_k).to(torch.int32).view(B, 1, 1).expand(B, H_q, 1)
        
        # 8. sparse_weights = ones (uniform attention within selected)
        sparse_weights = torch.ones((B, H_q, total_k), dtype=torch.float32, device=q.device)
        
        return SparsePattern(
            sparse_idx=sparse_idx,
            sparse_len=sparse_len,
            sparse_weights=sparse_weights,
        )


def get_scorer_from_env() -> SparseAttentionScorer:
    """Create scorer from environment variables."""
    from skylight.config import SkylightSparseConfig
    cfg = SkylightSparseConfig.from_env()
    return SparseAttentionScorer(
        topk=cfg.topk,
        sink_size=cfg.sink_size,
        local_size=cfg.local_size,
        channel_num=cfg.channel_num if cfg.channel_num != -1 else 8,
        max_seq_len=cfg.max_seq_len,
    )