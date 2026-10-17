"""FastKVzip gate port: loads the released weights strictly and matches the
reference implementation (rebuilt here with transformers' Qwen3RMSNorm)."""
from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn as nn

from offload_package.scoring.fastkvzip import (
    FastKVzipGate, gate_file_for, load_fastkvzip_gates,
)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _RefWeight(nn.Module):
    """Verbatim math of FastKVzip prefill/attention/gate.py::Weight."""

    def __init__(self, input_dim, output_dim, nhead, ngroup, dtype, sink):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        self.output_dim, self.nhead = output_dim, nhead
        self.q_proj = nn.Linear(input_dim, nhead * ngroup * output_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(input_dim, nhead * output_dim, bias=False, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(output_dim)
        self.k_norm = Qwen3RMSNorm(output_dim)
        self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))
        self.d = math.sqrt(output_dim)

    def forward(self, hidden_states):
        hidden_states = hidden_states.squeeze(0)
        nseq = hidden_states.shape[0]
        hidden_shape = (nseq, self.nhead, -1, self.output_dim)
        queries = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        queries = queries.transpose(0, 1).transpose(-1, -2)
        keys = keys.transpose(0, 1)
        logit = torch.matmul(keys, queries) / self.d + self.b.unsqueeze(2)
        logit_base = torch.matmul(self.k_base, queries) / self.d
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))
        score = score.mean(-1)
        return score.squeeze(-1).unsqueeze(0)


def test_gate_file_name():
    assert gate_file_for(MODEL, 32, 8) == "qwen3-4b-instruct-2507/q4_dim16_sink16.pt"


def test_released_gates_load_and_match_reference():
    try:
        gates = load_fastkvzip_gates(MODEL, 32, 8, "cuda")
    except Exception as e:  # offline / not downloaded
        pytest.skip(f"gate weights unavailable: {e}")
    assert len(gates) == 36
    g0 = gates[0]
    assert (g0.nhead, g0.ngroup, g0.output_dim, g0.sink) == (8, 4, 16, 16)
    torch.manual_seed(0)
    for layer in (0, 17, 35):
        g = gates[layer]
        ref = _RefWeight(g.q_proj.in_features, g.output_dim, g.nhead, g.ngroup,
                         g.q_proj.weight.dtype, g.sink).cuda()
        ref.load_state_dict(g.state_dict())
        h = torch.randn(1, 37, g.q_proj.in_features, device="cuda",
                        dtype=g.q_proj.weight.dtype)
        ours = g(h[0])                            # [H, N]
        theirs = ref(h)[0].float()                # [H, N]
        assert ours.shape == (8, 37)
        assert torch.allclose(ours, theirs, atol=1e-6, rtol=0), (layer, (ours - theirs).abs().max())
        assert bool(((ours > 0) & (ours < 1)).all())
