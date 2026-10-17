"""Batched all-layer gate == per-layer gates (heads combined by max)."""
from __future__ import annotations

import pytest
import torch

from offload_package.scoring.fastkvzip import BatchedFastKVzipGates, FastKVzipGate

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_batched_matches_per_layer():
    torch.manual_seed(0)
    gates = []
    for _ in range(4):
        g = FastKVzipGate(256, 16, 8, 4, torch.bfloat16, sink=16).cuda()
        with torch.no_grad():
            for p in g.parameters():
                p.copy_(torch.randn_like(p) * 0.2)
        gates.append(g.eval())
    batched = BatchedFastKVzipGates(gates).cuda()
    for n in (1, 3, 64):
        h = torch.randn(4, n, 256, device="cuda", dtype=torch.bfloat16)
        ref = torch.stack([g(h[i]).amax(0) for i, g in enumerate(gates)])
        out = batched(h)
        assert out.shape == (4, n)
        assert torch.allclose(out, ref, atol=2e-3, rtol=0), (n, (out - ref).abs().max())
