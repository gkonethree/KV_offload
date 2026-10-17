"""FastKVzip gate: query-agnostic per-token, per-KV-head keep scores.

Port of the gate from "Fast KVzip: Efficient and Accurate LLM Inference
with Gated KV Eviction" (Kim, Han, Yun; arXiv 2601.17668,
github.com/Janghyun1230/FastKVzip, ``prefill/attention/gate.py::Weight``).

Each attention layer has a tiny learned gate that reads the layer's
attention input -- the post-``input_layernorm`` hidden states, i.e. the
``hidden_states`` argument of the attention forward -- and emits one score
in (0, 1) per token per KV head:

    q = RMSNorm(q_proj(h)) : [N, H_kv, G, d]      (G = q heads per kv head)
    k = RMSNorm(k_proj(h)) : [N, H_kv, 1, d]
    logit      = k . q / sqrt(d) + b              : per (kv head, token, group)
    logit_sink = k_base . q / sqrt(d)             : 16 learned sink keys
    score      = mean_G 1 / (1 + sum_sink exp(logit_sink - logit))

Scores depend only on the token's own hidden state, so they are computed
once when the token is produced and never change (no re-scoring at
offload rounds). Released gates live in the HF repo ``Jang-Hyun/Fast-KVzip``
as ``<model-name-lowercase>/q{G}_dim16_sink16.pt``.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn

GATE_REPO = "Jang-Hyun/Fast-KVzip"


class _RMSNorm(nn.Module):
    """Same math as transformers' Qwen3RMSNorm (fp32 statistics)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(dtype)


class FastKVzipGate(nn.Module):
    """One layer's gate (``Weight`` in the reference implementation)."""

    def __init__(self, input_dim: int, output_dim: int, nhead: int,
                 ngroup: int, dtype, sink: int = 1) -> None:
        super().__init__()
        self.output_dim, self.nhead, self.ngroup, self.sink = output_dim, nhead, ngroup, sink
        self.q_proj = nn.Linear(input_dim, nhead * ngroup * output_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(input_dim, nhead * output_dim, bias=False, dtype=dtype)
        self.q_norm = _RMSNorm(output_dim)
        self.k_norm = _RMSNorm(output_dim)
        self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))
        self.d = math.sqrt(output_dim)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states`` [N, input_dim] -> scores [nhead, N] (float32)."""
        n = hidden_states.shape[0]
        shape = (n, self.nhead, -1, self.output_dim)
        queries = self.q_norm(self.q_proj(hidden_states).view(shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(shape))
        queries = queries.transpose(0, 1).transpose(-1, -2)   # H, N, d, G
        keys = keys.transpose(0, 1)                           # H, N, 1, d
        logit = torch.matmul(keys, queries) / self.d + self.b.unsqueeze(2)
        logit_base = torch.matmul(self.k_base, queries) / self.d  # H, N, sink, G
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))
        return score.mean(-1).squeeze(-1).float()            # H, N


class BatchedFastKVzipGates(nn.Module):
    """All layers' gates as one batched computation.

    Decode scores one token per request per step; run per layer that is ~15
    tiny kernels x 36 layers of pure launch overhead. Scores are only read at
    offload rounds, so the hook stashes each layer's attention input and this
    module scores every layer at once (stacked weights, bmm + einsum).
    Same math as ``FastKVzipGate`` with the heads combined by max:
    ``forward(H [L, N, D]) -> [L, N]`` float32.
    """

    def __init__(self, gates: list[FastKVzipGate]) -> None:
        super().__init__()
        g0 = gates[0]
        self.nhead, self.ngroup, self.output_dim = g0.nhead, g0.ngroup, g0.output_dim
        self.d = g0.d
        self.eps_q = g0.q_norm.variance_epsilon
        self.eps_k = g0.k_norm.variance_epsilon
        st = lambda f: torch.stack([f(g).detach() for g in gates])
        self.register_buffer("wq_t", st(lambda g: g.q_proj.weight.t()).contiguous())  # L, D, HGd
        self.register_buffer("bq", st(lambda g: g.q_proj.bias))                       # L, HGd
        self.register_buffer("wk_t", st(lambda g: g.k_proj.weight.t()).contiguous())  # L, D, Hd
        self.register_buffer("qn", st(lambda g: g.q_norm.weight))                     # L, d
        self.register_buffer("kn", st(lambda g: g.k_norm.weight))                     # L, d
        self.register_buffer("k_base", st(lambda g: g.k_base[:, 0]))                  # L, H, S, d
        self.register_buffer("b", st(lambda g: g.b[:, 0]))                            # L, H, G

    @staticmethod
    def _norm(x, w, eps):
        dtype = x.dtype
        xf = x.to(torch.float32)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return w * xf.to(dtype)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        L, n, _ = hidden.shape
        H, G, d = self.nhead, self.ngroup, self.output_dim
        q = torch.baddbmm(self.bq.unsqueeze(1), hidden, self.wq_t).view(L, n, H, G, d)
        k = torch.bmm(hidden, self.wk_t).view(L, n, H, d)
        q = self._norm(q, self.qn.view(L, 1, 1, 1, d), self.eps_q)
        k = self._norm(k, self.kn.view(L, 1, 1, d), self.eps_k)
        logit = torch.einsum("lnhd,lnhgd->lnhg", k, q) / self.d + self.b.unsqueeze(1)
        base = torch.einsum("lhsd,lnhgd->lnhsg", self.k_base, q) / self.d
        score = 1 / (1 + torch.exp(base - logit.unsqueeze(3)).sum(3))     # L, n, H, G
        return score.mean(-1).amax(-1).float()                            # L, n


class RandomTokenGate(nn.Module):
    """Control scorer with the gate's interface: i.i.d. uniform scores, so a
    round offloads a random subset of each layer's non-protected tokens."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self._gen = None

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._gen is None or self._gen.device != hidden_states.device:
            self._gen = torch.Generator(device=hidden_states.device).manual_seed(self.seed)
        return torch.rand(1, hidden_states.shape[0], device=hidden_states.device,
                          generator=self._gen)


class BatchedRandomGates(nn.Module):
    """``RandomTokenGate`` for all layers at once: ``[L, N, D] -> [L, N]``."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self._gen = None

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._gen is None or self._gen.device != hidden.device:
            self._gen = torch.Generator(device=hidden.device).manual_seed(self.seed)
        return torch.rand(hidden.shape[0], hidden.shape[1], device=hidden.device,
                          generator=self._gen)


def gate_file_for(model_name: str, num_attention_heads: int,
                  num_kv_heads: int) -> str:
    ngroup = int(num_attention_heads) // int(num_kv_heads)
    return f"{model_name.rstrip('/').split('/')[-1].lower()}/q{ngroup}_dim16_sink16.pt"


def load_fastkvzip_gates(model_name: str, num_attention_heads: int,
                         num_kv_heads: int, device) -> list[FastKVzipGate]:
    """Download (or reuse the HF cache of) the released gates, one per layer."""
    from huggingface_hub import hf_hub_download

    gate_id = gate_file_for(model_name, num_attention_heads, num_kv_heads)
    path = hf_hub_download(repo_id=GATE_REPO, filename=gate_id, repo_type="model")
    state = torch.load(path, map_location="cpu", weights_only=False)["module"]
    dtype = state[0]["q_proj.weight"].dtype
    head_group_out, input_dim = state[0]["q_proj.weight"].shape
    head_out = state[0]["k_proj.weight"].shape[0]
    output_dim = state[0]["q_norm.weight"].shape[-1]
    nhead = head_out // output_dim
    ngroup = head_group_out // head_out
    m = re.search(r"sink(\d+)", gate_id)
    sink = int(m.group(1)) if m else 0
    gates = []
    for weights in state:
        g = FastKVzipGate(input_dim, output_dim, nhead, ngroup, dtype, sink=sink)
        g.load_state_dict(weights)
        gates.append(g.to(device).eval())
    return gates
