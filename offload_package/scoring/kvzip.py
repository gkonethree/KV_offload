"""KVzip: query-agnostic KV importance by context reconstruction.

Port of "KVzip: Query-Agnostic KV Cache Compression with Context
Reconstruction" (Kim et al., arXiv 2505.23416, github.com/snu-mllab/KVzip,
``attention/score.py::KVScore._get_score`` and ``model/wrapper.py``).

The model is asked to repeat its own context. For every context chunk the
input ``"\\n\\nRepeat the previous context exactly." + <chat postfix> + chunk``
(later chunks: ``"... part of the previous context exactly, starting with"``
+ the last 8 tokens of the previous chunk) is run on top of the prefilled
context cache, and in every layer each KV pair is scored by the largest
attention probability any repeat query (over all query heads of its KV group)
gives it. The softmax runs over ``[sink | chunk | repeat tokens]`` only, with
the repeat tokens causal. The result is one score per (layer, KV head, token),
independent of any downstream query.

Unlike the FastKVzip gate (scores a token from its own hidden state, free at
decode time) this is a separate teacher-forced pass over the whole prompt, so
it runs once per request, right before the request's first offload round, on
a private HF copy of the model (``KVzipScorer.score``). Tokens generated
after that have no reconstruction score and use a neutral per-layer default
(``DefaultScoreGates``).
"""
from __future__ import annotations

import math
import threading

import torch
import torch.nn as nn

_ATTN_NAME = "kvzip_sdpa"
_registered = False
_reg_lock = threading.Lock()


class _ScoreState:
    """Shared by the attention function of every layer during one pass."""

    def __init__(self) -> None:
        self.active = False
        self.sink = 0
        self.start = 0
        self.end = 0
        self.q_len = 0
        self.out: list[torch.Tensor] = []     # per layer [1, Hkv, ctx_chunk]


STATE = _ScoreState()


def _register_attention() -> None:
    global _registered
    with _reg_lock:
        if _registered:
            return
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        def kvzip_attention(module, query, key, value, attention_mask, **kwargs):
            if STATE.active:
                _score(module.layer_idx, query, key)
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

        AttentionInterface.register(_ATTN_NAME, kvzip_attention)
        AttentionMaskInterface.register(_ATTN_NAME, sdpa_mask)
        _registered = True


def _score(layer_idx: int, query: torch.Tensor, key: torch.Tensor) -> None:
    """``KVScore._get_score``: query [1, Hq, q, d] (post-RoPE), key
    [1, Hkv, kv, d] = cache (sink + context) + this pass's q repeat tokens."""
    s = STATE
    _, hq, q_len, d = query.shape
    hkv = key.shape[1]
    q = query.view(1, hkv, hq // hkv, q_len, d)
    k = torch.cat([key[:, :, :s.sink], key[:, :, s.start:s.end], key[:, :, -q_len:]], dim=2)
    k = k.unsqueeze(2).transpose(-2, -1)                       # 1, Hkv, 1, d, k
    ctx_len = s.end - s.start
    w = torch.matmul(q, k) / math.sqrt(d)                      # 1, Hkv, G, q, k
    mask = torch.full((q_len, q_len), torch.finfo(w.dtype).min, device=w.device, dtype=w.dtype)
    mask = torch.triu(mask, diagonal=1)
    w[..., -q_len:, -q_len:] += mask
    w = torch.softmax(w, dim=-1)                               # model dtype, as upstream
    w = w[..., s.sink:s.sink + ctx_len]
    s.out[layer_idx] = w.amax(dim=(-3, -2))                    # max over group, q


class KVzipScorer:
    """Holds a private HF copy of the model and scores prompts."""

    def __init__(self, model_name: str, device, dtype=torch.bfloat16, *,
                 sink: int = 64, chunk_size: int = 2000, prev_postfix: int = 8,
                 trust_remote_code: bool = False) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _register_attention()
        self.device = torch.device(device)
        self.sink, self.chunk, self.prev_postfix = int(sink), int(chunk_size), int(prev_postfix)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation=_ATTN_NAME,
            trust_remote_code=trust_remote_code).to(self.device).eval()
        cfg = self.model.config
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        self.postfix_ids = self._chat_postfix()

    def _enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _chat_postfix(self) -> list[int]:
        """Tokens a chat template appends after the user turn (assistant
        header), as upstream ``postfix_ids``; empty without a template."""
        if not getattr(self.tok, "chat_template", None):
            return []
        mark = "KVZIPMARK"
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True)
        return self._enc(text.split(mark, 1)[1])

    def _repeat_inputs(self, ctx: list[int]) -> list[tuple[list[int], list[int]]]:
        chunks = [ctx[i:i + self.chunk] for i in range(0, len(ctx), self.chunk)]
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                q = self._enc("\n\nRepeat the previous context exactly.")
            else:
                q = self._enc("\n\nRepeat the part of the previous context exactly, starting with ")
                q = q + chunks[i - 1][-self.prev_postfix:]
            out.append((c, q + self.postfix_ids + c))
        return out

    @torch.inference_mode()
    def score(self, prompt_ids: list[int]) -> torch.Tensor:
        """``[L, Hkv, n_prompt]`` float32 on ``self.device``. The first
        ``sink`` positions (the sink, never scored) get +inf."""
        from transformers import DynamicCache

        n = len(prompt_ids)
        sink = min(self.sink, n)
        ctx = prompt_ids[sink:]
        res = torch.full((self.num_layers, self.num_kv_heads, n), float("inf"),
                         dtype=torch.float32, device=self.device)
        if not ctx:
            return res
        ids = torch.tensor([prompt_ids], device=self.device)
        cache = DynamicCache(config=self.model.config)
        STATE.active = False
        self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                   logits_to_keep=1)                        # prefill the cache
        s = STATE
        s.sink, s.active = sink, True
        s.start = sink
        scores = [None] * self.num_layers
        try:
            for chunk_ids, rep in self._repeat_inputs(ctx):
                s.end = s.start + len(chunk_ids)
                s.q_len = len(rep)
                s.out = [None] * self.num_layers
                self.model(input_ids=torch.tensor([rep], device=self.device),
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache.crop(-len(rep))                       # drop the repeat tokens
                for li in range(self.num_layers):
                    part = s.out[li][0].float()             # Hkv, chunk
                    scores[li] = part if scores[li] is None else torch.cat([scores[li], part], -1)
                s.start = s.end
        finally:
            s.active = False
            s.out = []
        for li in range(self.num_layers):
            res[li, :, sink:] = scores[li]
        return res

    def close(self) -> None:
        self.model = None
        torch.cuda.empty_cache()


class DefaultScoreGates(nn.Module):
    """Gate interface for tokens the reconstruction pass never sees (decode
    tokens): a constant per-layer (per-head) score. ``set_default`` refreshes
    it from each scored request's median, so new tokens rank mid-pack rather
    than always first or last to be offloaded."""

    def __init__(self, num_layers: int, nhead: int, per_head: bool, device) -> None:
        super().__init__()
        self.per_head = per_head
        self.register_buffer("value", torch.full((num_layers, nhead), 0.0, device=device))

    def set_default(self, scores: torch.Tensor) -> None:     # scores [L, H, n]
        self.value.copy_(scores.float().median(dim=-1).values)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # [L, n, D]
        L, n = hidden.shape[:2]
        v = self.value.unsqueeze(1).expand(L, n, -1)          # L, n, H
        return v if self.per_head else v.amax(-1)


class DefaultScoreGate(nn.Module):
    """Per-layer view of ``DefaultScoreGates`` (large prefill chunks)."""

    def __init__(self, batched: DefaultScoreGates, layer: int) -> None:
        super().__init__()
        self._b, self._li = [batched], layer                  # list: not a submodule

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # [n, D] -> [H, n]
        return self._b[0].value[self._li].unsqueeze(1).expand(-1, hidden.shape[0])


def write_prompt_scores(engine, slot: int, scores: torch.Tensor) -> None:
    """Install ``scores`` [L, Hkv, n] as request ``slot``'s keep scores."""
    n = scores.shape[-1]
    if getattr(engine, "headwise", False):
        engine.score[:, slot, :n] = scores.permute(0, 2, 1).to(engine.score.device)
    else:
        engine.score[:, slot, :n] = scores.amax(1).to(engine.score.device)
