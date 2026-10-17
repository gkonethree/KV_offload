"""Forced decoding for the benches: every config decodes the same reference tokens.

With BENCH_FORCE_TOKENS=<json list of token ids>, the bench adds this logits
processor. At decode step t it records, from the config's own logits, whether
the argmax equals ref[t] and the NLL of ref[t], then masks the logits so ref[t]
is sampled. Queries, and therefore sparse selections and fetches, then follow
the same text in every config, and the recorded agreement / NLL measure how far
each config's attention drifts from the reference.

Stats stay on the GPU (no per-step sync); `ForcedTokens.instance.summary()`
reads them after generation. Only for in-process engines
(VLLM_ENABLE_V1_MULTIPROCESSING=0), which the bench uses.
"""
import json
import os

import torch

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate, LogitsProcessor, MoveDirectionality)


class ForcedTokens(LogitsProcessor):
    instance = None

    def __init__(self, vllm_config, device, is_pin_memory):
        with open(os.environ['BENCH_FORCE_TOKENS']) as f:
            self.ref = [int(t) for t in json.load(f)]
        n = len(self.ref)
        self.nll = torch.zeros(n, dtype=torch.float32, device=device)
        self.agree = torch.zeros(n, dtype=torch.float32, device=device)
        self.seen = torch.zeros(n, dtype=torch.float32, device=device)
        self.rows: dict[int, list[int]] = {}   # batch index -> running output ids
        ForcedTokens.instance = self

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for i in batch_update.removed:
            self.rows.pop(i, None)
        for i, _params, _prompt, out in batch_update.added:
            self.rows[i] = out
        for a, b, d in batch_update.moved:
            ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
            if ra is not None:
                self.rows[b] = ra
            if d == MoveDirectionality.SWAP and rb is not None:
                self.rows[a] = rb

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for i, out in self.rows.items():
            t = len(out)
            if t >= len(self.ref):
                continue
            tok = self.ref[t]
            row = logits[i]
            self.nll[t] = -torch.log_softmax(row.float(), -1)[tok]
            self.agree[t] = (row.argmax() == tok).float()
            self.seen[t] = 1.0
            row.fill_(float('-inf'))
            row[tok] = 0.0
        return logits

    def reset(self) -> None:
        for b in (self.nll, self.agree, self.seen):
            b.zero_()

    def summary(self) -> dict:
        n = int(self.seen.sum().item())
        agree = self.agree.cpu()
        miss = (agree == 0) & (self.seen.cpu() > 0)
        return {
            'steps': n,
            'agree': float(agree.sum()) / max(n, 1),
            'nll': float(self.nll.sum()) / max(n, 1),
            'first_disagree': int(miss.nonzero()[0]) if miss.any() else -1,
        }
