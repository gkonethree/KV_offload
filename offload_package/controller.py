from __future__ import annotations

from collections import defaultdict


class KVCompressionController:
    """Per-request decode-token counters.

    No counter is advanced during prefill.  A request triggers an offload pass
    when its decode count reaches a positive multiple of W.
    """

    def __init__(self, W: int, page_size: int) -> None:
        if W <= 0:
            raise ValueError("W must be > 0")
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        if W % page_size:
            raise ValueError("W must be a multiple of page_size")
        self.W = int(W)
        self.page_size = int(page_size)
        self._started: set[str] = set()
        self._decode_tokens: defaultdict[str, int] = defaultdict(int)

    def on_prefill_complete(self, request_id: str) -> None:
        self._started.add(request_id)
        self._decode_tokens[request_id] = 0

    def on_decode(self, request_id: str, num_tokens: int = 1) -> bool:
        if request_id not in self._started:
            return False
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        self._decode_tokens[request_id] += int(num_tokens)
        n = self._decode_tokens[request_id]
        return n > 0 and n % self.W == 0

    def remove(self, request_id: str) -> None:
        self._started.discard(request_id)
        self._decode_tokens.pop(request_id, None)

    def count(self, request_id: str) -> int:
        return self._decode_tokens.get(request_id, 0)
