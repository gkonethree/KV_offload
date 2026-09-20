"""ApiServerAdapter: run any vendored long-context benchmark against an
OpenAI-compatible endpoint (a vLLM server with skylight sparse kernels).

Implements the one method benchmarks call: process_request. Sparsity lives in
the server (SKYLIGHT_SPARSE_METHOD), so this adapter is method-agnostic. Prompt
construction mirrors the hub HuggingFace adapter (_preprocess_context_and_questions)
so scores match: chat-template-wrap the context, append the question inside the
user turn, then answer_prefix; context is token-truncated to max_context_length.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from .adapter_base import ModelAdapter, Request, RequestResponse

_SEP = " __SAH_SEP__ "  # rare separator; survives tokenize=False chat template


class ApiServerAdapter(ModelAdapter):
    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:8000",
                 tokenizer_name: Optional[str] = None, timeout: float = 3600.0) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name or model_name)

    def _preprocess(self, context: str, questions: List[str], answer_prefix: str):
        # mirrors HuggingFaceAdapter._preprocess_context_and_questions
        c = context + _SEP
        if self.tok.chat_template is not None:
            c = self.tok.apply_chat_template(
                [{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
        head, _, tail = c.partition(_SEP)
        return head, [q + tail + answer_prefix for q in questions]

    def _complete(self, prompt_ids: List[int], max_new: int, gk: Dict[str, Any]) -> str:
        body = json.dumps({"model": self.model_name, "prompt": prompt_ids,  # token ids -> exact prompt
                           "max_tokens": int(max_new),
                           "temperature": float(gk.get("temperature", 0.0)),
                           "stop": gk.get("stop")}).encode()
        req = urllib.request.Request(self.base_url + "/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["text"]

    def process_request(self, request: Request, generation_kwargs: Dict[str, Any],
                        request_kwargs: Dict[str, Any]) -> RequestResponse:
        max_ctx = int(request_kwargs.get("max_context_length", 2**31))
        max_new = int(generation_kwargs.get("max_new_tokens", 256))
        single = not isinstance(request.questions, list)
        questions = [request.questions] if single else list(request.questions)
        context, questions = self._preprocess(request.context, questions, request.answer_prefix)
        ctx_ids = self.tok.encode(context)[:max_ctx]  # token-exact truncation == HF adapter
        out: List[str] = []
        for q in questions:
            q_ids = self.tok.encode(q, add_special_tokens=False)
            out.append(self._complete(ctx_ids + q_ids, max_new, generation_kwargs))
        return RequestResponse(responses=out[0] if single else out)
