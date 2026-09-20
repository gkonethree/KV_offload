"""Minimal adapter contract vendored from sparse-attention-hub adapters/base.py,
stripped of the sparse-attention machinery. Benchmarks only use Request /
RequestResponse + process_request."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class Request:
    context: str
    questions: Union[str, List[str]]
    answer_prefix: str = ""


@dataclass
class RequestResponse:
    responses: Union[str, List[str]]


class ModelAdapter(ABC):
    @abstractmethod
    def process_request(self, request: "Request", generation_kwargs: Dict[str, Any],
                        request_kwargs: Dict[str, Any]) -> "RequestResponse": ...
