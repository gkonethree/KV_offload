"""skylight long-context benchmark suite.

Benchmark definitions (datasets + prompts + scoring) vendored from
sparse-attention-hub; run against an OpenAI-compatible skylight server via
``api_server.ApiServerAdapter``. See ``run.py`` for the orchestrator.
"""
from .adapter_base import ModelAdapter, Request, RequestResponse  # noqa: F401
from .infinite_bench import InfiniteBench
from .longbench import LongBench
from .longbenchv2 import LongBenchv2
from .loogle import Loogle
from .ruler import Ruler

REGISTRY = {
    "longbench": LongBench,
    "longbenchv2": LongBenchv2,
    "ruler": Ruler,
    "infinite_bench": InfiniteBench,
    "loogle": Loogle,
}

__all__ = ["REGISTRY", "ModelAdapter", "Request", "RequestResponse",
           "LongBench", "LongBenchv2", "Ruler", "InfiniteBench", "Loogle"]
