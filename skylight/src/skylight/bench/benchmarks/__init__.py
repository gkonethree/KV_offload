"""Registry of agentic benchmark adapters supported by skylight bench.

Each adapter implements the AgenticBenchmark protocol (see base.py).
Adding a new benchmark: write a module + add one entry below.
"""
from .mini_swe_agent import MiniSweAgent

REGISTRY = {
    "mini-swe-agent": MiniSweAgent(),
}

__all__ = ["REGISTRY"]
