"""Protocol every agentic benchmark adapter must satisfy.

Structural typing: adapters do not need to inherit anything; if they
expose these attrs/methods with these signatures, they work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AgenticBenchmark(Protocol):
    """Adapter shape for one agentic benchmark."""

    name: str  # e.g. "mini-swe-agent"

    def default_instances_path(self) -> Path:
        """Default subset to run against if the user doesn't pass one."""
        ...

    def build_agent_cmd(
        self,
        model: str,
        base_url: str,
        instances: Path,
        out_dir: Path,
    ) -> list[str]:
        """argv that runs the agent against `base_url`, processing `instances`.

        The agent must write per-instance attempts (e.g. patches.jsonl)
        under `out_dir` so `parse_results` and the progress reporter can
        consume them.
        """
        ...

    def build_eval_cmd(self, out_dir: Path) -> list[str]:
        """argv that scores the agent's output in `out_dir`.

        Run AFTER build_agent_cmd finishes. Writes a results JSON under
        out_dir (or CWD) that `parse_results` can parse.
        """
        ...

    def parse_results(self, out_dir: Path) -> dict:
        """Read whatever the eval wrote and return headline numbers.

        MUST return at least: {"n_solved": int, "n_total": int,
        "pass_at_1": float}. May add "errors": list[str] on failure.
        """
        ...
