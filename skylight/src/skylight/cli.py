"""skylight CLI — thin wrapper that invokes vllm's OpenAI api_server
with our sparse attention backend wired in.

Usage:
    skylight serve [vllm api_server args...]

Behavior:
    1. Recognizes only the ``serve`` subcommand. Everything after it is
       forwarded to ``vllm.entrypoints.openai.api_server`` verbatim.
    2. Injects ``--attention-backend CUSTOM`` into the forwarded argv unless
       the caller already passed ``--attention-backend`` (override path).
    3. Ensures ``PROMETHEUS_MULTIPROC_DIR`` is set so the prometheus
       multi-process collector aggregates EngineCore worker metrics into
       the parent's ``/metrics`` endpoint. Operator-set value is respected.
    4. Replaces the current process with the vllm server via ``os.execvp``.
       No shared state, no env-var hacks, no return.

The plugin entry point in :mod:`skylight.plugin` runs automatically in
every vllm process (api_server + EngineCore workers), so backend
registration is a side effect of running vllm at all under this venv.
"""
from __future__ import annotations

import os
import sys
import tempfile

from skylight.runtime import configure_runtime


def _build_vllm_argv(skylight_args: list[str]) -> list[str]:
    """Pure function: turn ``skylight serve [args]`` into a vllm api_server argv.

    Args:
        skylight_args: argv after the program name (i.e. ``sys.argv[1:]``).

    Returns:
        The argv to pass to ``os.execvp``: ``[python, -m, vllm....api_server, ...]``.

    Raises:
        ValueError: if the args don't start with the ``serve`` subcommand.
    """
    if not skylight_args or skylight_args[0] != "serve":
        raise ValueError("usage: skylight serve [vllm api_server args...]")

    forwarded = list(skylight_args[1:])
    # Inject --attention-backend CUSTOM iff caller didn't set it themselves.
    has_backend = any(
        a == "--attention-backend" or a.startswith("--attention-backend=")
        for a in forwarded
    )
    if not has_backend:
        forwarded = ["--attention-backend", "CUSTOM", *forwarded]

    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *forwarded]


def _ensure_prometheus_multiproc_dir() -> None:
    """Ensure ``PROMETHEUS_MULTIPROC_DIR`` is set so workers' metric updates
    aggregate into the parent's ``/metrics`` endpoint.

    Operator-set value wins. Otherwise we create a temp dir; cleanup is
    delegated to the OS (small files, $TMPDIR is rotated periodically).
    """
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        return
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="skylight-prom-")


def main() -> None:
    """Entry point for the ``skylight`` console script."""
    try:
        cmd = _build_vllm_argv(sys.argv[1:])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    configure_runtime()
    _ensure_prometheus_multiproc_dir()

    # Replace the current process. From here on, we're vllm.
    os.execvp(cmd[0], cmd)


# Make ``python -m skylight.cli serve ...`` work the same as the
# ``skylight`` console script. Without this block, ``python -m`` loads the
# module (defining ``main``) but never invokes it — the process would exit
# cleanly with rc=0 and no output. The console script generated from
# ``[project.scripts]`` already does the right thing, but tests / users that
# invoke via ``python -m`` rely on this entry.
if __name__ == "__main__":
    main()
