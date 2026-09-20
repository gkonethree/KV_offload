"""Make mini-swe-agent v2's submission robust on both failure modes.

Background:
  The v2 protocol captures whatever the agent prints after the
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel as the submission text.
  Two empirical failure modes on Qwen3.5-27B + SWE-bench:

  (a) Agent edits source files AND writes a valid diff to patch.txt, then
      submits via `cat patch.txt`. Works fine under v2's stdout protocol.

  (b) Agent edits source files but at submit time runs
      `cat /testbed/path/to/file.py | head -N | tail -M` instead of
      `cat patch.txt`. v2 captures raw Python source as the "submission"
      and swebench fails with "Only garbage was found in the patch input."

Hybrid policy installed here:
  - Read what's after the sentinel (the v2 default capture).
  - If it parses as a unified diff (has `diff --git` AND `@@` markers),
    trust it. This preserves the (a) path verbatim.
  - Otherwise, fall back to `git diff` inside the still-running container.
    This recovers the (b) path: even if the agent printed garbage, the
    container's working tree is the source of truth.

The fallback also handles the rarer case where the agent edits files but
never produces a patch.txt; the harness is now self-sufficient.

Installed via side-effect import from `skylight.bench.litellm_model`.
"""
from __future__ import annotations
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import Submitted  # type: ignore

_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_WORKDIR = "/testbed"
_DIFF_HEADER = re.compile(r"^diff --git ", re.MULTILINE)
_HUNK = re.compile(r"^@@ ", re.MULTILINE)

_log = logging.getLogger(__name__)


def _is_valid_unified_diff(text: str) -> bool:
    """A well-formed unified diff has at least one `diff --git` header
    and at least one `@@ ` hunk marker on a non-`+`/`-` prefixed line."""
    if not text.strip():
        return False
    return bool(_DIFF_HEADER.search(text)) and bool(_HUNK.search(text))


def _check_finished(self, output: dict) -> None:
    raw = output.get("output", "")
    lines = raw.lstrip().splitlines(keepends=True)
    if not (lines and lines[0].strip() == _SENTINEL and output["returncode"] == 0):
        return

    # v2-style submission: everything after the sentinel line.
    agent_submission = "".join(lines[1:])

    if _is_valid_unified_diff(agent_submission):
        submission = agent_submission
        source = "agent_stdout"
    else:
        # Fallback: ask git inside the container.
        diff_out = self.execute({"command":
            f"git -C {_WORKDIR} add -A && "
            f"git -C {_WORKDIR} diff --cached --no-color HEAD"
        })
        submission = diff_out.get("output", "") if isinstance(diff_out, dict) else ""
        source = "git_diff_fallback"
        if not submission.strip():
            _log.warning(
                "submission triggered but neither agent stdout nor git diff "
                "produced a patch; submitting empty"
            )

    _log.info("submission captured via %s (%d bytes)", source, len(submission))
    raise Submitted({
        "role": "exit",
        "content": submission,
        "extra": {"exit_status": "Submitted", "submission": submission},
    })


DockerEnvironment._check_finished = _check_finished  # type: ignore[assignment]


def _docker_start_log_path() -> Path | None:
    raw = os.environ.get("SKYLIGHT_DOCKER_START_LOG")
    return Path(raw) if raw else None


def _append_docker_start_log(record: dict) -> None:
    path = _docker_start_log_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


_orig_start_container = DockerEnvironment._start_container


def _start_container_with_retry(self) -> None:
    """Retry docker run on failure: pull image then retry up to 3 times."""
    max_attempts = int(os.environ.get("SKYLIGHT_DOCKER_START_RETRIES", "3"))
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _orig_start_container(self)
            _append_docker_start_log({
                "event": "start_ok",
                "image": self.config.image,
                "attempt": attempt,
            })
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            stderr = (exc.stderr or "") if hasattr(exc, "stderr") else ""
            stdout = (exc.stdout or "") if hasattr(exc, "stdout") else ""
            _append_docker_start_log({
                "event": "start_failed",
                "image": self.config.image,
                "attempt": attempt,
                "returncode": exc.returncode,
                "cmd": list(exc.cmd) if exc.cmd else [],
                "stderr": stderr[-4000:],
                "stdout": stdout[-2000:],
            })
            _log.warning(
                "docker start failed (attempt %d/%d) image=%s rc=%s",
                attempt, max_attempts, self.config.image, exc.returncode,
            )
            if attempt >= max_attempts:
                break
            pull_cmd = [self.config.executable, "pull", self.config.image]
            try:
                pull = subprocess.run(
                    pull_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.pull_timeout,
                )
                _append_docker_start_log({
                    "event": "pull",
                    "image": self.config.image,
                    "attempt": attempt,
                    "returncode": pull.returncode,
                    "stderr": (pull.stderr or "")[-4000:],
                })
            except (subprocess.TimeoutExpired, OSError) as pull_exc:
                _append_docker_start_log({
                    "event": "pull_error",
                    "image": self.config.image,
                    "attempt": attempt,
                    "error": str(pull_exc),
                })
            time.sleep(min(30, 2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc


DockerEnvironment._start_container = _start_container_with_retry  # type: ignore[assignment]
