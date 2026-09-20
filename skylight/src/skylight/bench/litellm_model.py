"""Skylight's LitellmModel subclass: fail-fast retries + per-attempt trace.

Two responsibilities, both in service of agentic-bench runtime bounds and
observability:

  1. **Fail fast on server timeouts.** Upstream ``LitellmModel`` treats every
     non-auth exception as retriable; with the default 10 attempts and
     LiteLLM/httpx's fallthrough request timeout (~30 min observed), one
     hung vLLM backend will burn multiple hours per ``query()`` call
     before giving up. We add ``litellm.Timeout`` to
     ``abort_exceptions`` so the first per-request timeout exits the
     query immediately. Trace data justifies this: latency is strictly
     bimodal — healthy calls finish in ≤ 41s (p99 ≈ 31s) while every
     failure pins exactly at the request_timeout boundary. There is no
     "slow but recovering" population, so retrying a Timeout against the
     same hung server has zero recovery value.

  2. **Per-attempt structured trace.** Each ``_query()`` invocation
     appends one JSON line to ``$LITELLM_TRACE_JSONL`` with a stable
     ``call_id`` per logical ``query()``, a 1-indexed ``attempt`` within
     that call, a coarse error ``category`` (see ``ERROR_CATEGORIES``),
     and the exception class/message. This replaces the lost-on-uv-sync
     site-packages monkeypatch and makes retries first-class trace
     events instead of being buried in the agent log via tenacity's
     ``before_sleep_log``.

Registered via the bench ``-c`` overrides:
    model.model_class=skylight.bench.litellm_model.SkylightLitellmModel
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import litellm
import litellm.exceptions as le
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from . import mini_swe_patch  # noqa: F401  install git-based submission


# Documented set of coarse categories. The isinstance ladder in
# ``_categorize_litellm_error`` is the source of truth for precedence
# (most-specific subclasses first, because litellm's exception tree has
# overlaps like ``ContextWindowExceededError <: BadRequestError``).
ERROR_CATEGORIES = (
    "ok",
    "auth",
    "client_4xx",
    "context_overflow",
    "content_policy",
    "rate_limit",
    "timeout",
    "connection",
    "server_5xx",
    "server_other",
    "other",
)


def _categorize_litellm_error(exc: BaseException) -> str:
    """Map a LiteLLM exception to one of ``ERROR_CATEGORIES``."""
    if isinstance(exc, le.AuthenticationError):
        return "auth"
    if isinstance(exc, (le.PermissionDeniedError, le.NotFoundError)):
        return "auth"
    if isinstance(exc, le.ContextWindowExceededError):
        return "context_overflow"
    if isinstance(exc, le.ContentPolicyViolationError):
        return "content_policy"
    if isinstance(exc, le.UnsupportedParamsError):
        return "client_4xx"
    if isinstance(exc, le.BadRequestError):
        return "client_4xx"
    if isinstance(exc, le.RateLimitError):
        return "rate_limit"
    if isinstance(exc, le.Timeout):
        return "timeout"
    if isinstance(exc, le.APIConnectionError):
        return "connection"
    if isinstance(exc, (
        le.ServiceUnavailableError,
        le.InternalServerError,
        le.BadGatewayError,
    )):
        return "server_5xx"
    if isinstance(exc, le.APIError):
        return "server_other"
    return "other"


def _emit_trace_event(payload: dict[str, Any]) -> None:
    """Append one JSON line to ``$LITELLM_TRACE_JSONL`` if set.

    All failures are swallowed — observability must never break the
    agent loop. Returns silently when the env var is unset.
    """
    path = os.environ.get("LITELLM_TRACE_JSONL")
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


class SkylightLitellmModel(LitellmModel):
    """LitellmModel with fail-fast retries and per-attempt trace logging.

    See module docstring for rationale. Tenacity retry state is
    communicated through instance attrs that ``query()`` resets per
    invocation; each ``_query()`` invocation logs exactly one trace row.
    """

    # Append Timeout to the upstream abort list. In our localhost+vLLM
    # topology a Timeout always indicates a hung server, never a
    # transient blip worth retrying.
    abort_exceptions = LitellmModel.abort_exceptions + [le.Timeout]

    def query(self, messages: list[dict], **kwargs) -> dict:
        """Wrap upstream query() with a fresh call_id and attempt counter.

        ``super().query()`` invokes ``_query`` once per tenacity attempt;
        we read the instance attrs from there to label each trace row.
        """
        self._trace_call_id = uuid.uuid4().hex[:12]
        self._trace_attempt = 0
        return super().query(messages, **kwargs)

    def _query(self, messages: list[dict], **kwargs):
        # query() resets these; if a caller bypasses query() and hits
        # _query directly we still record something coherent.
        self._trace_attempt = getattr(self, "_trace_attempt", 0) + 1
        call_id = getattr(self, "_trace_call_id", "direct")
        attempt = self._trace_attempt

        final_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": [BASH_TOOL],
            **(self.config.model_kwargs | kwargs),
        }

        # Log the full request only on attempt 1 so retries don't bloat
        # the trace (the messages are identical across attempts of the
        # same call_id; replayers can dedupe by call_id).
        request_payload = final_kwargs if attempt == 1 else {"call_id_ref": call_id}

        t0 = time.time()
        try:
            resp = litellm.completion(**final_kwargs)
        except le.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            _emit_trace_event({
                "t": t0,
                "duration_s": time.time() - t0,
                "call_id": call_id,
                "attempt": attempt,
                "category": "auth",
                "exception_class": type(e).__name__,
                "error": str(e),
                "ok": False,
                "request": request_payload,
            })
            raise
        except Exception as e:
            _emit_trace_event({
                "t": t0,
                "duration_s": time.time() - t0,
                "call_id": call_id,
                "attempt": attempt,
                "category": _categorize_litellm_error(e),
                "exception_class": type(e).__name__,
                "error": str(e),
                "ok": False,
                "request": request_payload,
            })
            raise

        try:
            resp_dump = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
        except Exception:
            resp_dump = "<unserializable>"
        _emit_trace_event({
            "t": t0,
            "duration_s": time.time() - t0,
            "call_id": call_id,
            "attempt": attempt,
            "category": "ok",
            "ok": True,
            "request": request_payload,
            "response": resp_dump,
        })
        return resp
