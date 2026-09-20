"""Tests for skylight.bench.litellm_model.

Covers the three guarantees of SkylightLitellmModel:

  1. The error categorizer maps every relevant LiteLLM exception class
     to a coarse, greppable category.
  2. ``litellm.Timeout`` is in ``abort_exceptions`` — the central P0
     guarantee that a hung server fails fast on the first call rather
     than burning N×timeout retries.
  3. Each ``_query()`` invocation writes one structured trace row to
     ``$LITELLM_TRACE_JSONL`` with a stable ``call_id`` per logical
     ``query()`` and a monotonically-increasing ``attempt`` index.

Pure-Python; ``litellm.completion`` is patched out so no network or
model load is required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Both deps come from the [agentic] optional-dependency group. Bare
# `uv sync` (no extras) — which verify_commit.sh runs before pytest —
# prunes them, so these tests gracefully no-op rather than red-bar.
pytest.importorskip("litellm")
pytest.importorskip("minisweagent")

import litellm.exceptions as le  # noqa: E402

from skylight.bench.litellm_model import (  # noqa: E402
    ERROR_CATEGORIES,
    SkylightLitellmModel,
    _categorize_litellm_error,
    _emit_trace_event,
)


def _make_exc(cls, msg: str = "synthetic"):
    """Construct a litellm exception bypassing its bespoke ``__init__``.

    Each LiteLLM exception class takes a different positional signature
    (some require ``response: httpx.Response``, ``status_code``, etc.).
    For categorizer tests only ``isinstance`` matters, so we sidestep
    by calling the base ``Exception.__init__`` directly.
    """
    inst = cls.__new__(cls)
    Exception.__init__(inst, msg)
    return inst


# ----------------------------- categorizer ------------------------------------


@pytest.mark.parametrize("cls, expected", [
    (le.AuthenticationError, "auth"),
    (le.PermissionDeniedError, "auth"),
    (le.NotFoundError, "auth"),
    (le.ContextWindowExceededError, "context_overflow"),
    (le.ContentPolicyViolationError, "content_policy"),
    (le.UnsupportedParamsError, "client_4xx"),
    (le.BadRequestError, "client_4xx"),
    (le.RateLimitError, "rate_limit"),
    (le.Timeout, "timeout"),
    (le.APIConnectionError, "connection"),
    (le.ServiceUnavailableError, "server_5xx"),
    (le.InternalServerError, "server_5xx"),
    (le.BadGatewayError, "server_5xx"),
    (le.APIError, "server_other"),
])
def test_categorize_litellm_exceptions(cls, expected):
    assert _categorize_litellm_error(_make_exc(cls)) == expected


def test_categorize_unknown_exception_is_other():
    assert _categorize_litellm_error(RuntimeError("?")) == "other"


def test_categorize_specific_subclass_wins_over_parent():
    """ContextWindowExceededError inherits from BadRequestError; the
    isinstance ladder must yield the more-specific category."""
    cwe = _make_exc(le.ContextWindowExceededError)
    assert isinstance(cwe, le.BadRequestError)  # establishes hierarchy
    assert _categorize_litellm_error(cwe) == "context_overflow"


def test_error_categories_const_lists_all_categorizer_outputs():
    """The documented ``ERROR_CATEGORIES`` tuple must be a superset of
    everything ``_categorize_litellm_error`` can return."""
    produced = {
        _categorize_litellm_error(_make_exc(c))
        for c in [
            le.AuthenticationError, le.PermissionDeniedError, le.NotFoundError,
            le.ContextWindowExceededError, le.ContentPolicyViolationError,
            le.UnsupportedParamsError, le.BadRequestError, le.RateLimitError,
            le.Timeout, le.APIConnectionError, le.ServiceUnavailableError,
            le.InternalServerError, le.BadGatewayError, le.APIError,
        ]
    } | {"other", "ok"}
    assert produced <= set(ERROR_CATEGORIES)


# ----------------------------- abort_exceptions -------------------------------


def test_timeout_in_abort_exceptions():
    """Central P0 guarantee: a litellm.Timeout never triggers a retry."""
    assert le.Timeout in SkylightLitellmModel.abort_exceptions


def test_abort_exceptions_is_superset_of_upstream():
    """Upstream abort classes (auth, etc.) must still abort under our subclass."""
    from minisweagent.models.litellm_model import LitellmModel as Upstream
    for cls in Upstream.abort_exceptions:
        assert cls in SkylightLitellmModel.abort_exceptions


# ----------------------------- emit_trace -------------------------------------


def test_emit_trace_event_writes_jsonl(tmp_path: Path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("LITELLM_TRACE_JSONL", str(trace))
    _emit_trace_event({"hello": "world"})
    _emit_trace_event({"k": 1})
    lines = trace.read_text().strip().splitlines()
    assert json.loads(lines[0]) == {"hello": "world"}
    assert json.loads(lines[1]) == {"k": 1}


def test_emit_trace_event_silent_when_env_unset(monkeypatch):
    monkeypatch.delenv("LITELLM_TRACE_JSONL", raising=False)
    # Must not raise even with a non-JSON-serializable payload.
    _emit_trace_event({"x": object()})


def test_emit_trace_event_silent_on_io_failure(monkeypatch):
    """Observability must never break the agent loop."""
    monkeypatch.setenv("LITELLM_TRACE_JSONL", "/proc/nonexistent/path/trace.jsonl")
    _emit_trace_event({"x": 1})  # no exception


# ----------------------------- _query trace shape -----------------------------


def _make_model() -> SkylightLitellmModel:
    return SkylightLitellmModel(
        model_name="openai/test-model",
        model_kwargs={"api_base": "http://localhost:9999/v1", "api_key": "EMPTY"},
        cost_tracking="ignore_errors",
    )


def test_query_writes_one_trace_row_per_attempt_success(tmp_path: Path, monkeypatch):
    trace = tmp_path / "t.jsonl"
    monkeypatch.setenv("LITELLM_TRACE_JSONL", str(trace))
    model = _make_model()
    fake_resp = MagicMock()
    fake_resp.model_dump.return_value = {"id": "x"}
    # Drive _query directly with manual instance attrs so we don't need
    # to spin up the full retry/cost/parse pipeline.
    model._trace_call_id = "callid-001"
    model._trace_attempt = 0
    with patch("skylight.bench.litellm_model.litellm.completion", return_value=fake_resp):
        model._query([{"role": "user", "content": "hi"}])
    rows = [json.loads(l) for l in trace.read_text().strip().splitlines()]
    assert len(rows) == 1
    assert rows[0]["call_id"] == "callid-001"
    assert rows[0]["attempt"] == 1
    assert rows[0]["ok"] is True
    assert rows[0]["category"] == "ok"
    assert "duration_s" in rows[0]


def test_query_trace_row_carries_category_on_failure(tmp_path: Path, monkeypatch):
    trace = tmp_path / "t.jsonl"
    monkeypatch.setenv("LITELLM_TRACE_JSONL", str(trace))
    model = _make_model()
    boom = le.Timeout("simulated", model="m", llm_provider="p")
    model._trace_call_id = "callid-002"
    model._trace_attempt = 0
    with patch("skylight.bench.litellm_model.litellm.completion", side_effect=boom):
        with pytest.raises(le.Timeout):
            model._query([{"role": "user", "content": "hi"}])
    row = json.loads(trace.read_text().strip().splitlines()[-1])
    assert row["category"] == "timeout"
    assert row["ok"] is False
    assert row["exception_class"] == "Timeout"
    assert row["call_id"] == "callid-002"
    assert row["attempt"] == 1


def test_query_attempt_index_increments_on_repeated_calls(tmp_path: Path, monkeypatch):
    """Tenacity's retry loop calls _query repeatedly with the same
    call_id; the attempt counter must increment across those calls."""
    trace = tmp_path / "t.jsonl"
    monkeypatch.setenv("LITELLM_TRACE_JSONL", str(trace))
    model = _make_model()
    boom = le.ServiceUnavailableError("hi", model="m", llm_provider="p")
    model._trace_call_id = "callid-003"
    model._trace_attempt = 0
    with patch("skylight.bench.litellm_model.litellm.completion", side_effect=boom):
        for _ in range(3):
            with pytest.raises(le.ServiceUnavailableError):
                model._query([{"role": "user", "content": "hi"}])
    rows = [json.loads(l) for l in trace.read_text().strip().splitlines()]
    assert [r["attempt"] for r in rows] == [1, 2, 3]
    assert len({r["call_id"] for r in rows}) == 1
    assert all(r["category"] == "server_5xx" for r in rows)


def test_query_logs_full_request_on_attempt_1_only(tmp_path: Path, monkeypatch):
    """Attempt 1 carries the full request body so the trace is
    replayable; attempt 2+ logs only a call_id reference to avoid
    multiplying the trace size by the retry count."""
    trace = tmp_path / "t.jsonl"
    monkeypatch.setenv("LITELLM_TRACE_JSONL", str(trace))
    model = _make_model()
    boom = le.ServiceUnavailableError("hi", model="m", llm_provider="p")
    model._trace_call_id = "callid-004"
    model._trace_attempt = 0
    with patch("skylight.bench.litellm_model.litellm.completion", side_effect=boom):
        for _ in range(2):
            with pytest.raises(le.ServiceUnavailableError):
                model._query([{"role": "user", "content": "hi"}])
    rows = [json.loads(l) for l in trace.read_text().strip().splitlines()]
    # Attempt 1 logs full kwargs (model, messages, tools, ...)
    assert rows[0]["request"].get("model") == "openai/test-model"
    assert "messages" in rows[0]["request"]
    # Attempt 2 logs only the back-reference
    assert rows[1]["request"] == {"call_id_ref": "callid-004"}


def test_query_resets_call_id_per_logical_call(tmp_path: Path, monkeypatch):
    """SkylightLitellmModel.query() must assign a fresh call_id each
    time it's invoked, even on the same model instance.

    We observe ``model._trace_call_id`` at the moment ``_query`` is
    entered (captured from the patched ``litellm.completion``) and bail
    via ``Timeout`` so we don't have to fabricate a valid tool-call
    response just to satisfy the downstream parse/cost machinery in
    upstream ``query``. Timeout is in ``abort_exceptions``, so each
    ``query()`` call raises after exactly one attempt — exactly what
    we need to compare two distinct call_ids.
    """
    trace = tmp_path / "t.jsonl"
    monkeypatch.setenv("LITELLM_TRACE_JSONL", str(trace))
    model = _make_model()

    captured: list[str] = []

    def fake_completion(**kwargs):
        captured.append(model._trace_call_id)
        raise le.Timeout("stop", model="m", llm_provider="p")

    with patch(
        "skylight.bench.litellm_model.litellm.completion",
        side_effect=fake_completion,
    ):
        for _ in range(2):
            with pytest.raises(le.Timeout):
                model.query([{"role": "user", "content": "hi"}])

    assert len(captured) == 2
    assert captured[0] != captured[1], "call_id must be fresh per query() call"
    rows = [json.loads(line) for line in trace.read_text().strip().splitlines()]
    assert len({r["call_id"] for r in rows}) == 2
