"""SkylightSparseConfig — sparse-attention knobs parsed from environment.

The backend reads this exactly once at builder construction time. All four
fields are validated up front so misconfiguration surfaces at vllm startup
rather than mid-decode.

Forward-compat: when vllm's ``AttentionConfig`` accepts arbitrary fields
(pydantic ``extra='allow'``), add a sibling classmethod
``from_attention_config(attn_cfg)`` that reads from ``vllm_config.attention_config``
with these env vars as fallback. Not implemented today — env-only path is
all this commit ships.

Env vars (all default 0; missing = absent):

  SKYLIGHT_SPARSE_TOPK         fractional top-K in [0, 1] (e.g. 0.10 = 10%)
  SKYLIGHT_SPARSE_SINK         leading "sink" tokens always kept
  SKYLIGHT_SPARSE_LOCAL        trailing "local" window tokens always kept
  SKYLIGHT_SPARSE_CHANNEL_NUM  leading head channels used for score dot-product
                               (-1 = use full head_dim; default -1)

At least one of TOPK / SINK / LOCAL must be > 0; otherwise the backend
would degenerate to dense, in which case the user should set
``VLLM_ATTENTION_BACKEND=FLASHINFER`` instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SkylightSparseConfig:
    """Validated sparse-attention configuration."""

    topk: float
    """Fractional top-K of middle range, in [0, 1]. 0 disables top-K selection."""

    sink_size: int
    """Number of leading tokens unconditionally kept (attention sinks). >= 0."""

    local_size: int
    """Number of trailing tokens unconditionally kept (local window). >= 0."""

    channel_num: int
    """Leading Q/K channels used for selection score. -1 means full head_dim."""

    @classmethod
    def from_env(cls) -> "SkylightSparseConfig":
        """Parse SKYLIGHT_SPARSE_* env vars into a validated config.

        Raises:
            ValueError: if any field is malformed or all of TOPK/SINK/LOCAL are 0.
        """
        topk = _parse_topk("SKYLIGHT_SPARSE_TOPK")
        sink_size = _parse_nonneg_int("SKYLIGHT_SPARSE_SINK", default=0)
        local_size = _parse_nonneg_int("SKYLIGHT_SPARSE_LOCAL", default=0)
        channel_num = _parse_channel_num("SKYLIGHT_SPARSE_CHANNEL_NUM", default=-1)

        if topk <= 0 and sink_size == 0 and local_size == 0:
            raise ValueError(
                "SkylightSparseBackend requires at least one of "
                "SKYLIGHT_SPARSE_{TOPK,SINK,LOCAL} > 0. "
                "For a dense run, set VLLM_ATTENTION_BACKEND=FLASHINFER instead."
            )

        return cls(
            topk=topk,
            sink_size=sink_size,
            local_size=local_size,
            channel_num=channel_num,
        )


def _parse_topk(name: str) -> float:
    """Parse a [0, 1] float env var. Missing → 0.0."""
    raw = os.environ.get(name, "")
    if raw == "":
        return 0.0
    try:
        v = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} must be a float in [0, 1] (e.g. {name}=0.10)"
        ) from exc
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name}={v} must be in [0, 1]")
    return v


def _parse_nonneg_int(name: str, default: int) -> int:
    """Parse a non-negative int env var. Missing → ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} must be a non-negative integer (e.g. {name}=64)"
        ) from exc
    if v < 0:
        raise ValueError(f"{name}={v} must be >= 0")
    return v


def _parse_channel_num(name: str, default: int) -> int:
    """Parse channel_num. -1 = use full head_dim. Missing → ``default``. >0 must be int."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name}={raw!r} must be -1 or a positive int (e.g. {name}=8)"
        ) from exc
    if v == 0 or v < -1:
        raise ValueError(
            f"{name}={v} must be -1 (use full head_dim) or a positive int"
        )
    return v
