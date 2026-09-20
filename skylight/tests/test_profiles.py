"""Tests for canonical serving and benchmark profiles."""
from __future__ import annotations

import pytest

from skylight.profiles import BMMProfile


def test_default_bmm_profile_is_the_qualified_incremental_path() -> None:
    assert BMMProfile().to_env() == {
        "SKYLIGHT_SPARSE_METHOD": "block_minmax",
        "SKYLIGHT_SPARSE_TOPK": "0.1",
        "SKYLIGHT_SPARSE_SINK": "64",
        "SKYLIGHT_SPARSE_LOCAL": "64",
        "SKYLIGHT_SPARSE_CHANNEL_NUM": "-1",
        "SKYLIGHT_SPARSE_SUB_PAGE": "16",
        "SKYLIGHT_BLOCK_SIZE": "16",
        "SKYLIGHT_INCR_SLOT": "1",
        "SKYLIGHT_INCR_FULLCG": "1",
        "SKYLIGHT_INCR_PIPELINED": "1",
        "SKYLIGHT_FI_BSR": "0",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("target_density", 0.0, "target_density"),
        ("target_density", 1.1, "target_density"),
        ("sink", -1, "sink"),
        ("local", -1, "local"),
        ("channel_num", 0, "channel_num"),
        ("sub_page", 0, "sub_page"),
        ("block_size", 0, "block_size"),
    ),
)
def test_bmm_profile_rejects_invalid_policy(
    field: str,
    value: int | float,
    message: str,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        BMMProfile(**kwargs)
