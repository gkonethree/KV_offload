"""Contract tests for the million-context crossover experiment runner."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.long_context_crossover import (
    build_condition_command,
    condition_complete,
    condition_key,
    summarize_pair,
)


def test_build_condition_command_locks_workload_and_bmm_profile(tmp_path: Path) -> None:
    command = build_condition_command(
        backend="sparse",
        model="Qwen/Qwen3.5-9B",
        context=131072,
        generated=128,
        batch=4,
        gpu=0,
        output_dir=tmp_path,
        gpu_memory_utilization=0.99,
        enforce_eager=False,
        rope_overrides='{"max_position_embeddings":1048576}',
        kv_cache_dtype="fp8",
    )
    rendered = " ".join(command)
    assert command[:3] == ["env", "CUDA_VISIBLE_DEVICES=0", "SKYLIGHT_HF_OVERRIDES={\"max_position_embeddings\":1048576}"]
    assert "SKYLIGHT_KV_DTYPE=fp8" in command
    assert "--backend sparse" in rendered
    assert "--num-prompts 4" in rendered
    assert "--random-input-len 131072" in rendered
    assert "--random-output-len 128" in rendered
    assert "--max-model-len 131712" in rendered
    assert "--gpu-memory-utilization 0.99" in rendered
    assert "--max-num-seqs 4" in rendered
    assert "--no-enforce-eager" in rendered
    assert "--method block_minmax" in rendered
    assert "--topk 0.1" in rendered
    assert "--sink 64" in rendered
    assert "--local 64" in rendered
    assert "--channel-num -1" in rendered
    assert f"--artifacts-dir {tmp_path}" in rendered


def test_condition_key_is_stable_and_unambiguous() -> None:
    assert condition_key("cudagraph", "dense", 65536, 8, 128) == (
        "context-65536/batch-08/cudagraph/dense-gen128"
    )


def test_condition_complete_requires_successful_result(tmp_path: Path) -> None:
    assert not condition_complete(tmp_path)
    (tmp_path / "result.json").write_text(json.dumps({"failed": 0}))
    assert not condition_complete(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"success": True}))
    assert condition_complete(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({"success": False}))
    assert not condition_complete(tmp_path)


def test_summarize_pair_requires_both_throughput_and_tpot_wins() -> None:
    dense = {"output_throughput": 100.0, "mean_tpot_ms": 10.0, "failed": 0}
    bmm = {"output_throughput": 106.0, "mean_tpot_ms": 9.0, "failed": 0}
    summary = summarize_pair(dense, bmm)
    assert summary["classification"] == "win"
    assert summary["throughput_ratio"] == 1.06
    assert summary["tpot_speedup"] == 1.1111

    not_enough_throughput = {"output_throughput": 104.0, "mean_tpot_ms": 9.0, "failed": 0}
    assert summarize_pair(dense, not_enough_throughput)["classification"] == "neutral"

    failed = {"output_throughput": 120.0, "mean_tpot_ms": 8.0, "failed": 1}
    assert summarize_pair(dense, failed)["classification"] == "regression"
