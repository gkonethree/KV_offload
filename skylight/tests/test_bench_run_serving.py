"""Tests for skylight.bench.run_serving helper functions.

Covers the pure-Python helpers (build_serve_cmd, build_env,
build_benchmark_cmd, free_port). The full end-to-end orchestration —
spawning a server, running benchmark_serving, tearing down — is
exercised by manual runs and the existing integration smoke (which
already proves server startup, /health, and decode).
"""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import skylight.bench.run_serving as run_serving
from skylight.bench.run_serving import (
    ArtifactPaths,
    build_benchmark_cmd,
    build_env,
    build_serve_cmd,
    free_port,
    safe_environment_snapshot,
)
from skylight.runtime import Platform


# -------------------------------- build_serve_cmd ----------------------------


def test_sparse_serve_cmd_invokes_skylight_cli():
    """Sparse backend uses ``python -m skylight.cli serve``."""
    cmd = build_serve_cmd("sparse", "Qwen/Qwen3-0.6B", 8000, 2048, False)
    assert sys.executable == cmd[0]
    assert "-m" in cmd
    assert "skylight.cli" in cmd
    assert "serve" in cmd


def test_dense_serve_cmd_invokes_vllm_api_server():
    """Dense backend uses ``python -m vllm.entrypoints.openai.api_server``."""
    cmd = build_serve_cmd("dense", "Qwen/Qwen3-0.6B", 8000, 2048, False)
    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "skylight.cli" not in cmd


def test_dense_serve_cmd_forces_flashinfer_backend():
    """Dense run explicitly pins --attention-backend FLASHINFER (production baseline)."""
    cmd = build_serve_cmd("dense", "X", 8000, 2048, False)
    assert "--attention-backend" in cmd
    idx = cmd.index("--attention-backend")
    assert cmd[idx + 1] == "FLASHINFER"


def test_serve_cmd_pins_model_and_tokenizer_revisions():
    revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    cmd = build_serve_cmd(
        "dense",
        "Qwen/Qwen3.5-9B",
        8000,
        2048,
        False,
        model_revision=revision,
        tokenizer_revision=revision,
    )

    assert cmd[cmd.index("--revision") + 1] == revision
    assert cmd[cmd.index("--tokenizer-revision") + 1] == revision


def test_sparse_serve_cmd_does_not_force_attention_backend():
    """Sparse leaves --attention-backend off — skylight.cli injects CUSTOM itself."""
    cmd = build_serve_cmd("sparse", "X", 8000, 2048, False)
    assert "--attention-backend" not in cmd


def test_serve_cmd_passes_model_port_max_model_len():
    cmd = build_serve_cmd("sparse", "Qwen/Qwen3-0.6B", 9999, 4096, False)
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-0.6B"
    assert "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "9999"
    assert "--max-model-len" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "4096"


def test_serve_cmd_enforce_eager_toggle():
    """--enforce-eager appears only when enforce_eager=True."""
    cmd_on = build_serve_cmd("sparse", "X", 8000, 2048, enforce_eager=True)
    cmd_off = build_serve_cmd("sparse", "X", 8000, 2048, enforce_eager=False)
    assert "--enforce-eager" in cmd_on
    assert "--enforce-eager" not in cmd_off


def test_serve_cmd_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown --backend"):
        build_serve_cmd("triton", "X", 8000, 2048, False)


def test_serve_cmd_defaults_gpu_memory_utilization_to_half():
    """Default 0.5 matches the handoff's tested operational setting and
    coexists with other tenants on a shared B200. vLLM's own default
    (0.92) is unsafe on a shared GPU and isn't what we want."""
    for backend in ("sparse", "dense"):
        cmd = build_serve_cmd(backend, "X", 8000, 2048, False)
        assert "--gpu-memory-utilization" in cmd
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.5"


def test_serve_cmd_gpu_memory_utilization_override():
    """Caller can pass a different fraction (e.g. 0.85 for a dedicated host)."""
    cmd = build_serve_cmd(
        "sparse", "X", 8000, 2048, False, gpu_memory_utilization=0.85,
    )
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.85"


def test_serve_cmd_omits_tool_call_flags_by_default():
    """Perf bench sends raw prompts; tool-call parsing is opt-in to
    avoid wasted server overhead on non-agentic runs."""
    for backend in ("sparse", "dense"):
        cmd = build_serve_cmd(backend, "X", 8000, 2048, False)
        assert "--enable-auto-tool-choice" not in cmd
        assert "--tool-call-parser" not in cmd


def test_serve_cmd_enables_tool_call_parsing_when_parser_given():
    """Setting tool_call_parser appends BOTH --enable-auto-tool-choice
    AND --tool-call-parser <name>. vLLM requires both — passing just
    one yields BadRequestError on the first tool-call request."""
    for backend in ("sparse", "dense"):
        cmd = build_serve_cmd(
            backend, "X", 8000, 2048, False,
            tool_call_parser="qwen3_coder",
        )
        assert "--enable-auto-tool-choice" in cmd
        assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_coder"


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_serve_cmd_sets_matched_release_controls(backend):
    cmd = build_serve_cmd(backend, "X", 8000, 17408, False)
    assert cmd[cmd.index("--gdn-prefill-backend") + 1] == "triton"
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16"
    assert cmd[cmd.index("--block-size") + 1] == "16"


# -------------------------------- build_env ----------------------------------


def test_dense_env_does_not_set_skylight_sparse_vars():
    """Dense run does NOT export SKYLIGHT_SPARSE_* even when knobs are given."""
    env = build_env("dense", topk=0.10, sink=64, local=64)
    assert "SKYLIGHT_SPARSE_TOPK" not in env
    assert "SKYLIGHT_SPARSE_SINK" not in env
    assert "SKYLIGHT_SPARSE_LOCAL" not in env


def test_sparse_env_sets_all_three_knobs_when_provided():
    env = build_env("sparse", topk=0.10, sink=64, local=64)
    assert env["SKYLIGHT_SPARSE_TOPK"] == "0.1"
    assert env["SKYLIGHT_SPARSE_SINK"] == "64"
    assert env["SKYLIGHT_SPARSE_LOCAL"] == "64"


def test_sparse_bmm_env_uses_profile_defaults_for_unset_knobs():
    """An underspecified BMM run still resolves to the qualified profile."""
    env = build_env("sparse", topk=0.10, sink=None, local=None)
    assert env["SKYLIGHT_SPARSE_METHOD"] == "block_minmax"
    assert env["SKYLIGHT_SPARSE_TOPK"] == "0.1"
    assert env["SKYLIGHT_SPARSE_SINK"] == "64"
    assert env["SKYLIGHT_SPARSE_LOCAL"] == "64"
    assert env["SKYLIGHT_SPARSE_CHANNEL_NUM"] == "-1"


def test_sparse_bmm_env_uses_the_resolved_server_block_size():
    env = build_env(
        "sparse",
        topk=0.10,
        sink=64,
        local=64,
        block_size=32,
    )
    assert env["SKYLIGHT_BLOCK_SIZE"] == "32"


def test_sparse_env_sets_channel_num_when_provided():
    """channel_num — the doubly-sparse axis — flows through to the env."""
    env = build_env("sparse", topk=0.10, sink=64, local=64, channel_num=8)
    assert env["SKYLIGHT_SPARSE_CHANNEL_NUM"] == "8"


def test_sparse_env_channel_num_minus_one_sentinel_passes_through():
    """channel_num=-1 (full head_dim, no channel sparsity) is propagated literally."""
    env = build_env("sparse", topk=0.10, sink=64, local=64, channel_num=-1)
    assert env["SKYLIGHT_SPARSE_CHANNEL_NUM"] == "-1"


def test_dense_env_ignores_channel_num():
    """channel_num is sparse-only; dense run never exports it."""
    env = build_env("dense", topk=None, sink=None, local=None, channel_num=8)
    assert "SKYLIGHT_SPARSE_CHANNEL_NUM" not in env


def test_sparse_bmm_env_sets_complete_incremental_profile():
    env = build_env(
        "sparse",
        topk=0.10,
        sink=64,
        local=64,
        channel_num=-1,
        method="block_minmax",
    )
    assert {
        key: env[key]
        for key in (
            "SKYLIGHT_SPARSE_METHOD",
            "SKYLIGHT_SPARSE_TOPK",
            "SKYLIGHT_SPARSE_SINK",
            "SKYLIGHT_SPARSE_LOCAL",
            "SKYLIGHT_SPARSE_CHANNEL_NUM",
            "SKYLIGHT_SPARSE_SUB_PAGE",
            "SKYLIGHT_BLOCK_SIZE",
            "SKYLIGHT_INCR_SLOT",
            "SKYLIGHT_INCR_FULLCG",
            "SKYLIGHT_INCR_PIPELINED",
            "SKYLIGHT_FI_BSR",
        )
    } == {
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


def test_sparse_oracle_env_has_no_incremental_bmm_flags():
    env = build_env(
        "sparse",
        topk=0.10,
        sink=64,
        local=64,
        channel_num=-1,
        method="oracle",
    )
    assert env["SKYLIGHT_SPARSE_METHOD"] == "oracle"
    assert env["SKYLIGHT_SPARSE_TOPK"] == "0.1"
    assert "SKYLIGHT_INCR_SLOT" not in env
    assert "SKYLIGHT_SPARSE_SUB_PAGE" not in env


def test_dense_env_removes_inherited_sparse_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("SKYLIGHT_SPARSE_METHOD", "block_minmax")
    monkeypatch.setenv("SKYLIGHT_INCR_SLOT", "1")
    monkeypatch.setenv("SKYLIGHT_METRICS_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("SKYLIGHT_METRICS_SAMPLING", "1.0")

    env = build_env("dense", topk=None, sink=None, local=None)

    assert "SKYLIGHT_SPARSE_METHOD" not in env
    assert "SKYLIGHT_INCR_SLOT" not in env
    assert "SKYLIGHT_METRICS_LOG_DIR" not in env
    assert "SKYLIGHT_METRICS_SAMPLING" not in env


def test_sparse_metrics_are_complete_for_qualification(tmp_path):
    metrics_dir = tmp_path / "telemetry"
    env = build_env(
        "sparse",
        topk=0.10,
        sink=64,
        local=64,
        metrics_dir=metrics_dir,
    )
    assert env["SKYLIGHT_METRICS_LOG_DIR"] == str(metrics_dir)
    assert env["SKYLIGHT_METRICS_SAMPLING"] == "1.0"


def test_sparse_env_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown sparse method"):
        build_env(
            "sparse",
            topk=0.10,
            sink=64,
            local=64,
            method="mystery",
        )


def test_env_inherits_caller_environment():
    """Both backends inherit os.environ; we layer on top."""
    import os
    os.environ["SKYLIGHT_BENCH_TEST_INHERITED"] = "yes"
    try:
        env_dense = build_env("dense", topk=None, sink=None, local=None)
        env_sparse = build_env("sparse", topk=0.1, sink=None, local=None)
        assert env_dense.get("SKYLIGHT_BENCH_TEST_INHERITED") == "yes"
        assert env_sparse.get("SKYLIGHT_BENCH_TEST_INHERITED") == "yes"
    finally:
        del os.environ["SKYLIGHT_BENCH_TEST_INHERITED"]


# -------------------------------- build_benchmark_cmd ------------------------


def test_benchmark_cmd_invokes_vllm_bench_serve_subcommand():
    """We use ``vllm bench serve`` (CLI subcommand) — the version-stable public surface."""
    cmd = build_benchmark_cmd("X", 8000, 100, "random", 1.0, "out.json")
    assert cmd[0] == str(Path(sys.executable).with_name("vllm"))
    assert cmd[1:3] == ["bench", "serve"]


def test_benchmark_cmd_routes_to_completions_endpoint():
    cmd = build_benchmark_cmd("X", 8000, 100, "random", 1.0, "out.json")
    assert "--endpoint" in cmd
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/completions"
    assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "localhost"
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "8000"


def test_benchmark_cmd_passes_num_prompts_and_request_rate():
    cmd = build_benchmark_cmd("X", 8000, 250, "random", 5.5, "out.json")
    assert cmd[cmd.index("--num-prompts") + 1] == "250"
    assert cmd[cmd.index("--request-rate") + 1] == "5.5"


def test_benchmark_cmd_dataset_name_propagated():
    cmd = build_benchmark_cmd("X", 8000, 10, "sharegpt", 1.0, "out.json")
    assert cmd[cmd.index("--dataset-name") + 1] == "sharegpt"


def test_benchmark_cmd_uses_resolved_tokenizer_snapshot():
    snapshot = (
        "/models/Qwen--Qwen3.5-9B/snapshots/"
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    cmd = build_benchmark_cmd(
        "Qwen/Qwen3.5-9B",
        8000,
        10,
        "random",
        1.0,
        "out.json",
        tokenizer=snapshot,
    )

    assert cmd[cmd.index("--tokenizer") + 1] == snapshot


def test_benchmark_cmd_save_result_filename():
    cmd = build_benchmark_cmd("X", 8000, 100, "random", 1.0, "/tmp/r.json")
    assert "--save-result" in cmd
    assert "--result-filename" in cmd
    assert cmd[cmd.index("--result-filename") + 1] == "/tmp/r.json"


def test_benchmark_cmd_random_input_len_omitted_when_unset():
    """--random-input-len is omitted unless explicitly requested (vllm picks defaults)."""
    cmd = build_benchmark_cmd("X", 8000, 100, "random", 1.0, "out.json")
    assert "--random-input-len" not in cmd
    assert "--random-output-len" not in cmd


def test_benchmark_cmd_random_input_len_added_when_requested():
    """--random-input-len + --random-output-len flow through when provided."""
    cmd = build_benchmark_cmd(
        "X", 8000, 100, "random", 1.0, "out.json",
        random_input_len=16384, random_output_len=128,
    )
    assert cmd[cmd.index("--random-input-len") + 1] == "16384"
    assert cmd[cmd.index("--random-output-len") + 1] == "128"


def test_benchmark_cmd_defaults_to_warm_deterministic_generation():
    cmd = build_benchmark_cmd("X", 8000, 8, "random", float("inf"), "out.json")
    assert cmd[cmd.index("--num-warmups") + 1] == "2"
    assert cmd[cmd.index("--temperature") + 1] == "0.0"
    assert "--ignore-eos" in cmd


def test_benchmark_cmd_allows_generation_overrides():
    cmd = build_benchmark_cmd(
        "X",
        8000,
        8,
        "random",
        float("inf"),
        "out.json",
        num_warmups=1,
        temperature=0.7,
        ignore_eos=False,
    )
    assert cmd[cmd.index("--num-warmups") + 1] == "1"
    assert cmd[cmd.index("--temperature") + 1] == "0.7"
    assert "--ignore-eos" not in cmd


# -------------------------------- artifacts ----------------------------------


def test_artifact_paths_create_the_release_layout(tmp_path):
    paths = ArtifactPaths.create(tmp_path / "condition")
    assert paths.result == tmp_path / "condition" / "result.json"
    assert paths.server_log == tmp_path / "condition" / "server.log"
    assert paths.client_log == tmp_path / "condition" / "client.log"
    assert paths.resolved == tmp_path / "condition" / "resolved.json"
    assert paths.versions == tmp_path / "condition" / "versions.json"
    assert paths.telemetry == tmp_path / "condition" / "telemetry"
    assert paths.telemetry.is_dir()


def test_safe_environment_snapshot_is_allowlisted_and_redacted():
    snapshot = safe_environment_snapshot(
        {
            "SKYLIGHT_SPARSE_METHOD": "block_minmax",
            "CUDA_VISIBLE_DEVICES": "2",
            "CUDA_LAUNCH_BLOCKING": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NVIDIA_TF32_OVERRIDE": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NCCL_ALGO": "Ring",
            "TORCH_EXTENSIONS_DIR": "/var/tmp/extensions",
            "FLASHINFER_TOPK_ALGO": "radix",
            "PYTHONPYCACHEPREFIX": "/var/tmp/skylight/pycache",
            "GITHUB_TOKEN": "secret",
            "HF_TOKEN": "secret",
            "PASSWORD": "secret",
            "UNRELATED": "not-recorded",
        }
    )
    assert snapshot == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_LAUNCH_BLOCKING": "1",
        "CUDA_VISIBLE_DEVICES": "2",
        "FLASHINFER_TOPK_ALGO": "radix",
        "NCCL_ALGO": "Ring",
        "NVIDIA_TF32_OVERRIDE": "0",
        "PYTHONPYCACHEPREFIX": "/var/tmp/skylight/pycache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SKYLIGHT_SPARSE_METHOD": "block_minmax",
        "TORCH_EXTENSIONS_DIR": "/var/tmp/extensions",
    }


def test_installed_package_versions_are_complete_and_canonical(monkeypatch):
    class FakeDistribution:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        run_serving.metadata,
        "distributions",
        lambda: [
            FakeDistribution("huggingface_hub", "1.2.3"),
            FakeDistribution("Transformers", "5.0.0"),
            FakeDistribution("transformers", "5.0.0"),
            FakeDistribution("tokenizers", "0.22.0"),
        ],
    )

    assert run_serving._installed_package_versions() == {
        "huggingface-hub": ["1.2.3"],
        "tokenizers": ["0.22.0"],
        "transformers": ["5.0.0"],
    }


def test_collect_gpu_identity_is_structured_for_selected_device(monkeypatch):
    commands = []
    monkeypatch.setattr(
        run_serving,
        "_command_output",
        lambda command: commands.append(command)
        or "0, NVIDIA H200, GPU-test, 580.159.04, 9.0",
    )

    assert run_serving.collect_gpu_identity(0) == {
        "index": 0,
        "name": "NVIDIA H200",
        "uuid": "GPU-test",
        "driver_version": "580.159.04",
        "compute_capability": "9.0",
    }
    assert "--id=0" in commands[0]


def test_collect_gpu_identity_rejects_unavailable_nvidia_smi(monkeypatch):
    monkeypatch.setattr(
        run_serving,
        "_command_output",
        lambda command: "unavailable: FileNotFoundError: nvidia-smi",
    )

    with pytest.raises(RuntimeError, match="selected GPU identity"):
        run_serving.collect_gpu_identity(0)


def test_main_writes_artifacts_and_always_terminates_server(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()
    (artifact_dir / "server.log").write_text(
        "stale server log\n",
        encoding="utf-8",
    )
    events = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        assert (artifact_dir / "versions.json").is_file()
        assert (artifact_dir / "resolved.json").is_file()
        events.append("popen")
        return FakeProcess()

    def fake_run(cmd, **kwargs):
        result_path = Path(cmd[cmd.index("--result-filename") + 1])
        result_path.write_text(json.dumps({"completed": 1, "num_prompts": 1}))
        events.append("client")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        run_serving,
        "configure_runtime",
        lambda: Platform("blackwell", (10, 0), "sm100", "10.0"),
        raising=False,
    )
    monkeypatch.setattr(run_serving, "collect_versions", lambda platform: {"ok": True}, raising=False)
    monkeypatch.setattr(run_serving, "free_port", lambda: 8123)
    monkeypatch.setattr(run_serving.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_serving.subprocess, "run", fake_run)
    monkeypatch.setattr(run_serving, "wait_for_health", lambda *args: None)
    monkeypatch.setattr(
        run_serving,
        "terminate_group",
        lambda proc: events.append("terminate"),
    )

    result = run_serving.main(
        [
            "--backend", "dense",
            "--model", "X",
            "--num-prompts", "1",
            "--artifacts-dir", str(artifact_dir),
        ]
    )

    assert result == 0
    assert events == ["popen", "client", "terminate"]
    assert (artifact_dir / "result.json").is_file()
    assert (artifact_dir / "server.log").is_file()
    assert (artifact_dir / "server.log").read_text(encoding="utf-8") == ""
    assert (artifact_dir / "client.log").is_file()


def test_main_preserves_manifests_and_terminates_on_health_failure(
    monkeypatch,
    tmp_path,
):
    artifact_dir = tmp_path / "failed"
    terminated = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        run_serving,
        "configure_runtime",
        lambda: Platform("hopper", (9, 0), "sm90", "9.0"),
        raising=False,
    )
    monkeypatch.setattr(run_serving, "collect_versions", lambda platform: {"ok": True}, raising=False)
    monkeypatch.setattr(run_serving, "free_port", lambda: 8123)
    monkeypatch.setattr(run_serving.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        run_serving,
        "wait_for_health",
        lambda *args: (_ for _ in ()).throw(TimeoutError("not ready")),
    )
    monkeypatch.setattr(run_serving, "terminate_group", terminated.append)

    with pytest.raises(TimeoutError, match="not ready"):
        run_serving.main(
            [
                "--backend", "sparse",
                "--model", "X",
                "--artifacts-dir", str(artifact_dir),
            ]
        )

    assert len(terminated) == 1
    assert (artifact_dir / "versions.json").is_file()
    assert (artifact_dir / "resolved.json").is_file()
    assert (artifact_dir / "server.log").is_file()
    assert (artifact_dir / "client.log").is_file()


def test_main_returns_client_failure_and_still_terminates_server(
    monkeypatch,
    tmp_path,
):
    artifact_dir = tmp_path / "client-failed"
    terminated = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        run_serving,
        "configure_runtime",
        lambda: Platform("blackwell", (10, 0), "sm100", "10.0"),
    )
    monkeypatch.setattr(run_serving, "collect_versions", lambda platform: {"ok": True})
    monkeypatch.setattr(run_serving, "free_port", lambda: 8123)
    monkeypatch.setattr(run_serving.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        run_serving.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=17),
    )
    monkeypatch.setattr(run_serving, "wait_for_health", lambda *args: None)
    monkeypatch.setattr(run_serving, "terminate_group", terminated.append)

    result = run_serving.main(
        [
            "--backend", "dense",
            "--model", "X",
            "--artifacts-dir", str(artifact_dir),
        ]
    )

    assert result == 17
    assert len(terminated) == 1
    assert (artifact_dir / "client.log").is_file()
    assert (artifact_dir / "versions.json").is_file()
    assert (artifact_dir / "resolved.json").is_file()


# ---------------------------- reusable session -------------------------------


def test_serving_session_reuses_one_server_for_multiple_clients(
    monkeypatch,
    tmp_path,
):
    events = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        events.append(("start", command, kwargs["start_new_session"]))
        return process

    def fake_run(command, **kwargs):
        events.append(("client", command, "stdout" in kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_serving.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_serving.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_serving,
        "wait_for_health",
        lambda port, timeout, proc: events.append(("health", port, timeout, proc)),
    )
    monkeypatch.setattr(
        run_serving,
        "terminate_group",
        lambda proc: events.append(("stop", proc)),
    )

    with run_serving.ServingSession(
        ["server"],
        {"TEST": "1"},
        port=8123,
        timeout=30,
        server_log=tmp_path / "server.log",
    ) as session:
        first = session.run_benchmark(
            ["client", "one"],
            client_log=tmp_path / "one.log",
        )
        second = session.run_benchmark(
            ["client", "two"],
            client_log=tmp_path / "two.log",
        )

    assert first.returncode == second.returncode == 0
    assert [event[0] for event in events] == [
        "start", "health", "client", "client", "stop"
    ]
    assert events[0][2] is True
    assert events[1][1:3] == (8123, 30)
    assert events[-1][1] is process


def test_serving_session_terminates_when_health_check_fails(monkeypatch):
    terminated = []

    class FakeProcess:
        returncode = None

        def poll(self):
            return None

    process = FakeProcess()
    monkeypatch.setattr(
        run_serving.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        run_serving,
        "wait_for_health",
        lambda *args: (_ for _ in ()).throw(TimeoutError("not ready")),
    )
    monkeypatch.setattr(run_serving, "terminate_group", terminated.append)

    with pytest.raises(TimeoutError, match="not ready"):
        with run_serving.ServingSession(["server"], {}, 8123, 30):
            pass

    assert terminated == [process]


def test_serving_session_rejects_client_after_server_exit(monkeypatch):
    session = run_serving.ServingSession(["server"], {}, 8123, 30)
    session._proc = SimpleNamespace(returncode=9, poll=lambda: 9)

    with pytest.raises(RuntimeError, match="server exited"):
        session.run_benchmark(["client"])


def test_serving_session_health_requires_live_process_and_http_200(monkeypatch):
    class HealthyResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    session = run_serving.ServingSession(["server"], {}, 8123, 30)
    session._proc = SimpleNamespace(returncode=None, poll=lambda: None)
    monkeypatch.setattr(
        run_serving.urllib.request,
        "urlopen",
        lambda url, timeout: HealthyResponse(),
    )

    assert session.is_healthy() is True

    session._proc = SimpleNamespace(returncode=9, poll=lambda: 9)
    assert session.is_healthy() is False


def test_serving_session_health_returns_false_on_connection_error(monkeypatch):
    session = run_serving.ServingSession(["server"], {}, 8123, 30)
    session._proc = SimpleNamespace(returncode=None, poll=lambda: None)
    monkeypatch.setattr(
        run_serving.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(ConnectionError()),
    )

    assert session.is_healthy() is False


def test_serving_session_forwards_client_timeout(monkeypatch):
    observed = []
    session = run_serving.ServingSession(["server"], {}, 8123, 30)
    session._proc = SimpleNamespace(returncode=None, poll=lambda: None)
    monkeypatch.setattr(
        run_serving.subprocess,
        "run",
        lambda command, **kwargs: (
            observed.append((command, kwargs))
            or SimpleNamespace(returncode=0)
        ),
    )

    session.run_benchmark(["client"], timeout=45)

    assert observed == [(["client"], {"timeout": 45})]


# -------------------------------- free_port ----------------------------------


def test_free_port_in_user_range_and_bindable():
    """free_port() returns a port we can actually bind to."""
    port = free_port()
    assert 1024 <= port <= 65535
    # Sanity: bind again. If free_port handed us a stale port, this would fail.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", port))


def test_free_port_returns_different_ports_across_calls():
    """Tiny defensive check — port allocation isn't deterministic."""
    ports = {free_port() for _ in range(10)}
    # Not strictly required (the kernel may reuse a recently-released port),
    # but in practice we expect at least some variation.
    assert len(ports) >= 2, f"free_port() looked stuck at {ports!r}"
