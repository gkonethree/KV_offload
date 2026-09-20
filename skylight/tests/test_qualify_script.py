"""Contract tests for the matched Hopper/Blackwell qualification matrix."""
from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


SCRIPT = Path("scripts/qualify_serving.sh")


def _dry_run(
    hardware: str = "hopper",
    trials: int = 3,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--hardware",
            hardware,
            "--gpu",
            "0",
            "--trials",
            str(trials),
            "--output-root",
            "/tmp/skylight-qualification",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )


def _commands(result: subprocess.CompletedProcess[str]) -> list[list[str]]:
    return [
        shlex.split(line.removeprefix("RUN "))
        for line in result.stdout.splitlines()
        if line.startswith("RUN ")
    ]


def test_qualification_script_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_qualification_dry_run_has_exact_36_condition_matrix() -> None:
    result = _dry_run()
    assert result.returncode == 0, result.stderr
    commands = _commands(result)

    assert len(commands) == 36
    rendered = [" ".join(command) for command in commands]
    assert sum("--backend dense" in command for command in rendered) == 18
    assert sum("--backend sparse" in command for command in rendered) == 18
    assert sum("--enforce-eager" in command for command in rendered) == 18
    assert sum("--no-enforce-eager" in command for command in rendered) == 18


def test_qualification_defaults_to_one_trial() -> None:
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--hardware",
            "hopper",
            "--gpu",
            "0",
            "--output-root",
            "/tmp/skylight-qualification",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(_commands(result)) == 12


def test_qualification_dry_run_locks_release_controls_and_unique_artifacts() -> None:
    result = _dry_run()
    assert result.returncode == 0, result.stderr
    commands = _commands(result)
    artifacts = []

    for command in commands:
        assert command[:4] == [
            "env",
            "CUDA_VISIBLE_DEVICES=0",
            str(Path.cwd() / ".venv/bin/python"),
            "-m",
        ]
        assert "skylight.bench.run_serving" in command
        expected_pairs = {
            "--model": "Qwen/Qwen3.5-9B",
            "--dtype": "bfloat16",
            "--gpu-memory-utilization": "0.5",
            "--max-model-len": "17408",
            "--max-num-seqs": "8",
            "--block-size": "16",
            "--gdn-prefill-backend": "triton",
            "--random-output-len": "64",
            "--temperature": "0",
            "--num-warmups": "2",
            "--server-timeout": "1800",
        }
        for flag, value in expected_pairs.items():
            assert command[command.index(flag) + 1] == value
        assert "--ignore-eos" in command
        artifacts.append(command[command.index("--artifacts-dir") + 1])

    assert len(set(artifacts)) == 36


def test_qualification_matrix_has_paired_workloads_and_bmm_profile() -> None:
    result = _dry_run(trials=1)
    assert result.returncode == 0, result.stderr
    commands = _commands(result)
    assert len(commands) == 12

    expected = [
        ("eager", "dense", 1, 4096),
        ("eager", "sparse", 1, 4096),
        ("eager", "dense", 8, 4096),
        ("eager", "sparse", 8, 4096),
        ("eager", "dense", 8, 16384),
        ("eager", "sparse", 8, 16384),
        ("cudagraph", "dense", 1, 4096),
        ("cudagraph", "sparse", 1, 4096),
        ("cudagraph", "dense", 8, 4096),
        ("cudagraph", "sparse", 8, 4096),
        ("cudagraph", "dense", 8, 16384),
        ("cudagraph", "sparse", 8, 16384),
    ]
    for command, (mode, backend, batch, input_len) in zip(
        commands,
        expected,
        strict=True,
    ):
        assert command[command.index("--backend") + 1] == backend
        assert command[command.index("--num-prompts") + 1] == str(batch)
        assert command[command.index("--random-input-len") + 1] == str(input_len)
        assert (
            "--enforce-eager" in command
            if mode == "eager"
            else "--no-enforce-eager" in command
        )
        if backend == "sparse":
            assert command[command.index("--method") + 1] == "block_minmax"
            assert command[command.index("--topk") + 1] == "0.10"
            assert command[command.index("--sink") + 1] == "64"
            assert command[command.index("--local") + 1] == "64"
            assert command[command.index("--channel-num") + 1] == "-1"


def test_qualification_rejects_invalid_inputs_without_gpu_access() -> None:
    result = _dry_run(hardware="ada", trials=1)
    assert result.returncode == 2
    assert "--hardware must be hopper or blackwell" in result.stderr
