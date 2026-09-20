import json
import os
from pathlib import Path
import subprocess
import tomllib
from types import SimpleNamespace

import pytest
import yaml

from skylight.bench import density_sweep
from skylight.runtime import Platform


VALID_CONFIG = {
    "model": "Qwen/Qwen3.5-9B",
    "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "tokenizer_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "hardware": {
        "architecture": "hopper",
        "expected_name": "NVIDIA H200",
        "device": 0,
        "memory_utilization": 0.99,
    },
    "server": {
        "mode": "cudagraph",
        "max_model_len": 32896,
        "dtype": "bfloat16",
        "block_size": 16,
        "gdn_prefill_backend": "triton",
        "timeout_seconds": 1800,
        "client_timeout_seconds": 1800,
        "hf_overrides": {
            "text_config": {
                "max_position_embeddings": 1048576,
            }
        },
    },
    "workload": {
        "dataset": "random",
        "context_length": 32768,
        "output_length": 128,
        "concurrencies": [64, 1],
        "request_rate": "inf",
        "warmups": 2,
        "temperature": 0,
        "ignore_eos": True,
    },
    "bmm": {
        "method": "block_minmax",
        "densities": [0.001, 0.1],
        "sink": 64,
        "local": 64,
        "channel_num": -1,
    },
}


def write_config(tmp_path: Path, payload=None) -> Path:
    path = tmp_path / "sweep.yaml"
    path.write_text(
        yaml.safe_dump(VALID_CONFIG if payload is None else payload),
        encoding="utf-8",
    )
    return path


def test_parses_valid_config_and_builds_canonical_groups(tmp_path):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))

    assert config.model_revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert config.tokenizer_revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert config.workload.concurrencies == (64, 1)
    assert config.max_num_seqs == 64
    assert [group.name for group in config.groups()] == [
        "dense",
        "topk-0.001",
        "topk-0.1",
    ]
    assert [group.backend for group in config.groups()] == [
        "dense",
        "sparse",
        "sparse",
    ]
    assert [group.density for group in config.groups()] == [None, 0.001, 0.1]


def test_rejects_duplicate_mapping_keys(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        yaml.safe_dump(VALID_CONFIG) + "\nmodel: duplicate/model\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key.*model"):
        density_sweep.SweepConfig.from_yaml(path)


@pytest.mark.parametrize(
    ("field", "revision"),
    [
        ("model_revision", "main"),
        ("model_revision", "c202236"),
        ("tokenizer_revision", "not-a-commit"),
        ("tokenizer_revision", "C202236235762E1C871AD0CCB60C8EE5BA337B9A"),
    ],
)
def test_requires_immutable_hugging_face_commit_revisions(
    tmp_path,
    field,
    revision,
):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload[field] = revision

    with pytest.raises(ValueError, match=field):
        density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "surprise"),
        ("hardware", "surprise"),
        ("server", "surprise"),
        ("workload", "surprise"),
        ("bmm", "surprise"),
    ],
)
def test_rejects_unknown_schema_keys(tmp_path, section, key):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    target = payload if section is None else payload[section]
    target[key] = "not-allowed"

    with pytest.raises(ValueError, match="unknown key.*surprise"):
        density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))


def test_hf_overrides_remains_open_ended(tmp_path):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload["server"]["hf_overrides"]["future_transformers_key"] = {
        "nested": "accepted",
    }

    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))

    assert config.server.hf_overrides["future_transformers_key"] == {
        "nested": "accepted",
    }


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("concurrencies", [64, 64]),
        ("densities", [0.1, 0.1]),
        ("densities", [0.1, 0.1000000000001]),
    ],
)
def test_rejects_duplicate_values_and_canonical_density_paths(
    tmp_path,
    field,
    values,
):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    section = "workload" if field == "concurrencies" else "bmm"
    payload[section][field] = values

    with pytest.raises(ValueError, match="unique"):
        density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))


@pytest.mark.parametrize("density", [0, -0.1, 1.01, float("inf"), float("nan")])
def test_rejects_invalid_density(tmp_path, density):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload["bmm"]["densities"] = [density]

    with pytest.raises(ValueError, match="densities"):
        density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))


def test_rejects_eager_mode(tmp_path):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload["server"]["mode"] = "eager"

    with pytest.raises(ValueError, match="mode.*cudagraph"):
        density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))


def test_rejects_insufficient_max_model_len(tmp_path):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload["server"]["max_model_len"] = 32895

    with pytest.raises(ValueError, match="max_model_len"):
        density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("hardware", "architecture"), "ada"),
        (("hardware", "memory_utilization"), 0),
        (("workload", "context_length"), 0),
        (("workload", "output_length"), -1),
        (("workload", "concurrencies"), []),
        (("workload", "request_rate"), 0),
        (("workload", "warmups"), -1),
        (("server", "client_timeout_seconds"), 0),
        (("bmm", "densities"), []),
    ],
)
def test_rejects_other_invalid_values(tmp_path, path, value):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload[path[0]][path[1]] = value

    with pytest.raises(ValueError):
        density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))


def test_canonical_density_is_stable_and_compact():
    assert density_sweep.canonical_density(0.1000000000001) == "0.1"
    assert density_sweep.canonical_density(0.001) == "0.001"
    assert density_sweep.canonical_density(1.0) == "1"


def test_semantic_fingerprint_is_deterministic_and_order_independent():
    first = {
        "server": {"mode": "cudagraph", "max_model_len": 32896},
        "densities": [0.001, 0.1],
    }
    reordered = {
        "densities": [0.001, 0.1],
        "server": {"max_model_len": 32896, "mode": "cudagraph"},
    }

    fingerprint = density_sweep.semantic_fingerprint(first)

    assert fingerprint == density_sweep.semantic_fingerprint(reordered)
    assert len(fingerprint) == 64
    assert fingerprint != density_sweep.semantic_fingerprint(
        {**first, "densities": [0.001]}
    )


def test_resolve_tokenizer_snapshot_uses_exact_revision(monkeypatch):
    calls = []
    revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return f"/cache/snapshots/{revision}"

    monkeypatch.setattr(
        density_sweep,
        "snapshot_download",
        fake_snapshot_download,
        raising=False,
    )

    assert density_sweep.resolve_tokenizer_snapshot(
        "Qwen/Qwen3.5-9B",
        revision,
    ) == Path(f"/cache/snapshots/{revision}")
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3.5-9B",
            "revision": revision,
        }
    ]


class FakeSession:
    created = []
    fail_clients = set()
    spawn_fail_clients = set()
    timeout_clients = set()
    healthy_after_failure = True

    def __init__(self, serve_cmd, env, port, timeout, server_log=None):
        self.serve_cmd = serve_cmd
        self.env = env
        self.port = port
        self.timeout = timeout
        self.server_log = Path(server_log)
        self.group = self.server_log.parent.name
        self.client_warmups = []
        self.client_concurrencies = []
        self.client_timeouts = []
        self.closed = False
        type(self).created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def run_benchmark(self, command, client_log=None, timeout=None):
        concurrency = int(command[command.index("--num-prompts") + 1])
        warmups = int(command[command.index("--num-warmups") + 1])
        result_path = Path(command[command.index("--result-filename") + 1])
        self.client_concurrencies.append(concurrency)
        self.client_warmups.append(warmups)
        self.client_timeouts.append(timeout)
        if (self.group, concurrency) in type(self).spawn_fail_clients:
            raise OSError("client executable unavailable")
        if (self.group, concurrency) in type(self).timeout_clients:
            raise subprocess.TimeoutExpired(command, timeout)
        Path(client_log).write_text("fake benchmark\n", encoding="utf-8")
        if (self.group, concurrency) in type(self).fail_clients:
            return SimpleNamespace(returncode=7)

        throughput = (
            float(concurrency * 10)
            if self.group == "dense"
            else float(concurrency * 12)
        )
        result_path.write_text(
            json.dumps(
                {
                    "completed": concurrency,
                    "failed": 0,
                    "output_throughput": throughput,
                    "mean_tpot_ms": 1000.0 / throughput,
                    "max_concurrent_requests": concurrency,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    def is_healthy(self):
        return type(self).healthy_after_failure


@pytest.fixture
def sweep_runtime(monkeypatch):
    FakeSession.created = []
    FakeSession.fail_clients = set()
    FakeSession.spawn_fail_clients = set()
    FakeSession.timeout_clients = set()
    FakeSession.healthy_after_failure = True
    events = []

    def fake_configure_runtime(expected_hardware):
        events.append(("configure", os.environ.get("CUDA_VISIBLE_DEVICES")))
        return Platform("hopper", (9, 0), "sm90", "9.0")

    monkeypatch.setattr(
        density_sweep,
        "configure_runtime",
        fake_configure_runtime,
        raising=False,
    )
    monkeypatch.setattr(
        density_sweep,
        "detected_gpu_name",
        lambda: "NVIDIA H200",
        raising=False,
    )
    monkeypatch.setattr(
        density_sweep,
        "require_clean_sources",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        density_sweep,
        "resolve_tokenizer_snapshot",
        lambda model, revision: Path(
            f"/models/{model.replace('/', '--')}/snapshots/{revision}"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        density_sweep,
        "collect_versions",
        lambda platform: {
            "captured_at_utc": "changes-every-run",
            "platform": {
                "name": platform.name,
                "capability": list(platform.capability),
                "sm": platform.sm,
                "torch_arch_list": platform.torch_arch_list,
            },
            "python": "3.12.test",
            "torch": "test",
            "torch_cuda": "13.0",
            "vllm": "0.21.0",
            "flashinfer_python": "0.6.8.post1",
            "skylight": {"revision": "abc", "dirty": False},
            "skylight_kernels": {"revision": "def", "dirty": False},
            "nvidia_smi": "NVIDIA H200, GPU-test, 580.0, 9.0",
        },
        raising=False,
    )
    monkeypatch.setattr(
        density_sweep,
        "collect_gpu_identity",
        lambda device: {
            "index": device,
            "name": "NVIDIA H200",
            "uuid": "GPU-test",
            "driver_version": "580.0",
            "compute_capability": "9.0",
        },
        raising=False,
    )
    ports = iter(range(18000, 18100))
    monkeypatch.setattr(
        density_sweep,
        "free_port",
        lambda: next(ports),
        raising=False,
    )
    return events


def test_gpu_identity_failure_precedes_artifact_creation(
    tmp_path,
    sweep_runtime,
    monkeypatch,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"
    monkeypatch.setattr(
        density_sweep,
        "collect_gpu_identity",
        lambda device: (_ for _ in ()).throw(
            RuntimeError("selected GPU identity unavailable")
        ),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="selected GPU identity"):
        density_sweep.run_sweep(
            config,
            output_root,
            session_factory=FakeSession,
        )

    assert not output_root.exists()
    assert FakeSession.created == []


def test_rejects_out_of_yaml_experiment_overrides_before_startup(
    tmp_path,
    sweep_runtime,
    monkeypatch,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    monkeypatch.setenv("SKYLIGHT_TP", "2")

    with pytest.raises(ValueError, match="SKYLIGHT_TP.*YAML"):
        density_sweep.run_sweep(
            config,
            tmp_path / "results",
            session_factory=FakeSession,
        )

    assert FakeSession.created == []


def test_ambient_vllm_environment_participates_in_resume_identity(
    tmp_path,
    sweep_runtime,
    monkeypatch,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"
    monkeypatch.setenv("VLLM_TEST_EXECUTION_POLICY", "first")
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0

    monkeypatch.setenv("VLLM_TEST_EXECUTION_POLICY", "second")
    FakeSession.created = []
    with pytest.raises(ValueError, match="fingerprint mismatch.*new output root"):
        density_sweep.run_sweep(
            config,
            output_root,
            session_factory=FakeSession,
        )

    assert FakeSession.created == []


def test_relevant_source_changes_ignore_generated_artifacts(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = repository / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    bytecode = cache / "source.cpython-312.pyc"
    bytecode.write_bytes(b"original")
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "add",
            "source.py",
            "pkg/__pycache__/source.cpython-312.pyc",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    bytecode.write_bytes(b"generated update")
    (repository / ".venv").symlink_to("/tmp/test-venv")
    assert density_sweep._relevant_source_changes(repository) == ()

    (repository / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert density_sweep._relevant_source_changes(repository) == ("source.py",)


def test_run_sweep_reuses_one_server_per_group(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"

    returncode = density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    )

    assert returncode == 0
    assert sweep_runtime == [("configure", "0")]
    assert [(session.group, session.client_warmups) for session in FakeSession.created] == [
        ("dense", [2, 2]),
        ("topk-0.001", [2, 2]),
        ("topk-0.1", [2, 2]),
    ]
    assert all(
        session.client_concurrencies == [64, 1]
        for session in FakeSession.created
    )
    assert all(
        session.client_timeouts == [1800, 1800]
        for session in FakeSession.created
    )
    assert all(
        session.serve_cmd[
            session.serve_cmd.index("--max-num-seqs") + 1
        ] == "64"
        for session in FakeSession.created
    )
    assert all(session.closed for session in FakeSession.created)
    assert (
        output_root / "topk-0.001" / "concurrency-64" / "status.json"
    ).is_file()
    assert (output_root / "topk-0.001" / "server-resolved.json").is_file()


def test_completed_resume_starts_no_server(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0

    FakeSession.created = []
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0

    assert FakeSession.created == []


def test_fingerprint_mismatch_fails_before_server_start(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0
    manifest_path = output_root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fingerprint"] = "not-the-current-fingerprint"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    FakeSession.created = []
    with pytest.raises(ValueError, match="fingerprint mismatch.*new output root"):
        density_sweep.run_sweep(
            config,
            output_root,
            session_factory=FakeSession,
        )

    assert FakeSession.created == []


def test_case_status_fingerprint_mismatch_fails_before_server_start(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0
    status_path = output_root / "dense" / "concurrency-64" / "status.json"
    status = json.loads(status_path.read_text())
    status["fingerprint"] = "different-case-identity"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    FakeSession.created = []
    with pytest.raises(ValueError, match="fingerprint mismatch.*new output root"):
        density_sweep.run_sweep(
            config,
            output_root,
            session_factory=FakeSession,
        )

    assert FakeSession.created == []


def test_failed_case_artifacts_are_preserved_before_rerun(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0
    case_dir = output_root / "dense" / "concurrency-64"
    status_path = case_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["state"] = "failed"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    (case_dir / "marker.txt").write_text("preserve me", encoding="utf-8")

    FakeSession.created = []
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0

    assert [(session.group, session.client_concurrencies) for session in FakeSession.created] == [
        ("dense", [64]),
    ]
    assert FakeSession.created[0].client_warmups == [2]
    assert (
        case_dir / "attempts" / "attempt-1" / "marker.txt"
    ).read_text(encoding="utf-8") == "preserve me"
    assert json.loads(status_path.read_text())["state"] == "success"


@pytest.mark.parametrize(
    ("healthy", "expected_concurrencies"),
    [(True, [64, 1]), (False, [64])],
)
def test_client_failure_continues_only_while_server_is_healthy(
    tmp_path,
    sweep_runtime,
    healthy,
    expected_concurrencies,
):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload["bmm"]["densities"] = [0.1]
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))
    FakeSession.fail_clients = {("dense", 64)}
    FakeSession.healthy_after_failure = healthy

    returncode = density_sweep.run_sweep(
        config,
        tmp_path / "results",
        session_factory=FakeSession,
    )

    assert returncode == 1
    assert FakeSession.created[0].client_concurrencies == expected_concurrencies
    failed = json.loads(
        (
            tmp_path
            / "results"
            / "dense"
            / "concurrency-64"
            / "status.json"
        ).read_text()
    )
    assert failed["state"] == "failed"
    assert failed["returncode"] == 7


def test_client_timeout_unconditionally_stops_tainted_server_group(
    tmp_path,
    sweep_runtime,
):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload["bmm"]["densities"] = [0.1]
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))
    FakeSession.timeout_clients = {("dense", 64)}
    FakeSession.healthy_after_failure = True

    assert density_sweep.run_sweep(
        config,
        tmp_path / "results",
        session_factory=FakeSession,
    ) == 1

    assert FakeSession.created[0].client_concurrencies == [64]
    assert FakeSession.created[0].closed
    status = json.loads(
        (
            tmp_path
            / "results"
            / "dense"
            / "concurrency-64"
            / "status.json"
        ).read_text()
    )
    assert status["state"] == "failed"
    assert "TimeoutExpired" in status["error"]


def test_pre_spawn_client_failure_keeps_warmups_for_next_case(
    tmp_path,
    sweep_runtime,
):
    payload = yaml.safe_load(yaml.safe_dump(VALID_CONFIG))
    payload["bmm"]["densities"] = [0.1]
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path, payload))
    FakeSession.spawn_fail_clients = {("dense", 64)}

    assert density_sweep.run_sweep(
        config,
        tmp_path / "results",
        session_factory=FakeSession,
    ) == 1

    assert FakeSession.created[0].client_warmups == [2, 2]


def test_summary_joins_sparse_metrics_to_matching_dense_concurrency(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"

    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0

    summary = json.loads((output_root / "summary.json").read_text())
    sparse_64 = next(
        row
        for row in summary["rows"]
        if row["group"] == "topk-0.001" and row["concurrency"] == 64
    )
    assert sparse_64["output_throughput"] == 768.0
    assert sparse_64["dense_output_throughput"] == 640.0
    assert sparse_64["output_throughput_ratio"] == pytest.approx(1.2)
    assert sparse_64["dense_mean_tpot_ms"] == pytest.approx(1000 / 640)
    assert sparse_64["tpot_speedup"] == pytest.approx(1.2)
    assert len(summary["run_fingerprint"]) == 64
    assert summary["config"]["server"]["mode"] == "cudagraph"
    assert sparse_64["run_fingerprint"] == summary["run_fingerprint"]
    assert (output_root / "summary.csv").is_file()


def test_summary_excludes_success_with_mismatched_resolved_fingerprint(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0
    resolved_path = (
        output_root / "dense" / "concurrency-64" / "resolved.json"
    )
    resolved = json.loads(resolved_path.read_text())
    resolved["fingerprint"] = "stale"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")

    density_sweep.write_summary(output_root, config)

    rows = json.loads((output_root / "summary.json").read_text())["rows"]
    assert not any(
        row["group"] == "dense" and row["concurrency"] == 64
        for row in rows
    )


def test_hardware_name_mismatch_fails_before_server_start(
    tmp_path,
    sweep_runtime,
    monkeypatch,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    monkeypatch.setattr(
        density_sweep,
        "detected_gpu_name",
        lambda: "NVIDIA H100 80GB HBM3",
    )

    with pytest.raises(RuntimeError, match="expected.*NVIDIA H200.*detected.*H100"):
        density_sweep.run_sweep(
            config,
            tmp_path / "results",
            session_factory=FakeSession,
        )

    assert FakeSession.created == []


def test_group_and_case_manifests_include_config_and_run_identity(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))
    output_root = tmp_path / "results"

    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=FakeSession,
    ) == 0

    for path in (
        output_root / "dense" / "server-resolved.json",
        output_root / "dense" / "concurrency-64" / "resolved.json",
    ):
        manifest = json.loads(path.read_text())
        assert manifest["schema_version"] == 1
        assert manifest["config"]["server"]["mode"] == "cudagraph"
        assert manifest["identity"]["skylight"]["revision"] == "abc"
        assert "captured_at_utc" not in manifest["identity"]
        assert len(manifest["fingerprint"]) == 64

    server_manifest = json.loads(
        (output_root / "dense" / "server-resolved.json").read_text()
    )
    case_manifest = json.loads(
        (
            output_root
            / "dense"
            / "concurrency-64"
            / "resolved.json"
        ).read_text()
    )
    assert "--max-num-seqs" in server_manifest["serve_command"]
    assert server_manifest["serve_command"][
        server_manifest["serve_command"].index("--revision") + 1
    ] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert server_manifest["serve_command"][
        server_manifest["serve_command"].index("--tokenizer-revision") + 1
    ] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert server_manifest["environment"]["SKYLIGHT_HF_OVERRIDES"]
    assert "--num-prompts" in case_manifest["benchmark_command"]
    assert case_manifest["benchmark_command"][
        case_manifest["benchmark_command"].index("--tokenizer") + 1
    ] == (
        "/models/Qwen--Qwen3.5-9B/snapshots/"
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    assert case_manifest["warmups"] == 2
    case_status = json.loads(
        (
            output_root
            / "topk-0.001"
            / "concurrency-64"
            / "status.json"
        ).read_text()
    )
    assert case_status["started_at_utc"]
    assert case_status["finished_at_utc"]
    markers = [
        json.loads(line)
        for line in (
            output_root
            / "topk-0.001"
            / "telemetry"
            / "case-markers.jsonl"
        ).read_text().splitlines()
    ]
    assert [
        (marker["phase"], marker["concurrency"])
        for marker in markers[:2]
    ] == [("start", 64), ("finish", 64)]


def test_startup_failure_is_preserved_in_group_status(
    tmp_path,
    sweep_runtime,
):
    config = density_sweep.SweepConfig.from_yaml(write_config(tmp_path))

    class StartupFailure(FakeSession):
        def __enter__(self):
            raise RuntimeError("startup exploded")

    output_root = tmp_path / "results"
    assert density_sweep.run_sweep(
        config,
        output_root,
        session_factory=StartupFailure,
    ) == 1

    status = json.loads(
        (output_root / "dense" / "group-status.json").read_text()
    )
    assert status["state"] == "failed"
    assert "startup exploded" in status["error"]


def test_checked_h200_config_is_the_exact_32k_density_matrix():
    repository = Path(__file__).resolve().parents[1]
    config = density_sweep.SweepConfig.from_yaml(
        repository / "experiments" / "h200-32k-density.yaml"
    )

    assert config.model == "Qwen/Qwen3.5-9B"
    assert config.model_revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert config.tokenizer_revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert config.hardware.architecture == "hopper"
    assert config.hardware.expected_name == "NVIDIA H200"
    assert config.hardware.memory_utilization == 0.99
    assert config.server.mode == "cudagraph"
    assert config.server.max_model_len == 32896
    assert config.server.client_timeout_seconds == 1800
    assert config.workload.context_length == 32768
    assert config.workload.output_length == 128
    assert config.workload.concurrencies == (64, 32, 16, 8, 4, 2, 1)
    assert config.bmm.densities == (
        0.001,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )
    assert len(config.groups()) == 13
    assert len(config.groups()) * len(config.workload.concurrencies) == 91


def test_pyyaml_is_a_direct_bounded_dependency():
    repository = Path(__file__).resolve().parents[1]
    project = tomllib.loads(
        (repository / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert "pyyaml>=6,<7" in project["project"]["dependencies"]
    assert "huggingface-hub>=1,<2" in project["project"]["dependencies"]
