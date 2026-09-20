"""Persistent-server density sweep configuration and runner."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping

from huggingface_hub import snapshot_download
import yaml

from skylight.bench.run_serving import (
    ServingSession,
    build_benchmark_cmd,
    build_env,
    build_serve_cmd,
    collect_gpu_identity,
    collect_versions,
    free_port,
    safe_environment_snapshot,
)
from skylight.runtime import configure_runtime


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError(f"unhashable mapping key {key!r}") from exc
        if duplicate:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def canonical_density(value: float) -> str:
    """Return the stable decimal spelling used in density artifact paths."""
    return format(value, ".12g")


def semantic_fingerprint(payload: Mapping[str, object]) -> str:
    """Hash canonical JSON so mapping insertion order cannot affect resume."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HardwareConfig:
    architecture: str
    expected_name: str
    device: int
    memory_utilization: float


@dataclass(frozen=True)
class ServerConfig:
    mode: str
    max_model_len: int
    dtype: str
    block_size: int
    gdn_prefill_backend: str
    timeout_seconds: float
    client_timeout_seconds: float
    hf_overrides: Mapping[str, Any]


@dataclass(frozen=True)
class WorkloadConfig:
    dataset: str
    context_length: int
    output_length: int
    concurrencies: tuple[int, ...]
    request_rate: float
    warmups: int
    temperature: float
    ignore_eos: bool


@dataclass(frozen=True)
class BMMConfig:
    method: str
    densities: tuple[float, ...]
    sink: int
    local: int
    channel_num: int


@dataclass(frozen=True)
class ServerGroup:
    name: str
    backend: str
    density: float | None


@dataclass(frozen=True)
class SweepConfig:
    model: str
    model_revision: str
    tokenizer_revision: str
    hardware: HardwareConfig
    server: ServerConfig
    workload: WorkloadConfig
    bmm: BMMConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "SweepConfig":
        path = Path(path)
        try:
            payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {path}: {exc}") from exc

        root = _require_mapping(payload, "root")
        _require_keys(
            root,
            "root",
            {
                "model",
                "model_revision",
                "tokenizer_revision",
                "hardware",
                "server",
                "workload",
                "bmm",
            },
        )

        model = _nonempty_string(root["model"], "model")
        model_revision = _commit_revision(
            root["model_revision"],
            "model_revision",
        )
        tokenizer_revision = _commit_revision(
            root["tokenizer_revision"],
            "tokenizer_revision",
        )
        hardware = _parse_hardware(root["hardware"])
        server = _parse_server(root["server"])
        workload = _parse_workload(root["workload"])
        bmm = _parse_bmm(root["bmm"])

        required_model_len = workload.context_length + workload.output_length
        if server.max_model_len < required_model_len:
            raise ValueError(
                "server.max_model_len must be at least "
                "workload.context_length + workload.output_length "
                f"({required_model_len})"
            )
        return cls(
            model=model,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            hardware=hardware,
            server=server,
            workload=workload,
            bmm=bmm,
        )

    @property
    def max_num_seqs(self) -> int:
        return max(self.workload.concurrencies)

    def groups(self) -> tuple[ServerGroup, ...]:
        return (
            ServerGroup(name="dense", backend="dense", density=None),
            *(
                ServerGroup(
                    name=f"topk-{canonical_density(density)}",
                    backend="sparse",
                    density=density,
                )
                for density in self.bmm.densities
            ),
        )

    def semantic_payload(self) -> dict[str, object]:
        """Return a JSON-safe experiment definition with no runtime identity."""
        payload = asdict(self)
        request_rate = payload["workload"]["request_rate"]
        if math.isinf(request_rate):
            payload["workload"]["request_rate"] = "inf"
        return payload


def _require_mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{location} keys must be strings")
    return value


def _require_keys(
    payload: Mapping[str, object],
    location: str,
    allowed: set[str],
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {location}: {', '.join(unknown)}")
    missing = sorted(allowed - set(payload))
    if missing:
        raise ValueError(f"missing key(s) in {location}: {', '.join(missing)}")


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


_COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")


def _commit_revision(value: object, location: str) -> str:
    revision = _nonempty_string(value, location)
    if _COMMIT_REVISION.fullmatch(revision) is None:
        raise ValueError(f"{location} must be a 40-character lowercase commit SHA")
    return revision


def resolve_tokenizer_snapshot(model: str, revision: str) -> Path:
    """Resolve the benchmark tokenizer from the exact immutable HF snapshot."""
    return Path(snapshot_download(repo_id=model, revision=revision))


def _integer(value: object, location: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{location} must be at least {minimum}")
    return value


def _finite_number(
    value: object,
    location: str,
    *,
    minimum_exclusive: float | None = None,
    maximum_inclusive: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    if minimum_exclusive is not None and result <= minimum_exclusive:
        raise ValueError(f"{location} must be greater than {minimum_exclusive}")
    if maximum_inclusive is not None and result > maximum_inclusive:
        raise ValueError(f"{location} must be at most {maximum_inclusive}")
    return result


def _parse_hardware(value: object) -> HardwareConfig:
    payload = _require_mapping(value, "hardware")
    _require_keys(
        payload,
        "hardware",
        {"architecture", "expected_name", "device", "memory_utilization"},
    )
    architecture = _nonempty_string(payload["architecture"], "hardware.architecture")
    if architecture not in {"hopper", "blackwell"}:
        raise ValueError("hardware.architecture must be hopper or blackwell")
    return HardwareConfig(
        architecture=architecture,
        expected_name=_nonempty_string(
            payload["expected_name"],
            "hardware.expected_name",
        ),
        device=_integer(payload["device"], "hardware.device", minimum=0),
        memory_utilization=_finite_number(
            payload["memory_utilization"],
            "hardware.memory_utilization",
            minimum_exclusive=0,
            maximum_inclusive=1,
        ),
    )


def _parse_server(value: object) -> ServerConfig:
    payload = _require_mapping(value, "server")
    _require_keys(
        payload,
        "server",
        {
            "mode",
            "max_model_len",
            "dtype",
            "block_size",
            "gdn_prefill_backend",
            "timeout_seconds",
            "client_timeout_seconds",
            "hf_overrides",
        },
    )
    mode = _nonempty_string(payload["mode"], "server.mode")
    if mode != "cudagraph":
        raise ValueError("server.mode must be cudagraph")
    hf_overrides = _require_mapping(payload["hf_overrides"], "server.hf_overrides")
    return ServerConfig(
        mode=mode,
        max_model_len=_integer(
            payload["max_model_len"],
            "server.max_model_len",
            minimum=1,
        ),
        dtype=_nonempty_string(payload["dtype"], "server.dtype"),
        block_size=_integer(payload["block_size"], "server.block_size", minimum=1),
        gdn_prefill_backend=_nonempty_string(
            payload["gdn_prefill_backend"],
            "server.gdn_prefill_backend",
        ),
        timeout_seconds=_finite_number(
            payload["timeout_seconds"],
            "server.timeout_seconds",
            minimum_exclusive=0,
        ),
        client_timeout_seconds=_finite_number(
            payload["client_timeout_seconds"],
            "server.client_timeout_seconds",
            minimum_exclusive=0,
        ),
        hf_overrides=hf_overrides,
    )


def _parse_workload(value: object) -> WorkloadConfig:
    payload = _require_mapping(value, "workload")
    _require_keys(
        payload,
        "workload",
        {
            "dataset",
            "context_length",
            "output_length",
            "concurrencies",
            "request_rate",
            "warmups",
            "temperature",
            "ignore_eos",
        },
    )
    raw_concurrencies = payload["concurrencies"]
    if not isinstance(raw_concurrencies, list) or not raw_concurrencies:
        raise ValueError("workload.concurrencies must be a non-empty list")
    concurrencies = tuple(
        _integer(item, "workload.concurrencies", minimum=1)
        for item in raw_concurrencies
    )
    if len(concurrencies) != len(set(concurrencies)):
        raise ValueError("workload.concurrencies must be unique")

    raw_rate = payload["request_rate"]
    if isinstance(raw_rate, str) and raw_rate.lower() in {"inf", "infinity"}:
        request_rate = math.inf
    else:
        request_rate = _finite_number(
            raw_rate,
            "workload.request_rate",
            minimum_exclusive=0,
        )

    ignore_eos = payload["ignore_eos"]
    if not isinstance(ignore_eos, bool):
        raise ValueError("workload.ignore_eos must be a boolean")
    return WorkloadConfig(
        dataset=_nonempty_string(payload["dataset"], "workload.dataset"),
        context_length=_integer(
            payload["context_length"],
            "workload.context_length",
            minimum=1,
        ),
        output_length=_integer(
            payload["output_length"],
            "workload.output_length",
            minimum=1,
        ),
        concurrencies=concurrencies,
        request_rate=request_rate,
        warmups=_integer(payload["warmups"], "workload.warmups", minimum=0),
        temperature=_finite_number(
            payload["temperature"],
            "workload.temperature",
        ),
        ignore_eos=ignore_eos,
    )


def _parse_bmm(value: object) -> BMMConfig:
    payload = _require_mapping(value, "bmm")
    _require_keys(
        payload,
        "bmm",
        {"method", "densities", "sink", "local", "channel_num"},
    )
    method = _nonempty_string(payload["method"], "bmm.method")
    if method != "block_minmax":
        raise ValueError("bmm.method must be block_minmax")

    raw_densities = payload["densities"]
    if not isinstance(raw_densities, list) or not raw_densities:
        raise ValueError("bmm.densities must be a non-empty list")
    densities = tuple(
        _finite_number(
            item,
            "bmm.densities",
            minimum_exclusive=0,
            maximum_inclusive=1,
        )
        for item in raw_densities
    )
    paths = tuple(canonical_density(density) for density in densities)
    if len(paths) != len(set(paths)):
        raise ValueError("bmm.densities must have unique canonical paths")

    return BMMConfig(
        method=method,
        densities=densities,
        sink=_integer(payload["sink"], "bmm.sink", minimum=0),
        local=_integer(payload["local"], "bmm.local", minimum=0),
        channel_num=_integer(payload["channel_num"], "bmm.channel_num"),
    )


SCHEMA_VERSION = 1
_SUMMARY_FIELDS = (
    "run_fingerprint",
    "group",
    "backend",
    "density",
    "concurrency",
    "completed",
    "failed",
    "output_throughput",
    "mean_tpot_ms",
    "max_concurrent_requests",
    "dense_output_throughput",
    "dense_mean_tpot_ms",
    "output_throughput_ratio",
    "tpot_speedup",
)
_FORBIDDEN_EXPERIMENT_ENV = (
    "SKYLIGHT_BENCH_DETAILED",
    "SKYLIGHT_BENCH_PROFILE",
    "SKYLIGHT_KV_DTYPE",
    "SKYLIGHT_TP",
)
_CONFIG_CONTROLLED_ENV = {
    "SKYLIGHT_BLOCK_SIZE",
    "SKYLIGHT_FI_BSR",
    "SKYLIGHT_HF_OVERRIDES",
    "SKYLIGHT_METRICS_LOG_DIR",
    "SKYLIGHT_METRICS_SAMPLING",
}
_CONFIG_CONTROLLED_PREFIXES = ("SKYLIGHT_INCR_", "SKYLIGHT_SPARSE_")
_IGNORED_UNTRACKED_ROOTS = {".venv", "skylight_kernels"}


def detected_gpu_name() -> str:
    """Return the name of the first GPU made visible by the config."""
    import torch

    return str(torch.cuda.get_device_name(0))


def _validate_inherited_environment() -> None:
    present = [
        name
        for name in _FORBIDDEN_EXPERIMENT_ENV
        if name in os.environ
    ]
    if present:
        raise ValueError(
            f"{', '.join(present)} must be unset; this focused runner accepts "
            "benchmark policy only through YAML"
        )


def _ignored_source_path(path: str) -> bool:
    parts = Path(path).parts
    return (
        not parts
        or parts[0] in _IGNORED_UNTRACKED_ROOTS
        or "__pycache__" in parts
        or path.endswith(".pyc")
    )


def _git_paths(repository: Path, arguments: list[str]) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"cannot inspect source checkout {repository}: {error}")
    return tuple(
        item.decode(errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    )


def _relevant_source_changes(repository: Path) -> tuple[str, ...]:
    """List real source changes while ignoring generated host artifacts."""
    repository = Path(repository)
    tracked = _git_paths(
        repository,
        [
            "diff",
            "--name-only",
            "-z",
            "HEAD",
            "--",
            ".",
            ":(exclude)**/__pycache__/**",
            ":(exclude)**/*.pyc",
        ],
    )
    untracked = _git_paths(
        repository,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    return tuple(
        sorted(
            {
                path
                for path in (*tracked, *untracked)
                if not _ignored_source_path(path)
            }
        )
    )


def require_clean_sources() -> None:
    """Require committed Skylight and kernel source for resumable evidence."""
    repository = Path(__file__).resolve().parents[3]
    checkouts = {
        "skylight": repository,
        "skylight_kernels": repository / "skylight_kernels",
    }
    dirty = {
        name: changes
        for name, path in checkouts.items()
        if (changes := _relevant_source_changes(path))
    }
    if dirty:
        details = "; ".join(
            f"{name}: {', '.join(changes)}"
            for name, changes in dirty.items()
        )
        raise RuntimeError(
            "density sweeps require committed source; commit or stash: "
            f"{details}"
        )


def _fingerprinted_environment() -> dict[str, str]:
    snapshot = safe_environment_snapshot(dict(os.environ))
    return {
        name: value
        for name, value in snapshot.items()
        if name not in _CONFIG_CONTROLLED_ENV
        and not name.startswith(_CONFIG_CONTROLLED_PREFIXES)
    }


def _captured_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_json_line(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest {path} must contain a JSON object")
    return payload


def _immutable_identity(versions: Mapping[str, object]) -> dict[str, object]:
    """Remove observation time while retaining code, package, and GPU identity."""
    identity = {
        key: value
        for key, value in versions.items()
        if key != "captured_at_utc"
    }
    for name in ("skylight", "skylight_kernels"):
        source = identity.get(name)
        if isinstance(source, Mapping):
            normalized = dict(source)
            normalized.pop("dirty", None)
            normalized["source_clean"] = True
            identity[name] = normalized
    identity["environment"] = _fingerprinted_environment()
    return identity


def _run_manifest(
    config: SweepConfig,
    identity: Mapping[str, object],
) -> dict[str, object]:
    semantic = config.semantic_payload()
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "config": semantic,
        "identity": identity,
    }
    return {
        **fingerprint_payload,
        "fingerprint": semantic_fingerprint(fingerprint_payload),
    }


def _group_manifest(
    run_manifest: Mapping[str, object],
    group: ServerGroup,
) -> dict[str, object]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_manifest["fingerprint"],
        "config": run_manifest["config"],
        "identity": run_manifest["identity"],
        "group": asdict(group),
    }
    return {**payload, "fingerprint": semantic_fingerprint(payload)}


def _case_manifest(
    group_manifest: Mapping[str, object],
    concurrency: int,
) -> dict[str, object]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "group_fingerprint": group_manifest["fingerprint"],
        "config": group_manifest["config"],
        "identity": group_manifest["identity"],
        "group": group_manifest["group"],
        "concurrency": concurrency,
    }
    return {**payload, "fingerprint": semantic_fingerprint(payload)}


def _require_matching_fingerprint(
    path: Path,
    expected: str,
) -> None:
    if not path.exists():
        return
    actual = _read_json(path).get("fingerprint")
    if actual != expected:
        raise ValueError(
            f"fingerprint mismatch in {path}; choose a new output root"
        )


def _validate_resume(
    output_root: Path,
    config: SweepConfig,
    run_manifest: Mapping[str, object],
) -> None:
    _require_matching_fingerprint(
        output_root / "run-manifest.json",
        str(run_manifest["fingerprint"]),
    )
    for group in config.groups():
        group_manifest = _group_manifest(run_manifest, group)
        group_dir = output_root / group.name
        _require_matching_fingerprint(
            group_dir / "server-resolved.json",
            str(group_manifest["fingerprint"]),
        )
        _require_matching_fingerprint(
            group_dir / "group-status.json",
            str(group_manifest["fingerprint"]),
        )
        for concurrency in config.workload.concurrencies:
            case_manifest = _case_manifest(group_manifest, concurrency)
            case_dir = group_dir / f"concurrency-{concurrency}"
            for name in ("resolved.json", "status.json"):
                _require_matching_fingerprint(
                    case_dir / name,
                    str(case_manifest["fingerprint"]),
                )


def _result_succeeded(result: Mapping[str, object], concurrency: int) -> bool:
    return (
        result.get("completed") == concurrency
        and result.get("failed") == 0
    )


def case_complete(
    case_dir: Path,
    fingerprint: str,
    concurrency: int,
) -> bool:
    """Return whether one case is a successful matching resumable result."""
    case_dir = Path(case_dir)
    try:
        manifest = _read_json(case_dir / "resolved.json")
        status = _read_json(case_dir / "status.json")
        result = _read_json(case_dir / "result.json")
    except ValueError:
        return False
    return (
        manifest.get("fingerprint") == fingerprint
        and status.get("fingerprint") == fingerprint
        and status.get("state") == "success"
        and _result_succeeded(result, concurrency)
    )


def _archive_case_artifacts(case_dir: Path) -> None:
    existing = [
        path
        for path in case_dir.iterdir()
        if path.name != "attempts"
    ] if case_dir.is_dir() else []
    if not existing:
        return

    attempts_dir = case_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (attempts_dir / f"attempt-{attempt}").exists():
        attempt += 1
    destination = attempts_dir / f"attempt-{attempt}"
    destination.mkdir()
    for path in existing:
        shutil.move(str(path), destination / path.name)


def _status(
    state: str,
    fingerprint: str,
    *,
    started_at_utc: str | None = None,
    returncode: int | None = None,
    error: str | None = None,
) -> dict[str, object]:
    updated_at = _captured_at()
    started_at = started_at_utc or updated_at
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "state": state,
        "started_at_utc": started_at,
        "updated_at_utc": updated_at,
    }
    if state != "running":
        payload["finished_at_utc"] = updated_at
    if returncode is not None:
        payload["returncode"] = returncode
    if error is not None:
        payload["error"] = error
    return payload


def _summary_result(
    case_dir: Path,
    concurrency: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    try:
        resolved = _read_json(case_dir / "resolved.json")
        status = _read_json(case_dir / "status.json")
        result = _read_json(case_dir / "result.json")
    except ValueError:
        return None
    if (
        resolved.get("fingerprint") != fingerprint
        or status.get("fingerprint") != fingerprint
        or status.get("state") != "success"
        or not _result_succeeded(result, concurrency)
    ):
        return None
    return result


def _ratio(numerator: object, denominator: object) -> float | None:
    if not isinstance(numerator, (int, float)):
        return None
    if not isinstance(denominator, (int, float)) or denominator == 0:
        return None
    return float(numerator) / float(denominator)


def write_summary(output_root: Path, config: SweepConfig) -> None:
    """Atomically rebuild JSON/CSV summaries from all successful cases."""
    output_root = Path(output_root)
    run_manifest = _read_json(output_root / "run-manifest.json")
    fingerprint_payload = {
        "schema_version": run_manifest.get("schema_version"),
        "config": run_manifest.get("config"),
        "identity": run_manifest.get("identity"),
    }
    run_fingerprint = semantic_fingerprint(fingerprint_payload)
    manifest_config_fingerprint = semantic_fingerprint(
        {"config": fingerprint_payload["config"]}
    )
    requested_config_fingerprint = semantic_fingerprint(
        {"config": config.semantic_payload()}
    )
    if (
        fingerprint_payload["schema_version"] != SCHEMA_VERSION
        or manifest_config_fingerprint != requested_config_fingerprint
        or run_manifest.get("fingerprint") != run_fingerprint
    ):
        raise ValueError(
            "run manifest does not match the summary configuration"
        )

    dense: dict[int, Mapping[str, object]] = {}
    dense_group = config.groups()[0]
    dense_manifest = _group_manifest(run_manifest, dense_group)
    for concurrency in config.workload.concurrencies:
        case_manifest = _case_manifest(dense_manifest, concurrency)
        result = _summary_result(
            output_root / "dense" / f"concurrency-{concurrency}",
            concurrency,
            str(case_manifest["fingerprint"]),
        )
        if result is not None:
            dense[concurrency] = result

    rows = []
    for group in config.groups():
        group_manifest = _group_manifest(run_manifest, group)
        for concurrency in config.workload.concurrencies:
            case_manifest = _case_manifest(group_manifest, concurrency)
            result = _summary_result(
                output_root / group.name / f"concurrency-{concurrency}",
                concurrency,
                str(case_manifest["fingerprint"]),
            )
            if result is None:
                continue
            dense_result = dense.get(concurrency)
            dense_throughput = (
                dense_result.get("output_throughput")
                if dense_result is not None
                else None
            )
            dense_tpot = (
                dense_result.get("mean_tpot_ms")
                if dense_result is not None
                else None
            )
            rows.append(
                {
                    "run_fingerprint": run_fingerprint,
                    "group": group.name,
                    "backend": group.backend,
                    "density": group.density,
                    "concurrency": concurrency,
                    "completed": result.get("completed"),
                    "failed": result.get("failed"),
                    "output_throughput": result.get("output_throughput"),
                    "mean_tpot_ms": result.get("mean_tpot_ms"),
                    "max_concurrent_requests": result.get(
                        "max_concurrent_requests"
                    ),
                    "dense_output_throughput": dense_throughput,
                    "dense_mean_tpot_ms": dense_tpot,
                    "output_throughput_ratio": _ratio(
                        result.get("output_throughput"),
                        dense_throughput,
                    ),
                    "tpot_speedup": _ratio(
                        dense_tpot,
                        result.get("mean_tpot_ms"),
                    ),
                }
            )

    _atomic_write_json(
        output_root / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "config": config.semantic_payload(),
            "updated_at_utc": _captured_at(),
            "rows": rows,
        },
    )
    csv_path = output_root / "summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)


def _build_group_runtime(
    config: SweepConfig,
    group: ServerGroup,
    group_dir: Path,
    port: int,
) -> tuple[list[str], dict[str, str]]:
    hf_overrides = json.dumps(
        config.server.hf_overrides,
        sort_keys=True,
        separators=(",", ":"),
    )
    previous_overrides = os.environ.get("SKYLIGHT_HF_OVERRIDES")
    os.environ["SKYLIGHT_HF_OVERRIDES"] = hf_overrides
    try:
        command = build_serve_cmd(
            backend=group.backend,
            model=config.model,
            port=port,
            max_model_len=config.server.max_model_len,
            enforce_eager=False,
            gpu_memory_utilization=config.hardware.memory_utilization,
            max_num_seqs=config.max_num_seqs,
            gdn_prefill_backend=config.server.gdn_prefill_backend,
            dtype=config.server.dtype,
            block_size=config.server.block_size,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
        )
        environment = build_env(
            backend=group.backend,
            topk=group.density,
            sink=config.bmm.sink,
            local=config.bmm.local,
            channel_num=config.bmm.channel_num,
            method=config.bmm.method,
            block_size=config.server.block_size,
            metrics_dir=(
                group_dir / "telemetry"
                if group.backend == "sparse"
                else None
            ),
        )
    finally:
        if previous_overrides is None:
            os.environ.pop("SKYLIGHT_HF_OVERRIDES", None)
        else:
            os.environ["SKYLIGHT_HF_OVERRIDES"] = previous_overrides
    environment["SKYLIGHT_HF_OVERRIDES"] = hf_overrides
    return command, environment


def _benchmark_command(
    config: SweepConfig,
    port: int,
    concurrency: int,
    warmups: int,
    case_dir: Path,
    tokenizer_snapshot: Path,
) -> list[str]:
    return build_benchmark_cmd(
        model=config.model,
        port=port,
        num_prompts=concurrency,
        dataset_name=config.workload.dataset,
        request_rate=config.workload.request_rate,
        output_file=str(case_dir / "result.json"),
        random_input_len=config.workload.context_length,
        random_output_len=config.workload.output_length,
        num_warmups=warmups,
        temperature=config.workload.temperature,
        ignore_eos=config.workload.ignore_eos,
        tokenizer=str(tokenizer_snapshot),
    )


def run_sweep(
    config: SweepConfig,
    output_root: Path,
    session_factory=ServingSession,
) -> int:
    """Run missing cases, reusing one immutable server per density group."""
    _validate_inherited_environment()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.hardware.device)
    platform = configure_runtime(config.hardware.architecture)
    require_clean_sources()
    gpu_name = detected_gpu_name()
    if config.hardware.expected_name not in gpu_name:
        raise RuntimeError(
            f"expected GPU name containing {config.hardware.expected_name!r}; "
            f"detected {gpu_name!r}"
        )

    selected_gpu = collect_gpu_identity(config.hardware.device)
    versions = {
        **collect_versions(platform),
        "selected_gpu": selected_gpu,
    }
    identity = _immutable_identity(versions)
    manifest = _run_manifest(config, identity)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_resume(output_root, config, manifest)
    _atomic_write_json(output_root / "run-manifest.json", manifest)
    print(
        f"[density-sweep] gpu={gpu_name} output={output_root} "
        f"max_num_seqs={config.max_num_seqs}",
        flush=True,
    )
    tokenizer_snapshot: Path | None = None

    for group in config.groups():
        group_manifest = _group_manifest(manifest, group)
        group_dir = output_root / group.name
        cases = [
            (
                concurrency,
                group_dir / f"concurrency-{concurrency}",
                _case_manifest(group_manifest, concurrency),
            )
            for concurrency in config.workload.concurrencies
        ]
        incomplete = [
            item
            for item in cases
            if not case_complete(
                item[1],
                str(item[2]["fingerprint"]),
                item[0],
            )
        ]
        if not incomplete:
            print(f"[density-sweep] {group.name}: complete, skipping", flush=True)
            continue
        if tokenizer_snapshot is None:
            tokenizer_snapshot = resolve_tokenizer_snapshot(
                config.model,
                config.tokenizer_revision,
            )

        group_dir.mkdir(parents=True, exist_ok=True)
        if group.backend == "sparse":
            (group_dir / "telemetry").mkdir(exist_ok=True)
        _atomic_write_json(group_dir / "server-resolved.json", group_manifest)
        _atomic_write_json(group_dir / "versions.json", versions)
        group_status = group_dir / "group-status.json"
        group_started_at = _captured_at()
        _atomic_write_json(
            group_status,
            _status(
                "running",
                str(group_manifest["fingerprint"]),
                started_at_utc=group_started_at,
            ),
        )
        with (group_dir / "server.log").open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== attempt {_captured_at()} ===\n")

        port = free_port()
        serve_cmd, environment = _build_group_runtime(
            config,
            group,
            group_dir,
            port,
        )
        _atomic_write_json(
            group_dir / "server-resolved.json",
            {
                **group_manifest,
                "serve_command": serve_cmd,
                "environment": safe_environment_snapshot(environment),
            },
        )
        print(
            f"[density-sweep] {group.name}: starting one server for "
            f"{len(incomplete)} case(s)",
            flush=True,
        )
        stop_group = False
        group_error = None
        try:
            with session_factory(
                serve_cmd,
                environment,
                port,
                config.server.timeout_seconds,
                server_log=group_dir / "server.log",
            ) as session:
                for concurrency, case_dir, case_manifest in incomplete:
                    _archive_case_artifacts(case_dir)
                    case_dir.mkdir(parents=True, exist_ok=True)
                    fingerprint = str(case_manifest["fingerprint"])
                    warmups = config.workload.warmups
                    case_started_at = _captured_at()
                    command = _benchmark_command(
                        config,
                        port,
                        concurrency,
                        warmups,
                        case_dir,
                        tokenizer_snapshot,
                    )
                    _atomic_write_json(
                        case_dir / "resolved.json",
                        {
                            **case_manifest,
                            "benchmark_command": command,
                            "warmups": warmups,
                        },
                    )
                    _atomic_write_json(
                        case_dir / "status.json",
                        _status(
                            "running",
                            fingerprint,
                            started_at_utc=case_started_at,
                        ),
                    )
                    marker_path = group_dir / "telemetry" / "case-markers.jsonl"
                    if group.backend == "sparse":
                        _append_json_line(
                            marker_path,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "case_fingerprint": fingerprint,
                                "concurrency": concurrency,
                                "phase": "start",
                                "at_utc": case_started_at,
                            },
                        )
                    print(
                        f"[density-sweep] {group.name} concurrency={concurrency} "
                        f"warmups={warmups}",
                        flush=True,
                    )
                    timed_out = False
                    try:
                        result = session.run_benchmark(
                            command,
                            client_log=case_dir / "client.log",
                            timeout=config.server.client_timeout_seconds,
                        )
                        result_payload = (
                            _read_json(case_dir / "result.json")
                            if (case_dir / "result.json").exists()
                            else {}
                        )
                        succeeded = (
                            result.returncode == 0
                            and _result_succeeded(result_payload, concurrency)
                        )
                        _atomic_write_json(
                            case_dir / "status.json",
                            _status(
                                "success" if succeeded else "failed",
                                fingerprint,
                                started_at_utc=case_started_at,
                                returncode=result.returncode,
                                error=(
                                    None
                                    if succeeded
                                    else "benchmark did not produce a complete result"
                                ),
                            ),
                        )
                        print(
                            f"[density-sweep] {group.name} "
                            f"concurrency={concurrency} "
                            f"state={'success' if succeeded else 'failed'} "
                            f"rc={result.returncode}",
                            flush=True,
                        )
                    except Exception as exc:
                        timed_out = isinstance(exc, subprocess.TimeoutExpired)
                        succeeded = False
                        _atomic_write_json(
                            case_dir / "status.json",
                            _status(
                                "failed",
                                fingerprint,
                                started_at_utc=case_started_at,
                                error=f"{type(exc).__name__}: {exc}",
                            ),
                        )
                        print(
                            f"[density-sweep] {group.name} "
                            f"concurrency={concurrency} failed: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    if group.backend == "sparse":
                        _append_json_line(
                            marker_path,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "case_fingerprint": fingerprint,
                                "concurrency": concurrency,
                                "phase": "finish",
                                "state": "success" if succeeded else "failed",
                                "at_utc": _captured_at(),
                            },
                        )
                    write_summary(output_root, config)
                    if timed_out:
                        group_error = (
                            "benchmark client timed out; server session is tainted"
                        )
                        print(
                            f"[density-sweep] {group.name}: {group_error}; "
                            "stopping this group",
                            flush=True,
                        )
                        stop_group = True
                        break
                    if not succeeded and not session.is_healthy():
                        print(
                            f"[density-sweep] {group.name}: server unhealthy; "
                            "stopping this group",
                            flush=True,
                        )
                        stop_group = True
                        break
        except Exception as exc:
            stop_group = True
            group_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[density-sweep] {group.name}: {group_error}",
                flush=True,
            )

        group_complete = all(
            case_complete(
                case_dir,
                str(case_manifest["fingerprint"]),
                concurrency,
            )
            for concurrency, case_dir, case_manifest in cases
        )
        _atomic_write_json(
            group_status,
            _status(
                "success" if group_complete else "failed",
                str(group_manifest["fingerprint"]),
                started_at_utc=group_started_at,
                error=(
                    group_error
                    or (
                        "server became unhealthy or cases remain incomplete"
                        if stop_group and not group_complete
                        else None
                    )
                ),
            ),
        )

    write_summary(output_root, config)
    complete = True
    for group in config.groups():
        group_manifest = _group_manifest(manifest, group)
        for concurrency in config.workload.concurrencies:
            case_manifest = _case_manifest(group_manifest, concurrency)
            if not case_complete(
                output_root
                / group.name
                / f"concurrency-{concurrency}",
                str(case_manifest["fingerprint"]),
                concurrency,
            ):
                complete = False
    return 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    return run_sweep(
        SweepConfig.from_yaml(args.config),
        args.output_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
