"""Tests for the shared Hopper/Blackwell runtime contract."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from skylight import runtime


def _fake_cuda_home(tmp_path: Path) -> Path:
    cuda_home = tmp_path / "cu13"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "lib").mkdir()
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\n")
    nvcc.chmod(0o755)
    return cuda_home


def test_detect_platform_maps_sm90_to_hopper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_device_capability", lambda: (9, 0))
    assert runtime.detect_platform() == runtime.Platform(
        name="hopper",
        capability=(9, 0),
        sm="sm90",
        torch_arch_list="9.0",
    )


def test_detect_platform_maps_sm100_to_blackwell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_device_capability", lambda: (10, 0))
    assert runtime.detect_platform() == runtime.Platform(
        name="blackwell",
        capability=(10, 0),
        sm="sm100",
        torch_arch_list="10.0",
    )


def test_detect_platform_rejects_unsupported_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_device_capability", lambda: (8, 0))
    with pytest.raises(RuntimeError, match=r"H100.*B200"):
        runtime.detect_platform()


def test_configure_runtime_rejects_requested_hardware_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home = _fake_cuda_home(tmp_path)
    monkeypatch.setattr(runtime, "_device_capability", lambda: (9, 0))
    monkeypatch.setattr(runtime, "resolve_cuda_home", lambda: cuda_home)
    monkeypatch.setattr(runtime, "prepare_cuda_toolchain", lambda _: None)
    with pytest.raises(RuntimeError, match=r"requested blackwell.*detected hopper"):
        runtime.configure_runtime("blackwell")


def test_resolve_cuda_home_ignores_an_inherited_host_toolkit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_cuda = tmp_path / "host-cuda"
    host_cuda.mkdir()
    purelib = tmp_path / "site-packages"
    wheel_cuda = purelib / "nvidia" / "cu13"
    wheel_cuda.mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(host_cuda))
    monkeypatch.setattr(
        runtime.sysconfig,
        "get_paths",
        lambda: {"purelib": str(purelib)},
    )

    assert runtime.resolve_cuda_home() == wheel_cuda.resolve()


def test_configure_runtime_localizes_only_compile_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home = _fake_cuda_home(tmp_path)
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(runtime, "_device_capability", lambda: (9, 0))
    monkeypatch.setattr(runtime, "resolve_cuda_home", lambda: cuda_home)
    monkeypatch.setattr(runtime, "prepare_cuda_toolchain", lambda _: None)
    monkeypatch.setenv("SKYLIGHT_COMPILE_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("HF_HOME", "/operator/model-cache")
    monkeypatch.setenv("CUDA_HOME", "/operator/cuda")
    for name in (
        "FLASHINFER_WORKSPACE_BASE",
        "PYTHONPYCACHEPREFIX",
        "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_CUDA_ARCH_LIST",
        "VLLM_CACHE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    platform = runtime.configure_runtime()

    assert platform.sm == "sm90"
    assert os.environ["HF_HOME"] == "/operator/model-cache"
    assert os.environ["SKYLIGHT_PLATFORM"] == "hopper"
    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(
        cache_root / "extensions" / "sm90"
    )
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(
        cache_root / "torchinductor" / "sm90"
    )
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(
        cache_root / "flashinfer" / "sm90"
    )
    assert os.environ["PYTHONPYCACHEPREFIX"] == str(
        cache_root / "pycache" / "sm90"
    )
    assert os.environ["VLLM_CACHE_ROOT"] == str(
        cache_root / "vllm" / "sm90"
    )
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0"
    assert os.environ["CUDA_HOME"] == str(cuda_home)


def test_configure_runtime_preserves_operator_cache_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home = _fake_cuda_home(tmp_path)
    monkeypatch.setattr(runtime, "_device_capability", lambda: (10, 0))
    monkeypatch.setattr(runtime, "resolve_cuda_home", lambda: cuda_home)
    monkeypatch.setattr(runtime, "prepare_cuda_toolchain", lambda _: None)
    operator_extensions = tmp_path / "operator" / "extensions"
    operator_inductor = tmp_path / "operator" / "inductor"
    operator_flashinfer = tmp_path / "operator" / "flashinfer"
    operator_pycache = tmp_path / "operator" / "pycache"
    operator_vllm = tmp_path / "operator" / "vllm"
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(operator_extensions))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(operator_inductor))
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(operator_flashinfer))
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(operator_pycache))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(operator_vllm))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "operator-arch")

    runtime.configure_runtime()

    assert os.environ["TORCH_EXTENSIONS_DIR"] == str(operator_extensions)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(operator_inductor)
    assert os.environ["FLASHINFER_WORKSPACE_BASE"] == str(operator_flashinfer)
    assert os.environ["PYTHONPYCACHEPREFIX"] == str(operator_pycache)
    assert os.environ["VLLM_CACHE_ROOT"] == str(operator_vllm)
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "operator-arch"


def test_prepare_cuda_toolchain_creates_required_links_idempotently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home = _fake_cuda_home(tmp_path)
    versioned = cuda_home / "lib" / "libcudart.so.13"
    versioned.write_text("runtime")
    driver = tmp_path / "driver" / "libcuda.so.1"
    driver.parent.mkdir()
    driver.write_text("driver")
    monkeypatch.setattr(runtime, "DRIVER_LIBRARY_CANDIDATES", (driver,))

    runtime.prepare_cuda_toolchain(cuda_home)
    runtime.prepare_cuda_toolchain(cuda_home)

    assert (cuda_home / "lib64").is_symlink()
    assert os.readlink(cuda_home / "lib64") == "lib"
    assert (cuda_home / "lib" / "libcudart.so").resolve() == versioned
    assert (cuda_home / "lib" / "stubs" / "libcuda.so").resolve() == driver


def test_prepare_cuda_toolchain_reports_missing_driver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cuda_home = _fake_cuda_home(tmp_path)
    missing = tmp_path / "missing" / "libcuda.so"
    monkeypatch.setattr(runtime, "DRIVER_LIBRARY_CANDIDATES", (missing,))

    with pytest.raises(RuntimeError, match=str(missing)):
        runtime.prepare_cuda_toolchain(cuda_home)


def test_write_runtime_env_serializes_only_allowlisted_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    platform = runtime.Platform("hopper", (9, 0), "sm90", "9.0")

    def fake_configure(expected_hardware: str | None = None) -> runtime.Platform:
        assert expected_hardware == "hopper"
        for name in runtime.RUNTIME_ENV_KEYS:
            monkeypatch.setenv(name, f"value for {name}")
        monkeypatch.setenv("GITHUB_TOKEN", "do-not-write")
        monkeypatch.setenv("HF_TOKEN", "do-not-write")
        return platform

    monkeypatch.setattr(runtime, "configure_runtime", fake_configure)
    output = tmp_path / "runtime.env"

    assert runtime.write_runtime_env(output, "hopper") == platform

    text = output.read_text()
    assert "GITHUB_TOKEN" not in text
    assert "HF_TOKEN" not in text
    assert "do-not-write" not in text
    for name in runtime.RUNTIME_ENV_KEYS:
        assert f"{name}=" in text


def test_print_json_cli_reports_resolved_platform(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runtime,
        "configure_runtime",
        lambda expected_hardware=None: runtime.Platform(
            "blackwell", (10, 0), "sm100", "10.0"
        ),
    )

    assert runtime.main(["--hardware", "blackwell", "--print-json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "capability": [10, 0],
        "name": "blackwell",
        "sm": "sm100",
        "torch_arch_list": "10.0",
    }
