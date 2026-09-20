"""Release-stack pinning contracts."""
from __future__ import annotations

from pathlib import Path
import tomllib


EXPECTED_DEPENDENCIES = {
    "torch==2.11.0+cu130",
    "vllm==0.21.0",
    "flashinfer-python==0.6.8.post1",
    "nvidia-cuda-nvcc==13.0.88",
    "nvidia-cuda-cccl==13.0.85",
    "nvidia-cuda-crt==13.0.88",
    "nvidia-cuda-runtime==13.0.96",
    "nvidia-cuda-cupti==13.0.85",
    "nvidia-cuda-nvrtc==13.0.88",
    "nvidia-nvjitlink==13.0.88",
    "nvidia-nvvm==13.0.88",
    "nvidia-nvtx==13.0.85",
}


def test_release_dependencies_are_explicitly_pinned() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = set(data["project"]["dependencies"])
    assert EXPECTED_DEPENDENCIES <= dependencies


def test_lock_contains_only_the_qualified_cuda_build_packages() -> None:
    data = tomllib.loads(Path("uv.lock").read_text())
    versions = {package["name"]: package["version"] for package in data["package"]}
    assert versions["torch"] == "2.11.0+cu130"
    assert versions["vllm"] == "0.21.0"
    assert versions["flashinfer-python"] == "0.6.8.post1"
    assert versions["nvidia-cuda-nvcc"] == "13.0.88"
    assert versions["nvidia-cuda-cccl"] == "13.0.85"
    assert versions["nvidia-cuda-crt"] == "13.0.88"
    assert versions["nvidia-nvvm"] == "13.0.88"
