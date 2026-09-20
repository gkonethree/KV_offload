"""Shared runtime setup for the qualified Hopper and Blackwell paths."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import sys
import sysconfig
from typing import Literal, Optional, Sequence


HardwareName = Literal["hopper", "blackwell"]

DRIVER_LIBRARY_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/local/cuda/lib64/stubs/libcuda.so"),
    Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    Path("/usr/lib64/libcuda.so.1"),
)

RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "SKYLIGHT_PLATFORM",
    "SKYLIGHT_COMPILE_CACHE_DIR",
    "CUDA_HOME",
    "FLASHINFER_WORKSPACE_BASE",
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPYCACHEPREFIX",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "VLLM_CACHE_ROOT",
)


@dataclass(frozen=True)
class Platform:
    """Resolved GPU architecture used by setup and every vLLM process."""

    name: HardwareName
    capability: tuple[int, int]
    sm: Literal["sm90", "sm100"]
    torch_arch_list: Literal["9.0", "10.0"]


def _device_capability() -> tuple[int, int]:
    """Return device zero's CUDA capability without importing torch at module load."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed; run ./setup.sh before starting Skylight."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Skylight requires one visible H100 (SM90) "
            "or B200 (SM100)."
        )
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


def detect_platform() -> Platform:
    """Resolve the first visible GPU to the supported release architectures."""
    capability = _device_capability()
    if capability == (9, 0):
        return Platform("hopper", capability, "sm90", "9.0")
    if capability == (10, 0):
        return Platform("blackwell", capability, "sm100", "10.0")
    raise RuntimeError(
        f"Unsupported CUDA capability {capability[0]}.{capability[1]}; "
        "Skylight currently supports H100 (SM90) and B200 (SM100)."
    )


def resolve_cuda_home() -> Path:
    """Find the CUDA 13 wheel toolchain installed in the active environment."""
    candidate = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    if not candidate.is_dir():
        raise RuntimeError(
            f"CUDA_HOME {candidate} does not exist. Run ./setup.sh to install "
            "the pinned CUDA 13.0 toolchain."
        )
    return candidate.resolve()


def _replace_broken_symlink(path: Path) -> None:
    if path.is_symlink() and not path.exists():
        path.unlink()


def prepare_cuda_toolchain(cuda_home: Path) -> None:
    """Repair the pip CUDA wheel layout needed by torch CUDA extensions."""
    cuda_home = Path(cuda_home)
    lib_dir = cuda_home / "lib"
    if not lib_dir.is_dir():
        raise RuntimeError(f"CUDA wheel library directory is missing: {lib_dir}")

    lib64 = cuda_home / "lib64"
    _replace_broken_symlink(lib64)
    if not lib64.exists() and not lib64.is_symlink():
        lib64.symlink_to("lib", target_is_directory=True)

    for versioned in sorted(lib_dir.glob("lib*.so.*")):
        stem = versioned.name.split(".so.", 1)[0]
        unversioned = lib_dir / f"{stem}.so"
        _replace_broken_symlink(unversioned)
        if not unversioned.exists() and not unversioned.is_symlink():
            unversioned.symlink_to(versioned.name)

    stubs_dir = lib_dir / "stubs"
    stubs_dir.mkdir(parents=True, exist_ok=True)
    driver_link = stubs_dir / "libcuda.so"
    _replace_broken_symlink(driver_link)
    if not driver_link.exists() and not driver_link.is_symlink():
        driver = next(
            (candidate for candidate in DRIVER_LIBRARY_CANDIDATES if candidate.exists()),
            None,
        )
        if driver is None:
            checked = ", ".join(str(path) for path in DRIVER_LIBRARY_CANDIDATES)
            raise RuntimeError(
                "Could not locate the NVIDIA driver library required for CUDA "
                f"extension linking. Checked: {checked}"
            )
        driver_link.symlink_to(driver.resolve())


def _prepend_path(name: str, paths: Sequence[Path]) -> None:
    existing = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    prefixes = [str(path) for path in paths]
    combined: list[str] = []
    for item in [*prefixes, *existing]:
        if item not in combined:
            combined.append(item)
    os.environ[name] = os.pathsep.join(combined)


def configure_runtime(expected_hardware: str | None = None) -> Platform:
    """Configure platform-specific caches and CUDA discovery for this process."""
    platform = detect_platform()
    expected = None if expected_hardware in (None, "auto") else expected_hardware
    if expected not in (None, "hopper", "blackwell"):
        raise ValueError(
            f"unknown hardware {expected_hardware!r}; expected auto, hopper, or blackwell"
        )
    if expected is not None and platform.name != expected:
        raise RuntimeError(
            f"requested {expected} hardware but detected {platform.name} "
            f"({platform.sm})"
        )

    cache_root = Path(
        os.environ.setdefault(
            "SKYLIGHT_COMPILE_CACHE_DIR",
            f"/var/tmp/skylight-{os.getuid()}",
        )
    ).expanduser()
    default_extensions = cache_root / "extensions" / platform.sm
    default_inductor = cache_root / "torchinductor" / platform.sm
    default_flashinfer = cache_root / "flashinfer" / platform.sm
    default_pycache = cache_root / "pycache" / platform.sm
    default_vllm = cache_root / "vllm" / platform.sm
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(default_extensions))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(default_inductor))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(default_flashinfer))
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(default_pycache))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(default_vllm))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", platform.torch_arch_list)
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["PYTHONPYCACHEPREFIX"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["VLLM_CACHE_ROOT"]).mkdir(parents=True, exist_ok=True)

    cuda_home = resolve_cuda_home()
    prepare_cuda_toolchain(cuda_home)
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"CUDA compiler is missing: {nvcc}")

    os.environ["SKYLIGHT_PLATFORM"] = platform.name
    os.environ["CUDA_HOME"] = str(cuda_home)
    _prepend_path("PATH", (Path(sys.executable).parent, cuda_home / "bin"))
    _prepend_path("LD_LIBRARY_PATH", (cuda_home / "lib", cuda_home / "lib" / "stubs"))
    return platform


def write_runtime_env(
    path: Path,
    expected_hardware: str | None = None,
) -> Platform:
    """Write the safe runtime allowlist as a sourceable POSIX shell file."""
    platform = configure_runtime(expected_hardware)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"export {name}={shlex.quote(os.environ[name])}"
        for name in RUNTIME_ENV_KEYS
    ]
    output.write_text("\n".join(lines) + "\n")
    return platform


def main(argv: Optional[list[str]] = None) -> int:
    """Small setup/qualification CLI; not part of the public Skylight CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hardware",
        choices=("auto", "hopper", "blackwell"),
        default="auto",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--write-env", type=Path)
    output.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)
    expected = None if args.hardware == "auto" else args.hardware

    if args.write_env is not None:
        platform = write_runtime_env(args.write_env, expected)
        print(json.dumps(asdict(platform), sort_keys=True))
        return 0

    platform = configure_runtime(expected)
    print(json.dumps(asdict(platform), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
