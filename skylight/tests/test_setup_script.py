"""Static and help-path contracts for the self-contained setup script."""
from __future__ import annotations

from pathlib import Path
import subprocess


SETUP = Path("setup.sh")


def _run_setup(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SETUP), *args],
        text=True,
        capture_output=True,
    )


def test_setup_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(SETUP)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_setup_help_is_gpu_and_network_free() -> None:
    result = _run_setup("--help")
    assert result.returncode == 0
    assert "auto|hopper|blackwell" in result.stdout
    assert "SKYLIGHT_COMPILE_CACHE_DIR" in result.stdout
    assert "SKYLIGHT_ENV_DIR" in result.stdout
    assert "SKYLIGHT_KERNELS_URL" in result.stdout


def test_setup_rejects_unknown_arguments_before_host_checks() -> None:
    result = _run_setup("--unknown")
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_setup_pins_uv_and_kernels() -> None:
    text = SETUP.read_text()
    assert "0.11.28" in text
    assert "b88bd0f4758ebb90cd388786af3e4f4fbd9c3cde" in text


def test_setup_never_reads_credential_env_or_token_variables() -> None:
    text = SETUP.read_text()
    assert "source ~/.env" not in text
    assert "source /root/.env" not in text
    assert "GITHUB_TOKEN" not in text
    assert "AUTHORIZATION" not in text


def test_setup_configures_cuda_before_single_kernel_build() -> None:
    text = SETUP.read_text()
    bootstrap = text.index("--no-install-package skylight-kernels")
    runtime = text.index("--write-env")
    exact_sync = text.index("--no-build-isolation-package skylight-kernels")
    assert bootstrap < runtime < exact_sync
    assert "setup.py build_ext --inplace" not in text
    assert '"$UV" pip install' not in text


def test_setup_bootstraps_without_kernels_then_exactly_syncs_the_workspace() -> None:
    text = SETUP.read_text()
    assert text.count("--no-install-package skylight-kernels") == 1
    assert text.count("--no-build-isolation-package skylight-kernels") == 1
    assert "--inexact" not in text
    assert text.count('"$UV" sync') == 2


def test_setup_can_keep_the_project_venv_on_node_local_storage() -> None:
    text = SETUP.read_text()
    assert 'VENV="${SKYLIGHT_ENV_DIR:-$ROOT/.venv}"' in text
    assert 'export UV_PROJECT_ENVIRONMENT="$VENV"' in text
    assert 'ln -s "$VENV" "$ROOT/.venv"' in text
