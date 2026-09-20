"""Tests for skylight.cli."""
from __future__ import annotations

import os
import sys

import pytest

from skylight.cli import _build_vllm_argv, main


@pytest.fixture(autouse=True)
def _stub_runtime_configuration(monkeypatch):
    calls = []

    def fake_configure_runtime():
        calls.append("configured")

    monkeypatch.setattr(
        "skylight.cli.configure_runtime",
        fake_configure_runtime,
        raising=False,
    )
    return calls


# ----------------------------- _build_vllm_argv ------------------------------


def test_build_argv_basic_serve():
    """``serve`` with no extra args injects --attention-backend CUSTOM."""
    cmd = _build_vllm_argv(["serve"])
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    assert cmd[3:] == ["--attention-backend", "CUSTOM"]


def test_build_argv_forwards_extra_args():
    """Extra args after `serve` are forwarded verbatim (after the injected flag)."""
    cmd = _build_vllm_argv(["serve", "--model", "Qwen/Qwen3-0.6B", "--port", "9999"])
    assert "--attention-backend" in cmd
    assert "CUSTOM" in cmd
    # Confirm user args are present.
    assert "--model" in cmd
    assert "Qwen/Qwen3-0.6B" in cmd
    assert "--port" in cmd
    assert "9999" in cmd


def test_build_argv_respects_user_backend_override():
    """If the user already passed --attention-backend, do NOT inject CUSTOM."""
    cmd = _build_vllm_argv(["serve", "--attention-backend", "FLASHINFER", "--model", "x"])
    # CUSTOM should NOT have been injected.
    assert cmd.count("--attention-backend") == 1
    # FLASHINFER preserved.
    backend_idx = cmd.index("--attention-backend")
    assert cmd[backend_idx + 1] == "FLASHINFER"


def test_build_argv_respects_equals_form_override():
    """Accept --attention-backend=X (equals form) as a user override too."""
    cmd = _build_vllm_argv(["serve", "--attention-backend=TRITON_ATTN", "--model", "x"])
    # No bare --attention-backend (we'd have injected one if we missed the = form).
    assert "--attention-backend" not in cmd
    assert "--attention-backend=TRITON_ATTN" in cmd


def test_build_argv_rejects_missing_subcommand():
    """Empty argv must raise (caller prints usage and exits 2)."""
    with pytest.raises(ValueError, match="usage: skylight serve"):
        _build_vllm_argv([])


def test_build_argv_rejects_unknown_subcommand():
    """Anything other than `serve` is unsupported today."""
    with pytest.raises(ValueError, match="usage: skylight serve"):
        _build_vllm_argv(["bench", "--model", "x"])


# ----------------------------- main() ----------------------------------------


def test_main_exits_2_on_no_args(monkeypatch, capsys):
    """``skylight`` with no args prints usage to stderr and exits 2."""
    monkeypatch.setattr(sys, "argv", ["skylight"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "usage: skylight serve" in err


def test_main_exits_2_on_bad_subcommand(monkeypatch, capsys):
    """``skylight bench`` exits 2 with usage."""
    monkeypatch.setattr(sys, "argv", ["skylight", "bench"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    assert "usage: skylight serve" in capsys.readouterr().err


def test_python_m_invocation_reaches_main():
    """``python -m skylight.cli`` (no args) must reach main() and exit 2.

    Regression test: without ``if __name__ == "__main__": main()`` in
    ``cli.py``, ``python -m`` loaded the module but never invoked main(),
    producing rc=0 with no output. That broke the e2e test driver
    (subprocess.Popen([sys.executable, "-m", "skylight.cli", "serve", ...]))
    which expected the server to start.
    """
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "skylight.cli"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2, (
        f"expected exit 2 (usage), got rc={result.returncode}; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "usage: skylight serve" in result.stderr, (
        f"missing usage message in stderr: {result.stderr!r}"
    )


def test_main_sets_prometheus_multiproc_dir_when_unset(monkeypatch, tmp_path):
    """If PROMETHEUS_MULTIPROC_DIR isn't set, main() creates one before exec."""
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    captured: dict = {}

    def fake_execvp(file, args):
        captured["env"] = dict(os.environ)

    monkeypatch.setattr("os.execvp", fake_execvp)
    monkeypatch.setattr(sys, "argv", ["skylight", "serve", "--model", "X"])

    main()

    assert "PROMETHEUS_MULTIPROC_DIR" in captured["env"]
    mp_dir = captured["env"]["PROMETHEUS_MULTIPROC_DIR"]
    assert os.path.isdir(mp_dir), f"multiproc dir not created on disk: {mp_dir!r}"
    assert "skylight-prom-" in mp_dir, f"unexpected dir naming: {mp_dir!r}"


def test_main_preserves_operator_prometheus_multiproc_dir(monkeypatch, tmp_path):
    """If PROMETHEUS_MULTIPROC_DIR is operator-set, main() does NOT override it."""
    operator_dir = str(tmp_path / "operator-prom")
    os.makedirs(operator_dir, exist_ok=True)
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", operator_dir)
    captured: dict = {}

    def fake_execvp(file, args):
        captured["env"] = dict(os.environ)

    monkeypatch.setattr("os.execvp", fake_execvp)
    monkeypatch.setattr(sys, "argv", ["skylight", "serve", "--model", "X"])

    main()

    assert captured["env"]["PROMETHEUS_MULTIPROC_DIR"] == operator_dir


def test_main_calls_execvp_with_built_argv(monkeypatch):
    """On valid input, main() calls os.execvp with the build_argv output."""
    captured: dict = {}

    def fake_execvp(file: str, args: list[str]) -> None:
        captured["file"] = file
        captured["args"] = args
        # Don't actually exec — just record. main() doesn't return after a real
        # execvp, but the test exits cleanly because we returned normally.

    monkeypatch.setattr("os.execvp", fake_execvp)
    monkeypatch.setattr(sys, "argv", ["skylight", "serve", "--model", "x"])

    main()

    assert captured["file"] == sys.executable
    assert captured["args"][:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert "--attention-backend" in captured["args"]
    assert "CUSTOM" in captured["args"]
    assert "--model" in captured["args"]
    assert "x" in captured["args"]


def test_main_configures_runtime_before_exec(
    monkeypatch,
    _stub_runtime_configuration,
):
    events = []

    def fake_configure_runtime():
        events.append("runtime")

    def fake_execvp(file, args):
        events.append("exec")

    monkeypatch.setattr("skylight.cli.configure_runtime", fake_configure_runtime)
    monkeypatch.setattr("os.execvp", fake_execvp)
    monkeypatch.setattr(sys, "argv", ["skylight", "serve", "--model", "x"])

    main()

    assert events == ["runtime", "exec"]
