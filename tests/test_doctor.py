"""`frun doctor`: checks report problems with a fix, and never crash."""
import builtins
import sys
import types

import pytest
from typer.testing import CliRunner

from fusion_runtime.cli import _checks
from fusion_runtime.cli._checks import FAIL, INFO, OK, WARN, CheckResult
from fusion_runtime.cli.app import app

runner = CliRunner()


def test_versions_match_ignores_build_tags():
    assert _checks.versions_match("2.9.1+cpu", "2.9.1")
    assert not _checks.versions_match("2.9.1", "2.11.0")


def test_mismatched_torch_pair_fails_with_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(__version__="2.9.1"))
    monkeypatch.setitem(sys.modules, "torchaudio", types.SimpleNamespace(__version__="2.11.0"))
    result = _checks.check_torch_pair()
    assert result.status == FAIL
    assert "don't match" in result.message
    assert "torchaudio==2.9.1" in result.fix


def test_a_crashing_check_is_reported_not_raised():
    def check_broken():
        raise RuntimeError("boom")

    (title, results), = _checks.run_checks([("System", [check_broken])])
    assert results[0].status == FAIL
    assert "broken check crashed: RuntimeError: boom" in results[0].message


def test_low_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path))

    class Usage:
        free = 3e9

    monkeypatch.setattr(_checks.shutil, "disk_usage", lambda _: Usage)
    result = _checks.check_disk()
    assert result.status == WARN and "production model" in result.fix


def test_a_port_is_in_use_when_something_answers(monkeypatch):
    """Asked by connecting, not by binding: a program on the wildcard address still lets
    us bind 127.0.0.1 on macOS, so a bind test would call a taken port free."""
    import socket

    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert _checks.port_in_use("127.0.0.1", port)
    assert not _checks.port_in_use("127.0.0.1", port)


def test_ipv6_squatter_is_noticed(monkeypatch):
    import socket

    try:
        server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:  # pragma: no cover - machine without IPv6
        pytest.skip("no IPv6 on this machine")
    with server:
        server.bind(("::1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert _checks.port_answers_over_ipv6(port)
        assert not _checks.port_in_use("127.0.0.1", port), "IPv4 is still free; only clients using 'localhost' hit it"


def test_busy_port_is_a_warning(monkeypatch):
    monkeypatch.setattr(_checks, "port_in_use", lambda host, port: True)
    assert _checks.check_port().status == WARN


def _no_models(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))


def test_missing_development_models_fail(monkeypatch, tmp_path):
    _no_models(monkeypatch, tmp_path)
    monkeypatch.setattr(_checks, "_cuda_device_count", lambda: 0)
    results = _checks.check_models()
    dev = next(r for r in results if r.message.startswith("development profile"))
    assert dev.status == FAIL and dev.fix == "frun models pull"


def test_missing_production_models_only_matter_with_a_gpu(monkeypatch, tmp_path):
    _no_models(monkeypatch, tmp_path)
    monkeypatch.setattr(_checks, "_cuda_device_count", lambda: 0)
    prod = next(r for r in _checks.check_models() if r.message.startswith("production profile"))
    assert prod.status == INFO
    monkeypatch.setattr(_checks, "_cuda_device_count", lambda: 1)
    prod = next(r for r in _checks.check_models() if r.message.startswith("production profile"))
    assert prod.status == WARN and prod.fix == "frun models pull --config production"


def test_silero_not_downloaded(monkeypatch, tmp_path):
    _no_models(monkeypatch, tmp_path)
    result = _checks.check_silero_loads()
    assert result.status == FAIL and result.fix == "frun models pull --vad"


def test_missing_audio_extra_is_a_warning(monkeypatch):
    real_import = builtins.__import__

    def no_sounddevice(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("No module named 'sounddevice'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sounddevice)
    result, = _checks.check_audio()
    assert result.status == WARN and "fusion-runtime[talk]" in result.fix


def _fake_run(sections):
    return lambda: sections


def test_doctor_exits_1_and_shows_fixes_when_something_fails(monkeypatch):
    monkeypatch.setattr(_checks, "run_checks", _fake_run([
        ("Models", [CheckResult(OK, "all fine"), CheckResult(FAIL, "missing qwen", "frun models pull")]),
        ("Audio", [CheckResult(WARN, "no mic", "plug one in")]),
    ]))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "✗ missing qwen" in result.output and "→ frun models pull" in result.output
    assert "1 problem, 1 warning" in result.output


def test_doctor_exits_0_with_only_warnings(monkeypatch):
    monkeypatch.setattr(_checks, "run_checks", _fake_run([("Audio", [CheckResult(WARN, "no mic", "plug one in")])]))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "No problems, 1 warning." in result.output


def test_ok_results_hide_fix_text(monkeypatch):
    monkeypatch.setattr(_checks, "run_checks", _fake_run([("System", [CheckResult(OK, "Python 3.11", "unused")])]))
    result = runner.invoke(app, ["doctor"])
    assert "unused" not in result.output and "All good." in result.output
