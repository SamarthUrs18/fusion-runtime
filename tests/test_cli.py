"""`frun` CLI: argument parsing, and staying light enough that --help is instant."""
import json
import subprocess
import sys

from typer.testing import CliRunner

from fusion_runtime.cli.app import app
from fusion_runtime.cli.version import package_version

runner = CliRunner()

# Modules that must never load just to parse arguments or print help.
HEAVY_MODULES = [
    "numpy", "pydantic", "scipy", "torch", "torchaudio", "faster_whisper",
    "ctranslate2", "llama_cpp", "onnxruntime", "kokoro_onnx", "fastapi", "uvicorn",
]


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version_prints_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"fusion-runtime {package_version()}"


def test_unknown_command_fails():
    result = runner.invoke(app, ["no-such-command"])
    assert result.exit_code != 0


def test_importing_cli_loads_no_heavy_modules():
    code = (
        "import sys, json; import fusion_runtime.cli; "
        f"print(json.dumps([m for m in {HEAVY_MODULES!r} if m in sys.modules]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []


def test_module_entry_point_runs():
    out = subprocess.run(
        [sys.executable, "-m", "fusion_runtime.cli", "version"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert out.stdout.startswith("fusion-runtime ")


def test_package_exports_still_importable():
    from fusion_runtime import DEVELOPMENT_CONFIG, PipelineOrchestrator, run_single_turn

    assert DEVELOPMENT_CONFIG.llm.model
    assert callable(run_single_turn) and PipelineOrchestrator


def test_unknown_package_attribute_raises():
    import fusion_runtime
    import pytest

    with pytest.raises(AttributeError):
        fusion_runtime.does_not_exist


# ---- frun models -------------------------------------------------------------

import fusion_runtime.catalog as catalog_pkg
from fusion_runtime.catalog import download as download_mod


def _empty_model_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))


def _record_pulls(monkeypatch, fail=None):
    pulled = []

    def fake_pull(entry, root, force=False, log=print):
        if fail:
            raise download_mod.DownloadError(fail)
        pulled.append(entry.id)

    monkeypatch.setattr(download_mod, "pull", fake_pull)
    monkeypatch.setattr(download_mod, "check_disk_space", lambda needed, root: None)
    return pulled


def test_models_list_shows_missing_and_how_to_fix(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "qwen2.5-7b-q4" in result.output and "missing" in result.output
    assert "(FUSION_MODEL_DIR)" in result.output
    assert "Run: frun models pull --config production" in result.output


def test_models_pull_defaults_to_development_profile(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    pulled = _record_pulls(monkeypatch)
    result = runner.invoke(app, ["models", "pull"])
    assert result.exit_code == 0, result.output
    assert pulled == ["whisper-tiny.en", "qwen2.5-0.5b-q4", "kokoro-v1.0", "silero-vad"]


def test_models_pull_stage_flag_uses_chosen_profile(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    pulled = _record_pulls(monkeypatch)
    result = runner.invoke(app, ["models", "pull", "--llm", "--config", "production"])
    assert result.exit_code == 0, result.output
    assert pulled == ["qwen2.5-7b-q4"]


def test_models_pull_by_id(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    pulled = _record_pulls(monkeypatch)
    result = runner.invoke(app, ["models", "pull", "kokoro-v1.0"])
    assert result.exit_code == 0, result.output
    assert pulled == ["kokoro-v1.0"]


def test_models_pull_skips_installed(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    pulled = _record_pulls(monkeypatch)
    monkeypatch.setattr(catalog_pkg, "is_installed", lambda entry, root: True)
    result = runner.invoke(app, ["models", "pull"])
    assert result.exit_code == 0
    assert pulled == []
    assert "already installed" in result.output


def test_models_pull_unknown_id_fails(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    result = runner.invoke(app, ["models", "pull", "nope"])
    assert result.exit_code == 1
    assert "Unknown model: nope" in result.output


def test_models_pull_reports_download_failure(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    _record_pulls(monkeypatch, fail="whisper-tiny.en: download failed: ConnectionError")
    result = runner.invoke(app, ["models", "pull", "--whisper"])
    assert result.exit_code == 1
    assert "download failed" in result.output and "run the same command again" in result.output


def test_models_pull_refuses_without_disk_space(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)

    def no_space(needed, root):
        raise download_mod.NotEnoughDiskSpace("Need 4.7 GB but only 1.0 GB is free")

    monkeypatch.setattr(download_mod, "check_disk_space", no_space)
    result = runner.invoke(app, ["models", "pull", "--llm", "--config", "production"])
    assert result.exit_code == 1
    assert "only 1.0 GB is free" in result.output


# ---- frun up -----------------------------------------------------------------

import socket

import uvicorn


def _all_models_installed(monkeypatch):
    monkeypatch.setattr(catalog_pkg, "is_installed", lambda entry, root: True)


def _record_uvicorn(monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app_path, **kw: calls.append((app_path, kw)))
    return calls


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_up_starts_server_with_chosen_profile(monkeypatch):
    _all_models_installed(monkeypatch)
    calls = _record_uvicorn(monkeypatch)
    monkeypatch.setenv("FUSION_CONFIG", "unset-before-test")  # restored after the test
    port = _free_port()
    result = runner.invoke(app, ["up", "--port", str(port), "--config", "production"])
    assert result.exit_code == 0, result.output
    assert calls == [("fusion_runtime.server:app",
                      {"host": "127.0.0.1", "port": port, "workers": 1, "log_level": "info",
                       "reload": False, "reload_includes": None, "reload_dirs": None})]
    assert os.environ["FUSION_CONFIG"] == "production"
    assert f"frun talk --url ws://localhost:{port}/v1/voice/ws" in result.output


def test_up_default_port_suggests_plain_talk(monkeypatch):
    _all_models_installed(monkeypatch)
    _record_uvicorn(monkeypatch)
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    monkeypatch.setenv("FUSION_CONFIG", "unset-before-test")
    result = runner.invoke(app, ["up"])
    assert "run `frun talk` in another terminal" in result.output


def test_up_refuses_when_models_missing(tmp_path, monkeypatch):
    _empty_model_env(tmp_path, monkeypatch)
    calls = _record_uvicorn(monkeypatch)
    result = runner.invoke(app, ["up", "--config", "production"])
    assert result.exit_code == 1
    assert "qwen2.5-7b-q4" in result.output
    assert "Run: frun models pull --config production" in result.output
    assert calls == []


def test_up_warns_when_something_else_holds_the_port_over_ipv6(monkeypatch):
    _all_models_installed(monkeypatch)
    _record_uvicorn(monkeypatch)
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    monkeypatch.setattr("fusion_runtime.cli._checks.port_answers_over_ipv6", lambda port: True)
    monkeypatch.setenv("FUSION_CONFIG", "unset-before-test")
    result = runner.invoke(app, ["up"])
    assert result.exit_code == 0
    assert "IPv6" in result.output and "lsof" in result.output


def test_talk_says_when_another_program_answers(monkeypatch):
    import websockets

    from fusion_runtime.cli import _talk_client

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise websockets.exceptions.InvalidMessage("did not receive a valid HTTP response")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(_talk_client.websockets, "connect", Boom)
    monkeypatch.setattr("fusion_runtime.cli.talk._audio_available", lambda: True)
    result = runner.invoke(app, ["talk"])
    assert result.exit_code == 1
    assert "isn't fusion-runtime" in result.output and "lsof" in result.output


def test_talk_defaults_to_ipv4(monkeypatch):
    from fusion_runtime.cli import talk as talk_module

    assert talk_module.DEFAULT_URL.startswith("ws://127.0.0.1:")  # not "localhost": that is IPv6 first on macOS


def test_up_refuses_busy_port(monkeypatch):
    _all_models_installed(monkeypatch)
    calls = _record_uvicorn(monkeypatch)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = runner.invoke(app, ["up", "--port", str(port)])
    assert result.exit_code == 1
    assert f"port {port} is already in use" in result.output
    assert calls == []


def test_up_warns_when_exposed_to_network(monkeypatch):
    _all_models_installed(monkeypatch)
    _record_uvicorn(monkeypatch)
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    monkeypatch.setenv("FUSION_CONFIG", "unset-before-test")
    result = runner.invoke(app, ["up", "--host", "0.0.0.0"])
    assert "no authentication" in result.output


# ---- frun talk ---------------------------------------------------------------

import os

from fusion_runtime.cli import _talk_client


class _FakeClient:
    instances = []
    raise_on_run = None

    def __init__(self, uri, echo_cancellation, verbose=False):
        self.uri, self.echo_cancellation, self.verbose = uri, echo_cancellation, verbose
        _FakeClient.instances.append(self)

    async def run(self):
        if _FakeClient.raise_on_run:
            raise _FakeClient.raise_on_run


def _fake_client(monkeypatch, raise_on_run=None):
    _FakeClient.instances, _FakeClient.raise_on_run = [], raise_on_run
    monkeypatch.setattr(_talk_client, "VoiceChatClient", _FakeClient)
    monkeypatch.setattr("fusion_runtime.cli.talk._audio_available", lambda: True)
    monkeypatch.delenv("FUSION_AEC", raising=False)


def test_talk_passes_url_and_echo_cancellation(monkeypatch):
    _fake_client(monkeypatch)
    result = runner.invoke(app, ["talk", "--url", "ws://box:9000/v1/voice/ws", "--no-aec"])
    assert result.exit_code == 0, result.output
    client, = _FakeClient.instances
    assert (client.uri, client.echo_cancellation) == ("ws://box:9000/v1/voice/ws", False)


def test_talk_echo_cancellation_on_by_default_and_env_can_disable(monkeypatch):
    _fake_client(monkeypatch)
    runner.invoke(app, ["talk"])
    assert _FakeClient.instances[-1].echo_cancellation is True
    monkeypatch.setenv("FUSION_AEC", "0")
    runner.invoke(app, ["talk"])
    assert _FakeClient.instances[-1].echo_cancellation is False


def test_talk_explains_when_server_is_not_running(monkeypatch):
    _fake_client(monkeypatch, raise_on_run=ConnectionRefusedError(61, "Connection refused"))
    result = runner.invoke(app, ["talk"])
    assert result.exit_code == 1
    assert "Is the server running?" in result.output and "frun up" in result.output


def test_talk_without_audio_extra_says_how_to_install(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_sounddevice(name, *args, **kwargs):
        if name == "sounddevice":
            raise ImportError("No module named 'sounddevice'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sounddevice)
    result = runner.invoke(app, ["talk"])
    assert result.exit_code == 1
    assert "pip install 'fusion-runtime[talk]'" in result.output


def test_version_comes_only_from_pyproject():
    import tomllib
    from pathlib import Path

    import fusion_runtime
    from fusion_runtime.server import app as server_app

    declared = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())["project"]["version"]
    assert fusion_runtime.__version__ == declared
    assert package_version() == declared
    assert server_app.version == declared


def test_up_passes_log_settings_to_the_server(monkeypatch):
    _all_models_installed(monkeypatch)
    calls = _record_uvicorn(monkeypatch)
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    for var in ("FUSION_CONFIG", "FUSION_LOG_FORMAT", "FUSION_LOG_LEVEL", "FUSION_LOG_CONTENT"):
        monkeypatch.setenv(var, "unset-before-test")
    result = runner.invoke(app, ["up", "--log-format", "json", "--log-level", "debug", "--log-content"])
    assert result.exit_code == 0, result.output
    assert (os.environ["FUSION_LOG_FORMAT"], os.environ["FUSION_LOG_LEVEL"], os.environ["FUSION_LOG_CONTENT"]) == ("json", "debug", "1")
    assert "writes what users say" in result.output
    assert calls[0][1]["log_level"] == "warning"  # JSON mode keeps uvicorn's own text logs quiet


def test_up_defaults_keep_content_out_of_logs(monkeypatch):
    _all_models_installed(monkeypatch)
    _record_uvicorn(monkeypatch)
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    for var in ("FUSION_CONFIG", "FUSION_LOG_FORMAT", "FUSION_LOG_LEVEL", "FUSION_LOG_CONTENT"):
        monkeypatch.setenv(var, "unset-before-test")
    runner.invoke(app, ["up"])
    assert (os.environ["FUSION_LOG_FORMAT"], os.environ["FUSION_LOG_CONTENT"]) == ("pretty", "0")


def test_talk_passes_verbose(monkeypatch):
    _fake_client(monkeypatch)
    runner.invoke(app, ["talk", "--verbose"])
    assert _FakeClient.instances[-1].verbose is True


def test_talk_formats_turn_summary_timeline_and_errors():
    summary = {"outcome": "completed", "ttfa_ms": 436.2, "response_ms": 310.0, "end_of_turn_wait_ms": 204.0,
               "stt_transcribe_ms": 208.0, "llm_first_token_ms": 115.0, "llm_tokens_per_second": 58.3,
               "tts_first_chunk_ms": 220.0, "playback_delay_ms": 35.0}
    line = _talk_client.format_turn_summary(summary)
    assert line.startswith("📊 completed: TTFA 436ms · response 310ms")
    assert "llm 115ms 58 tok/s" in line and "playback +35ms" in line

    interrupted = _talk_client.format_turn_summary({"outcome": "interrupted", "response_ms": 500, "interrupted": True,
                                                   "interruption_stop_ms": 12})
    assert "interrupted (stopped in 12ms)" in interrupted and "TTFA" not in interrupted

    timeline = _talk_client.format_timeline([{"event": "speech_start", "t_ms": 0.0}, {"event": "audio_first_sent", "t_ms": 1436.4}])
    assert "1436 ms  audio_first_sent" in timeline

    error = _talk_client.format_error({"type": "error", "code": "auth_failed", "stage": "llm",
                                       "message": "401", "fix": "Check the API key", "retryable": False})
    assert error == "❌ [llm] auth_failed: 401\n   → Check the API key"
