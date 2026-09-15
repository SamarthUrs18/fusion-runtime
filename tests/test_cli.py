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
