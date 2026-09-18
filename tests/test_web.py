"""The browser client: the console loads, the script is embeddable, and both ship in the package."""
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fusion_runtime.server as server
from fusion_runtime import web

ROOT = Path(__file__).resolve().parents[1]


class FakeOrchestrator:
    """Enough of an orchestrator to start the server without loading models."""

    def __init__(self, config):
        self.config = config
        self.ready = True

    async def initialize(self):
        pass

    async def shutdown(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FUSION_LOG_FORMAT", "off")
    monkeypatch.setattr(server, "PipelineOrchestrator", FakeOrchestrator)
    with TestClient(server.app) as client:
        yield client


def test_console_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # the console runs on the same script a customer's site embeds
    assert web.CLIENT_ROUTE in response.text


def test_client_script_can_be_embedded_from_another_site(client):
    response = client.get(web.CLIENT_ROUTE)
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.headers["access-control-allow-origin"] == "*"
    assert "FusionRuntime" in response.text


def test_the_script_holds_no_secrets():
    """A page is public. Authentication is a short-lived token, never a key."""
    source = web.client_js()
    assert "api_key" not in source.lower()
    assert "token=" in source  # how a page authenticates instead


def test_config_message_tells_the_client_both_sample_rates():
    """The browser shouldn't have to assume what rate replies arrive in."""
    source = ROOT.joinpath("fusion_runtime/server.py").read_text()
    assert '"output_sample_rate": orchestrator.config.tts.sample_rate' in source


def test_web_files_are_packaged():
    """Without this, `frun up` works from a checkout and 404s for everyone who pip installs."""
    patterns = tomllib.loads(ROOT.joinpath("pyproject.toml").read_text())
    packaged = patterns["tool"]["setuptools"]["package-data"]["fusion_runtime"]
    assert "web/*.html" in packaged and "web/*.js" in packaged
    assert web.CONSOLE_FILE.is_file() and web.CLIENT_FILE.is_file()


def test_missing_files_say_it_is_a_packaging_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(web, "CONSOLE_FILE", tmp_path / "gone.html")
    with pytest.raises(web.WebAssetMissing, match="Reinstall"):
        web.console_html()


# ---- the audio thread, run outside a browser (needs node; skipped where it isn't) ----

def _capture(hardware_rate: int, target_rate: int = 16000) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node isn't installed, so the browser client's audio code can't be run here")
    output = subprocess.run(
        [node, str(ROOT / "tests/js/capture_check.js"), str(web.CLIENT_FILE), str(hardware_rate), str(target_rate)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    return json.loads(output.stdout)


@pytest.mark.parametrize("hardware_rate", [48000, 44100, 16000, 22050])
def test_microphone_is_resampled_to_the_rate_the_server_asked_for(hardware_rate):
    """A second of audio has to arrive as a second of 16 kHz audio, whatever the sound card runs at.

    Nothing crashes when this is wrong — Whisper is simply fed audio at the
    wrong speed and transcribes nonsense.
    """
    result = _capture(hardware_rate)
    assert result["samples"] == 16000
    assert result["chunk_samples"] == 640  # 40 ms per message
    # a 0.5 sine is 0.354 RMS: resampling that averages too widely would flatten it
    assert 0.33 < result["level"] < 0.37
