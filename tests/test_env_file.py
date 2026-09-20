""".env files: settings and tokens without exporting them by hand, and never logged."""
from pathlib import Path

from fusion_runtime.env import load_env_file


def test_the_usual_shapes_are_understood(tmp_path):
    """Parsing is python-dotenv's; this pins the shapes agents and deployments actually write."""
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "HF_TOKEN=hf_secret\n"
        "export FUSION_TURN_WAIT_MS=800\n"
        'QUOTED="spaces and #hashes"\n'
        "SINGLE='also quoted'\n"
        "EMPTY=\n"
        "not a setting\n"
    )
    environ = {}
    load_env_file([tmp_path], environ)
    assert environ["HF_TOKEN"] == "hf_secret"
    assert environ["FUSION_TURN_WAIT_MS"] == "800"
    assert environ["QUOTED"] == "spaces and #hashes"
    assert environ["SINGLE"] == "also quoted"
    assert environ["EMPTY"] == ""
    assert "not a setting" not in environ


def test_the_real_environment_wins(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=from_file\nFUSION_TURN_WAIT_MS=800\n")
    environ = {"HF_TOKEN": "from_shell"}
    loaded = load_env_file([tmp_path], environ)
    assert environ["HF_TOKEN"] == "from_shell", "a value already set must not be overwritten"
    assert environ["FUSION_TURN_WAIT_MS"] == "800"
    assert loaded == ["FUSION_TURN_WAIT_MS"]  # names only: values are secrets


def test_several_directories_and_missing_files(tmp_path):
    project, agent_dir = tmp_path / "project", tmp_path / "agents"
    project.mkdir(), agent_dir.mkdir()
    (project / ".env").write_text("A=1\n")
    (agent_dir / ".env").write_text("A=2\nB=3\n")
    environ = {}
    assert load_env_file([project, agent_dir, tmp_path / "gone"], environ) == ["A", "B"]
    assert environ == {"A": "1", "B": "3"}, "the first file that sets a name wins"


def test_no_env_file_is_fine(tmp_path):
    assert load_env_file([tmp_path], {}) == []


def test_frun_reads_it(tmp_path, monkeypatch):
    from fusion_runtime.cli.app import app
    from typer.testing import CliRunner

    (tmp_path / ".env").write_text("FUSION_TEST_MARKER=on\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FUSION_TEST_MARKER", raising=False)
    CliRunner().invoke(app, ["version"])
    import os

    assert os.environ["FUSION_TEST_MARKER"] == "on"


def test_the_example_file_only_lists_variables_that_exist():
    """A .env.example naming settings nothing reads is worse than none.

    It is a starting file, not a reference: the rarely-changed knobs (limits,
    token lifetime, a keys file) live in the README, so this checks that
    everything listed is real and that the essentials are present — not that
    every variable appears.
    """
    import re

    text = Path(__file__).resolve().parent.parent.joinpath(".env.example").read_text()
    named = {m.group(1) for m in re.finditer(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, re.M)}
    known = {
        "HF_TOKEN", "OPENAI_API_KEY",  # secrets read by the runtimes and the downloader
        "FUSION_AGENT", "FUSION_CONFIG", "FUSION_MODEL_DIR", "FUSION_AUTO_DOWNLOAD",
        "FUSION_TURN_WAIT_MS", "FUSION_INTERRUPT_AFTER_MS", "FUSION_TURN_DETECTOR",
        "FUSION_LLM_URL", "FUSION_LLM_MODEL", "FUSION_LLM_API_KEY_ENV",
        "FUSION_LOG_FORMAT", "FUSION_LOG_LEVEL", "FUSION_LOG_CONTENT", "FUSION_AEC",
        "FUSION_ACCEPTED_KEYS", "FUSION_ACCEPTED_KEYS_FILE", "FUSION_SESSION_TOKEN_TTL_S",
        "FUSION_TRUSTED_PROXY", "FUSION_API_KEY", "FUSION_ALLOWED_ORIGINS",
        "FUSION_MAX_SESSIONS", "FUSION_MAX_SESSIONS_PER_KEY", "FUSION_MAX_MESSAGE_BYTES",
        "FUSION_MAX_TURN_AUDIO_S", "FUSION_MAX_SESSION_S", "FUSION_IDLE_TIMEOUT_S",
        "FUSION_CONNECTIONS_PER_MINUTE", "FUSION_TOKENS_PER_MINUTE",
    }
    assert named <= known, f"names nothing reads: {named - known}"
    essentials = {"HF_TOKEN", "FUSION_ACCEPTED_KEYS", "FUSION_API_KEY", "FUSION_AGENT",
                  "FUSION_TURN_WAIT_MS", "FUSION_LLM_URL"}
    assert essentials <= named, f"missing from the example: {essentials - named}"
