"""An agent file describes the whole agent: prompt, models, settings — and the server uses it."""
import pytest
from fusion_runtime.agent import (
    DEFAULT_PROMPT,
    LLM,
    TTS,
    VAD,
    Agent,
    AgentError,
    Stage,
    Turns,
    load_agent,
)
from fusion_runtime.cli.app import app
from typer.testing import CliRunner

cli = CliRunner()


def write_agent(tmp_path, body: str, name: str = "agent.py"):
    path = tmp_path / name
    path.write_text(body)
    return path


AGENT_FILE = '''
from fusion_runtime import Agent, LLM, TTS, Turns

agent = Agent(
    name="orders",
    prompt="You take orders for ShopKart.",
    llm=LLM("qwen2.5-0.5b-q4", max_tokens=120),
    tts=TTS("kokoro-v1.0", voice="af_heart"),
    turns=Turns(wait_ms=800),
)
'''


# ---- the file --------------------------------------------------------------------------------

def test_loads_an_agent_and_its_settings(tmp_path):
    agent = load_agent(write_agent(tmp_path, AGENT_FILE))
    assert agent.name == "orders" and agent.prompt.startswith("You take orders")
    config = agent.config({})
    assert config.llm.model == "qwen2.5-0.5b-q4" and config.llm.max_tokens == 120
    assert config.tts.voice == "af_heart"
    assert config.turn_detection.min_silence_ms == 800
    assert agent.describe()["agent"] == "orders" and "prompt" not in agent.describe()


def test_a_plugin_runtime_can_be_named_for_a_stage():
    agent = Agent(prompt="hi", tts=TTS("voice-pack://studio", runtime="my_pkg.tts:Engine", speed=1.2))
    config = agent.config({})
    assert (config.tts.runtime, config.tts.model, config.tts.speed) == ("my_pkg.tts:Engine", "voice-pack://studio", 1.2)


def test_runtime_prefix_and_runtime_settings(tmp_path):
    agent = Agent(prompt="hi", llm=LLM("vllm:hf:org/model", max_model_len=8192, max_tokens=64))
    config = agent.config({})
    assert (config.llm.runtime, config.llm.model) == ("vllm", "hf:org/model")
    assert config.llm.options == {"max_model_len": 8192}  # unknown settings go to the runtime
    assert config.llm.max_tokens == 64  # known ones become config fields


@pytest.mark.parametrize("ref, expected", [
    ("qwen2.5-0.5b-q4", (None, "qwen2.5-0.5b-q4")),
    ("hf:org/model", (None, "hf:org/model")),
    ("http://localhost:8000/v1", (None, "http://localhost:8000/v1")),
    ("vllm:hf:org/model", ("vllm", "hf:org/model")),
    ("my_pkg.turns:Detector", ("my_pkg.turns:Detector", "")),  # a plugin runtime, no model file
])
def test_model_reference_splitting(ref, expected):
    assert Stage(ref).split() == expected


def test_language_and_turn_detector_reach_the_config():
    agent = Agent(prompt="hi", language="hi",
                  turns=Turns("my_pkg.turns:Detector", interrupt_after_ms=450, resume_window_ms=0))
    config = agent.config({})
    assert config.stt.language == "hi" and config.tts.language == "hi"
    assert config.turn_detection.runtime == "my_pkg.turns:Detector" and not config.turn_detection.model
    assert config.turn_detection.barge_in_min_speech_ms == 450
    assert config.turn_detection.resume_window_ms == 0


def test_the_environment_still_wins_over_the_agent():
    agent = Agent(prompt="hi", turns=Turns(wait_ms=800), llm="qwen2.5-0.5b-q4")
    config = agent.config({"FUSION_TURN_WAIT_MS": "300", "FUSION_LLM_URL": "http://localhost:9000/v1",
                           "FUSION_LLM_MODEL": "served-model"})
    assert config.turn_detection.min_silence_ms == 300
    assert config.llm.api_base == "http://localhost:9000/v1" and config.llm.model == "served-model"


def test_defaults_without_an_agent_file():
    agent = Agent()
    assert agent.prompt == DEFAULT_PROMPT and agent.config({}).llm.model


# ---- mistakes, with useful messages ----------------------------------------------------------

def test_empty_prompt_is_rejected():
    with pytest.raises(AgentError, match="prompt must not be empty"):
        Agent(prompt="   ")


def test_tools_become_tool_objects_and_bad_ones_are_explained():
    def order_status(order_id: str) -> str:
        """Look up where an order is."""
        return "shipped"

    agent = Agent(prompt="hi", tools=[order_status])
    assert [t.name for t in agent.tools] == ["order_status"]
    assert agent.describe()["tools"] == ["order_status"]
    with pytest.raises(AgentError, match="tool name"):
        Agent(prompt="hi", tools=[lambda: None])
    with pytest.raises(AgentError, match="two tools are named"):
        Agent(prompt="hi", tools=[order_status, order_status])


def test_bad_timing_values_are_rejected():
    with pytest.raises(AgentError, match="wait_ms"):
        Turns(wait_ms=-5)


def test_a_stage_class_used_for_the_wrong_stage_is_caught():
    with pytest.raises(AgentError, match="TTS.*given as the llm model; use LLM"):
        Agent(prompt="hi", llm=TTS("kokoro-v1.0"))


def test_vad_settings_and_limits():
    assert Agent(prompt="hi", vad=VAD(threshold=0.7)).config({}).vad.threshold == 0.7
    with pytest.raises(AgentError, match="between 0 and 1"):
        VAD(threshold=1.4)
    with pytest.raises(AgentError, match="can't be swapped yet"):
        VAD("my_pkg.vad:Engine")


def test_turns_and_vad_reject_the_wrong_type():
    with pytest.raises(AgentError, match="turns takes Turns"):
        Agent(prompt="hi", turns={"wait_ms": 500})
    with pytest.raises(AgentError, match="vad takes VAD"):
        Agent(prompt="hi", vad=0.5)


def test_missing_file(tmp_path):
    with pytest.raises(AgentError, match="no agent file at"):
        load_agent(tmp_path / "nope.py")


def test_folder_is_rejected(tmp_path):
    with pytest.raises(AgentError, match="folder"):
        load_agent(tmp_path)


def test_file_without_an_agent_says_what_to_add(tmp_path):
    path = write_agent(tmp_path, "x = 1\n")
    with pytest.raises(AgentError, match="defines no Agent"):
        load_agent(path)


def test_two_agents_need_one_named_agent(tmp_path):
    body = "from fusion_runtime import Agent\na = Agent(prompt='one')\nb = Agent(prompt='two')\n"
    with pytest.raises(AgentError, match="defines 2 agents"):
        load_agent(write_agent(tmp_path, body))


def test_an_error_inside_the_file_names_it(tmp_path):
    path = write_agent(tmp_path, "raise RuntimeError('boom')\n", name="broken.py")
    with pytest.raises(AgentError, match=r"broken.py failed while loading: RuntimeError: boom"):
        load_agent(path)


def test_an_agent_can_import_files_next_to_it(tmp_path):
    (tmp_path / "prompts.py").write_text("GREETING = 'You are the order line.'\n")
    body = "from fusion_runtime import Agent\nimport prompts\nagent = Agent(prompt=prompts.GREETING)\n"
    assert load_agent(write_agent(tmp_path, body)).prompt == "You are the order line."


# ---- the CLI ---------------------------------------------------------------------------------

def _no_server(monkeypatch):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: calls.update(k))
    monkeypatch.setattr("fusion_runtime.cli._checks.port_in_use", lambda host, port: False)
    monkeypatch.setattr("fusion_runtime.catalog.is_installed", lambda entry, root: True)
    return calls


def test_up_with_an_agent_file_tells_the_server_about_it(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_AGENT", "")
    calls = _no_server(monkeypatch)
    path = write_agent(tmp_path, AGENT_FILE)
    result = cli.invoke(app, ["up", str(path), "--port", "8123"])
    import os

    assert result.exit_code == 0, result.output
    assert os.environ["FUSION_AGENT"] == str(path.resolve())
    assert "agent" in result.output and "800 ms of silence" in result.output
    assert calls["port"] == 8123


def test_up_reports_a_broken_agent_file_and_does_not_start(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_AGENT", "")
    calls = _no_server(monkeypatch)
    path = write_agent(tmp_path, "x = 1\n")
    result = cli.invoke(app, ["up", str(path)])
    assert result.exit_code == 1 and "defines no Agent" in result.output
    assert calls == {}


def test_up_without_an_agent_uses_the_one_the_environment_names(monkeypatch, tmp_path):
    """A container has no command line to put a path on, so FUSION_AGENT has to
    work. This used to do the opposite: the variable was unset here, so a
    deployment that set it got the default profile and no warning."""
    import os

    agent = tmp_path / "agent.py"
    agent.write_text("from fusion_runtime import Agent\nagent = Agent(name='env-agent', prompt='hi')\n")
    monkeypatch.setenv("FUSION_AGENT", str(agent))
    _no_server(monkeypatch)
    result = cli.invoke(app, ["up"])
    assert result.exit_code == 0, result.output
    assert os.environ["FUSION_AGENT"] == str(agent.resolve())


def test_up_with_neither_falls_back_to_the_profile(monkeypatch):
    import os

    monkeypatch.delenv("FUSION_AGENT", raising=False)
    _no_server(monkeypatch)
    result = cli.invoke(app, ["up"])
    assert result.exit_code == 0, result.output
    assert "FUSION_AGENT" not in os.environ  # nothing stale left for the server to pick up


# ---- the server ------------------------------------------------------------------------------

def test_server_uses_the_agents_prompt(tmp_path, monkeypatch):
    from fusion_runtime import server

    path = write_agent(tmp_path, AGENT_FILE)
    monkeypatch.setattr(server, "agent", load_agent(path))
    assert server.agent.prompt.startswith("You take orders")


def test_moving_a_served_model_by_url_keeps_its_name():
    # Found on a pod: vLLM had to move ports, and the URL alone should have been enough
    agent = Agent(prompt="hi", llm=LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ@main"))
    config = agent.config({"FUSION_LLM_URL": "http://localhost:8002/v1"})
    assert config.llm.api_base == "http://localhost:8002/v1" and config.llm.model == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    named = Agent(prompt="hi", llm=LLM("http://localhost:9000/v1", model_name="served"))
    assert named.config({"FUSION_LLM_URL": "http://localhost:9001/v1"}).llm.model == "served"
    with pytest.raises(ValueError, match="FUSION_LLM_MODEL"):  # a local model has no name on any server
        Agent(prompt="hi", llm=LLM("qwen2.5-0.5b-q4")).config({"FUSION_LLM_URL": "http://localhost:8002/v1"})


def test_missing_models_are_named_for_the_agent_file_not_a_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_MODEL_DIR", str(tmp_path / "empty"))
    path = write_agent(tmp_path, AGENT_FILE)
    result = cli.invoke(app, ["up", str(path)])
    assert result.exit_code == 1
    assert "profile" not in result.output and f"frun models pull {path}" in result.output.replace("\n", "")
