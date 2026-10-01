"""frun up starting vLLM / SGLang: the command it builds, when it starts nothing, and the process
it looks after. A fake engine (a script that answers /health like vLLM) stands in for the real one."""
import os
import signal
import socket
import sys
import textwrap
import time

import pytest
from fusion_runtime import llm_server
from fusion_runtime.agent import LLM, Agent
from fusion_runtime.llm_server import LaunchError

FAKE_ENGINE = textwrap.dedent('''\
    #!{python}
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    args = sys.argv
    port = int(args[args.index("--port") + 1])
    name = args[args.index("--served-model-name") + 1]
    print("fake engine: loading " + name, flush=True)
    if os.environ.get("FAKE_ENGINE_MODE") == "crash":
        print("fake engine: CUDA out of memory", flush=True)
        sys.exit(3)
    if os.environ.get("FUSION_ACCEPTED_KEYS"):
        print("fake engine: was given fusion's keys", flush=True)
        sys.exit(4)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b""
            if self.path == "/v1/models":
                body = json.dumps({{"data": [{{"id": name}}]}}).encode()
            elif self.path != "/health":
                self.send_response(404); self.end_headers(); return
            self.send_response(200); self.end_headers(); self.wfile.write(body)

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
''')


@pytest.fixture(autouse=True)
def _no_llm_url(monkeypatch):
    monkeypatch.delenv("FUSION_LLM_URL", raising=False)
    monkeypatch.delenv(llm_server.ENGINE_ENV_ENV, raising=False)


@pytest.fixture
def engine_env(tmp_path):
    """A folder laid out like a venv with vLLM in it: bin/vllm."""
    bin_dir = tmp_path / "vllm-env" / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / "vllm"
    script.write_text(FAKE_ENGINE.format(python=sys.executable))
    script.chmod(0o755)
    return tmp_path / "vllm-env"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def llm_config(llm):
    return Agent(llm=llm).config().llm


def test_vllm_command_has_the_settings_a_shared_gpu_needs(engine_env):
    launch = llm_server.plan(llm_config(LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ@abc123", engine_env=str(engine_env))),
                             has_tools=True)
    command = launch.command
    assert command[:3] == [str(engine_env / "bin" / "vllm"), "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ"]
    joined = " ".join(command)
    assert "--host 127.0.0.1 --port 8002" in joined  # never reachable from another machine
    assert "--served-model-name Qwen/Qwen2.5-7B-Instruct-AWQ" in joined and "--revision abc123" in joined
    assert "--gpu-memory-utilization 0.6 --max-model-len 4096" in joined
    assert "--enable-auto-tool-choice --tool-call-parser hermes" in joined
    assert launch.base_url == "http://127.0.0.1:8002/v1" and launch.health_url == "http://127.0.0.1:8002/health"
    assert launch.env["PATH"].startswith(str(engine_env / "bin"))  # ninja, for kernels compiled on first use
    assert any("tool_parser" in note for note in launch.notes)


def test_sglang_gets_its_own_flag_names(monkeypatch):
    monkeypatch.setattr(llm_server, "_imports", lambda python, module: True)  # "installed" next to fusion
    launch = llm_server.plan(llm_config(LLM("sglang:hf:org/model", gpu_memory=0.5, max_model_len=8192,
                                            max_callers=12, extra_args="--seed 1")), has_tools=True)
    joined = " ".join(launch.command)
    assert "-m sglang.launch_server --model-path org/model --host 127.0.0.1 --port 30000" in joined
    assert "--mem-fraction-static 0.5 --context-length 8192 --max-running-requests 12" in joined
    assert "--tool-call-parser qwen25" in joined and "--enable-auto-tool-choice" not in joined
    assert launch.command[-2:] == ["--seed", "1"]


def test_no_tools_no_parser_and_settings_are_checked(engine_env):
    base = dict(engine_env=str(engine_env))
    launch = llm_server.plan(llm_config(LLM("vllm:hf:org/model", url="http://localhost:9100/v1", **base)),
                             has_tools=False)
    assert "--tool-call-parser" not in launch.command and launch.notes == []
    assert "--port 9100" in " ".join(launch.command)  # a local url= moves where it's started
    with pytest.raises(LaunchError, match="gpu_memory must be between"):
        llm_server.plan(llm_config(LLM("vllm:hf:org/model", gpu_memory=60, **base)), has_tools=False)
    with pytest.raises(LaunchError, match="extra_args must be"):
        llm_server.plan(llm_config(LLM("vllm:hf:org/model", extra_args={"seed": 1}, **base)), has_tools=False)


def test_fusion_keys_stay_out_of_the_engines_environment(engine_env):
    launch = llm_server.plan(llm_config(LLM("vllm:hf:org/model", engine_env=str(engine_env))), has_tools=False,
                             environ={"PATH": "/usr/bin", "FUSION_ACCEPTED_KEYS": "frun_x", "FUSION_API_KEY": "frun_y",
                                      "HF_HOME": "/workspace/hf"})
    assert "FUSION_ACCEPTED_KEYS" not in launch.env and "FUSION_API_KEY" not in launch.env
    assert launch.env["HF_HOME"] == "/workspace/hf"


@pytest.mark.parametrize("llm", [
    LLM("qwen2.5-0.5b-q4"),                                         # in-process
    LLM("llama_server:hf:org/model"),                               # you start llama-server
    LLM("vllm:hf:org/model", url="http://gpu-box:8002/v1"),         # a server on another machine
    LLM("vllm:hf:org/model", launch=False),                         # you start it yourself
    LLM("http://127.0.0.1:8002/v1", runtime="vllm", model_name="m"),
])
def test_nothing_is_started_unless_the_agent_asks_for_a_local_engine(llm):
    assert llm_server.launchable(llm_config(llm)) is None


def test_fusion_llm_url_means_the_server_is_elsewhere():
    config = Agent(llm=LLM("vllm:hf:org/model")).config({"FUSION_LLM_URL": "http://gpu:8002/v1"})
    assert llm_server.launchable(config.llm) is None


def test_a_missing_engine_says_how_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_server.shutil, "which", lambda name: None)
    with pytest.raises(LaunchError, match=r"(?s)isn't installed where fusion can find it.*pip install vllm"
                                          r".*FUSION_LLM_ENGINE_ENV.*start vLLM yourself at http://127.0.0.1:8002/v1"):
        llm_server.plan(llm_config(LLM("vllm:hf:org/model")), has_tools=False)
    with pytest.raises(LaunchError, match=r"isn't installed in .*empty-env"):
        llm_server.plan(llm_config(LLM("vllm:hf:org/model", engine_env=str(tmp_path / "empty-env"))), has_tools=False)


def test_the_server_starts_restarts_after_a_crash_and_stops(engine_env):
    port = free_port()
    launch = llm_server.plan(llm_config(LLM("vllm:hf:org/model", url=f"http://127.0.0.1:{port}/v1",
                                            engine_env=str(engine_env))), has_tools=False)
    assert llm_server.already_serving(launch) is False  # nothing there yet: fusion starts it
    lines = []
    server = llm_server.LLMServer(launch, lambda level, message: lines.append((level, message)))
    server.first_restart_delay_s = 0.1
    server.start()
    try:
        assert llm_server.already_serving(launch) is True  # a second frun up would use it, not start another
        assert ("server", "fake engine: loading org/model") in lines  # its log comes through

        first = server.pid
        os.kill(first, signal.SIGKILL)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not (server.pid != first and llm_server.already_serving(launch)):
            time.sleep(0.1)
        assert server.restarts == 1 and server.pid != first
        assert any(level == "error" and "exited" in message for level, message in lines)
    finally:
        server.stop()
    assert llm_server.already_serving(launch) is False


def test_a_server_that_dies_while_starting_shows_its_last_lines(engine_env, monkeypatch):
    monkeypatch.setenv("FAKE_ENGINE_MODE", "crash")
    launch = llm_server.plan(llm_config(LLM("vllm:hf:org/model", url=f"http://127.0.0.1:{free_port()}/v1",
                                            engine_env=str(engine_env))), has_tools=False)
    server = llm_server.LLMServer(launch, lambda level, message: None)
    with pytest.raises(LaunchError, match=r"(?s)exited while starting \(exit code 3\).*CUDA out of memory"):
        server.start()


def test_another_model_on_the_port_is_named_rather_than_started_over(engine_env):
    port = free_port()
    mine = llm_server.plan(llm_config(LLM("vllm:hf:org/model", url=f"http://127.0.0.1:{port}/v1",
                                          engine_env=str(engine_env))), has_tools=False)
    server = llm_server.LLMServer(mine, lambda level, message: None)
    server.start()
    try:
        other = llm_server.plan(llm_config(LLM("vllm:hf:org/other", url=f"http://127.0.0.1:{port}/v1",
                                               engine_env=str(engine_env))), has_tools=False)
        with pytest.raises(LaunchError, match="serves org/model, not org/other"):
            llm_server.already_serving(other)
    finally:
        server.stop()


def test_without_a_log_file_every_line_goes_to_the_terminal():
    import io

    out = io.StringIO()
    printer = llm_server.LogPrinter(llm_server.ENGINES["vllm"], json_format=False, stream=out)
    for level, message in [("starting", "starting vLLM"), ("server", "loading weights"), ("ready", "vLLM ready in 40 s"),
                           ("server", "Avg prompt throughput: 812 tokens/s")]:
        printer(level, message)
    assert out.getvalue().splitlines() == ["   vllm > starting vLLM", "   vllm | loading weights",
                                           "   vllm > vLLM ready in 40 s", "   vllm | Avg prompt throughput: 812 tokens/s"]


def test_with_a_log_file_the_terminal_keeps_the_start_and_trouble_only(tmp_path):
    import io
    import json

    out, log = io.StringIO(), tmp_path / "logs" / "vllm.log"
    printer = llm_server.LogPrinter(llm_server.ENGINES["vllm"], json_format=True, file=log, stream=out)
    printer("starting", "starting vLLM")
    printer("server", "loading weights")  # during the start: both
    printer("ready", "vLLM ready in 40 s")
    printer("server", "Avg prompt throughput: 812 tokens/s")  # routine: file only
    printer("server", "WARNING 09-30 KV cache is 95% full")  # trouble: both
    printer("error", "vLLM exited (code 1)")
    printer.close()

    shown = [json.loads(line)["message"] for line in out.getvalue().splitlines()]
    assert "Avg prompt throughput: 812 tokens/s" not in shown
    assert shown == ["starting vLLM", "loading weights", "vLLM ready in 40 s",
                     "WARNING 09-30 KV cache is 95% full", "vLLM exited (code 1)"]
    written = log.read_text().splitlines()
    assert "loading weights" in written and "Avg prompt throughput: 812 tokens/s" in written
    assert any(line.endswith("fusion: vLLM exited (code 1)") for line in written)


def test_a_log_file_that_cant_be_written_is_named(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(LaunchError, match="can't write the LLM server's log"):
        llm_server.LogPrinter(llm_server.ENGINES["vllm"], json_format=False, file=blocker / "vllm.log")
