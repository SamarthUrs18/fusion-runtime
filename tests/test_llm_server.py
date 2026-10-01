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
            elif self.path == "/metrics":
                body = ('vllm:num_requests_running{{model_name="' + name + '"}} 2.0\\n'
                        'vllm:num_requests_waiting{{model_name="' + name + '"}} 1.0\\n'
                        'vllm:kv_cache_usage_perc{{model_name="' + name + '"}} 0.25\\n').encode()
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


def fake_nvidia_smi(cards, apps=""):
    import subprocess

    def run(args, **kwargs):
        out = cards if "--query-gpu" in args[1] else apps
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")
    return run


def test_the_gpu_is_read_with_whatever_holds_memory_on_it():
    gpu = llm_server.read_gpu({}, run=fake_nvidia_smi(
        "0, NVIDIA GeForce RTX 3090, 24576, 9216, GPU-aaa\n1, NVIDIA L4, 23034, 23000, GPU-bbb\n",
        "GPU-aaa, 4242, /root/vllm-env/bin/python3, 14848\nGPU-bbb, 7, other, 10\n"))
    assert (gpu.index, gpu.name, gpu.total_gb, gpu.free_gb) == (0, "NVIDIA GeForce RTX 3090", 24.0, 9.0)
    assert gpu.users == ((4242, "/root/vllm-env/bin/python3", 14.5),)
    second = llm_server.read_gpu({"CUDA_VISIBLE_DEVICES": "1"}, run=fake_nvidia_smi(
        "0, A, 24576, 1, GPU-aaa\n1, NVIDIA L4, 23034, 23000, GPU-bbb\n"))
    assert second.name == "NVIDIA L4"


def test_a_gpu_already_in_use_stops_the_start_and_says_by_what(engine_env):
    launch = llm_server.plan(llm_config(LLM("vllm:hf:org/model", engine_env=str(engine_env))), has_tools=False)
    busy = llm_server.GPU(0, "NVIDIA GeForce RTX 3090", 24.0, 9.0, ((4242, "/root/vllm-env/bin/python3", 14.5),))
    with pytest.raises(LaunchError, match=r"(?s)vLLM needs 14.4 GB of GPU 0.*only 9.0 GB is free.*pid 4242.*kill <pid>"):
        llm_server.check_gpu_memory(launch, busy, speech_need_gb=2.5)
    free = llm_server.GPU(0, "NVIDIA GeForce RTX 3090", 24.0, 23.5)
    assert llm_server.check_gpu_memory(launch, free, speech_need_gb=2.5) == []
    assert llm_server.check_gpu_memory(launch, None, speech_need_gb=2.5) == []  # no NVIDIA GPU: nothing to check


def test_too_little_left_for_speech_is_a_warning_with_a_number(engine_env):
    launch = llm_server.plan(llm_config(LLM("vllm:hf:org/model", gpu_memory=0.9, engine_env=str(engine_env))),
                             has_tools=False)
    warnings = llm_server.check_gpu_memory(launch, llm_server.GPU(0, "L4", 22.0, 21.8), speech_need_gb=2.5)
    assert len(warnings) == 1 and "leaving 2.2 GB" in warnings[0] and "lower gpu_memory (to 0.86)" in warnings[0]


def test_speech_needs_more_with_a_bigger_whisper_on_the_gpu():
    from fusion_runtime.config import PROFILES

    on_cpu = llm_server.speech_gb(PROFILES["development"])  # Whisper on the CPU: Kokoro and contexts only
    on_gpu = llm_server.speech_gb(PROFILES["production"])  # whisper-small on CUDA
    assert on_cpu == 1.6 and on_gpu == 2.5


VLLM_METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="org/model"} 7.0
vllm:num_requests_waiting{engine="0",model_name="org/model"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="org/model"} 0.62
vllm:gpu_cache_usage_perc{engine="0",model_name="org/model"} 0.62
"""


def test_the_engines_own_numbers_are_read_from_its_metrics():
    assert llm_server.parse_engine_metrics(VLLM_METRICS, "vllm") == {"running": 7, "waiting": 3, "kv_cache_used": 0.62}
    older = "vllm:gpu_cache_usage_perc{model_name=\"m\"} 0.3\n"  # before vLLM renamed it
    assert llm_server.parse_engine_metrics(older, "vllm") == {"kv_cache_used": 0.3}
    sglang = "sglang:num_running_reqs{tp_rank=\"0\"} 4\nsglang:num_queue_reqs{tp_rank=\"0\"} 0\nsglang:token_usage{tp_rank=\"0\"} 0.2\n"
    assert llm_server.parse_engine_metrics(sglang, "sglang") == {"running": 4, "waiting": 0, "kv_cache_used": 0.2}
    assert "--enable-metrics" in llm_server.ENGINES["sglang"].command("python", "m", 30000)


async def test_the_monitor_notices_the_server_going_down_and_coming_back():
    import httpx
    from fusion_runtime.telemetry import telemetry
    from fusion_runtime.telemetry.sinks import ListSink

    up = {"now": True}

    def answer(request):
        if not up["now"]:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, text=VLLM_METRICS if request.url.path == "/metrics" else "")

    sink = ListSink()
    telemetry.add_sink(sink)
    try:
        monitor = llm_server.EngineMonitor("vLLM", "http://127.0.0.1:8002/v1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
            await monitor.poll(client)
            assert monitor.snapshot()["state"] == "up" and monitor.snapshot()["waiting"] == 3
            assert monitor.turn_fields() == {"llm_server_running": 7, "llm_server_waiting": 3, "kv_cache_used": 0.62}
            up["now"] = False
            await monitor.poll(client)
            assert monitor.snapshot()["state"] == "down" and "waiting" not in monitor.snapshot()
            up["now"] = True
            await monitor.poll(client)
    finally:
        telemetry.remove_sink(sink)
    names = [e.name for e in sink.events if e.name in ("llm_server.up", "llm_server.down")]
    assert names == ["llm_server.down", "llm_server.up"]  # the first healthy check isn't news


def test_the_engines_numbers_become_prometheus_gauges():
    from fusion_runtime.telemetry.events import Event
    from fusion_runtime.telemetry.metrics import TelemetryMetrics
    from prometheus_client import generate_latest

    metrics = TelemetryMetrics()
    metrics.handle(Event(name="llm_server.state", stage="llm",
                         attrs={"engine": "vllm", "state": "up", "running": 7, "waiting": 3, "kv_cache_used": 0.62}))
    text = generate_latest(metrics.registry).decode()
    assert 'fusion_llm_server_up{engine="vllm"} 1.0' in text
    assert 'fusion_llm_server_requests_waiting{engine="vllm"} 3.0' in text
    assert 'fusion_llm_server_kv_cache_used{engine="vllm"} 0.62' in text


def test_an_engine_that_cant_run_is_an_error_not_a_crash(engine_env):
    (engine_env / "bin" / "vllm").write_bytes(b"\x00\x01not a program")
    launch = llm_server.plan(llm_config(LLM("vllm:hf:org/model", url=f"http://127.0.0.1:{free_port()}/v1",
                                            engine_env=str(engine_env))), has_tools=False)
    with pytest.raises(LaunchError, match=r"couldn't run .*vllm: .*Reinstall vLLM"):
        llm_server.LLMServer(launch, lambda level, message: None).start()
