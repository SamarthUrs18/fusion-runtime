"""Starting and looking after the LLM server, when the agent names vLLM or SGLang.

    LLM("vllm:hf:Qwen/Qwen2.5-7B-Instruct-AWQ")      `frun up` starts vLLM on port 8002
    LLM("sglang:hf:Qwen/Qwen2.5-7B-Instruct-AWQ")    ...SGLang on port 30000

`frun up` starts the server before loading speech, so the server takes its share
of GPU memory first; streams its log; restarts it if it exits; and stops it on
the way out. A server that already answers at the address is used as it is.

The engine lives in its own Python environment, because it pins its own torch
and CUDA: point FUSION_LLM_ENGINE_ENV (or engine_env=) at that environment, or
have its command on PATH. fusion never installs it.

The server listens on 127.0.0.1 only. It has no keys of its own, so it is never
reachable from another machine; callers reach the agent, and the agent reaches it.
"""
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlsplit

ENGINE_ENV_ENV = "FUSION_LLM_ENGINE_ENV"  # the Python environment vLLM or SGLang is installed in
LOG_FILE_ENV = "FUSION_LLM_LOG"  # a file for the server's full log; the terminal then shows only trouble

# Settings read here, when fusion starts the server. Everything else goes to the HTTP client.
LAUNCH_OPTIONS = ("launch", "engine_env", "gpu_memory", "max_model_len", "max_callers",
                  "tool_parser", "extra_args", "start_timeout_s")

GPU_MEMORY = 0.6  # the LLM's share of the card; speech-to-text and text-to-speech use the rest
MAX_MODEL_LEN = 4096  # a voice call's context; longer reserves memory a call never uses
START_TIMEOUT_S = 900.0  # a first start downloads the model and compiles kernels: SGLang took ~10 min
LOG_TAIL = 40  # lines kept to show when the server fails to start


class LaunchError(RuntimeError):
    """The LLM server can't be started, or what's at its address isn't it. The message says what to do."""


@dataclass(frozen=True)
class Engine:
    name: str  # the runtime prefix: "vllm"
    display: str  # "vLLM"
    port: int
    install: str  # what goes after `pip install`
    tool_parser: str  # the parser for the default model, Qwen 2.5
    # flag names for the shared settings
    gpu_memory_flag: str
    max_model_len_flag: str
    max_callers_flag: str

    def command(self, python_or_bin: str, model: str, port: int) -> List[str]:
        if self.name == "vllm":
            return [python_or_bin, "serve", model, "--host", "127.0.0.1", "--port", str(port)]
        # --enable-metrics: SGLang's /metrics is off otherwise, and fusion reads its queue from it
        return [python_or_bin, "-m", "sglang.launch_server", "--model-path", model,
                "--host", "127.0.0.1", "--port", str(port), "--enable-metrics"]

    def tool_flags(self, parser: str) -> List[str]:
        if self.name == "vllm":
            return ["--enable-auto-tool-choice", "--tool-call-parser", parser]
        return ["--tool-call-parser", parser]


ENGINES: Dict[str, Engine] = {
    "vllm": Engine("vllm", "vLLM", 8002, "vllm", "hermes",
                   "--gpu-memory-utilization", "--max-model-len", "--max-num-seqs"),
    "sglang": Engine("sglang", "SGLang", 30000, '"sglang[all]"', "qwen25",
                     "--mem-fraction-static", "--context-length", "--max-running-requests"),
}


@dataclass
class LaunchPlan:
    engine: Engine
    model_name: str  # what the server answers to, and what fusion asks it for
    base_url: str  # http://127.0.0.1:8002/v1
    command: List[str]
    env: Dict[str, str]
    start_timeout_s: float = START_TIMEOUT_S
    notes: List[str] = field(default_factory=list)  # said once at startup: defaults chosen for the user
    gpu_memory: float = GPU_MEMORY  # the engine's share of the card

    @property
    def health_url(self) -> str:
        parts = urlsplit(self.base_url)
        return f"{parts.scheme}://{parts.netloc}/health"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"


def launchable(llm) -> Optional[Engine]:
    """The engine to start for this LLM config, or None when fusion shouldn't start anything.

    Nothing is started for a model given as a URL, a server on another machine,
    `launch=False`, or when FUSION_LLM_URL has already moved the LLM elsewhere.
    """
    engine = ENGINES.get(llm.runtime or "")
    if engine is None or (llm.model or "").startswith(("http://", "https://")):
        return None
    options = llm.options
    if options.get("launch", True) is False:
        return None
    url = options.get("url")
    if url and not _is_local(urlsplit(url).hostname):
        return None
    return engine


def plan(llm, *, has_tools: bool, environ: Optional[Mapping[str, str]] = None) -> LaunchPlan:
    """How to start the server for this LLM config. Raises LaunchError with the fix."""
    from fusion_runtime.resolver import _served_model_name

    env_vars = os.environ if environ is None else environ
    engine = launchable(llm)
    if engine is None:
        raise LaunchError(f"{llm.runtime}:{llm.model} isn't a server fusion starts")
    options = dict(llm.options)
    port = engine.port
    if options.get("url"):
        port = urlsplit(options["url"]).port
        if port is None:
            # fusion talks to the url as written (port 80 for http://localhost/v1), so starting the
            # engine on its usual port instead would leave every reply failing
            raise LaunchError(f"url={options['url']!r} has no port. Name the one {engine.display} should use, "
                              f"like http://127.0.0.1:{engine.port}/v1")
    ref = llm.model
    model, revision = ref, None
    if ref.startswith("hf:"):
        model = ref[3:]
        if "@" in model:
            model, revision = model.split("@", 1)
    model_name = options.get("model_name") or _served_model_name(ref)

    binary, bin_dir = _find_engine(engine, options.get("engine_env") or env_vars.get(ENGINE_ENV_ENV),
                                   f"http://127.0.0.1:{port}/v1")
    command = engine.command(binary, model, port)
    command += ["--served-model-name", model_name]
    if revision:
        command += ["--revision", revision]
    gpu_memory = _number(options, "gpu_memory", GPU_MEMORY, 0.05, 0.95)
    command += [engine.gpu_memory_flag, str(gpu_memory)]
    command += [engine.max_model_len_flag, str(int(_number(options, "max_model_len", MAX_MODEL_LEN, 256, None)))]
    if options.get("max_callers") is not None:
        command += [engine.max_callers_flag, str(int(_number(options, "max_callers", 0, 1, None)))]

    notes = []
    parser = options.get("tool_parser")
    if parser is None and has_tools:
        parser = engine.tool_parser
        notes.append(f"tool calls are read in the {parser} format (Qwen 2.5's); "
                     f"for another model family set tool_parser= (see the {engine.display} docs)")
    if parser:
        command += engine.tool_flags(str(parser))

    extra = options.get("extra_args") or []
    if isinstance(extra, str):
        extra = shlex.split(extra)
    if not isinstance(extra, (list, tuple)) or not all(isinstance(a, (str, int, float)) for a in extra):
        raise LaunchError('extra_args must be a string ("--seed 1") or a list of strings')
    command += [str(a) for a in extra]

    child_env = {k: v for k, v in env_vars.items() if not _is_secret(k)}
    if bin_dir:
        # Both engines compile kernels on first use and call `ninja` from their own environment
        child_env["PATH"] = bin_dir + os.pathsep + child_env.get("PATH", "")
    timeout = _number(options, "start_timeout_s", START_TIMEOUT_S, 10, None)
    return LaunchPlan(engine, model_name, f"http://127.0.0.1:{port}/v1", command, child_env, timeout, notes,
                      gpu_memory)


def already_serving(launch: LaunchPlan, timeout_s: float = 2.0) -> bool:
    """True when a server already answers at the address with this model; False when nothing does.

    Raises LaunchError when something answers there that isn't serving the model,
    rather than starting a second server that can't get the port.
    """
    import httpx

    try:
        response = httpx.get(launch.models_url, timeout=timeout_s)
    except httpx.ConnectError:
        return False
    except httpx.HTTPError as e:
        raise LaunchError(f"something is on {launch.base_url} but didn't answer as an LLM server ({e}). "
                          f"Stop it, or move the LLM: LLM(..., url=\"http://localhost:<port>/v1\")") from None
    try:
        served = [m.get("id") for m in response.json().get("data", [])]
    except (ValueError, AttributeError):
        served = None
    if response.status_code != 200 or served is None:
        raise LaunchError(f"something is on {launch.base_url} but it isn't an OpenAI-compatible server "
                          f"(GET /models gave HTTP {response.status_code}). Stop it, or move the LLM: "
                          f"LLM(..., url=\"http://localhost:<port>/v1\")")
    if launch.model_name not in served:
        raise LaunchError(f"the server on {launch.base_url} serves {', '.join(map(str, served)) or 'no models'}, "
                          f"not {launch.model_name}. Stop it, or move the LLM: LLM(..., url=\"http://localhost:<port>/v1\")")
    return True


# -- GPU memory, before starting ------------------------------------------------------------------

# What speech takes on the GPU beside the engine, in GB: rough, measured on a 3090 with
# CTranslate2 float16 Whisper and Kokoro on onnxruntime-gpu. Used to warn, never to refuse.
WHISPER_GB = {"tiny": 0.3, "base": 0.4, "small": 0.9, "medium": 2.0, "large": 3.6, "turbo": 2.0, "distil": 1.6}
KOKORO_GB = 0.8
CUDA_CONTEXT_GB = 0.8  # each library's CUDA context in fusion's process, together


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    total_gb: float
    free_gb: float
    users: tuple = ()  # (pid, process name, GB) of what holds memory on it now


def read_gpu(environ: Optional[Mapping[str, str]] = None, run=subprocess.run) -> Optional[GPU]:
    """The card the engine will use (the first in CUDA_VISIBLE_DEVICES), or None without nvidia-smi."""
    env = os.environ if environ is None else environ
    if shutil.which("nvidia-smi") is None and run is subprocess.run:
        return None
    visible = (env.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    index = int(visible) if visible.isdigit() else 0
    try:
        cards = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,uuid",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        apps = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in cards.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit() or int(parts[0]) != index:
            continue
        try:
            total, free = float(parts[2]) / 1024, float(parts[3]) / 1024
        except ValueError:
            return None
        users = []
        for app in apps.strip().splitlines():
            fields = [f.strip() for f in app.split(",")]
            if len(fields) == 4 and fields[0] == parts[4]:
                try:
                    users.append((int(fields[1]), fields[2], round(float(fields[3]) / 1024, 1)))
                except ValueError:
                    pass
        return GPU(index, parts[1], round(total, 1), round(free, 1), tuple(users))
    return None


def speech_gb(config) -> float:
    """What Whisper and Kokoro take on the GPU, roughly, for this pipeline's settings."""
    need = 0.0
    stt = getattr(config, "stt", None)
    if stt is not None and getattr(stt, "device", "cpu") in ("cuda", "auto"):
        model = str(getattr(stt, "model", "")).lower()
        need += next((gb for size, gb in WHISPER_GB.items() if size in model), WHISPER_GB["small"])
    need += KOKORO_GB  # on the GPU whenever onnxruntime-gpu is installed; small enough to always count
    return round(need + CUDA_CONTEXT_GB, 1)


def check_gpu_memory(launch: LaunchPlan, gpu: Optional[GPU], speech_need_gb: float) -> List[str]:
    """Refuse a start that can't work; return warnings for one that may not leave speech enough.

    The engine reserves its share of the card's *total* memory at start and fails if that much
    isn't free, a few minutes in and with a message about KV cache blocks. The usual cause is
    something left running: an earlier engine, another frun up, a notebook.
    """
    if gpu is None:
        return []
    engine = launch.engine.display
    wanted = launch.gpu_memory * gpu.total_gb
    if gpu.free_gb + 0.3 < wanted:  # a little slack: nvidia-smi rounds, and the engine allows some
        holders = "\n".join(f"    pid {pid}  {name}  {gb} GB" for pid, name, gb in gpu.users) or "    (nvidia-smi lists none)"
        raise LaunchError(
            f"{engine} needs {wanted:.1f} GB of GPU {gpu.index} ({gpu.name}): gpu_memory={launch.gpu_memory} of "
            f"{gpu.total_gb:.1f} GB, but only {gpu.free_gb:.1f} GB is free. Using it now:\n{holders}\n"
            f"  Stop what you don't need (an earlier {engine} or frun up is the usual one: kill <pid>), "
            f"or lower gpu_memory.")
    left = gpu.total_gb * (1 - launch.gpu_memory)
    if left < speech_need_gb:
        return [f"{engine} takes {wanted:.1f} GB, leaving {left:.1f} GB for speech-to-text and text-to-speech, "
                f"which need about {speech_need_gb:.1f} GB. If they fail to load, lower gpu_memory "
                f"(to {max(0.1, 1 - (speech_need_gb + 0.5) / gpu.total_gb):.2f}) or use a smaller Whisper."]
    return []


class LLMServer:
    """One engine process: started, watched, restarted with backoff, stopped."""

    def __init__(self, launch: LaunchPlan, log: Callable[[str, str], None]):
        self.launch = launch
        self._log = log  # (level, message)
        self._process: Optional[subprocess.Popen] = None
        self._tail: deque = deque(maxlen=LOG_TAIL)
        self._stopping = threading.Event()
        self._watcher: Optional[threading.Thread] = None
        self.restarts = 0
        self.first_restart_delay_s = 2.0

    @property
    def name(self) -> str:
        return self.launch.engine.display

    def start(self) -> None:
        """Start the server and wait until it answers. Raises LaunchError if it exits or takes too long."""
        self._spawn()
        self._wait_ready(self._process)
        self._watcher = threading.Thread(target=self._watch, name="llm-server-watch", daemon=True)
        self._watcher.start()

    def stop(self, timeout_s: float = 20.0) -> None:
        self._stopping.set()
        process = self._process
        if process is None or process.poll() is not None:
            return
        self._log("info", f"stopping {self.name}")
        _signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self._log("warning", f"{self.name} didn't stop in {timeout_s:.0f} s; killing it")
            _signal_group(process, signal.SIGKILL)
            process.wait(5)

    # -- internals ---------------------------------------------------------------------------

    def _spawn(self) -> None:
        if self._stopping.is_set():  # stop() came between a crash and its restart: don't start a new one
            raise LaunchError(f"stopped before {self.name} was restarted")
        self._tail.clear()
        self._log("starting", f"starting {self.name}")
        try:
            self._process = self._popen()
        except OSError as e:  # the program is there but can't run: wrong platform, a broken venv, permissions
            raise LaunchError(f"couldn't run {self.launch.command[0]}: {e.strerror or e}. "
                              f"Reinstall {self.name} in that environment") from None
        threading.Thread(target=self._pump, args=(self._process,), name="llm-server-log", daemon=True).start()

    def _popen(self) -> subprocess.Popen:
        return subprocess.Popen(
            self.launch.command, env=self.launch.env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
            # Its own process group: Ctrl+C reaches fusion, which then stops the server itself,
            # rather than both racing to shut down and the watcher restarting a server mid-exit.
            start_new_session=True,
            preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None,
        )

    def _pump(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            line = line.rstrip()
            if line:
                self._tail.append(line)
                self._log("server", line)

    def _wait_ready(self, process: subprocess.Popen) -> None:
        import httpx

        started = time.monotonic()
        next_note = 30.0
        while True:
            if self._stopping.is_set():
                # stop() may have looked before this process existed, so it's ours to end
                _signal_group(process, signal.SIGTERM)
                raise LaunchError(f"stopped while {self.name} was starting")
            code = process.poll()
            if code is not None:
                time.sleep(0.2)  # let the log pump catch the last lines
                tail = "\n".join(f"    {line}" for line in list(self._tail)[-20:])
                raise LaunchError(f"{self.name} exited while starting (exit code {code}). Its last lines:\n{tail}")
            try:
                if httpx.get(self.launch.health_url, timeout=2.0).status_code == 200:
                    self._log("ready", f"{self.name} ready in {time.monotonic() - started:.0f} s")
                    return
            except httpx.HTTPError:
                pass
            waited = time.monotonic() - started
            if waited > self.launch.start_timeout_s:
                _signal_group(process, signal.SIGTERM)
                raise LaunchError(f"{self.name} didn't answer within {self.launch.start_timeout_s:.0f} s. "
                                  "A first start downloads the model and compiles kernels; if that's what's "
                                  "happening, raise start_timeout_s")
            if waited > next_note:
                self._log("info", f"{self.name} still starting ({waited:.0f} s); a first start downloads "
                                  "the model and compiles kernels")
                next_note += 60.0
            time.sleep(1.0)

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    def _watch(self) -> None:
        delay = self.first_restart_delay_s
        while not self._stopping.is_set():
            process = self._process
            started = time.monotonic()
            code = process.wait()
            if self._stopping.is_set():
                return
            if time.monotonic() - started > 300:
                delay = self.first_restart_delay_s  # it ran for a while: this is a new problem, not the same one again
            self.restarts += 1
            self._log("error", f"{self.name} exited (code {code}); calls can't get replies until it's back. "
                               f"Restarting in {delay:.0f} s")
            if self._stopping.wait(delay):
                return
            delay = min(delay * 2, 60.0)
            try:
                self._spawn()
                self._wait_ready(self._process)
            except LaunchError as e:
                self._log("error", str(e))
            except OSError as e:  # the program went away (environment deleted?): keep trying, say why
                self._log("error", f"couldn't start {self.name} again: {e}")


# -- watching the engine while calls run ---------------------------------------------------------

# Prometheus series each engine publishes, by what fusion calls them. Several names per field:
# engines rename them between versions (vLLM's gpu_cache_usage_perc became kv_cache_usage_perc).
ENGINE_SERIES = {
    "vllm": {"running": ("vllm:num_requests_running",), "waiting": ("vllm:num_requests_waiting",),
             "kv_cache_used": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")},
    "sglang": {"running": ("sglang:num_running_reqs",), "waiting": ("sglang:num_queue_reqs",),
               "kv_cache_used": ("sglang:token_usage",)},
}
SERVED_BY = {"vLLM": "vllm", "SGLang": "sglang", "llama-server": "llama_server"}


def parse_engine_metrics(text: str, engine: str) -> Dict[str, float]:
    """Requests running and waiting, and the share of KV cache in use, from an engine's /metrics.

    Values are summed over label sets (one per model). When an engine publishes a field under
    two names, the first name in ENGINE_SERIES that appears is used, so it isn't counted twice.
    """
    by_name: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        by_name[name] = by_name.get(name, 0.0) + value
    found: Dict[str, float] = {}
    for field_name, names in ENGINE_SERIES.get(engine, {}).items():
        name = next((n for n in names if n in by_name), None)
        if name is not None:
            value = by_name[name]
            found[field_name] = round(value, 3) if field_name == "kv_cache_used" else int(value)
    return found


class EngineMonitor:
    """Asks the LLM server how it's doing every few seconds, for /health, /metrics and each turn's log.

    Its own view, not fusion's: how many requests it's decoding, how many wait, and how full
    its KV cache is. A queue that grows while the cache is full is the sign of too many callers.
    """

    def __init__(self, served_by: str, base_url: str, interval_s: float = 5.0):
        self.engine = SERVED_BY.get(served_by, served_by)
        self.display = served_by
        parts = urlsplit(base_url)
        root = f"{parts.scheme}://{parts.netloc}"
        self.health_url, self.metrics_url = root + "/health", root + "/metrics"
        self.interval_s = interval_s
        self.state: Dict[str, Any] = {"engine": self.engine, "url": base_url, "state": "unknown"}
        self._task = None

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.state)

    def turn_fields(self) -> Dict[str, Any]:
        """For a turn's llm.request: what the server was dealing with when this turn asked."""
        fields = {f"llm_server_{k}": self.state[k] for k in ("running", "waiting") if k in self.state}
        if "kv_cache_used" in self.state:
            fields["kv_cache_used"] = self.state["kv_cache_used"]
        return fields

    async def poll(self, client) -> None:
        import httpx

        from fusion_runtime.telemetry import telemetry

        before = self.state.get("state")
        try:
            healthy = (await client.get(self.health_url, timeout=3.0)).status_code == 200
        except httpx.HTTPError:
            healthy = False
        state: Dict[str, Any] = {"engine": self.engine, "url": self.state["url"], "state": "up" if healthy else "down"}
        if healthy and self.engine in ENGINE_SERIES:
            try:
                response = await client.get(self.metrics_url, timeout=3.0)
                if response.status_code == 200:
                    state.update(parse_engine_metrics(response.text, self.engine))
            except httpx.HTTPError:
                pass
        state["checked_at"] = round(time.time(), 1)
        self.state = state
        telemetry.emit("llm_server.state", level="debug", stage="llm",
                       **{k: v for k, v in state.items() if k != "checked_at"})
        if before != state["state"] and not (before == "unknown" and healthy):
            telemetry.emit("llm_server.up" if healthy else "llm_server.down",
                           level="info" if healthy else "error", stage="llm", engine=self.display, url=state["url"],
                           hint=None if healthy else "replies fail until it's back; frun up restarts a server it started")

    async def run(self) -> None:
        import asyncio

        import httpx

        async with httpx.AsyncClient() as client:
            while True:
                await self.poll(client)
                await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        import asyncio

        self._task = asyncio.get_running_loop().create_task(self.run())

    async def stop(self) -> None:
        import asyncio
        import contextlib

        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


class LogPrinter:
    """Where the server's lines and our notes about it go.

    Without a file, everything goes to the terminal, in the server's log format: a
    container's platform collects what's printed. With one (frun up --llm-log), the
    file gets every line, and the terminal only the start, warnings and errors:
    vLLM prints throughput every few seconds, and a first start hundreds of lines.

    Levels: "server" is a line from the engine; "starting" and "ready" bracket a
    start; the rest ("info", "warning", "error") are fusion's own notes.
    """

    def __init__(self, engine: Engine, json_format: bool, file: Optional[Path] = None, stream=None):
        self.engine = engine
        self.json_format = json_format
        self.stream = stream or sys.stderr
        self.path = Path(file).expanduser() if file else None
        self._file = None
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - open for the run, closed in close()
            except OSError as e:
                raise LaunchError(f"can't write the LLM server's log to {self.path}: {e}") from None
        self._starting = True
        self._lock = threading.Lock()

    def __call__(self, level: str, message: str) -> None:
        with self._lock:
            if level == "starting":
                self._starting = True
            if self._file is not None:
                stamp = datetime.now().isoformat(sep=" ", timespec="seconds")
                self._file.write(message + "\n" if level == "server" else f"--- {stamp} fusion: {message}\n")
            if level == "ready":
                self._starting = False
            if self._file is None or level != "server" or self._starting or _looks_like_trouble(message):
                self.stream.write(self._format(level, message) + "\n")
                self.stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _format(self, level: str, message: str) -> str:
        if self.json_format:
            record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                      "level": level if level in ("warning", "error") else "info",
                      "event": "llm_server.log" if level == "server" else "llm_server",
                      "engine": self.engine.name, "message": message}
            return json.dumps(record, separators=(",", ":"))
        return f"{self.engine.name:>7} {'|' if level == 'server' else '>'} {message}"


_TROUBLE = ("WARNING", "ERROR", "CRITICAL", "Traceback", "Exception", "Error:")


def _looks_like_trouble(line: str) -> bool:
    return any(word in line for word in _TROUBLE)


# -- helpers -----------------------------------------------------------------------------------


def _find_engine(engine: Engine, engine_env: Optional[str], url: str):
    """(the program to run, its environment's bin folder or None). Raises LaunchError with the install steps."""
    if engine_env:
        root = Path(engine_env).expanduser()
        bin_dir = root / ("Scripts" if os.name == "nt" else "bin")
        program = bin_dir / ("vllm" if engine.name == "vllm" else "python")
        if program.exists() and (engine.name == "vllm" or _imports(str(program), "sglang")):
            return str(program), str(bin_dir)
        raise LaunchError(f"{engine.display} isn't installed in {root}.\n"
                          f"  Install it there: {_install_hint(engine, root)}")
    if engine.name == "vllm":
        found = shutil.which("vllm")
        if found:
            return found, str(Path(found).parent)
    elif _imports(sys.executable, "sglang"):
        return sys.executable, None
    raise LaunchError(
        f"the agent runs its model on {engine.display}, which isn't installed where fusion can find it.\n"
        f"  It needs its own environment (it brings its own torch and CUDA):\n"
        f"    {_install_hint(engine, Path('~/' + engine.name + '-env'))}\n"
        f"  then tell fusion where it is: export {ENGINE_ENV_ENV}=~/{engine.name}-env\n"
        f"  (or start {engine.display} yourself at {url} and fusion will use it)")


def _install_hint(engine: Engine, root: Path) -> str:
    return f"python3 -m venv {root} && {root}/bin/pip install {engine.install}"


def _imports(python: str, module: str) -> bool:
    if python == sys.executable:
        import importlib.util
        return importlib.util.find_spec(module) is not None
    try:
        return subprocess.run([python, "-c", f"import importlib.util,sys; "
                                             f"sys.exit(importlib.util.find_spec({module!r}) is None)"],
                              capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _number(options: Mapping[str, Any], name: str, default: float, low: float, high: Optional[float]) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LaunchError(f"{name} must be a number, got {value!r}")
    if value < low or (high is not None and value > high):
        bounds = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise LaunchError(f"{name} must be {bounds}, got {value!r}")
    return value


def _is_local(host: Optional[str]) -> bool:
    if host in ("localhost", None, ""):
        return True
    from fusion_runtime.security import is_loopback
    return is_loopback(host)


def _is_secret(name: str) -> bool:
    """fusion's own keys stay out of the engine's environment: it has no use for them."""
    return name.startswith(("FUSION_ACCEPTED_KEYS", "FUSION_API_KEY"))


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _die_with_parent() -> None:  # pragma: no cover - runs in the child, Linux only
    """If fusion is killed outright (kill -9, OOM), take the server with it instead of leaving it on the GPU."""
    import ctypes

    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
