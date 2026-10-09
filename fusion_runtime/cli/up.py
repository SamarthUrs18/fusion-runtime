"""`frun up`: start the voice server."""
import os
import shlex
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlsplit

import typer

from fusion_runtime.cli._common import Profile, short_path

PUBLIC_URL_ENV = "FUSION_PUBLIC_URL"  # the address people open, when it isn't this machine's (a proxy, a domain)

class LogFormat(str, Enum):
    pretty = "pretty"
    json = "json"


class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


def up(
    agent: str = typer.Argument(
        None, metavar="[AGENT.PY]",
        help="An agent file: its prompt, models and turn settings. Without one, a profile of defaults is used.",
    ),
    host: str = typer.Option(
        "127.0.0.1", help="Address to listen on. 0.0.0.0 accepts connections from other machines."
    ),
    port: int = typer.Option(8000, help="Port to listen on."),
    config: Optional[Profile] = typer.Option(
        None, "--config", "-c",
        help="Which models and devices to use: development (laptops, on the CPU), production (an NVIDIA GPU) "
             "or hybrid (speech here, the LLM from a server). Defaults to $FUSION_CONFIG, then development.",
    ),
    log_format: Optional[LogFormat] = typer.Option(
        None, "--log-format",
        help="pretty: readable lines for a terminal. json: one JSON object per line, for log collectors and "
             "deployments. Defaults to $FUSION_LOG_FORMAT, then pretty.",
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.info, "--log-level", help="debug adds every partial transcript, audio chunk and client message.",
    ),
    log_content: bool = typer.Option(
        False, "--log-content",
        help="Include transcripts and replies in logs. Off by default: logs show only text lengths.",
    ),
    llm_url: str = typer.Option(
        None, "--llm-url", help="Use an OpenAI-compatible endpoint for the LLM, e.g. http://localhost:8080/v1 "
                                "(vLLM, llama-server, a hosted API). Also: FUSION_LLM_URL.",
    ),
    turn_detector: str = typer.Option(
        None, "--turn-detector",
        help="A turn detector model: a plugin name or module:Class. Default: end the turn after a pause. "
             "Also: FUSION_TURN_DETECTOR.",
    ),
    turn_wait_ms: int = typer.Option(
        None, "--turn-wait-ms", min=0,
        help="Silence (ms) before the agent answers. Longer suits people who pause mid-sentence. "
             "Default 500. Also: FUSION_TURN_WAIT_MS.",
    ),
    interrupt_after_ms: int = typer.Option(
        None, "--interrupt-after-ms", min=0,
        help="How long (ms) the caller must talk over the agent before it stops. Longer ignores coughs "
             "and echo; shorter stops sooner. Default 300. Also: FUSION_INTERRUPT_AFTER_MS.",
    ),
    llm_model: str = typer.Option(None, "--llm-model", help="The model's name on that endpoint. Also: FUSION_LLM_MODEL."),
    reload: bool = typer.Option(
        False, "--reload", help="Restart when the agent file changes. For development, not production.",
    ),
    llm_log: Path = typer.Option(
        None, "--llm-log", help="When frun up starts vLLM or SGLang: write its full log to this file, and keep "
                                "only its start, warnings and errors in the terminal. Also: FUSION_LLM_LOG.",
    ),
    llm_api_key_env: str = typer.Option(
        None, "--llm-api-key-env", help="Name of the environment variable holding the endpoint's API key "
                                        "(never the key itself). Also: FUSION_LLM_API_KEY_ENV.",
    ),
) -> None:
    """Start the voice server. Talk to it from another terminal with `frun talk`."""
    # The flag wins, then $FUSION_CONFIG, then development. Without this the flag's
    # default silently beat the variable and then overwrote it, so a container
    # started with FUSION_CONFIG=production quietly ran the development profile:
    # a 0.5B model on CPU instead of a 7B on the GPU, with nothing said about it.
    if config is None:
        wanted = (os.environ.get("FUSION_CONFIG") or "").strip()
        if wanted:
            try:
                config = Profile(wanted)
            except ValueError:
                raise typer.BadParameter(
                    f"FUSION_CONFIG={wanted!r} isn't a profile. "
                    f"Use one of: {', '.join(p.value for p in Profile)}",
                ) from None
        else:
            config = Profile.development

    # Same order as --config: the flag, then the variable. The images set FUSION_LOG_FORMAT=json,
    # and a flag default of pretty used to overwrite it, so containers logged for a terminal.
    if log_format is None:
        wanted = (os.environ.get("FUSION_LOG_FORMAT") or "").strip().lower()
        try:
            log_format = LogFormat(wanted) if wanted else LogFormat.pretty
        except ValueError:
            raise typer.BadParameter(
                f"FUSION_LOG_FORMAT={wanted!r} isn't a log format. Use one of: "
                f"{', '.join(f.value for f in LogFormat)}") from None

    from fusion_runtime.cli._checks import missing_models, port_answers_over_ipv6, port_in_use
    from fusion_runtime.config import (
        ACCEPTED_KEYS_ENV,
        AGENT_ENV,
        API_KEY_ENV,
        INTERRUPT_AFTER_ENV,
        LLM_KEY_ENV_ENV,
        LLM_MODEL_ENV,
        LLM_URL_ENV,
        TURN_DETECTOR_ENV,
        TURN_WAIT_ENV,
        model_dir,
        profile_label,
    )

    for variable, value in ((LLM_URL_ENV, llm_url), (LLM_MODEL_ENV, llm_model), (LLM_KEY_ENV_ENV, llm_api_key_env),
                            (TURN_DETECTOR_ENV, turn_detector),
                            (TURN_WAIT_ENV, str(turn_wait_ms) if turn_wait_ms is not None else None),
                            (INTERRUPT_AFTER_ENV, str(interrupt_after_ms) if interrupt_after_ms is not None else None)):
        if value:
            os.environ[variable] = value  # the server reads these at startup
    # A container has no command line to put an agent path on, so the environment
    # has to be able to name one. Without this, FUSION_AGENT was not only ignored
    # here, it was actively unset below.
    agent = agent or os.getenv(AGENT_ENV) or None
    agent_file = None
    try:
        if agent:
            from fusion_runtime.agent import load_agent

            agent_file = Path(agent).expanduser().resolve()
            loaded = load_agent(agent_file)  # fail here, with a clear message, not inside the server
            profile = loaded.config()
            has_tools = bool(loaded.tools)
        else:
            from fusion_runtime.cli._common import profile_config

            profile = profile_config(config)
            has_tools = False
        llm = profile.llm
    except ValueError as e:  # AgentError is a ValueError
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if llm.api_key_env and not os.getenv(llm.api_key_env):
        typer.echo(
            f"Error: the LLM endpoint needs an API key in ${llm.api_key_env}, which isn't set.\n"
            f"Run: export {llm.api_key_env}=<your key>   (or point --llm-url at a local server)",
            err=True,
        )
        raise typer.Exit(1)

    if agent_file is None:
        missing = missing_models(config)
    else:
        from fusion_runtime.catalog import entries_for_profile, is_installed
        missing = [e.id for e in entries_for_profile(profile) if not is_installed(e, model_dir())]
    if missing:
        if agent_file is not None:  # --config is ignored with an agent file, so don't name its profile
            needs, pull = f"{short_path(agent_file)} needs", f"frun models pull {short_path(agent_file)}"
        else:
            flag = "" if config is Profile.development else f" --config {config.value}"
            needs, pull = f"the {profile_label(config.value)} needs", f"frun models pull{flag}"
        typer.echo(
            f"Error: {needs} models that aren't installed: {', '.join(missing)}\n"
            f"  (model directory: {short_path(model_dir())})\n"
            f"Run: {pull}",
            err=True,
        )
        raise typer.Exit(1)

    try:
        in_use = port_in_use(host, port)
    except OSError as e:
        typer.echo(f"Error: can't listen on {host}:{port}: {e}", err=True)
        raise typer.Exit(1)
    if in_use:
        typer.echo(
            f"Error: port {port} is already in use (another `frun up` still running?).\n"
            f"Stop it, or use another port: frun up --port {port + 1}",
            err=True,
        )
        raise typer.Exit(1)

    # The LLM server fusion starts (vllm:, sglang:), planned now so a missing engine or a
    # clash with our own port stops the command before anything loads.
    from fusion_runtime import llm_server

    launch = None
    if llm_server.launchable(llm):
        try:
            launch = llm_server.plan(llm, has_tools=has_tools)
            if urlsplit(launch.base_url).port == port:
                raise llm_server.LaunchError(
                    f"{launch.engine.display} and this server would both use port {port}. "
                    f"Use another: frun up --port {port + 1}")
            if llm_server.already_serving(launch):
                typer.echo(f"LLM: {launch.engine.display} is already running at {launch.base_url} "
                           f"with {launch.model_name}; using it as it is")
                launch = None
            else:
                gpu = llm_server.read_gpu()
                for warning in llm_server.check_gpu_memory(launch, gpu, llm_server.speech_gb(profile)):
                    typer.echo(f"Warning: {warning}")
                if gpu is not None:
                    launch.notes.append(f"GPU {gpu.index}: {gpu.name}, {gpu.free_gb:.1f} of {gpu.total_gb:.1f} GB free; "
                                        f"{launch.engine.display} takes {launch.gpu_memory:.0%}")
        except llm_server.LaunchError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
    if os.getenv("FUSION_IMAGE") == "gpu" and not shutil.which("nvidia-smi"):
        # The NVIDIA container toolkit puts nvidia-smi in a container started with a GPU
        typer.echo("Warning: this is fusion's GPU image, but the container has no NVIDIA GPU, so speech runs\n"
                   "  slowly on the CPU and the vLLM container can't start. On a GPU machine, start it with\n"
                   "  --gpus all (or docker/docker-compose.gpu.yml). On a laptop, use the CPU image:\n"
                   "  ghcr.io/samarthurs18/fusion-runtime:cpu (docker/docker-compose.yml)")
    if _runs_in_process(llm) and shutil.which("nvidia-smi"):
        typer.echo("Note: the LLM runs inside this process, which suits about 4 callers at once. For more,\n"
                   "  run it on vLLM or SGLang: LLM(\"vllm:hf:<org>/<model>\"), and frun up starts it "
                   "(https://fusion-runtime.dev/docs#callers)")

    if host in ("127.0.0.1", "localhost") and port_answers_over_ipv6(port):
        typer.echo(
            f"Warning: another program is listening on port {port} over IPv6. This server uses IPv4, but\n"
            f"  clients that use the name \"localhost\" may reach that program instead. Find it with:\n"
            f"  lsof -nP -iTCP:{port} -sTCP:LISTEN     (or use: frun up --port {port + 1})"
        )

    # Keys are read here as well as in the server, so a mistake stops the command
    # rather than leaving a server running that isn't protected the way you think.
    from fusion_runtime.security import ConfigurationError, KeySet, is_loopback

    auth_error = None
    try:
        keys = KeySet.from_environment()
        auth_state = f"on ({len(keys)} key{'s' if len(keys) != 1 else ''})" if keys \
            else "off — this server answers on localhost only"
    except ConfigurationError as e:
        keys, auth_state = None, "misconfigured"
        auth_error = f"Error: {e}"
    reachable_from_elsewhere = not (is_loopback(host) or host == "localhost")
    if keys is not None and not keys and reachable_from_elsewhere:
        # The likely mistake, named: the two variables do different jobs, and only
        # one of them protects a server.
        has_client_key = bool(os.getenv(API_KEY_ENV))
        mixup = (f"\n  ({API_KEY_ENV} is set, but that is the key a client sends. A server needs "
                 f"{ACCEPTED_KEYS_ENV}.)" if has_client_key else "")
        auth_error = (
            f"Error: {host} is reachable from other machines and no keys are set, so anyone who can\n"
            f"  reach this port could use your GPU. Generate a key first:\n"
            f"    frun key new\n"
            f"  then put it in .env (or export it) as {ACCEPTED_KEYS_ENV}=<the key>.{mixup}"
        )

    if auth_error:
        typer.echo(auth_error, err=True)
        raise typer.Exit(1)

    talk_hint = "frun talk"
    if (host, port) not in (("127.0.0.1", 8000), ("localhost", 8000), ("0.0.0.0", 8000)):
        talk_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        talk_hint = f"frun talk --url ws://{talk_host}:{port}/v1/voice/ws"
    browser_host = "localhost" if host == "0.0.0.0" else host
    public = public_url(host, port)
    from fusion_runtime.cli import banner

    if banner.should_show(sys.stdout, json_logs=log_format is LogFormat.json):
        from fusion_runtime import __version__
        from fusion_runtime.agent import greeting_for

        rows, notes = _banner_rows(profile, llm, launch, agent_file, greeting_for(loaded if agent_file else None),
                                   config, auth_state, browser_host, port, talk_hint, public)
        banner.show(__version__, rows, notes)
    elif log_format is LogFormat.json:
        # A container's log is read by a machine: the same facts as one event, so every line is JSON.
        turns = profile.turn_detection
        if launch is not None:
            llm_where = {"llm_server": launch.engine.display, "llm_url": launch.base_url, "llm_started_here": True}
        elif llm.api_base or llm.provider.value == "openai":
            llm_where = {"llm_url": llm.api_base or "https://api.openai.com/v1"}
        else:
            llm_where = {"llm_in_process": True}
        _json_event("frun.up", level=log_level.value, agent=short_path(agent_file) if agent_file else None,
                    profile=None if agent_file else config.value, url=f"http://{host}:{port}", public_url=public,
                    llm=launch.model_name if launch is not None else llm.model, **llm_where,
                    llm_notes=list(launch.notes) if launch is not None and launch.notes else None,
                    turn_detector=turns.runtime or "silence", silence_ms=turns.min_silence_ms,
                    interrupt_after_ms=turns.barge_in_min_speech_ms, auth=auth_state, log_content=log_content)
    else:
        where = f"agent {short_path(agent_file)}" if agent_file else profile_label(config.value)
        typer.echo(f"Starting fusion-runtime ({where}) on http://{host}:{port}")
        turns = profile.turn_detection
        typer.echo(f"Turn detection: {turns.runtime or 'silence'}, agent answers after {turns.min_silence_ms} ms of silence"
                   + (" (shorter or longer when the detector is sure)" if turns.runtime else "")
                   + f"; talking over it for {turns.barge_in_min_speech_ms} ms interrupts")
        if launch is not None:
            typer.echo(f"LLM: {launch.model_name} on {launch.engine.display}, which this command starts "
                       f"at {launch.base_url} and stops on exit")
            for note in launch.notes:
                typer.echo(f"  ({note})")
        elif llm.api_base or llm.provider.value == "openai":
            typer.echo(f"LLM: {llm.model} at {llm.api_base or 'https://api.openai.com/v1'}"
                       + (f" (key from ${llm.api_key_env})" if llm.api_key_env else ""))
        typer.echo(f"Auth: {auth_state}")
        if public:
            typer.echo(f"Loading models. Once it says 'Models ready', get a browser link for {public} with\n"
                       f"  {_link_command(port, public)}\n  (or run `{talk_hint}` on this machine).\n")
        else:
            typer.echo(f"Loading models. Once it says 'Models ready', open http://{browser_host}:{port} in a browser "
                       f"and click Talk\n  (or run `{talk_hint}` in another terminal).\n")

    if log_content:
        note = "--log-content writes what users say, and the bot's replies, into the logs"
        if log_format is LogFormat.json:
            _json_event("logs.content_on", level=log_level.value, severity="warning", hint=note)
        else:
            typer.echo(f"Note: {note}.")
    # read by the server at startup
    os.environ["FUSION_CONFIG"] = config.value
    if agent_file:
        os.environ["FUSION_AGENT"] = str(agent_file)
    else:
        os.environ.pop("FUSION_AGENT", None)
    os.environ["FUSION_LOG_FORMAT"] = log_format.value
    os.environ["FUSION_LOG_LEVEL"] = log_level.value
    os.environ["FUSION_LOG_CONTENT"] = "1" if log_content else "0"
    from fusion_runtime.security.limits import Limits

    engine = None
    if launch is not None:
        # Before speech loads: the engine takes its share of GPU memory first, and speech the rest.
        log_file = llm_log or os.getenv(llm_server.LOG_FILE_ENV) or None
        try:
            printer = llm_server.LogPrinter(launch.engine, json_format=log_format is LogFormat.json, file=log_file)
        except llm_server.LaunchError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        if log_format is LogFormat.json:
            _json_event("llm_server.starting", level=log_level.value, server=launch.engine.display,
                        command=shlex.join(launch.command), log_file=short_path(printer.path) if printer.path else None)
        else:
            typer.echo(f"Starting {launch.engine.display}: {shlex.join(launch.command)}")
            if printer.path:
                typer.echo(f"{launch.engine.display}'s full log: {short_path(printer.path)}")
        engine = llm_server.LLMServer(launch, printer)
        try:
            engine.start()
        except llm_server.LaunchError as e:
            engine.stop()
            printer.close()
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        except KeyboardInterrupt:
            engine.stop()
            printer.close()
            raise typer.Exit(130)
    def stop_engine() -> None:
        if engine is not None:
            engine.stop()
            printer.close()

    # Refuse an oversized frame in the library, before it is buffered for us.
    settings = {"host": host, "port": port, "workers": 1, "ws_max_size": Limits.from_environment().max_message_bytes,
                "log_level": "warning" if log_format is LogFormat.json else "info"}
    if reload:
        settings.update(reload=True, reload_includes=[agent_file.name] if agent_file else None,
                        reload_dirs=[str(agent_file.parent)] if agent_file else None)
    try:
        _serve("fusion_runtime.server:app", on_abort=stop_engine, **settings)
    finally:
        stop_engine()


def _json_event(name: str, level: str, severity: str = "info", **fields) -> None:
    """One start-up event in the server's own JSON format, before the server has configured logging."""
    from fusion_runtime.telemetry import telemetry

    telemetry.configure(format="json", level=level)
    telemetry.emit(name, level=severity, stage="server", **fields)


def _serve(app_path: str, on_abort, **settings) -> None:
    """uvicorn.run, except that Ctrl+C while the models load stops at once.

    uvicorn only acts on Ctrl+C once start-up has finished, so pressing it while models load
    (minutes, the first time) did nothing until they were ready, then shut down with a traceback.
    Nothing is serving yet, so there's nothing to finish: stop the LLM server if we started one,
    say so in one line, and exit. os._exit, because the loading threads can't be interrupted and
    a normal exit would wait for them.
    """
    import uvicorn

    if settings.get("reload"):  # the reloader runs the server in a child process it restarts itself
        uvicorn.run(app_path, **settings)
        return

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            if self.started:
                return super().handle_exit(sig, frame)
            os.write(2, b"\nStopped while the models were loading.\n")  # not sys.stderr: os._exit won't flush it
            try:
                on_abort()
            finally:
                os._exit(130)

    Server(uvicorn.Config(app_path, **settings)).run()


def public_url(host: str, port: int, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Where people outside reach this server, when that can be known: FUSION_PUBLIC_URL (a domain
    behind Caddy, any proxy), or a Runpod pod's proxy. On a pod, localhost is the pod itself, so a
    localhost link can't be opened from a laptop.

    The pod's address is only guessed for a server listening on every interface: Runpod's proxy
    can't reach one bound to 127.0.0.1.
    """
    environ = os.environ if environ is None else environ
    if environ.get(PUBLIC_URL_ENV, "").strip():
        return environ[PUBLIC_URL_ENV].strip().rstrip("/")
    if host in ("0.0.0.0", "::") and environ.get("RUNPOD_POD_ID"):
        return f"https://{environ['RUNPOD_POD_ID']}-{port}.proxy.runpod.net"
    return None


def _link_command(port: int, public: str) -> str:
    return f"frun token --url http://127.0.0.1:{port} --public-url {public}"


def _banner_rows(profile, llm, launch, agent_file, greeting, config, auth_state, browser_host, port, talk_hint,
                 public=None):
    """What the welcome screen says: the same facts as the plain start-up lines, one row each."""
    from fusion_runtime.config import profile_label

    agent = f"{short_path(agent_file)}" if agent_file else f"{profile_label(config.value)}, no agent file"
    if greeting:
        agent += " · greets callers"
    if launch is not None:
        model = f"{launch.model_name} on {launch.engine.display} · starting it now"
    elif llm.api_base or llm.provider.value == "openai":
        model = f"{llm.model} at {llm.api_base or 'https://api.openai.com/v1'}"
    else:
        model = f"{_model_name('llm', llm)} · in this process"
    device = {"cuda": "on the GPU", "cpu": "on the CPU"}.get(getattr(profile.stt, "device", ""), "on the GPU if there is one")
    turns = profile.turn_detection
    rows = [
        ("agent", agent),
        ("llm", model),
        ("speech", f"{_model_name('stt', profile.stt)} · {_model_name('tts', profile.tts)} · {device}"),
        ("turns", (f"{turns.runtime} detector · " if turns.runtime else "")
                  + f"answers after {turns.min_silence_ms} ms of silence · "
                  f"{turns.barge_in_min_speech_ms} ms of talking over it interrupts"),
        ("auth", auth_state),
        ("talk", f"{public} · a one-time browser link: {_link_command(port, public)}" if public
                 else f"http://{browser_host}:{port} in a browser · or: {talk_hint}"),
    ]
    notes = list(launch.notes) if launch is not None else []
    notes.append("loading models; the link works once the log says 'Models ready'")
    return rows, notes


def _model_name(stage: str, stage_config) -> str:
    """The name the log uses for this model (catalog id, the name on its server), not its file path."""
    try:
        from fusion_runtime.resolver import resolve_stage_config

        resolved = resolve_stage_config(stage, stage_config)
        return resolved.catalog_id or resolved.spec.options.get("model_name") or Path(resolved.spec.model).name
    except Exception:  # the server reports a model it can't find with the full reason; this is only a label
        return Path(str(stage_config.model)).name


def _runs_in_process(llm) -> bool:
    return llm.runtime in (None, "llama_cpp") and llm.provider.value == "llama_cpp" and not llm.api_base
