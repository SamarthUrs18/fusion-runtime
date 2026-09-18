"""`frun up`: start the voice server."""
import os
from enum import Enum

import typer

from pathlib import Path

from fusion_runtime.cli._common import Profile, short_path


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
    config: Profile = typer.Option(Profile.development, "--config", "-c", help="Which models and devices to use."),
    log_format: LogFormat = typer.Option(
        LogFormat.pretty, "--log-format",
        help="pretty: readable lines for a terminal. json: one JSON object per line, for log collectors and deployments.",
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
    llm_api_key_env: str = typer.Option(
        None, "--llm-api-key-env", help="Name of the environment variable holding the endpoint's API key "
                                        "(never the key itself). Also: FUSION_LLM_API_KEY_ENV.",
    ),
) -> None:
    """Start the voice server. Talk to it from another terminal with `frun talk`."""
    from fusion_runtime.cli._checks import missing_models, port_answers_over_ipv6, port_in_use
    from fusion_runtime.config import (
        INTERRUPT_AFTER_ENV, LLM_KEY_ENV_ENV, LLM_MODEL_ENV, LLM_URL_ENV, TURN_DETECTOR_ENV, TURN_WAIT_ENV, model_dir,
    )

    for variable, value in ((LLM_URL_ENV, llm_url), (LLM_MODEL_ENV, llm_model), (LLM_KEY_ENV_ENV, llm_api_key_env),
                            (TURN_DETECTOR_ENV, turn_detector),
                            (TURN_WAIT_ENV, str(turn_wait_ms) if turn_wait_ms is not None else None),
                            (INTERRUPT_AFTER_ENV, str(interrupt_after_ms) if interrupt_after_ms is not None else None)):
        if value:
            os.environ[variable] = value  # the server reads these at startup
    agent_file = None
    try:
        if agent:
            from fusion_runtime.agent import load_agent

            agent_file = Path(agent).expanduser().resolve()
            loaded = load_agent(agent_file)  # fail here, with a clear message, not inside the server
            profile = loaded.config()
        else:
            from fusion_runtime.cli._common import profile_config

            profile = profile_config(config)
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
        flag = "" if config is Profile.development else f" --config {config.value}"
        typer.echo(
            f"Error: the {config.value} profile needs models that aren't installed: {', '.join(missing)}\n"
            f"  (model directory: {short_path(model_dir())})\n"
            f"Run: frun models pull{flag}",
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

    if host in ("127.0.0.1", "localhost") and port_answers_over_ipv6(port):
        typer.echo(
            f"Warning: another program is listening on port {port} over IPv6. This server uses IPv4, but\n"
            f"  clients that use the name \"localhost\" may reach that program instead. Find it with:\n"
            f"  lsof -nP -iTCP:{port} -sTCP:LISTEN     (or use: frun up --port {port + 1})"
        )

    talk_hint = "frun talk"
    if (host, port) not in (("127.0.0.1", 8000), ("localhost", 8000), ("0.0.0.0", 8000)):
        talk_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        talk_hint = f"frun talk --url ws://{talk_host}:{port}/v1/voice/ws"
    where = f"agent {short_path(agent_file)}" if agent_file else f"{config.value} profile"
    typer.echo(f"Starting fusion-runtime ({where}) on http://{host}:{port}")
    turns = profile.turn_detection
    typer.echo(f"Turn detection: {turns.runtime or 'silence'}, agent answers after {turns.min_silence_ms} ms of silence"
               + (" (shorter or longer when the detector is sure)" if turns.runtime else "")
               + f"; talking over it for {turns.barge_in_min_speech_ms} ms interrupts")
    if llm.api_base or llm.provider.value == "openai":
        typer.echo(f"LLM: {llm.model} at {llm.api_base or 'https://api.openai.com/v1'}"
                   + (f" (key from ${llm.api_key_env})" if llm.api_key_env else ""))
    if host == "0.0.0.0":
        typer.echo("Warning: there's no authentication yet, so anyone who can reach this machine can use it.")
    typer.echo(f"Loading models. Once it says 'Models ready', run `{talk_hint}` in another terminal.\n")

    if log_content:
        typer.echo("Note: --log-content writes what users say, and the bot's replies, into the logs.")
    # read by the server at startup
    os.environ["FUSION_CONFIG"] = config.value
    if agent_file:
        os.environ["FUSION_AGENT"] = str(agent_file)
    else:
        os.environ.pop("FUSION_AGENT", None)
    os.environ["FUSION_LOG_FORMAT"] = log_format.value
    os.environ["FUSION_LOG_LEVEL"] = log_level.value
    os.environ["FUSION_LOG_CONTENT"] = "1" if log_content else "0"
    import uvicorn

    uvicorn.run("fusion_runtime.server:app", host=host, port=port, workers=1,
                reload=reload, reload_includes=[agent_file.name] if reload and agent_file else None,
                reload_dirs=[str(agent_file.parent)] if reload and agent_file else None,
                log_level="warning" if log_format is LogFormat.json else "info")
