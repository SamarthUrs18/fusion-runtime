"""`frun up`: start the voice server."""
import os

import typer

from fusion_runtime.cli._common import Profile, short_path


def up(
    host: str = typer.Option(
        "127.0.0.1", help="Address to listen on. 0.0.0.0 accepts connections from other machines."
    ),
    port: int = typer.Option(8000, help="Port to listen on."),
    config: Profile = typer.Option(Profile.development, "--config", "-c", help="Which models and devices to use."),
) -> None:
    """Start the voice server. Talk to it from another terminal with `frun talk`."""
    from fusion_runtime.cli._checks import missing_models, port_in_use
    from fusion_runtime.config import model_dir

    missing = missing_models(config)
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

    talk_hint = "frun talk"
    if (host, port) not in (("127.0.0.1", 8000), ("localhost", 8000), ("0.0.0.0", 8000)):
        talk_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        talk_hint = f"frun talk --url ws://{talk_host}:{port}/v1/voice/ws"
    typer.echo(f"Starting fusion-runtime ({config.value} profile) on http://{host}:{port}")
    if host == "0.0.0.0":
        typer.echo("Warning: there's no authentication yet, so anyone who can reach this machine can use it.")
    typer.echo(f"Loading models. Once it says 'Models ready', run `{talk_hint}` in another terminal.\n")

    os.environ["FUSION_CONFIG"] = config.value  # read by the server at startup
    import uvicorn

    uvicorn.run("fusion_runtime.server:app", host=host, port=port, workers=1)
