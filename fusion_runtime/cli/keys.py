"""`frun key new`, `frun keys list`, `frun token`: the key's life cycle.

A key is generated here, printed once and stored nowhere — the runtime reads it
from the environment. Losing one means generating another, not recovering it.
"""
import os

import typer

keys_app = typer.Typer(help="Look at the keys this machine is configured with.", no_args_is_help=True)
key_app = typer.Typer(help="Generate a key for a server to accept.", no_args_is_help=True)


@key_app.command("new")
def key_new(
    name: str = typer.Argument(None, metavar="[NAME]",
                               help="A label for you, such as web or mobile. It carries no permissions."),
) -> None:
    """Generate a key. Shown once — the runtime never stores it."""
    from fusion_runtime.config import ACCEPTED_KEYS_ENV, API_KEY_ENV
    from fusion_runtime.security import fingerprint, new_key

    entry = new_key(name)
    secret = entry.split(":", 1)[1] if name else entry
    typer.echo(entry)
    typer.echo()
    typer.echo("Where the server runs (.env next to your agent, or the pod's environment):")
    typer.echo(f"  {ACCEPTED_KEYS_ENV}={entry}")
    typer.echo()
    # Both, because on one machine you are usually both: the server accepts keys,
    # `frun talk` and `frun token` present one. The client side never takes the
    # label — that belongs to the list of keys a server accepts.
    typer.echo("Where a client runs — this machine, for `frun talk` and `frun token`:")
    typer.echo(f"  {API_KEY_ENV}={secret}")
    typer.echo()
    typer.echo(f"This is the only time it is shown. Fingerprint: {fingerprint(secret)}")


@keys_app.command("list")
def keys_list() -> None:
    """Show the keys this machine's environment configures — names and fingerprints, never the keys."""
    from fusion_runtime.config import ACCEPTED_KEYS_ENV, ACCEPTED_KEYS_FILE_ENV
    from fusion_runtime.security import ConfigurationError, KeySet

    try:
        keys = KeySet.from_environment()
    except ConfigurationError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if not keys:
        typer.echo(f"No keys configured ({ACCEPTED_KEYS_ENV} is empty), so a server started here "
                   "answers on localhost only.")
        typer.echo("Generate one: frun key new")
        return
    typer.echo(f"{'name':<12} {'fingerprint':<12}")
    for entry in keys.describe():
        typer.echo(f"{entry['name']:<12} {entry['fingerprint']:<12}")
    source = os.getenv(ACCEPTED_KEYS_FILE_ENV)
    typer.echo(f"\nFrom {ACCEPTED_KEYS_ENV}" + (f" and {source}" if source else ""))
    typer.echo("Live session counts per key are in the telemetry, not here: this command reads "
               "configuration, not a running server.")


def token(
    url: str = typer.Option("http://127.0.0.1:8000", "--url", help="The server to ask."),
    key: str = typer.Option(None, "--key", help="The key to present. Also: FUSION_API_KEY."),
) -> None:
    """Mint a session token and print a console URL you can open.

    Tokens live in the server's memory, so this asks the server for one rather
    than making it up locally.
    """
    import httpx

    from fusion_runtime.config import API_KEY_ENV
    from fusion_runtime.security import client_key

    key = key or client_key(url)
    if not key:
        typer.echo(f"Error: no key given. Pass --key, or set {API_KEY_ENV} (a .env file works).\n"
                   "Don't have one? Generate it with: frun key new", err=True)
        raise typer.Exit(1)
    base = url.rstrip("/")
    try:
        response = httpx.post(f"{base}/v1/sessions", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    except httpx.HTTPError as e:
        typer.echo(f"Error: can't reach {base} ({e}). Is the server running? Start it with: frun up", err=True)
        raise typer.Exit(1)
    if response.status_code == 401:
        detail = response.json().get("error", {})
        typer.echo(f"Error: {detail.get('message', 'the key was not accepted')}", err=True)
        if detail.get("fix"):
            typer.echo(f"  → {detail['fix']}", err=True)
        raise typer.Exit(1)
    response.raise_for_status()
    body = response.json()
    typer.echo(f"{base}/?token={body['token']}")
    typer.echo(f"\nOpen that in a browser within {body['expires_in']} seconds. It works once.")
