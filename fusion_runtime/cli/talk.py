"""`frun talk`: talk to a running server with your microphone."""
import os
import sys

import typer

# 127.0.0.1, not localhost: on macOS "localhost" resolves to IPv6 (::1) first, so a
# stray IPv6 server on the same port answers instead of `frun up` (which binds IPv4).
DEFAULT_URL = "ws://127.0.0.1:8000/v1/voice/ws"


def _audio_available() -> bool:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        typer.echo("Error: `frun talk` needs the audio extra. Run: pip install 'fusion-runtime[talk]'", err=True)
        return False
    except OSError as e:  # sounddevice installed, but the PortAudio system library isn't
        hint = "sudo apt install libportaudio2" if sys.platform.startswith("linux") else "install PortAudio"
        typer.echo(f"Error: can't load the audio library ({e}). Fix: {hint}", err=True)
        return False
    return True


def talk(
    url: str = typer.Option(DEFAULT_URL, "--url", help="WebSocket URL of the server."),
    no_aec: bool = typer.Option(
        False, "--no-aec",
        help="Turn off echo cancellation. The mic is then muted while the bot talks, so you can't interrupt.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print each turn's full timeline, not just the one-line summary.",
    ),
) -> None:
    """Talk to the server with your microphone and speakers. You can interrupt the bot."""
    if not _audio_available():
        raise typer.Exit(1)

    import asyncio

    import websockets

    from fusion_runtime.cli import _talk_client

    echo_cancellation = not no_aec and os.environ.get("FUSION_AEC", "1") not in ("0", "false", "False")
    typer.echo(f"Connecting to {url}")
    if sys.platform == "darwin":
        typer.echo("If the mic bar never moves: System Settings → Privacy & Security → Microphone → allow your terminal.")

    client = _talk_client.VoiceChatClient(uri=url, echo_cancellation=echo_cancellation, verbose=verbose)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        typer.echo("\nStopped.")
    except (ConnectionRefusedError, OSError) as e:
        typer.echo(
            f"\nError: can't reach {url} ({e.strerror or e}).\n"
            "Is the server running? Start it in another terminal with: frun up",
            err=True,
        )
        raise typer.Exit(1)
    except websockets.exceptions.InvalidURI:
        typer.echo(f"Error: {url} isn't a valid WebSocket URL (expected ws://host:port/v1/voice/ws).", err=True)
        raise typer.Exit(1)
    except websockets.exceptions.InvalidMessage as e:
        typer.echo(
            f"\nError: something is answering on {url}, but it isn't fusion-runtime ({e}).\n"
            "Another program is probably using that port. Check with: lsof -nP -iTCP:8000 -sTCP:LISTEN\n"
            "Then stop it, or start the server on another port: frun up --port 8001",
            err=True,
        )
        raise typer.Exit(1)
    except websockets.exceptions.InvalidStatus as e:
        typer.echo(
            f"Error: the server refused the connection (HTTP {e.response.status_code}). "
            "Check the URL ends in /v1/voice/ws.",
            err=True,
        )
        raise typer.Exit(1)
    except websockets.exceptions.ConnectionClosed:
        typer.echo("\nThe server closed the connection.", err=True)
        raise typer.Exit(1)
