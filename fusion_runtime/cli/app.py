"""The Typer app every `frun` command registers on."""
import typer

from fusion_runtime.cli.version import version

app = typer.Typer(
    name="frun",
    help="Run, talk to and deploy fusion-runtime, a self-hosted voice agent runtime.",
    no_args_is_help=True,
    add_completion=False,
    # Tracebacks must never print local variables: deploy commands will hold API keys.
    pretty_exceptions_show_locals=False,
)


@app.callback()
def _root() -> None:
    # Having a callback keeps `frun <command>` as a command group even while
    # there's only one command (Typer otherwise runs a lone command as `frun`).
    pass


app.command()(version)
