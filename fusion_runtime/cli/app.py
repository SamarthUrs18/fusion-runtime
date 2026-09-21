"""The Typer app every `frun` command registers on."""
import typer

from fusion_runtime.cli.doctor import doctor
from fusion_runtime.cli.keys import key_app, keys_app, token
from fusion_runtime.cli.models import models_app
from fusion_runtime.cli.talk import talk
from fusion_runtime.cli.up import up
from fusion_runtime.cli.version import version

app = typer.Typer(
    name="frun",
    help="Run, talk to and deploy fusion-runtime, a self-hosted voice agent runtime.",
    no_args_is_help=True,
    add_completion=False,
    # Tracebacks must never print local variables: deploy commands will hold API keys.
    pretty_exceptions_show_locals=False,
)


def _show_version(value: bool) -> None:
    """`--version` is the first thing anyone types after installing a CLI.

    `frun version` did the job, but the flag returned a red "No such option"
    error, which is a poor first impression for something the user is entitled
    to assume works. Eager, so it answers before any argument parsing that
    could fail.
    """
    if value:
        from fusion_runtime.cli.version import package_version

        typer.echo(f"fusion-runtime {package_version()}")
        raise typer.Exit()


@app.callback()
def _root(
    show_version: bool = typer.Option(
        False, "--version", "-V", callback=_show_version, is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    # Settings and tokens can live in a .env file next to the project, so they
    # don't have to be exported by hand. The real environment still wins.
    from fusion_runtime.env import load_env_file

    load_env_file()


app.command()(up)
app.command()(talk)
app.add_typer(key_app, name="key")
app.add_typer(keys_app, name="keys")
app.command()(token)
app.add_typer(models_app, name="models")
app.command()(doctor)
app.command()(version)
