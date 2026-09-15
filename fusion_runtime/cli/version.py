"""`frun version`: show the installed version."""
import typer


def package_version() -> str:
    from fusion_runtime import __version__

    return __version__


def version() -> None:
    """Show the installed fusion-runtime version."""
    typer.echo(f"fusion-runtime {package_version()}")
