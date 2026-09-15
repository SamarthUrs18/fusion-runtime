"""`frun version`: show the installed version."""
from importlib.metadata import PackageNotFoundError, version as distribution_version

import typer


def package_version() -> str:
    try:
        return distribution_version("fusion-runtime")
    except PackageNotFoundError:  # source checkout that was never pip-installed
        from fusion_runtime import __version__

        return __version__


def version() -> None:
    """Show the installed fusion-runtime version."""
    typer.echo(f"fusion-runtime {package_version()}")
