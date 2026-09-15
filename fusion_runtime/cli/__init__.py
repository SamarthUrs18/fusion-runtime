"""frun: the fusion-runtime command-line tool.

One file per command in this package. Keep module-level imports light:
`frun --help` must never load torch, models or the pipeline. Heavy imports
go inside the command functions.
"""
from fusion_runtime.cli.app import app


def main() -> None:
    app()
