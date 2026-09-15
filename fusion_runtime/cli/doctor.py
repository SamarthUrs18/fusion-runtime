"""`frun doctor`: check this machine can run fusion-runtime, and say how to fix what can't."""
import typer

SYMBOLS = {"ok": "✓", "info": "–", "warn": "!", "fail": "✗"}


def doctor() -> None:
    """Check Python, libraries, GPU, models and audio. Changes nothing."""
    typer.echo("Checking your setup (loads a few libraries, takes a few seconds)...\n")
    from fusion_runtime.cli._checks import run_checks

    problems = warnings = 0
    for title, results in run_checks():
        typer.echo(title)
        for result in results:
            typer.echo(f"  {SYMBOLS[result.status]} {result.message}")
            if result.fix and result.status != "ok":
                typer.echo(f"    → {result.fix}")
            problems += result.status == "fail"
            warnings += result.status == "warn"
        typer.echo("")

    if problems:
        typer.echo(f"{problems} problem{'s' if problems != 1 else ''}, {warnings} warning{'s' if warnings != 1 else ''}. "
                   "Fix the ✗ lines first.")
        raise typer.Exit(1)
    if warnings:
        typer.echo(f"No problems, {warnings} warning{'s' if warnings != 1 else ''}.")
    else:
        typer.echo("All good.")
