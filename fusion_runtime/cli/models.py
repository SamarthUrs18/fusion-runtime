"""`frun models list` and `frun models pull`."""
from typing import List, Optional

import typer

from fusion_runtime.cli._common import Profile, profile_config, short_path

models_app = typer.Typer(help="Download and inspect models.", no_args_is_help=True)


@models_app.command("list")
def list_models() -> None:
    """Show catalog models, whether they're installed, and which profile uses them."""
    from fusion_runtime.catalog import entries_for_profile, format_size, is_installed, load_catalog
    from fusion_runtime.config import model_dir, model_dir_source

    root = model_dir()
    catalog = load_catalog()
    used_by = {model_id: [] for model_id in catalog}
    for profile in Profile:
        for entry in entries_for_profile(profile_config(profile), catalog):
            used_by[entry.id].append(profile.value)

    typer.echo(f"Model directory: {short_path(root)}  ({model_dir_source()})\n")
    rows = [("STAGE", "MODEL", "SIZE", "STATUS", "USED BY")]
    for entry in catalog.values():
        status = "installed" if is_installed(entry, root) else "missing"
        rows.append((entry.stage, entry.id, format_size(entry.total_bytes), status, ", ".join(used_by[entry.id]) or "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        typer.echo("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())

    for profile in Profile:
        missing = [e.id for e in entries_for_profile(profile_config(profile), catalog) if not is_installed(e, root)]
        if missing:
            flag = "" if profile is Profile.development else f" --config {profile.value}"
            typer.echo(f"\n{profile.value} is missing {', '.join(missing)}. Run: frun models pull{flag}")


@models_app.command("pull")
def pull(
    ids: Optional[List[str]] = typer.Argument(
        None, help="Catalog model IDs (see `frun models list`). Default: everything the profile needs."
    ),
    config: Profile = typer.Option(Profile.development, "--config", "-c", help="Profile whose models to pull."),
    whisper: bool = typer.Option(False, "--whisper", help="Only the profile's speech-to-text model."),
    llm: bool = typer.Option(False, "--llm", help="Only the profile's LLM."),
    kokoro: bool = typer.Option(False, "--kokoro", help="Only the profile's text-to-speech model."),
    vad: bool = typer.Option(False, "--vad", help="Only the voice activity detector."),
    force: bool = typer.Option(False, "--force", help="Download again even if already installed."),
) -> None:
    """Download models into the model directory."""
    from fusion_runtime.catalog import (
        UnknownModelError,
        entries_for_profile,
        format_size,
        get_entries,
        is_installed,
        load_catalog,
    )
    from fusion_runtime.catalog.download import DownloadError, bytes_to_download, check_disk_space
    from fusion_runtime.catalog.download import pull as pull_entry
    from fusion_runtime.config import model_dir

    catalog = load_catalog()
    try:
        chosen = get_entries(ids or [], catalog)
    except UnknownModelError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    stages = {s for s, on in (("stt", whisper), ("llm", llm), ("tts", kokoro), ("vad", vad)) if on}
    if stages or not ids:
        profile_entries = entries_for_profile(profile_config(config), catalog)
        chosen += [e for e in profile_entries if not stages or e.stage in stages]
        for stage in sorted(stages - {e.stage for e in profile_entries}):
            typer.echo(f"The {config.value} profile has no local {stage} model to pull.", err=True)
    chosen = list({e.id: e for e in chosen}.values())  # de-duplicate, keep order

    root = model_dir()
    todo = [e for e in chosen if force or not is_installed(e, root)]
    typer.echo(f"Model directory: {short_path(root)}")
    for entry in chosen:
        if entry not in todo:
            typer.echo(f"✓ {entry.id} already installed")
    if not todo:
        return

    needed = sum(bytes_to_download(e, root, force) for e in todo)
    try:
        check_disk_space(needed, root)
    except DownloadError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Downloading {len(todo)} model{'s' if len(todo) > 1 else ''} ({format_size(needed)})")
    for entry in todo:
        typer.echo(f"↓ {entry.id}  {entry.description} [{entry.license}]")
        try:
            pull_entry(entry, root, force=force, log=typer.echo)
        except DownloadError as e:
            typer.echo(f"✗ {e}", err=True)
            typer.echo("Check your internet connection and run the same command again.", err=True)
            raise typer.Exit(1)
        typer.echo(f"✓ {entry.id} ready")
