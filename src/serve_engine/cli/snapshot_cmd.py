"""serve snapshot — list / remove / GC engine warm-restore snapshots.

Snapshots are the bind-mounted torch.compile inductor caches that let a
re-deployment with the same shape skip compilation. Spec:
docs/superpowers/specs/2026-05-13-snapshot-system-design.md §8.
"""
from __future__ import annotations

import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc

snapshot_app = typer.Typer(help="Engine snapshot management")
app.add_typer(snapshot_app, name="snapshot")


@snapshot_app.command("ls")
def snapshot_ls():
    """List all registered snapshots, newest-used first."""
    rows = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/snapshots"))
    if not rows:
        typer.echo("no snapshots registered")
        return
    typer.echo(
        f"{'KEY':<10}{'ENGINE':<8}{'MODEL':<36}"
        f"{'GPU_ARCH':<10}{'SIZE_MB':<10}{'LAST_USED'}"
    )
    for r in rows:
        # Compact (hf_repo[-30:]) keeps the table readable for long names.
        model = r["hf_repo"]
        if len(model) > 35:
            model = "…" + model[-34:]
        typer.echo(
            f"{r['key_prefix']:<10}{r['engine']:<8}{model:<36}"
            f"{r['gpu_arch']:<10}{r['size_mb']:<10}{r['last_used_at']}"
        )


@snapshot_app.command("rm")
def snapshot_rm(
    key: str = typer.Argument(
        ...,
        help="Snapshot key (full or `all` to wipe everything).",
    ),
):
    """Remove a snapshot. Pass `all` to wipe every registered snapshot."""
    try:
        asyncio.run(ipc.delete(config.SOCK_PATH, f"/admin/snapshots/{key}"))
        typer.echo(f"removed: {key}")
    except RuntimeError as e:
        typer.echo(f"remove failed: {e}", err=True)
        raise typer.Exit(1) from e


@snapshot_app.command("gc")
def snapshot_gc(
    keep_last: int = typer.Option(
        2, "--keep-last",
        help="Keep this many most-recently-used per (engine, hf_repo). "
             "0 disables the per-model rule.",
    ),
    max_disk_gb: float = typer.Option(
        None, "--max-disk-gb",
        help="Cap total snapshot disk usage; LRU-evict globally to fit.",
    ),
):
    """Evict old snapshots. The per-model rule trims accumulated state for
    the same engine+model; --max-disk-gb adds a global cap on top."""
    body = {"keep_last_per_model": keep_last}
    if max_disk_gb is not None:
        body["max_disk_gb"] = max_disk_gb
    result = asyncio.run(
        ipc.post(config.SOCK_PATH, "/admin/snapshots/gc", json=body),
    )
    typer.echo(
        f"removed {result['removed']} snapshot(s); "
        f"{result['remaining_mb']} MB still cached"
    )
