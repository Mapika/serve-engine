from __future__ import annotations

import asyncio
import time

import typer

from serve_engine import config
from serve_engine.cli import app, ipc
from serve_engine.doctor.runner import run_all, summarise


@app.command("setup")
def setup():
    """First-run wizard: doctor, start daemon, create admin key, print URL."""
    typer.echo("=== serve-engine setup ===")
    typer.echo()
    typer.echo("Step 1: environment diagnostic")
    results = run_all()
    _, _, fail = summarise(results)
    for r in results:
        glyph = {"ok": "✓", "warn": "!", "fail": "✗"}[r.status]
        typer.echo(f"  {glyph} {r.name}: {r.detail}")
    if fail:
        typer.secho(
            "\n✗ doctor reports failures; fix and re-run `serve setup`.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(1)

    typer.echo()
    typer.echo("Step 2: starting daemon")
    try:
        asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
        typer.echo("  daemon already running")
    except Exception:
        # subprocess-spawn via existing daemon_cmd path
        import subprocess
        import sys
        log_path = config.LOGS_DIR / "daemon.log"
        config.SERVE_DIR.mkdir(parents=True, exist_ok=True)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [sys.executable, "-m", "serve_engine.daemon"],
            stdout=open(log_path, "ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (config.SERVE_DIR / "daemon.pid").write_text(str(proc.pid))
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
                typer.echo(f"  daemon started (pid {proc.pid})")
                break
            except Exception:
                time.sleep(0.3)
        else:
            typer.secho("  daemon failed to come up; check logs", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)

    typer.echo()
    typer.echo("Step 3: create admin key")
    label = typer.prompt("Key label", default="admin")
    body = {"name": label, "tier": "admin"}
    result = asyncio.run(ipc.post(config.SOCK_PATH, "/admin/keys", json=body))
    typer.echo(f"  id:     {result['id']}")
    typer.echo(f"  secret: {result['secret']}")
    typer.echo()
    typer.echo("Save this secret — it won't be shown again.")
    typer.echo()
    typer.secho(
        f"Done. Open http://127.0.0.1:{config.DEFAULT_PUBLIC_PORT}/ and paste the secret.",
        fg=typer.colors.GREEN,
    )
