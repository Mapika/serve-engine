from __future__ import annotations

import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("pin")
def pin(model_name: str = typer.Argument(...)):
    """Mark the deployment for <model_name> as pinned (never auto-evicted)."""
    deps = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/deployments"))
    models = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/models"))
    model = next((m for m in models if m["name"] == model_name), None)
    if model is None:
        typer.echo(f"model {model_name!r} not registered", err=True)
        raise typer.Exit(1)
    ready = [
        d for d in deps
        if d.get("status") == "ready" and d.get("model_id") == model["id"]
    ]
    if not ready:
        typer.echo(f"no ready deployment for {model_name!r}", err=True)
        raise typer.Exit(1)
    dep_id = ready[0]["id"]
    asyncio.run(ipc.post(config.SOCK_PATH, f"/admin/deployments/{dep_id}/pin"))
    typer.echo(f"pinned deployment #{dep_id} ({model_name})")
