from __future__ import annotations

import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("stop")
def stop():
    """Stop the currently active deployment."""
    asyncio.run(ipc.delete(config.SOCK_PATH, "/admin/deployments/current"))
    typer.echo("stopped")
