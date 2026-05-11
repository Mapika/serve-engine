from __future__ import annotations

import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("pull")
def pull(
    hf_repo: str = typer.Argument(
        ..., help="HuggingFace repo id, e.g. meta-llama/Llama-3.2-1B-Instruct"
    ),
    name: str = typer.Option(
        None, "--name", "-n", help="Local name for the model (default: repo basename)"
    ),
    revision: str = typer.Option("main", "--revision"),
):
    """Register a model with the daemon (download happens at first load)."""
    local_name = name or hf_repo.split("/")[-1].lower()
    body = {"name": local_name, "hf_repo": hf_repo, "revision": revision}
    try:
        result = asyncio.run(ipc.post(config.SOCK_PATH, "/admin/models", json=body))
    except RuntimeError as e:
        typer.echo(f"pull failed: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"registered: {result['name']} ({result['hf_repo']}@{result['revision']})")
