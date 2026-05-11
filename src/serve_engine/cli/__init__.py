from __future__ import annotations

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="serve — single-node inference orchestrator",
)

from serve_engine.cli import pull_cmd  # noqa: F401,E402  registers command
