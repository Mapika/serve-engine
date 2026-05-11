from __future__ import annotations

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="serve — single-node inference orchestrator",
)

from serve_engine.cli import (  # noqa: E402
    daemon_cmd,  # noqa: F401  registers `daemon` sub-app
    logs_cmd,  # noqa: F401  registers command
    ls_cmd,  # noqa: F401  registers command
    pin_cmd,  # noqa: F401  registers command
    ps_cmd,  # noqa: F401  registers command
    pull_cmd,  # noqa: F401  registers command
    run_cmd,  # noqa: F401  registers command
    stop_cmd,  # noqa: F401  registers command
    unpin_cmd,  # noqa: F401  registers command
)
