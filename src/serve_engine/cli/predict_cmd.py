"""serve predict — show predictor candidates and tick-loop stats.

Spec: docs/superpowers/specs/2026-05-13-predictive-layer-design.md §7.

v2.0 scope ships `serve predict` (current candidates) and
`serve predict --stats` (tick counters). `--replay` is a follow-up:
it needs a usage_events export format + an LRU baseline emulator that
hasn't landed.
"""
from __future__ import annotations

import asyncio

import typer

from serve_engine import config
from serve_engine.cli import app, ipc


@app.command("predict")
def predict(
    stats: bool = typer.Option(
        False, "--stats",
        help="Show tick-loop counters instead of the candidate list.",
    ),
):
    """Show the predictor's current top candidates (default) or its
    runtime stats (--stats)."""
    if stats:
        result = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/predictor/stats"))
        typer.echo(f"enabled:                       {result['enabled']}")
        if result["enabled"]:
            typer.echo(f"tick_interval_s:               {result['tick_interval_s']}")
            typer.echo(f"max_prewarm_per_tick:          {result['max_prewarm_per_tick']}")
        typer.echo(f"preloads_attempted:            {result['preloads_attempted']}")
        typer.echo(f"preloads_succeeded:            {result['preloads_succeeded']}")
        typer.echo(f"preloads_skipped_already_warm: {result['preloads_skipped_already_warm']}")
        typer.echo(f"preloads_skipped_no_deployment:{result['preloads_skipped_no_deployment']}")
        return

    rows = asyncio.run(ipc.get(config.SOCK_PATH, "/admin/predictor/candidates"))
    if not rows:
        typer.echo("no candidates (predictor has nothing to suggest right now)")
        return
    typer.echo(f"{'MODEL':<36}{'SCORE':<8}REASON")
    for r in rows:
        if r["adapter_name"]:
            model = f"{r['base_name']}:{r['adapter_name']}"
        else:
            model = r["base_name"]
        if len(model) > 35:
            model = model[:32] + "..."
        typer.echo(f"{model:<36}{r['score']:<8.3f}{r['reason']}")
