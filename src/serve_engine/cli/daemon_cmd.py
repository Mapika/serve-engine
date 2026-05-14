from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time

import typer

from serve_engine import config
from serve_engine.cli import app, ipc

daemon_app = typer.Typer(help="Daemon control")
app.add_typer(daemon_app, name="daemon")

PID_FILE = config.SERVE_DIR / "daemon.pid"


def spawn_daemon(
    host: str = config.DEFAULT_PUBLIC_HOST,
    port: int = config.DEFAULT_PUBLIC_PORT,
    *,
    timeout_s: float = 30.0,
    poll_s: float = 0.5,
) -> int:
    """Spawn the daemon process, write its PID, poll /healthz until ready.

    Returns the spawned PID. Raises TimeoutError if the daemon does not become
    healthy within `timeout_s`. Caller is responsible for ensuring no daemon
    is already running.
    """
    config.SERVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOGS_DIR / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "serve_engine.daemon", "--host", host, "--port", str(port)],
        stdout=open(log_path, "ab"),  # file must outlive this Popen call
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
            return proc.pid
        except Exception:
            time.sleep(poll_s)
    raise TimeoutError(f"daemon failed to become ready within {timeout_s:.0f}s")


def _is_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@daemon_app.command("start")
def daemon_start(
    host: str = typer.Option(config.DEFAULT_PUBLIC_HOST),
    port: int = typer.Option(config.DEFAULT_PUBLIC_PORT),
):
    """Start the daemon in the background."""
    if _is_running():
        typer.echo("daemon already running")
        raise typer.Exit(0)
    try:
        pid = spawn_daemon(host, port)
    except TimeoutError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    typer.echo(f"daemon started (pid {pid}) on http://{host}:{port}")


@daemon_app.command("stop")
def daemon_stop():
    """Stop the daemon."""
    if not _is_running():
        typer.echo("daemon not running")
        raise typer.Exit(0)
    pid = int(PID_FILE.read_text().strip())
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except OSError:
            break
    if PID_FILE.exists():
        PID_FILE.unlink()
    typer.echo("daemon stopped")


@daemon_app.command("status")
def daemon_status():
    """Show daemon status."""
    if not _is_running():
        typer.echo("daemon: not running")
        raise typer.Exit(1)
    pid = int(PID_FILE.read_text().strip())
    try:
        body = asyncio.run(ipc.get(config.SOCK_PATH, "/healthz"))
        typer.echo(f"daemon: running (pid {pid}), healthz: {body}")
    except Exception as e:
        typer.echo(f"daemon: pid file present (pid {pid}) but unhealthy: {e}", err=True)
        raise typer.Exit(2) from e
