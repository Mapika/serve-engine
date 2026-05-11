from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode, foreign keys, and autocommit.

    `isolation_level=None` puts the connection in autocommit mode: every DML
    statement commits immediately. This is necessary because the daemon
    shares a single long-lived connection across handlers that don't manage
    transactions explicitly — without autocommit, writes are lost on shutdown.
    Plan 02 will likely move to connection-per-request and re-introduce
    explicit transactions where useful.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def init_schema(conn: sqlite3.Connection) -> None:
    _ensure_migrations_table(conn)
    mig_dir = files("serve_engine.store.migrations")
    for entry in sorted(mig_dir.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".sql"):
            continue
        already = conn.execute(
            "SELECT 1 FROM _migrations WHERE filename=?", (entry.name,)
        ).fetchone()
        if already:
            continue
        sql = entry.read_text()
        # executescript implicitly COMMITs any open transaction at entry and
        # auto-commits the script's statements; we then record the migration
        # via a single autocommitted insert. The script itself is idempotent
        # (CREATE TABLE IF NOT EXISTS) so re-application on a partial failure
        # is safe.
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO _migrations (filename) VALUES (?)", (entry.name,)
        )
