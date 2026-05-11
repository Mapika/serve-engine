from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and foreign keys enabled.

    Uses sqlite3's default deferred-transaction mode: implicit transactions
    around DML, explicit commit via `with conn:` blocks. Callers that need
    autocommit must opt in per-statement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
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
        # `executescript` issues an implicit COMMIT before running, so we cannot
        # wrap *itself* in a transaction. Instead: apply the script, then
        # immediately record it in `_migrations` inside an explicit transaction
        # that we roll back if the INSERT fails. If the process dies between
        # the script and the INSERT, the next run re-applies; the SQL uses
        # `IF NOT EXISTS` so re-application is a no-op. This is the
        # documented best we can do given executescript's transaction semantics.
        conn.executescript(sql)
        with conn:
            conn.execute(
                "INSERT INTO _migrations (filename) VALUES (?)", (entry.name,)
            )
