from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path


class _PrefetchedCursor:
    """In-memory cursor returned by LockedConnection.execute().

    The underlying sqlite3 cursor is fully consumed inside the connection
    lock; this wrapper hands rows back to callers without touching the
    connection again. Forwards `lastrowid` and `rowcount` since several
    store-layer functions read them after INSERT/DELETE.
    """

    __slots__ = ("_idx", "_rows", "lastrowid", "rowcount")

    def __init__(self, rows: list, lastrowid: int | None, rowcount: int) -> None:
        self._rows = rows
        self._idx = 0
        self.lastrowid = lastrowid
        self.rowcount = rowcount

    def fetchone(self):
        if self._idx >= len(self._rows):
            return None
        r = self._rows[self._idx]
        self._idx += 1
        return r

    def fetchall(self):
        rest = self._rows[self._idx:]
        self._idx = len(self._rows)
        return rest

    def __iter__(self):
        while self._idx < len(self._rows):
            yield self._rows[self._idx]
            self._idx += 1


class LockedConnection:
    """sqlite3.Connection wrapper that serializes access via an RLock.

    The daemon shares one long-lived connection across FastAPI sync deps that
    run in the anyio worker-thread pool. sqlite3 with `check_same_thread=False`
    permits cross-thread use but does not serialize - concurrent execute()
    calls corrupt cursor state, surfacing as 'bad parameter or other API
    misuse', empty rows, or NoneType subscript errors.

    Per-execute locking alone is insufficient: callers chain `.execute(...)
    .fetchone()`, and a different thread's execute() between those two calls
    can corrupt the live cursor. So execute() consumes the cursor fully
    inside the lock and returns an in-memory _PrefetchedCursor; subsequent
    fetchone()/fetchall() touch only Python lists, not the connection.

    For multi-statement atomic sections (SELECT-then-UPDATE), wrap the block
    in `with conn.locked():` so the pattern is one logical operation against
    other threads.

    The wrapper forwards less-common attribute access to the underlying
    connection, so callers can keep using `sqlite3.Connection`-typed signatures.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()

    def execute(self, *args, **kwargs) -> _PrefetchedCursor:
        with self._lock:
            cur = self._conn.execute(*args, **kwargs)
            rows = cur.fetchall()
            return _PrefetchedCursor(rows, cur.lastrowid, cur.rowcount)

    def executemany(self, *args, **kwargs) -> _PrefetchedCursor:
        with self._lock:
            cur = self._conn.executemany(*args, **kwargs)
            rows = cur.fetchall()
            return _PrefetchedCursor(rows, cur.lastrowid, cur.rowcount)

    def executescript(self, *args, **kwargs):
        with self._lock:
            return self._conn.executescript(*args, **kwargs)

    def commit(self):
        with self._lock:
            return self._conn.commit()

    def rollback(self):
        with self._lock:
            return self._conn.rollback()

    def close(self):
        with self._lock:
            return self._conn.close()

    @contextmanager
    def locked(self):
        """Hold the connection lock for a multi-statement atomic section."""
        with self._lock:
            yield self

    def __getattr__(self, name):
        # Fallback for less-common methods (interrupt, set_trace_callback, etc.)
        return getattr(self._conn, name)


def connect(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode, foreign keys, and autocommit.

    `isolation_level=None` puts the connection in autocommit mode: every DML
    statement commits immediately. This is necessary because the daemon
    shares a single long-lived connection across handlers that don't manage
    transactions explicitly - without autocommit, writes are lost on shutdown.
    Plan 02 will likely move to connection-per-request and re-introduce
    explicit transactions where useful.

    Returns a `LockedConnection` (duck-typed as `sqlite3.Connection`) that
    serializes access across the FastAPI worker-thread pool.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    raw.row_factory = sqlite3.Row
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("PRAGMA foreign_keys=ON")
    return LockedConnection(raw)  # type: ignore[return-value]


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
