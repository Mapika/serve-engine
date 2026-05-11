from __future__ import annotations

import sqlite3


def record(
    conn: sqlite3.Connection,
    *,
    key_id: int,
    tokens_in: int,
    tokens_out: int,
    model_name: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO key_usage_events (key_id, tokens_in, tokens_out, model_name)
        VALUES (?, ?, ?, ?)
        """,
        (key_id, tokens_in, tokens_out, model_name),
    )


def totals_in_window(
    conn: sqlite3.Connection,
    *,
    key_id: int,
    window_s: int,
) -> tuple[int, int]:
    """Returns (request_count, total_tokens) for events with ts > now - window_s."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS n,
            COALESCE(SUM(tokens_in + tokens_out), 0) AS tok
        FROM key_usage_events
        WHERE key_id = ?
          AND ts > datetime('now', ?)
        """,
        (key_id, f"-{window_s} seconds"),
    ).fetchone()
    return int(row["n"]), int(row["tok"])


def purge_older_than_s(conn: sqlite3.Connection, *, max_age_s: float) -> int:
    """Delete usage events older than `max_age_s` seconds. Returns rows deleted."""
    cur = conn.execute(
        "DELETE FROM key_usage_events WHERE ts < datetime('now', ?)",
        (f"-{max_age_s} seconds",),
    )
    return cur.rowcount
