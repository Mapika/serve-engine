from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiKey:
    id: int
    name: str
    prefix: str
    tier: str
    rpm_override: int | None
    tpm_override: int | None
    rpd_override: int | None
    tpd_override: int | None
    rph_override: int | None
    tph_override: int | None
    rpw_override: int | None
    tpw_override: int | None
    revoked_at: str | None


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _row_to_key(row: sqlite3.Row) -> ApiKey:
    return ApiKey(
        id=row["id"],
        name=row["name"],
        prefix=row["prefix"],
        tier=row["tier"],
        rpm_override=row["rpm_override"],
        tpm_override=row["tpm_override"],
        rpd_override=row["rpd_override"],
        tpd_override=row["tpd_override"],
        rph_override=row["rph_override"],
        tph_override=row["tph_override"],
        rpw_override=row["rpw_override"],
        tpw_override=row["tpw_override"],
        revoked_at=row["revoked_at"],
    )


def create(
    conn: sqlite3.Connection,
    *,
    name: str,
    tier: str = "standard",
    rpm_override: int | None = None,
    tpm_override: int | None = None,
    rpd_override: int | None = None,
    tpd_override: int | None = None,
    rph_override: int | None = None,
    tph_override: int | None = None,
    rpw_override: int | None = None,
    tpw_override: int | None = None,
) -> tuple[str, ApiKey]:
    """Generate a new key. Returns (secret, ApiKey). The secret is only available here."""
    body = secrets.token_urlsafe(32)
    secret = f"sk-{body}"
    prefix = secret[:12]
    key_hash = _hash(secret)
    cur = conn.execute(
        """
        INSERT INTO api_keys
            (name, prefix, key_hash, tier,
             rpm_override, tpm_override, rpd_override, tpd_override,
             rph_override, tph_override, rpw_override, tpw_override)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name, prefix, key_hash, tier,
            rpm_override, tpm_override, rpd_override, tpd_override,
            rph_override, tph_override, rpw_override, tpw_override,
        ),
    )
    fetched = get_by_id(conn, cur.lastrowid)
    assert fetched is not None
    return secret, fetched


def get_by_id(conn: sqlite3.Connection, key_id: int) -> ApiKey | None:
    row = conn.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
    return _row_to_key(row) if row else None


def verify(conn: sqlite3.Connection, secret: str) -> ApiKey | None:
    """Look up a key by secret; returns None if missing or revoked."""
    candidate_hash = _hash(secret)
    row = conn.execute(
        "SELECT * FROM api_keys WHERE key_hash=? AND revoked_at IS NULL",
        (candidate_hash,),
    ).fetchone()
    if row is None:
        return None
    if not hmac.compare_digest(row["key_hash"], candidate_hash):
        return None
    conn.execute(
        "UPDATE api_keys SET last_used_at=CURRENT_TIMESTAMP WHERE id=?",
        (row["id"],),
    )
    return _row_to_key(row)


def list_all(conn: sqlite3.Connection) -> list[ApiKey]:
    rows = conn.execute(
        "SELECT * FROM api_keys ORDER BY id"
    ).fetchall()
    return [_row_to_key(r) for r in rows]


def revoke(conn: sqlite3.Connection, key_id: int) -> None:
    conn.execute(
        "UPDATE api_keys SET revoked_at=CURRENT_TIMESTAMP WHERE id=?",
        (key_id,),
    )


def count_active(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM api_keys WHERE revoked_at IS NULL"
    ).fetchone()
    return int(row["n"])
