"""Snapshot index store.

Tracks the (key, hf_repo, engine, gpu_arch, quant, shape) → on-disk
snapshot mapping. The blob itself lives under the configured
SNAPSHOTS_DIR; this row is the metadata.

Companion design: docs/superpowers/specs/2026-05-13-snapshot-system-design.md
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    id: int
    key: str
    hf_repo: str
    revision: str
    engine: str
    engine_image: str
    gpu_arch: str
    quantization: str | None
    max_model_len: int
    dtype: str
    tensor_parallel: int
    target_concurrency: int
    local_path: str
    size_mb: int
    created_at: str
    last_used_at: str
    source_peer_id: str | None
    updated_at: str


def compute_snapshot_key(
    *,
    hf_repo: str,
    revision: str,
    engine: str,
    engine_image: str,
    gpu_arch: str,
    quantization: str | None,
    max_model_len: int,
    dtype: str,
    tensor_parallel: int,
    target_concurrency: int,
) -> str:
    """Content-addressable snapshot key.

    SHA-256 over a deterministic ordering of every field that affects
    the engine's loaded state. Any field change → different key →
    snapshot miss → cold load (which then writes a fresh snapshot).

    Notably, the key does NOT include any non-deterministic fields
    (timestamps, peer IDs, deployment IDs). Two boxes computing the
    key for the same plan inputs MUST get the same key — federation
    relies on this for snapshot deduplication.
    """
    payload = "|".join((
        hf_repo,
        revision,
        engine,
        engine_image,
        gpu_arch,
        quantization or "none",
        str(max_model_len),
        dtype,
        str(tensor_parallel),
        str(target_concurrency),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
    return Snapshot(
        id=row["id"],
        key=row["key"],
        hf_repo=row["hf_repo"],
        revision=row["revision"],
        engine=row["engine"],
        engine_image=row["engine_image"],
        gpu_arch=row["gpu_arch"],
        quantization=row["quantization"],
        max_model_len=row["max_model_len"],
        dtype=row["dtype"],
        tensor_parallel=row["tensor_parallel"],
        target_concurrency=row["target_concurrency"],
        local_path=row["local_path"],
        size_mb=row["size_mb"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        source_peer_id=row["source_peer_id"],
        updated_at=row["updated_at"],
    )


def add(
    conn: sqlite3.Connection,
    *,
    key: str,
    hf_repo: str,
    revision: str,
    engine: str,
    engine_image: str,
    gpu_arch: str,
    quantization: str | None,
    max_model_len: int,
    dtype: str,
    tensor_parallel: int,
    target_concurrency: int,
    local_path: str,
    size_mb: int,
) -> Snapshot:
    """Register a snapshot. Raises sqlite3.IntegrityError on duplicate key
    (UNIQUE constraint) — caller should treat duplicates as already-present
    and call `get_by_key` to retrieve."""
    cur = conn.execute(
        """
        INSERT INTO snapshots (
            key, hf_repo, revision, engine, engine_image, gpu_arch,
            quantization, max_model_len, dtype, tensor_parallel,
            target_concurrency, local_path, size_mb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            key, hf_repo, revision, engine, engine_image, gpu_arch,
            quantization, max_model_len, dtype, tensor_parallel,
            target_concurrency, local_path, size_mb,
        ),
    )
    fetched = get_by_id(conn, cur.lastrowid)
    assert fetched is not None
    return fetched


def get_by_key(conn: sqlite3.Connection, key: str) -> Snapshot | None:
    row = conn.execute(
        "SELECT * FROM snapshots WHERE key=?", (key,),
    ).fetchone()
    return _row_to_snapshot(row) if row else None


def get_by_id(conn: sqlite3.Connection, snap_id: int) -> Snapshot | None:
    row = conn.execute(
        "SELECT * FROM snapshots WHERE id=?", (snap_id,),
    ).fetchone()
    return _row_to_snapshot(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[Snapshot]:
    rows = conn.execute(
        "SELECT * FROM snapshots ORDER BY last_used_at DESC",
    ).fetchall()
    return [_row_to_snapshot(r) for r in rows]


def list_for_engine(conn: sqlite3.Connection, engine: str) -> list[Snapshot]:
    rows = conn.execute(
        "SELECT * FROM snapshots WHERE engine=? ORDER BY last_used_at DESC",
        (engine,),
    ).fetchall()
    return [_row_to_snapshot(r) for r in rows]


def touch(conn: sqlite3.Connection, snap_id: int) -> None:
    """Bump last_used_at after a successful warm-restore. Drives LRU
    eviction in the GC layer."""
    conn.execute(
        "UPDATE snapshots SET last_used_at=CURRENT_TIMESTAMP WHERE id=?",
        (snap_id,),
    )


def delete(conn: sqlite3.Connection, snap_id: int) -> None:
    """Remove the index row. Caller is responsible for deleting the
    on-disk blob at `local_path` separately."""
    conn.execute("DELETE FROM snapshots WHERE id=?", (snap_id,))


def total_size_mb(conn: sqlite3.Connection) -> int:
    """For the disk-quota GC: total MB across all snapshot rows."""
    row = conn.execute(
        "SELECT COALESCE(SUM(size_mb), 0) AS total FROM snapshots",
    ).fetchone()
    return int(row["total"]) if row and row["total"] is not None else 0


def lru_for_engine_model(
    conn: sqlite3.Connection,
    engine: str,
    hf_repo: str,
    *,
    keep_n: int,
) -> list[Snapshot]:
    """Return snapshots for (engine, hf_repo) beyond the keep_n most
    recently used — these are eviction candidates for the
    'keep latest N per (engine, model)' GC policy."""
    rows = conn.execute(
        """
        SELECT * FROM snapshots
        WHERE engine=? AND hf_repo=?
        ORDER BY last_used_at DESC
        """,
        (engine, hf_repo),
    ).fetchall()
    return [_row_to_snapshot(r) for r in rows[keep_n:]]
