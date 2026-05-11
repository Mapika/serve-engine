from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

Status = Literal["pending", "loading", "ready", "stopping", "stopped", "failed"]
ACTIVE_STATUSES: tuple[Status, ...] = ("pending", "loading", "ready")


@dataclass(frozen=True)
class Deployment:
    id: int
    model_id: int
    backend: str
    image_tag: str
    gpu_ids: list[int]
    tensor_parallel: int
    max_model_len: int | None
    dtype: str
    container_id: str | None
    container_name: str | None
    container_port: int | None
    status: Status
    last_error: str | None


def _row_to_dep(row: sqlite3.Row) -> Deployment:
    gpu_csv = row["gpu_ids"] or ""
    gpu_ids = [int(x) for x in gpu_csv.split(",") if x]
    return Deployment(
        id=row["id"],
        model_id=row["model_id"],
        backend=row["backend"],
        image_tag=row["image_tag"],
        gpu_ids=gpu_ids,
        tensor_parallel=row["tensor_parallel"],
        max_model_len=row["max_model_len"],
        dtype=row["dtype"],
        container_id=row["container_id"],
        container_name=row["container_name"],
        container_port=row["container_port"],
        status=row["status"],
        last_error=row["last_error"],
    )


def create(
    conn: sqlite3.Connection,
    *,
    model_id: int,
    backend: str,
    image_tag: str,
    gpu_ids: list[int],
    tensor_parallel: int,
    max_model_len: int | None,
    dtype: str,
) -> Deployment:
    gpu_csv = ",".join(str(g) for g in gpu_ids)
    cur = conn.execute(
        """
        INSERT INTO deployments
            (model_id, backend, image_tag, gpu_ids, tensor_parallel, max_model_len, dtype)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (model_id, backend, image_tag, gpu_csv, tensor_parallel, max_model_len, dtype),
    )
    return get_by_id(conn, cur.lastrowid)  # type: ignore[return-value]


def get_by_id(conn: sqlite3.Connection, dep_id: int) -> Deployment | None:
    row = conn.execute("SELECT * FROM deployments WHERE id=?", (dep_id,)).fetchone()
    return _row_to_dep(row) if row else None


def update_status(
    conn: sqlite3.Connection,
    dep_id: int,
    status: Status,
    *,
    last_error: str | None = None,
) -> None:
    if last_error is not None:
        conn.execute(
            "UPDATE deployments SET status=?, last_error=? WHERE id=?",
            (status, last_error, dep_id),
        )
    else:
        conn.execute("UPDATE deployments SET status=? WHERE id=?", (status, dep_id))


def set_container(
    conn: sqlite3.Connection,
    dep_id: int,
    *,
    container_id: str,
    container_name: str,
    container_port: int,
) -> None:
    conn.execute(
        """
        UPDATE deployments
        SET container_id=?, container_name=?, container_port=?, started_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (container_id, container_name, container_port, dep_id),
    )


def find_active(conn: sqlite3.Connection) -> Deployment | None:
    placeholders = ",".join(["?"] * len(ACTIVE_STATUSES))
    row = conn.execute(
        f"SELECT * FROM deployments WHERE status IN ({placeholders}) ORDER BY id DESC LIMIT 1",
        ACTIVE_STATUSES,
    ).fetchone()
    return _row_to_dep(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[Deployment]:
    rows = conn.execute("SELECT * FROM deployments ORDER BY id").fetchall()
    return [_row_to_dep(r) for r in rows]
