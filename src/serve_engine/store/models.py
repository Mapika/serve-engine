from __future__ import annotations

import sqlite3
from dataclasses import dataclass


class AlreadyExists(Exception):
    pass


@dataclass(frozen=True)
class Model:
    id: int
    name: str
    hf_repo: str
    revision: str
    local_path: str | None


def _row_to_model(row: sqlite3.Row) -> Model:
    return Model(
        id=row["id"],
        name=row["name"],
        hf_repo=row["hf_repo"],
        revision=row["revision"],
        local_path=row["local_path"],
    )


def add(
    conn: sqlite3.Connection,
    *,
    name: str,
    hf_repo: str,
    revision: str = "main",
) -> Model:
    try:
        cur = conn.execute(
            "INSERT INTO models (name, hf_repo, revision) VALUES (?, ?, ?)",
            (name, hf_repo, revision),
        )
    except sqlite3.IntegrityError as e:
        raise AlreadyExists(f"model {name!r} already exists") from e
    return Model(id=cur.lastrowid, name=name, hf_repo=hf_repo, revision=revision, local_path=None)


def get_by_name(conn: sqlite3.Connection, name: str) -> Model | None:
    row = conn.execute("SELECT * FROM models WHERE name=?", (name,)).fetchone()
    return _row_to_model(row) if row else None


def get_by_id(conn: sqlite3.Connection, model_id: int) -> Model | None:
    row = conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    return _row_to_model(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[Model]:
    rows = conn.execute("SELECT * FROM models ORDER BY id").fetchall()
    return [_row_to_model(r) for r in rows]


def set_local_path(conn: sqlite3.Connection, model_id: int, path: str) -> None:
    conn.execute("UPDATE models SET local_path=? WHERE id=?", (path, model_id))


def delete(conn: sqlite3.Connection, model_id: int) -> None:
    conn.execute("DELETE FROM models WHERE id=?", (model_id,))
